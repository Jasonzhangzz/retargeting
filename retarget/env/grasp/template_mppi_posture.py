"""Grasp-family articulation and finger-order constraints, independent of 1x q."""
import math
import mujoco
import numpy as np
import torch
from .template import WujiContactKinematics

POSTURE_TYPES = {6,7,9,12,13,16,18,20,22,23,24,25,27,28,29,30,32,33}

class GraspPosture:
    def __init__(self, model, template, number, device, settings):
        self.settings=settings
        self.number=number
        self.device=torch.device(device)
        names=['right_palm_link']
        local=[[0.,0.,0.]]
        for finger,label in zip(range(2,6),('index','middle','ring','pinky')):
            names += [f'right_finger{finger}_link3',f'right_finger{finger}_link4',f'right_finger{finger}_link4']
            local += [[0.,0.,0.],[0.,0.,0.],model.site(f'right_{label}_tip').pos.tolist()]
        self.fk=WujiContactKinematics(model,[model.body(n).id for n in names],device)
        self.local=self.fk.tensor(local)
        neutral=mujoco.MjData(model);mujoco.mj_kinematics(model,neutral)
        axis=neutral.xpos[model.body('right_finger5_link2').id]-neutral.xpos[model.body('right_finger2_link2').id]
        palm=model.body('right_palm_link').id
        axis=neutral.xmat[palm].reshape(3,3).T@axis
        self.axis=self.fk.tensor(axis/np.linalg.norm(axis))
        self.margin=max(settings.finger_order_margin_m,.010 if number==29 else 0.)
        self.active={int(n[len('right_finger')]) for n in template.body_names if n.startswith('right_finger')}
        self.idle=[f for f in range(2,6) if f not in self.active]
        self.template_angles=np.asarray(template.finger_qpos).reshape(5,4)

    def bounds(self,low,high):
        # Normal opening remains free. Small MCP extension is allowed; PIP/DIP
        # cannot reverse to cancel an exaggerated bend in the neighboring joint.
        for finger in range(2,6):
            start=6+4*(finger-1)
            low[start]=max(float(low[start]),math.radians(-5))
            low[start+2:start+4]=low[start+2:start+4].clamp_min(0.)
            high[start+2]=min(float(high[start+2]),math.radians(self.settings.pip_flexion_cap_deg))
            if finger in self.idle:
                low[start+1]=max(float(low[start+1]),math.radians(-10))
                high[start+1]=min(float(high[start+1]),math.radians(10))
                # Park unused fingers in a broad open or folded band, never at
                # a particular seed angle. Contact fingers have no such band.
                if min(self.template_angles[finger-1,[0,2,3]]) > math.radians(55):
                    for joint,angle in ((0,45),(2,45),(3,30)):
                        low[start+joint]=max(float(low[start+joint]),math.radians(angle))
        # Thumb opposition is retained, including intended extension in palmar
        # and inferior grasps. Only excessive extension beyond the template's
        # extension envelope is removed; flexion remains free to the hardware limit.
        for joint in (2,3):
            low[6+joint]=max(float(low[6+joint]),min(self.template_angles[0,joint],0.)-math.radians(5))

    def single(self,q):
        p,r=self.fk.frames(q.to(self.fk.dtype))
        points=p+(r@self.local[...,None])[...,0]
        fingers=points[1:].reshape(4,3,3)
        axis=r[0]@self.axis
        gaps=((fingers[1:]-fingers[:-1])*axis).sum(-1)
        curl=q[6:].reshape(5,4)[1:]
        difference=(curl[:,2]-curl[:,3]).abs()
        curvature=(difference-math.radians(60)).clamp_min(0)/math.radians(30)
        order=(self.margin-gaps).clamp_min(0)/self.settings.finger_order_scale_m
        residual=torch.cat((order.flatten()*math.sqrt(self.settings.finger_order_weight),curvature))
        return residual,gaps

    def batch(self,q):
        residual,gaps=torch.vmap(self.single)(q)
        violation=(self.margin-gaps).clamp_min(0).amax((1,2))
        return residual.square().sum(-1).to(q.dtype),violation.to(q.dtype),gaps

    def metadata(self):
        return dict(policy='grasp_articulation_and_lateral_order_v2',
            active_fingers=sorted(self.active),idle_fingers=self.idle,
            idle_policy='broad folded/open band; no fixed angles',
            order_margin_m=self.margin,order_scope='adjacent index/middle/ring/pinky PIP, DIP and tips in palm frame',
            reference_1x_pose_used=False,template_thumb_extension_envelope_used=True)
