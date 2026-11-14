"""Current-frame Robokit retargeting on the exact native WUJI2 kinematics."""
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation
from retarget.env.grasp.hand import hand_tree
from retarget.env.grasp.pose_gpu import palm_frame


class RobokitInitializer:
    def __init__(self, surface, adapter, *, use_history=True):
        from robokit.robo import Robot
        from robokit.robo.robot_spec import RobotSpec
        from robokit.helpers.hand_retargeting import HandSpec, HandRetargeterOnline, HandRetargeterOnlineConfig
        from robokit.utils.ee_coord_utils import HandCoordinateSpec
        from robokit.utils.warp_utils import wp_vec7

        self.adapter=adapter;self.device=adapter.device;self.wp_vec7=wp_vec7
        tree=hand_tree('wuji2');links={'wrist':'right_palm_link'};chains=[]
        for f,name in enumerate(('thumb','index','middle','ring','pinky')):
            labels=('cmc','mcp','ip') if f==0 else ('mcp','pip','dip')
            for k,label in enumerate(labels,2):links[f'{name}_{label}']=f'right_finger{f+1}_link{k}'
            tip=f'right_{name}_retarget_tip';links[f'{name}_tip']=tip
            body=tree.find(f".//body[@name='right_finger{f+1}_link4']")
            ET.SubElement(body,'body',name=tip,pos=' '.join(map(str,adapter.site_local[f].cpu().tolist())))
            chains.append(tuple(f'{name}_{label}' for label in (*labels,'tip')))
        target_names=('wrist',)+tuple(n for chain in chains for n in chain)
        # Re-root the existing MJCF at its palm. Only the six virtual wrist
        # joints are removed; axes, origins and all 20 native joints are kept.
        robot_spec=RobotSpec.parse(ET.tostring(tree,encoding='unicode'),base_link_name='right_palm_link',load_meshes=True)
        # Keep mechanical connections out of the collision objective, while
        # checking all branches (including the ring/pinky branches).
        native_bodies={b:i for i,b in enumerate(surface.body_ids)};link_owner=[]
        for name in robot_spec.link_names:
            if name.endswith('_retarget_tip'):
                owner=4*(('thumb','index','middle','ring','pinky').index(name.split('_')[1])+1)
            else:
                body=surface.model.body(name.removesuffix('_anchor')).id
                while body not in native_bodies:body=int(surface.model.body_parentid[body])
                owner=native_bodies[body]
            link_owner.append(owner)
        ignored=[]
        for i,a in enumerate(link_owner):
            for k,b in enumerate(link_owner[i+1:],i+1):
                cross=a>0 and b>0 and (a-1)//4!=(b-1)//4
                palm=min(a,b)==0 and max(a,b)>0 and (max(a,b)-1)%4>=2
                distant=a>0 and b>0 and (a-1)//4==(b-1)//4 and abs(a-b)>2
                if not (cross or palm or distant):ignored.append((i,k))
        robot_spec.self_collision_ignored_pairs=np.asarray(ignored,dtype=np.int32).reshape(-1,2)
        self.robot=Robot(robot_spec)
        native_names=[surface.model.joint(i).name for i in range(6,26)]
        names=list(self.robot.spec.actuated_joint_names)
        if len(names)!=20 or set(names)!=set(native_names):raise ValueError('Robokit/native WUJI2 joint names disagree')
        self.to_native=np.array([names.index(n) for n in native_names])
        self.to_robokit=np.array([native_names.index(n) for n in names])
        limits=self.robot.spec.actuated_joint_limits[self.to_native]
        if not np.allclose(limits,np.stack((adapter.low[6:].cpu(),adapter.high[6:].cpu()),-1),atol=1e-6):
            raise ValueError('Robokit/native WUJI2 joint limits disagree')
        spec=HandSpec(floating_base=True,target_names=target_names,target_link_names=links,
            target_chains=tuple(chains),root_target_name='wrist',contact_target_names=tuple(c[-1] for c in chains),
            root_link_coord_spec=HandCoordinateSpec(palm_forward_axis='y',four_fingers_up_axis='-z'))
        config=HandRetargeterOnlineConfig(reference_origin='wrist',base_init_sample_range=(0.,0.,0.))
        config.task('positions').weight={n:(24. if n.endswith('_tip') else .5 if n=='wrist' else 4.) for n in target_names}
        config.task('directions').weight={(links[a],links[b],a,b):10. for chain in chains for a,b in zip(chain,chain[1:])}
        config.task('rest').weight=.001
        # Capsule fitting is conservative around motor housings. Keep this a
        # light initializer prior; exact native mesh checks gate CEM outputs.
        collision=config.task('self_collision');collision.weight=1.;collision.margin=.001
        collision.representation='capsule';collision.max_active_pairs=512
        # Joint warm start is managed below, independently of CEM. Keep internal
        # EMA and temporal objectives off so every target is the current frame.
        config.task('smoothness').weight=0.
        self.helper=HandRetargeterOnline(config,self.robot,spec,device=str(self.device));self.helper.warmup(1)
        self.mid_q=np.asarray(self.robot.spec.midrange_q,dtype=np.float32)[None]
        self.link_ids=[self.robot.spec.link_names.index(links[n]) for n in target_names]
        self.last_report={};self.use_history=use_history;self.reset()
        self.artifact=dict(method='robokit_hand_retargeter_online',source='current_frame_MANO_21_joints',
            uses_cem_history=False,uses_retarget_history=use_history,history_policy='previous_Robokit_joints_current_source_palm',actuated_joints=20,native_asset_sha256=surface.asset_hash,
            candidate_schedule=list(config.solver.num_candidates),target_names=list(target_names),
            self_collision_enabled=True,self_collision_representation='native_mesh_fitted_capsules',self_collision_weight=collision.weight)

    def reset(self):
        self.previous_q=None;self.previous_points=None;self.previous_result=None

    def __call__(self, joints, pad_normals=None):
        began=time.perf_counter();points=np.asarray(joints,dtype=np.float32)
        if points.shape!=(21,3) or not np.isfinite(points).all():raise ValueError('Robokit requires 21 finite current-frame joints in metres')
        if self.use_history and self.previous_result is not None and np.array_equal(points,self.previous_points):
            # A repeated target is the same retargeting problem. Re-solving it
            # from successively different warm starts otherwise introduces
            # tiny numerical centre shifts that bifurcate the CEM population.
            self.last_report={**self.last_report,'elapsed_ms':(time.perf_counter()-began)*1000,
                'uses_retarget_history':True,'initial_state':'identical_current_source_cached_Robokit_solution',
                'reused_identical_source':True,'history_reset_reason':None}
            return tuple(v.clone() for v in self.previous_result)
        j=torch.as_tensor(points,device=self.device);_,basis=palm_frame(j[0],j[[1,5,9,13,17]])
        r=(basis@self.adapter.robot_basis.T).cpu().numpy()
        if not np.allclose(r.T@r,np.eye(3),atol=1e-5) or np.linalg.det(r)<.99:raise ValueError('Degenerate source palm frame')
        pose=np.r_[points[0],Rotation.from_matrix(r).as_quat()[[3,0,1,2]]].astype(np.float32)[None]
        base=wp.from_numpy(pose,dtype=self.wp_vec7,device=str(self.device))
        # Reuse only this initializer's joint solution. The wrist and every
        # target still come from this frame; CEM output never enters this cache.
        jump=self.previous_points is not None and np.max(np.linalg.norm(
            (points-points[0])-(self.previous_points-self.previous_points[0]),axis=1))>.04
        history=self.use_history and self.previous_q is not None and not jump
        initial=self.robot.state(q=self.previous_q if history else self.mid_q,T_world_base=base)
        self.helper.reset()
        packed=self.helper.solve_numpy(points,init_state=initial)[0]
        if packed.shape!=(27,) or not np.isfinite(packed).all() or not np.isclose(np.linalg.norm(packed[3:7]),1.,atol=1e-4):
            raise ValueError('Robokit returned a nonfinite or invalid WUJI2 proposal')
        theta=packed[7:][self.to_native]
        if np.any(theta<self.adapter.low[6:].cpu().numpy()-1e-5) or np.any(theta>self.adapter.high[6:].cpu().numpy()+1e-5):
            raise ValueError('Robokit proposal violates native WUJI2 limits')
        rotation=Rotation.from_quat(packed[[4,5,6,3]]).as_matrix()
        self.previous_q=packed[None,7:].copy();self.previous_points=points.copy()
        self.last_report=dict(method=self.artifact['method'],elapsed_ms=(time.perf_counter()-began)*1000,
            source='current_frame_MANO_21_joints',uses_cem_history=False,uses_retarget_history=history,
            initial_state='previous_Robokit_joints_current_source_palm' if history else 'joint_midrange_current_source_palm',
            reused_identical_source=False,
            history_reset_reason='source_jump_over_40mm' if jump else None,
            self_collision_enabled=True,self_collision_representation='native_mesh_fitted_capsules',collision_role='soft_initializer_objective_final_mesh_gate_required')
        result=tuple(torch.as_tensor(a,device=self.device,dtype=torch.float32) for a in (packed[:3],rotation,theta))
        self.previous_result=tuple(v.clone() for v in result)
        return result
