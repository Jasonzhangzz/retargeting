"""Runtime-only EMF conditioning; raw capture/calibration records stay untouched."""
from __future__ import annotations
import math
import numpy as np
from scipy.spatial.transform import Rotation

class SensorPoseFilter:
    def __init__(self):
        self.timestamp=None;self.positions=None;self.rotations=None;self.accepted_p=None;self.accepted_r=None
        self.pending=[None]*5;self.velocity=np.zeros((5,3));self.angular_velocity=np.zeros((5,3));self.rejected=0
        self.history_p=[];self.history_r=[]
        self.motion_until=np.zeros(5)

    def update(self,positions,rotations,timestamp_ms):
        p=np.asarray(positions,dtype=float);r=np.asarray(rotations,dtype=float)
        dt=(timestamp_ms-self.timestamp)/1000 if self.timestamp is not None else 0.
        if self.positions is None or not 0<dt<=.25:
            self.timestamp=timestamp_ms;self.positions=p.copy();self.rotations=r.copy();self.accepted_p=p.copy();self.accepted_r=r.copy();self.pending=[None]*5
            self.velocity[:]=0;self.angular_velocity[:]=0
            self.history_p=[p.copy()];self.history_r=[r.copy()]
            self.motion_until[:]=0
            return p.copy(),r.copy(),{'reset':True,'rejected_fingers':[],'rejected_total':self.rejected,'mode':'persistent_jump_gate_one_euro'}
        gate_p=p.copy();gate_r=r.copy();bad=[]
        pos_delta=np.linalg.norm(p-self.accepted_p,axis=1)
        rot_delta=Rotation.from_matrix(self.accepted_r.swapaxes(-2,-1)@r).magnitude()
        for f in range(5):
            abrupt=dt<=.12 and timestamp_ms>=self.motion_until[f] and (pos_delta[f]>.012 or rot_delta[f]>math.radians(14))
            if abrupt:
                previous=self.pending[f]
                supported=False
                if previous is not None:
                    pp,rr=previous
                    supported=(pos_delta[f]<=.012 or np.linalg.norm(p[f]-pp)<=.7*pos_delta[f]) and (rot_delta[f]<=math.radians(14) or Rotation.from_matrix(rr.T@r[f]).magnitude()<=.7*rot_delta[f])
                if not supported:
                    self.pending[f]=(p[f].copy(),r[f].copy());gate_p[f]=self.accepted_p[f];gate_r[f]=self.accepted_r[f];bad.append(f);continue
                self.motion_until[f]=timestamp_ms+150
            self.pending[f]=None;self.accepted_p[f]=p[f];self.accepted_r[f]=r[f]
        self.rejected+=len(bad)
        self.history_p=(self.history_p+[gate_p.copy()])[-3:];self.history_r=(self.history_r+[gate_r.copy()])[-3:]
        averaged=[]
        if len(self.history_p)==3:
            hp=np.array(self.history_p);hr=np.array(self.history_r)
            path=np.linalg.norm(np.diff(hp,axis=0),axis=-1).sum(0)
            turn=Rotation.from_matrix((hr[:-1].swapaxes(-2,-1)@hr[1:]).reshape(-1,3,3)).magnitude().reshape(2,5).sum(0)
            net_p=np.linalg.norm(hp[-1]-hp[0],axis=1)
            net_r=Rotation.from_matrix(hr[0].swapaxes(-2,-1)@hr[-1]).magnitude()
            for f in range(5):
                # Smooth oscillating noise; coherent movement/steps use the
                # latest sample and retain the fast response.
                quiet_p=path[f]>.0001 and net_p[f]<.65*path[f]
                quiet_r=turn[f]>.003 and net_r[f]<.65*turn[f]
                if quiet_p:gate_p[f]=hp[:,f].mean(0)
                if quiet_r:gate_r[f]=Rotation.from_matrix(hr[:,f]).mean().as_matrix()
                if quiet_p or quiet_r:averaged.append(f)
        derivative_alpha=-math.expm1(-dt/.04)
        self.velocity+=derivative_alpha*((gate_p-self.positions)/dt-self.velocity)
        delta=Rotation.from_matrix(self.rotations.swapaxes(-2,-1)@gate_r).as_rotvec()
        self.angular_velocity+=derivative_alpha*(delta/dt-self.angular_velocity)
        cutoff_p=5.+60*np.linalg.norm(self.velocity,axis=1)
        cutoff_r=4.+1.5*np.linalg.norm(self.angular_velocity,axis=1)
        alpha_p=1-np.exp(-2*math.pi*cutoff_p*dt);alpha_r=1-np.exp(-2*math.pi*cutoff_r*dt)
        self.positions+=alpha_p[:,None]*(gate_p-self.positions)
        self.rotations=self.rotations@Rotation.from_rotvec(delta*alpha_r[:,None]).as_matrix()
        self.timestamp=timestamp_ms
        return self.positions.copy(),self.rotations.copy(),{'reset':False,'rejected_fingers':bad,'rejected_total':self.rejected,'oscillation_averaged_fingers':averaged,
            'raw_position_step_mm':(pos_delta*1000).tolist(),'raw_orientation_step_deg':np.rad2deg(rot_delta).tolist(),
            'position_tau_ms':(1000/(2*math.pi*cutoff_p)).tolist(),'orientation_tau_ms':(1000/(2*math.pi*cutoff_r)).tolist(),
            'mode':'persistent_jump_gate_one_euro'}
