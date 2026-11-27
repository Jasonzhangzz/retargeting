"""CEM ablation: normalized CARD contact plus exact native first-difference smooth.

Audit only: does not alter policy losses or start training. Reuses the common
16-pose search and diagnostics so budgets and initial states stay comparable.
"""
import torch
import audit_reach_seeded_ablation as runner
from retarget.baseline.card.math import CardSettings, axis_angle_near, exp_so3


def native_smooth_parts(states):
    """Native CARD per-transition energy / 1e6, averaged over all T frames.

No previous optimized pose is available at the window boundary; its smooth
energy is zero, as in CARD with previous=None. Thus divide by T, not T-1.
"""
    s=CardSettings()
    previous=states[:-1];current=states[1:];length=len(states)
    pr=axis_angle_near(exp_so3(previous[...,23:]))
    dr=axis_angle_near(exp_so3(current[...,23:]),pr)-pr
    return dict(native_smooth_joint=s.wtheta**2*(current[...,:20]-previous[...,:20]).square().sum((0,2))/length/1e6,
        native_smooth_translation=s.wt**2*((current[...,20:23]-previous[...,20:23])*1000).square().sum((0,2))/length/1e6,
        native_smooth_rotation=s.womega**2*dr.square().sum((0,2))/length/1e6)


class NativeSmoothPoseCost(runner.PoseCost):
    def evaluate_states(self, states):
        parts,states,gap=super().evaluate_states(states)
        native=native_smooth_parts(states)
        parts['motion_jitter']=sum(native.values())  # common ablation slot; manifest explicitly identifies replacement
        parts.update(native)
        parts['total']=parts['contact']+parts['motion_jitter']+parts['contact_jitter']
        return parts,states,gap

    def parity(self, parameters):
        parts,states,_=self.evaluate(parameters)
        s=CardSettings();raw=torch.zeros(states.shape[1],device=states.device)
        for t in range(1,len(states)):
            pt,pj=states[t-1,:,20:23],states[t-1,:,:20]
            pr=exp_so3(states[t-1,:,23:]);r=exp_so3(states[t,:,23:])
            # Literal expression from TactileRetarget's three smooth components.
            raw+=s.wtheta**2*(states[t,:,:20]-pj).square().sum(-1)
            raw+=s.wt**2*((states[t,:,20:23]-pt)*1000).square().sum(-1)
            raw+=s.womega**2*(axis_angle_near(r,axis_angle_near(pr))-axis_angle_near(pr)).square().sum(-1)
        torch.testing.assert_close(parts['motion_jitter'],raw/len(states)/1e6,rtol=3e-5,atol=1e-7)
        torch.testing.assert_close(parts['total'],parts['contact']+raw/len(states)/1e6+parts['contact_jitter'],rtol=3e-5,atol=1e-7)
        return {k:float(v.mean()) for k,v in parts.items()}


if __name__=='__main__':
    runner.PoseCost=NativeSmoothPoseCost
    runner.SPEC=dict(runner.SPEC,version='audit_native_first_difference_smooth',
        motion_regularizer='exact CARD smooth_joint+smooth_translation+smooth_rotation',
        native_smooth_weights=dict(joint=.1,translation=CardSettings().wt,rotation=10.),
        normalization='(native_contact_mm_energy + native_smooth_mm_energy)/1e6; mean over 16 frames',
        window_boundary='no external preceding optimized pose; first smooth contribution zero',
        variants=dict(full='contact + native smooth; gap jitter disabled',no_jitter='contact + native smooth'),
        excluded='style, contact material-point fix, IK motion reference, collision costs; not full native CARD')
    runner.main()
