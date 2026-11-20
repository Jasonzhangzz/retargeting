"""Causal developer continuity, original sampling, and unchanged offline defaults."""
from types import SimpleNamespace
import numpy as np
import torch
import warp as wp
from retarget.baseline.card.math import CardSettings
from retarget.baseline.card.online import OnlineCardRetarget,motion_draws


def test_motion_reference_preserves_ik_motion_instead_of_freezing_pose():
    from dataclasses import replace
    owner=OnlineCardRetarget.__new__(OnlineCardRetarget);owner.device=torch.device('cpu')
    owner.settings=replace(CardSettings(),wmotion_joint=300.,motion_sample_fraction=.125)
    owner.style_targets=lambda *args:{}
    owner.previous=(None,None,torch.full((20,),.5));owner.previous_initial_joint=torch.full((20,),.4)
    owner.initializer=SimpleNamespace(last_report={})
    owner.adapter=SimpleNamespace(low=torch.full((26,),-1.),high=torch.ones(26))
    for key in ('sdf','previous_source_wrist'):setattr(owner,key,None)
    for key in ('self_contact_pairs','self_contact_targets','paired_local','paired_normal','paired_candidates'):setattr(owner,key,{})
    context=owner.context(None,None,[],initial=(None,None,torch.full((20,),.43)))
    torch.testing.assert_close(context['motion_joint'],torch.full((20,),.53))
    owner.initializer.last_report={'history_reset_reason':'source_jump'}
    assert owner.context(None,None,[],initial=(None,None,torch.full((20,),.43)))['motion_joint'] is None


def test_source_motion_draws_only_replace_local_tail_and_respect_bounds():
    n,d=8,26;device='cpu'
    def arr(x,dtype=float):return wp.array(np.asarray(x),dtype=dtype,device=device)
    x=wp.zeros((n,d),device=device);noise=wp.zeros_like(x)
    wp.launch(motion_draws,(n,d),[x,noise,arr(np.zeros((1,d))),arr(np.full((1,d),.4)),arr([1],int),
        arr(np.full((1,d),-.5)),arr(np.full((1,d),.5)),2,.15,arr(np.zeros((1,d))),arr(np.ones(d)),7,arr([0],int),0],device=device)
    np.testing.assert_array_equal(x.numpy()[:6],0.)
    np.testing.assert_allclose(x.numpy()[6:],.4)
    np.testing.assert_array_equal(noise.numpy(),x.numpy())


def test_single_frame_interface_and_offline_defaults(monkeypatch):
    from retarget.baseline.card.retarget import TactileRetarget
    defaults=CardSettings()
    assert defaults.wmotion_joint==0. and defaults.motion_sample_fraction==0.
    monkeypatch.setattr(TactileRetarget,'step',lambda self,frame,contact_enabled:{'source_timestamp_ms':frame['timestamp_ms']})
    result=OnlineCardRetarget.__new__(OnlineCardRetarget).step({'timestamp_ms':123})
    assert result['source_timestamp_ms']==123
    assert result['buffer_delay_frames']==0 and result['buffer_delay_ms']==0
