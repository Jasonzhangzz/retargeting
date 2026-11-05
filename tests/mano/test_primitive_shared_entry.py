"""Atlas integration must use the current shared solver and original adapter."""
from pathlib import Path
import pytest
import torch
from scripts import fix_personalized_grasp_grid as atlas
from retarget.solver import cem
from retarget.env.grasp.mano_patch_consistent import PatchConsistentAdapter
from retarget.env.grasp.primitive_problem import AnatomicalPrimitiveDecoder,frozen_wrist_problem
from retarget.solver.cem import SamplingSettings as Settings


def test_atlas_uses_shared_entry_points_without_an_adapter_override():
 assert atlas.solve_patch_cem is cem.solve_patch_cem
 assert atlas.PatchConsistentAdapter is PatchConsistentAdapter
 assert 'PrimitivePatchCEM' not in vars(atlas)
 assert Path(cem.__file__).name=='cem.py'
 assert not Path(cem.__file__).with_name('mano_patch_cem.py').exists()


@pytest.mark.skipif(not Path('output/mano_wuji_primitives_20261105/saved_left').exists(),reason='Snapshot required')
def test_shared_cem_obeys_atlas_fixed_wrist_and_continuity_bounds():
 model=AnatomicalPrimitiveDecoder(Path('output/mano_wuji_primitives_20261105/saved_left'),'cpu')
 q=torch.zeros(1,6+model.components);q[:,:6]=torch.tensor([.01,.02,.03,.1,.2,.3])
 problem=frozen_wrist_problem(q,model,q,continuity_radius=.1)
 class Task:
  retain_feasible=True
  def validate(self,value,rows):
   cost=(value[:,6:]-.05).square().sum(-1)
   return cost,{'patch_joint_accepted':cost<1e-4}
  def evaluate_candidates(self,value,rows):
   cost,parts=self.validate(value,rows)
   return cost,parts['patch_joint_accepted']
 result=atlas.solve_patch_cem(problem,Task(),seed=42)
 torch.testing.assert_close(result['parameters'][:,:6],q[:,:6],rtol=0,atol=0)
 assert (result['parameters'][:,6:].abs()<=.100001).all()
 assert result['cost'][0]<(q[:,6:]-.05).square().sum()


def test_generic_dispatch_uses_the_same_cem_module():
 from retarget.solver.base import Problem,solve
 mean=torch.tensor([[.7,-.4]])
 problem=Problem(mean,torch.ones_like(mean)*.2,lambda x:x.square().sum(-1))
 from retarget.solver.base import Settings as Selector
 settings=Selector()
 direct=cem.solve(problem,seed=7)
 dispatched=solve(problem,settings,7)
 torch.testing.assert_close(direct.parameters,dispatched.parameters,atol=0,rtol=0)
 assert dispatched.metadata['implementation']=='current_patch_cem'
