"""Optimize the full grasp while keeping a genuine distal/middle index contact."""
import torch
from .template_mppi_review import ReviewObjective


class IndexEngagedObjective(ReviewObjective):
    def evaluate(self,controls):
        cost=super().evaluate(controls)
        valid=self.parts['fitted'][:,self.index_slots].any(-1)
        self.parts['valid'] &= valid
        return torch.where(valid,cost,torch.inf)
