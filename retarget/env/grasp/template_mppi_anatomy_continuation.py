"""Continue the fresh extension anchor without losing adjacent-size bounds."""
from .template_mppi_anatomy_review import ExtensionDiskObjective
from .template_mppi_continuation import ContinuationObjective


class ContinuousExtensionObjective(ExtensionDiskObjective,ContinuationObjective):
    def optimize(self,initial_q,solver,seed):
        return super().optimize(self.parent_q,solver,seed)
