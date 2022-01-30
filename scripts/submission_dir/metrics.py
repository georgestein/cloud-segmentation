
import numpy as np
import torch
import torchmetrics

class Intersection(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("intersection", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.intersection += torch.logical_and(preds.view(-1), target.view(-1))
        # self.correct += torch.sum(preds == target)
        # self.total += target.numel()

    def compute(self):
        return self.intersection.float()
        # return self.correct.float() / self.total
