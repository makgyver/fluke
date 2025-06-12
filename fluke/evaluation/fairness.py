from collections import defaultdict
import torch
from torchmetrics import Metric

from ..data import FastDataLoader  # NOQA
from . import Evaluator  # NOQA

__all__ = ["FairnessEval", "DemographicParityDifference", "add_sensitive_feature"]


def _FAIR_METRICS():
    return {"demographic_parity": DemographicParityDifference()}


def add_sensitive_feature(
    data_loader: FastDataLoader, sensitive_feature: int, exclude_sensitive: bool = True
):
    new_data_loader = data_loader.clone()
    new_data_loader.tensors = (
        (
            torch.cat(
                new_data_loader.tensors[0][:, :sensitive_feature],
                new_data_loader.tensors[0][:, sensitive_feature + 1 :],
            )
            if exclude_sensitive
            else new_data_loader.tensors[0]
        ),
        new_data_loader.tensors[1],
        new_data_loader.tensors[0][:, sensitive_feature],
    )

    return new_data_loader


# Draft - chatgpt
class DemographicParityDifference(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("group_positive", default=defaultdict(int), dist_reduce_fx=None)
        self.add_state("group_total", default=defaultdict(int), dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, y_true: torch.Tensor, sensitive_attr: torch.Tensor):
        preds = preds.detach().cpu()
        sensitive_attr = sensitive_attr.detach().cpu()

        for group in torch.unique(sensitive_attr):
            group_val = group.item()
            mask = sensitive_attr == group
            positives = (preds[mask] == 1).sum().item()
            total = mask.sum().item()

            self.group_positive[group_val] += positives
            self.group_total[group_val] += total

    def compute(self):
        rates = []
        for group in self.group_total:
            total = self.group_total[group]
            pos = self.group_positive[group]
            rate = pos / total if total > 0 else 0.0
            rates.append(rate)

        return max(rates) - min(rates) if rates else 0.0


class FairnessEval(Evaluator):

    def __init__(self, eval_every: int, **metrics):
        super().__init__(eval_every)
        if metrics:
            self.metrics = metrics
        else:
            self.metrics = _FAIR_METRICS()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"FairnessEval(eval_every={self.eval_every})" + f"[{', '.join(self.metrics.keys())}]"
