from torchmetrics import Accuracy, F1Score, Precision, Recall

from . import Evaluator

__all__ = ["ClassificationEval", "_CLF_METRICS"]


def _CLF_METRICS(n_classes: int):

    return {
        "accuracy": Accuracy(task="multiclass", num_classes=n_classes, top_k=1),
        # macro
        "macro_precision": Precision(
            task="multiclass",
            num_classes=n_classes,
            top_k=1,
            average="macro",
        ),
        "macro_recall": Recall(
            task="multiclass",
            num_classes=n_classes,
            top_k=1,
            average="macro",
        ),
        "macro_f1": F1Score(
            task="multiclass",
            num_classes=n_classes,
            top_k=1,
            average="macro",
        ),
        # micro
        "micro_precision": Precision(
            task="multiclass",
            num_classes=n_classes,
            top_k=1,
            average="micro",
        ),
        "micro_recall": Recall(
            task="multiclass",
            num_classes=n_classes,
            top_k=1,
            average="micro",
        ),
        "micro_f1": F1Score(
            task="multiclass",
            num_classes=n_classes,
            top_k=1,
            average="micro",
        ),
    }


class ClassificationEval(Evaluator):

    def __init__(self, eval_every: int, n_classes: int, **metrics):
        super().__init__(eval_every)
        self.n_classes = n_classes
        if metrics:
            self.metrics = metrics
        else:
            self.metrics = _CLF_METRICS(n_classes)
