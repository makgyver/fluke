import torch
import sys
sys.path.append(".")
sys.path.append("..")

from fluke.data import FastDataLoader  # NOQA
from fluke.evaluation import Evaluator, ClassificationEval  # NOQA


def test_classification_eval():
    loader = FastDataLoader(torch.FloatTensor([[1, 2, 3],
                                               [4, 5, 6],
                                               [7, 8, 9],
                                               [10, 11, 12]]),
                            torch.LongTensor([0, 1, 0, 2]),
                            num_labels=2,
                            batch_size=1,
                            shuffle=False,
                            skip_singleton=False)

    clf_eval = ClassificationEval(loss_fn=torch.nn.CrossEntropyLoss(),
                                  n_classes=3,
                                  device=torch.device("cpu"))

    assert clf_eval.n_classes == 3
    assert clf_eval.device == torch.device("cpu")

    # model that only classifies with 0
    class ModelZero(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 3)

    # model that only classifies with 1
    class ModelPerfect(torch.nn.Module):
        def forward(self, x):
            if torch.allclose(x, torch.FloatTensor([1, 2, 3])):
                return torch.FloatTensor([[1, 0, 0]])
            elif torch.allclose(x, torch.FloatTensor([4, 5, 6])):
                return torch.FloatTensor([[0, 1, 0]])
            elif torch.allclose(x, torch.FloatTensor([7, 8, 9])):
                return torch.FloatTensor([[1, 0, 0]])
            elif torch.allclose(x, torch.FloatTensor([10, 11, 12])):
                return torch.FloatTensor([[0, 0, 1]])

    # test with model that only classifies with 0
    zero_eval = clf_eval.evaluate(ModelZero(), loader)
    perfect_eval = clf_eval.evaluate(ModelPerfect(), loader)

    assert zero_eval["accuracy"] == 0.5
    assert zero_eval["micro_precision"] == 0.5
    assert zero_eval["micro_recall"] == 0.5
    assert zero_eval["micro_f1"] == 0.5
    assert zero_eval["macro_precision"] == 0.16667
    assert zero_eval["macro_recall"] == 0.33333
    assert zero_eval["macro_f1"] == 0.22222
    assert zero_eval["loss"] > 1

    assert perfect_eval["accuracy"] == 1
    assert perfect_eval["micro_precision"] == 1
    assert perfect_eval["micro_recall"] == 1
    assert perfect_eval["micro_f1"] == 1
    assert perfect_eval["macro_precision"] == 1
    assert perfect_eval["macro_recall"] == 1
    assert perfect_eval["macro_f1"] == 1
    assert perfect_eval["loss"] < 1

    assert clf_eval.evaluate(None, None) == {}

    assert str(clf_eval) == "ClassificationEval(n_classes=3,device=cpu)" + \
        "[accuracy,precision,recall,f1,CrossEntropyLoss]"
    assert repr(clf_eval) == str(clf_eval)


if __name__ == "__main__":
    test_classification_eval()
    # 97% coverage for fluke/evaluation.py
