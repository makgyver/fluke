from ...data import FastDataLoader  # NOQA
from ...evaluation import Evaluator  # NOQA
from ...evaluation.fairness import add_sensitive_feature  # NOQA


def make_client_fair_eval(party_class: type, sensitive_feature: int) -> type:

    class FairEvalClient(party_class):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
            if test_set is not None:
                fair_test_set = add_sensitive_feature(test_set, sensitive_feature, False)
                return super().evaluate(evaluator, fair_test_set)
            return {}

        def __repr__(self, indent: int = 0):
            return self.__str__(indent=indent)

        def __str__(self, indent: int = 0) -> str:
            tostr = super().__str__(indent)
            tostr = tostr.replace(self.__class__.__name__, party_class.__name__)
            return f"FairEval_{tostr}"

    return FairEvalClient


def make_server_fair_eval(party_class: type, sensitive_feature: int) -> type:

    class FairEvalServer(party_class):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
            if test_set is not None:
                fair_test_set = add_sensitive_feature(test_set, sensitive_feature, False)
                return super().evaluate(evaluator, fair_test_set)
            return {}

        def __repr__(self, indent: int = 0):
            return self.__str__(indent=indent)

        def __str__(self, indent: int = 0) -> str:
            tostr = super().__str__(indent)
            tostr = tostr.replace(self.__class__.__name__, party_class.__name__)
            return f"FairEval_{tostr}"

    return FairEvalServer


def make_fair_eval(alg_class: type) -> type:

    class FairnessFL(alg_class):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert hasattr(self, "hyper_params") and hasattr(
                self.hyper_params, "sensitive_attribute"
            ), "Hyper parameters must contain 'sensitive_attribute' for fairness evaluation."

        def get_client_class(self):
            return make_client_fair_eval(
                super().get_client_class(), self.hyper_params.sensitive_attribute
            )

        def get_server_class(self):
            return make_server_fair_eval(
                super().get_server_class(), self.hyper_params.sensitive_attribute
            )

    return FairnessFL
