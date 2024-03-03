# FEDERATED BOOSTING ALGORITHMS
from enum import Enum

from .adaboostf import AdaboostF
from .adaboostf2 import AdaboostF2
from .distboostf import DistboostF
from .preweakf import PreweakF
from .preweakf2 import PreweakF2


class FedAdaboostAlgorithmsEnum(Enum):
    ADABOOSTF = 'adaboostf'
    ADABOOSTF2 = 'adaboostf2'
    DISTBOOSTF = 'distboostf'
    PREWEAKF = 'preweakf'
    PREWEAKF2 = 'preweakf2'

    @classmethod
    def contains(cls, member: object) -> bool:
        if isinstance(member, str):
            return member in cls._value2member_map_.keys()
        elif isinstance(member, FedAdaboostAlgorithmsEnum):
            return member.value in cls._member_names_

    def algorithm(self):
        algos = {
            'adaboostf': AdaboostF,
            'adaboostf2': AdaboostF2,
            'distboostf': DistboostF,
            'preweakf': PreweakF,
            'preweakf2': PreweakF2
        }

        return algos[self.value]