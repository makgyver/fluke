"""Implementation of the [FedAVGM19]_ algorithm.

References:
    .. [FedAVGM19] Tzu-Ming Harry Hsu, Hang Qi, Matthew Brown. Measuring the Effects of
       Non-Identical Data Distribution for Federated Visual Classification. In: arXiv (2019).
       URL: https://arxiv.org/abs/1909.06335
"""
from torch.nn import Module
import torch
from collections import OrderedDict
from typing import Iterable
import sys
sys.path.append(".")
sys.path.append("..")

from ..data import FastDataLoader  # NOQA
from ..algorithms import CentralizedFL  # NOQA
from ..utils.model import diff_model, STATE_DICT_KEYS_TO_IGNORE  # NOQA
from ..server import Server  # NOQA
from ..client import Client  # NOQA


class FedAVGMServer(Server):
    def __init__(self,
                 model: Module,
                 test_data: FastDataLoader,
                 clients: Iterable[Client],
                 eval_every: int = 1,
                 weighted: bool = True,
                 momentum: float = 0.9):
        """_summary_

        Args:
            model (Module): _description_
            test_data (FastDataLoader): _description_
            clients (Iterable[Client]): _description_
            eval_every (int, optional): _description_. Defaults to 1.
            weighted (bool, optional): _description_. Defaults to True.
            momentum (float, optional): _description_. Defaults to 0.9.
        """
        super().__init__(model, test_data, clients, eval_every, weighted)
        self.hyper_params.update(momentum=momentum)

    @torch.no_grad()
    def aggregate(self, eligible: Iterable[Client]) -> None:
        """Aggregate the models of the eligible clients using the FedAVGM algorithm.

        Formally, given the global model parameters :math:`\\theta^{t-1}` at round :math:`t-1`,
        the model parameters of the eligible clients :math:`\\theta_{i}^{t-1}`, and the weights
        :math:`w_i` of the clients, the update of the global model parameters for round :math:`t`
        are computed as:

        .. math::

            \\theta^t = \\mu \\theta^{t-1} - \\sum_{i=1}^{N} w_i (\\theta^{t-1} - \\theta^{t-1}_i)

        where :math:`\\mu` is the `momentum` hyper-parameter.

        Args:
            eligible (Iterable[Client]): The eligible clients.
        """
        avg_model_sd = OrderedDict()
        clients_sd = self.get_client_models(eligible)
        clients_diff = [diff_model(self.model.state_dict(), client_model)
                        for client_model in clients_sd]
        weights = self._get_client_weights(eligible)

        for key in self.model.state_dict().keys():
            if key.endswith(STATE_DICT_KEYS_TO_IGNORE):
                avg_model_sd[key] = self.model.state_dict()[key].clone()
                continue
            for i, client_diff in enumerate(clients_diff):
                if key not in avg_model_sd:
                    avg_model_sd[key] = weights[i] * client_diff[key]
                else:
                    avg_model_sd[key] += weights[i] * client_diff[key]

        for key, param in self.model.named_parameters():
            param.data = self.hyper_params.momentum * param.data - avg_model_sd[key].data


class FedAVGM(CentralizedFL):

    def get_server_class(self) -> Server:
        return FedAVGMServer
