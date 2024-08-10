"""Implementation of the [SCAFFOLD20]_ algorithm.

References:
    .. [SCAFFOLD20] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J. Reddi,
       Sebastian U. Stich, Ananda Theertha Suresh. SCAFFOLD: Stochastic Controlled Averaging for
       Federated Learning. In: ICML (2020). URL: https://arxiv.org/abs/1910.06378
"""
# from torch.optim import Optimizer
from torch.nn import Module
import torch
from typing import Iterable
from copy import deepcopy
import sys
sys.path.append(".")
sys.path.append("..")

from .. import GlobalSettings  # NOQA
from ..utils import OptimizerConfigurator, clear_cache  # NOQA
from ..utils.model import safe_load_state_dict  # NOQA
from ..data import FastDataLoader  # NOQA
from ..algorithms import CentralizedFL  # NOQA
from ..server import Server  # NOQA
from ..client import Client  # NOQA
from ..comm import Message  # NOQA


# class SCAFFOLDOptimizer(Optimizer):
#     def __init__(self, params, lr: float = 0.001, weight_decay: float = 0.01):
#         defaults = dict(lr=lr, weight_decay=weight_decay)
#         super(SCAFFOLDOptimizer, self).__init__(params, defaults)

#     def step(self, server_controls, client_controls, closure=None):

#         loss = None
#         if closure is not None:
#             loss = closure

#         for group in self.param_groups:
#             for p, c, ci in zip(group['params'], server_controls, client_controls):
#                 if p.grad is None:
#                     continue
#                 dp = p.grad.data + c.data - ci.data
#                 p.data = p.data - dp.data * group['lr']

#         return loss


class SCAFFOLDClient(Client):
    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: torch.nn.Module,
                 local_epochs: int = 3,
                 **kwargs):
        super().__init__(index=index, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=loss_fn, local_epochs=local_epochs,
                         **kwargs)
        self.control = None
        self.delta_c = None
        self.delta_y = None
        self.server_control = None

    def receive_model(self) -> None:
        model, server_control = self.channel.receive(self, self.server, msg_type="model").payload
        self.server_control = server_control
        if self.model is None:
            self.model = model
            self.control = [torch.zeros_like(p.data, device=self.device)
                            for p in self.model.parameters()]  # if p.requires_grad]
            # self.delta_y = [torch.zeros_like(p.data)
            #                 for p in self.model.parameters() if p.requires_grad]
            # self.delta_c = [torch.zeros_like(p.data)
            #                 for p in self.model.parameters() if p.requires_grad]
        else:
            safe_load_state_dict(self.model, model.state_dict())

    def _move_to(self, control: list[torch.tensor], device: str) -> None:
        for c in control:
            c.data = c.data.to(device)

    def _get_next_batch(self):
        try:
            X, y = next(self.train_set)
            X, y = X.to(self.device), y.to(self.device)
        except StopIteration:
            self.train_iterator = iter(self.train_set)
            X, y = next(self.train_iterator)

        return X, y

    def fit(self, override_local_epochs: int = 0):
        epochs = override_local_epochs if override_local_epochs else self.hyper_params.local_epochs
        self.receive_model()
        server_model = deepcopy(self.model).to(self.device)
        self.model.to(self.device)
        self.model.train()

        # self._move_to(self.control, self.device)
        # self._move_to(self.server_control, self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.optimizer_cfg(self.model)

        for _ in range(epochs):
            loss = None
            # for _, (X, y) in enumerate(self.train_set):
            # X, y = X.to(self.device), y.to(self.device)
            X, y = self._get_next_batch()
            self.optimizer.zero_grad()
            y_hat = self.model(X)
            loss = self.hyper_params.loss_fn(y_hat, y)
            loss.backward()
            for param, c, c_i in zip(self.model.parameters(),
                                     self.server_control,
                                     self.control):
                if param.requires_grad:
                    param.grad.data += (c - c_i).to(self.device)
            self.optimizer.step()  # self.server_control, self.control)
        self.scheduler.step()

        with torch.no_grad():

            self.delta_y = []
            c_plus = []
            self.delta_c = []

            # compute delta_y (difference of model before and after training)
            for param_l, param_g in zip(self.model.parameters(), server_model.parameters()):
                self.delta_y.append(param_l - param_g)

            # compute c_plus
            coef = 1 / (self.hyper_params.local_epochs * self.scheduler.get_last_lr()[0])
            for c_l, c_g, diff in zip(self.control, self.server_control, self.delta_y):
                c_plus.append(c_l - c_g - coef * diff)

            # compute delta_c
            for c_p, c_l in zip(c_plus, self.control):
                self.delta_c.append(c_p - c_l)

            self.control = c_plus

            # params = zip(self.model.parameters(), server_model.parameters(), self.delta_y)
            # for local_model, server_model, delta_y in params:
            #     delta_y.data = local_model.data.detach() - server_model.data.detach()

            # new_controls = [torch.zeros_like(p.data, device=self.device)
            #                 for p in self.model.parameters() if p.requires_grad]
            # coeff = 1. / (self.hyper_params.local_epochs  # * len(self.train_set)
            #             * self.scheduler.get_last_lr()[0])
            # params = zip(self.control, self.server_control, new_controls, self.delta_y)
            # for local_control, server_control, new_control, delta_y in params:
            #     new_control.data = local_control.data - server_control.data - delta_y.data * coeff

            # for local_control, new_control, delta_c in zip(self.control, new_controls, self.delta_c):
            #     delta_c.data = new_control.data - local_control.data
            #     local_control.data = new_control.data

        self.model.to("cpu")
        server_model.to("cpu")
        self._move_to(self.control, "cpu")
        self._move_to(self.server_control, "cpu")
        clear_cache()
        self.send_model()

    def send_model(self):
        self.channel.send(Message((self.delta_y, self.delta_c), "model", self), self.server)


class SCAFFOLDServer(Server):
    def __init__(self,
                 model: Module,
                 test_data: FastDataLoader,
                 clients: Iterable[Client],
                 eval_every: int = 1,
                 global_step: float = 1.):
        super().__init__(model, test_data, clients, eval_every, False)
        self.device = GlobalSettings().get_device()
        self.model.to(self.device)
        self.control = [torch.zeros_like(p.data, device=self.device)
                        for p in self.model.parameters()]  # if p.requires_grad]
        self.hyper_params.update(global_step=global_step)

    def broadcast_model(self, eligible: Iterable[Client]) -> None:
        self.channel.broadcast(Message((self.model, self.control), "model", self), eligible)

    @torch.no_grad()
    def aggregate(self, eligible: Iterable[Client]) -> None:
        self.model.to(self.device)

        y_delta_cache = []
        c_delta_cache = []

        for client in eligible:
            cl_delta_y, cl_delta_c = self.channel.receive(self, client, "model").payload
            y_delta_cache.append(cl_delta_y)
            c_delta_cache.append(cl_delta_c)

        # update global model
        avg_weight = torch.tensor([1 / len(eligible) for _ in eligible], device=self.device)
        for param, y_del in zip(self.model.parameters(), zip(*y_delta_cache)):
            x_del = torch.sum(avg_weight * torch.stack(y_del, dim=-1), dim=-1)
            param.data += self.hyper_params.global_step * x_del
            # print(param)

        # update global control
        for c_g, c_del in zip(self.control, zip(*c_delta_cache)):
            c_del = torch.sum(avg_weight * torch.stack(c_del, dim=-1), dim=-1)
            c_g.data += (len(eligible) / self.n_clients) * c_del

        # delta_y = [torch.zeros_like(p.data, device=self.device)
        #            for p in self.model.parameters() if p.requires_grad]
        # delta_c = [torch.zeros_like(p.data, device=self.device)
        #            for p in self.model.parameters() if p.requires_grad]

        # for client in eligible:
        #     cl_delta_y, cl_delta_c = self.channel.receive(self, client, "model").payload
        #     deltas = zip(cl_delta_c, cl_delta_y, delta_c, delta_y)
        #     for client_delta_c, client_delta_y, server_delta_c, server_delta_y in deltas:
        #         server_delta_y.data = server_delta_y.data + client_delta_y.data.clone()
        #         server_delta_c.data = server_delta_c.data + client_delta_c.data.clone()

        # for server_delta_c, server_delta_y in zip(delta_c, delta_y):
        #     # * (eligible[i].n_examples / tot_examples)
        #     server_delta_y.data = server_delta_y.data / len(eligible)
        #     server_delta_c.data = server_delta_c.data / self.n_clients

        # params_deltas = zip(self.model.parameters(), self.control, delta_y, delta_c)
        # for param, server_control, server_delta_y, server_delta_c in params_deltas:
        #     param.data = param.data + self.hyper_params.global_step * server_delta_y
        #     server_control.data = server_control.data + server_delta_c.data


class SCAFFOLD(CentralizedFL):
    """SCAFFOLD Federated Learning Environment.

    https://arxiv.org/pdf/1910.06378.pdf

    Parameters
    ----------
    n_clients : int
        Number of clients in the FL environment.
    n_rounds : int
        Number of communication rounds.
    n_epochs : int
        Number of epochs per communication round.
    optimizer_cfg : OptimizerConfigurator
        Optimizer configurator for the clients.
    model : torch.nn.Module
        Model to be trained.
    loss_fn : Callable
        Loss function.
    eligible_perc : float, optional
        Percentage of clients to be selected for each communication round, by default 0.5.
    """

    # def get_optimizer_class(self) -> torch.optim.Optimizer:
    #     return SCAFFOLDOptimizer

    def get_client_class(self) -> Client:
        return SCAFFOLDClient

    def get_server_class(self) -> Server:
        return SCAFFOLDServer
