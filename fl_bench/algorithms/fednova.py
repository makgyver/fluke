from typing import Callable, Iterable
import multiprocessing as mp
from torch.nn import Module
from copy import deepcopy

from collections import OrderedDict

import sys
from fl_bench.data import DataSplitter; sys.path.append(".")
from utils import OptimizerConfigurator
from algorithms import CentralizedFL

from server import Server

import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np

from fl_bench.client import Client
from fl_bench.utils import OptimizerConfigurator
from fl_bench.data import DataSplitter, FastTensorDataLoader
from fl_bench import GlobalSettings



# code taken from https://github.com/JYWa/FedNova

class FedNovaOptimizer(Optimizer):
    r"""Implements federated normalized averaging (FedNova).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Parameters
    ----------
    params (iterable): 
        Iterable of parameters to optimize or dicts defining parameter groups
    ratio (float):
        relative sample size of client
    gmf (float):
        global/server/slow momentum factor
    mu (float):
        parameter for proximal local SGD
    lr (float):
        learning rate
    momentum (float, optional):
        momentum factor (default: 0)
    weight_decay (float, optional):
        weight decay (L2 penalty) (default: 0)
    dampening (float, optional):
        dampening for momentum (default: 0)
    nesterov (bool, optional):
        enables Nesterov momentum (default: False)

    Example
    -------
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> optimizer.zero_grad()
    >>> loss_fn(model(input), target).backward()
    >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    Note
    ----
    The implementation of SGD with Momentum/Nesterov subtly differs from
    Sutskever et. al. and implementations in some other frameworks.

    Considering the specific case of Momentum, the update can be written as

    .. math::
                v = \rho * v + g \\
                p = p - lr * v

    where p, g, v and :math:`\rho` denote the parameters, gradient,
    velocity, and momentum respectively.

    This is in contrast to Sutskever et. al. and
    other frameworks which employ an update of the form

    .. math::
            v = \rho * v + lr * g \\
            p = p - v

    The Nesterov version is analogously modified.
    """

    def __init__(self, params, gmf=0, lr=required, mu=0, momentum=0.9, dampening=0,
                 weight_decay=0, nesterov=False, variance=0):
        self.gmf = gmf
        # self.ratio = ratio
        self.lr = lr
        self.momentum = momentum
        self.mu = mu
        self.local_normalizing_vec = 0
        self.local_counter = 0
        self.local_steps = 0

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, variance=variance)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FedNovaOptimizer, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(FedNova, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        
        Parameters
        ----------
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']


            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                param_state = self.state[p]
                if 'old_init' not in param_state:
                    param_state['old_init'] = torch.clone(p.data).detach()

                local_lr = group['lr']

                # apply momentum updates
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                # apply proximal updates
                if self.mu != 0:
                    d_p.add_(self.mu, p.data - param_state['old_init'])

                # update accumalated local updates
                if 'cum_grad' not in param_state:
                    param_state['cum_grad'] = torch.clone(d_p).detach()
                    param_state['cum_grad'].mul_(local_lr)

                else:
                    param_state['cum_grad'].add_(local_lr, d_p)

                p.data.add_(-local_lr, d_p) #DEPRECATED!!

        # compute local normalizing vector a_i
        if self.momentum != 0:
            self.local_counter = self.local_counter * self.momentum + 1
            self.local_normalizing_vec += self.local_counter

        self.etamu = local_lr * self.mu
        if self.etamu != 0:
            self.local_normalizing_vec *= (1 - self.etamu)
            self.local_normalizing_vec += 1

        if self.momentum == 0:
            self.local_normalizing_vec += 1

        self.local_steps += 1

        return loss

    def update_learning_rate(self, epoch, round):
        """
        1) Decay learning rate exponentially (epochs 30, 60, 80)
        ** note: target_lr is the reference learning rate from which to scale down
        """
        if epoch == int(round / 2):
            lr = self.lr/10 
            for param_group in self.param_groups:
                param_group['lr'] = lr

        if epoch == int(round * 0.75):
            lr = self.lr/100 
            for param_group in self.param_groups:
                param_group['lr'] = lr

class FedNovaClient(Client):
    def __init__(self,
                 train_set: FastTensorDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Callable,
                 validation_set: FastTensorDataLoader=None,
                 local_epochs: int=3,
                 total_train_size:int=0,
                 pattern: str = 'constant',
                 max_epochs: int = 3): 
        assert optimizer_cfg.optimizer == FedNovaOptimizer, \
            "FedNovaClient only supports FedNovaOptimizer"

        super().__init__(train_set, optimizer_cfg, loss_fn, validation_set, local_epochs)

        self.total_train_size = total_train_size
        self.max_epochs = max_epochs

        self.ratio =  train_set.size  / self.total_train_size
        self.pattern = pattern
        self.local_epochs = local_epochs
        

    def update_local_epochs(self, current_round: int=0, c:int = 0):
        """Function to update the number of locale epochs according to FedNova algorithm:
        
        both arguments are used in order to change the seed according to each client

        Parameters
        ----------
        current_round : int,
            current_round of training
        c: int,
            index of the client
        """
        if self.pattern == "constant":
            self.local_epochs = self.local_epochs

        if self.pattern == "uniform_random":
            np.random.seed(2020+current_round+c)
            self.local_epochs = np.random.randint(low=2, high=self.max_epochs, size=1)[0]
    

class FedNovaServer(Server):
    def __init__(self,
                 model: Module,
                 clients: Iterable[Client],
                 eligible_perc: float=1.,
                 tau_eff: float=0.,
                 global_step: float=1.,
                 weighted: bool = False): 

        super().__init__(model, clients, eligible_perc)
        self.control = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.global_step = global_step
        self.tau_eff = tau_eff
        self.weighted = weighted 


    def fit(self, n_rounds: int=10) -> None:
        """Run the federated learning algorithm.

        Parameters
        ----------
        n_rounds : int, optional
            The number of rounds to run, by default 10.
        """
        if GlobalSettings().get_workers() > 1:
            return self._fit_multiprocess(n_rounds)

        with GlobalSettings().get_live_renderer():
            progress_fl = GlobalSettings().get_progress_bar("FL")
            progress_client = GlobalSettings().get_progress_bar("clients")
            client_x_round = int(self.n_clients*self.eligible_perc)
            task_rounds = progress_fl.add_task("[red]FL Rounds", total=n_rounds*client_x_round)
            task_local = progress_client.add_task("[green]Local Training", total=client_x_round)
            total_rounds = self.rounds + n_rounds
            for round in range(self.rounds, total_rounds):
                self.notify_start_round(round + 1, self.model)
                eligible = self.get_eligible_clients()
                self.notify_selected_clients(round + 1, eligible)
                self.broadcast(eligible)
                client_evals = []
                for c, client in enumerate(eligible):
                    tau_eff_cuda = 0
                    client.update_local_epochs(current_round = round, c = c)
                    
                    client_eval = client.local_train()
                    if client_eval:
                        client_evals.append(client_eval)

                    if client.optimizer.mu != 0:
                        tau_eff_cuda = torch.tensor(client.optimizer.local_steps*client.ratio)
                    else:
                        tau_eff_cuda = torch.tensor(client.optimizer.local_normalizing_vec*client.ratio)

                    self.tau_eff += tau_eff_cuda
                    client.optimizer.update_learning_rate(round, n_rounds)

                    progress_client.update(task_id=task_local, completed=c+1)
                    progress_fl.update(task_id=task_rounds, advance=1)
                self.aggregate(eligible)
                self.notify_end_round(round + 1, self.model, client_evals)
                self.rounds += 1 
                if self.checkpoint_path is not None:
                    self.save(self.checkpoint_path)
            progress_fl.remove_task(task_rounds)
            progress_client.remove_task(task_local)

        # with Progress() as progress:
        #     client_x_round = int(self.n_clients*self.eligible_perc)
        #     task_rounds = progress.add_task("[red]FL Rounds", total=n_rounds*client_x_round)
        #     task_local = progress.add_task("[green]Local Training", total=client_x_round)

        #     for round in range(n_rounds):
        #         eligible = self.get_eligible_clients()
        #         self.broadcast(eligible)

        #         client_evals = []
        #         for c, client in enumerate(eligible):
        #             tau_eff_cuda = 0
        #             client.update_local_epochs(current_round = round, c = c)
                    
        #             client_eval = client.local_train()
        #             if client_eval:
        #                 client_evals.append(client_eval)

        #             if client.optimizer.mu != 0:
        #                 tau_eff_cuda = torch.tensor(client.optimizer.local_steps*client.ratio)
        #             else:
        #                 tau_eff_cuda = torch.tensor(client.optimizer.local_normalizing_vec*client.ratio)

        #             self.tau_eff += tau_eff_cuda
        #             client.optimizer.update_learning_rate(round, n_rounds)

        #             progress.update(task_id=task_local, completed=c+1)
        #             progress.update(task_id=task_rounds, advance=1)
        #         self.aggregate(eligible)
        #         self.notify_all(self.model, round + 1, client_evals)

    def _fit_multiprocess(self, n_rounds: int=10) -> None:
        """Run the federated learning algorithm using multiprocessing.

        Parameters
        ----------
        n_rounds : int, optional
            The number of rounds to run, by default 10.
        """
        progress_fl = GlobalSettings().get_progress_bar("FL")
        progress_client = GlobalSettings().get_progress_bar("clients")
        def callback_progress(result):
            progress_fl.update(task_id=task_rounds, advance=1)
            progress_client.update(task_id=task_local, advance=1)

        client_x_round = int(self.n_clients*self.eligible_perc)
        task_rounds = progress_fl.add_task("[red]FL Rounds", total=n_rounds*client_x_round)
        task_local = progress_client.add_task("[green]Local Training", total=client_x_round)

        total_rounds = self.rounds + n_rounds

        with GlobalSettings().get_live_renderer():
            for round in range(self.rounds, total_rounds):
                self.notify_start_round(round + 1, self.model)
                client_evals = []
                eligible = self.get_eligible_clients()
                self.notify_selected_clients(round + 1, eligible)
                self.broadcast(eligible)
                progress_client.update(task_id=task_local, completed=0)
                print(eligible)
                with mp.Pool(processes=GlobalSettings().get_workers()) as pool:
                    for c, client in enumerate(eligible):
                        
                        # if client.optimizer is None:
                        #     client.optimizer, client.scheduler = client.optimizer_cfg(client.model)
                        client.update_local_epochs(current_round = round, c = c)

                        client_eval = pool.apply_async(self._local_train, 
                                                    args=(client,), 
                                                    callback=callback_progress)
                        if client_eval:
                            client_evals.append(client_eval)
                    pool.close()
                    pool.join()
                for c, client in enumerate(eligible):
                    tau_eff_cuda = 0
                    if client.optimizer.mu != 0:
                        tau_eff_cuda = torch.tensor(client.optimizer.local_steps*client.ratio)
                    else:
                        tau_eff_cuda = torch.tensor(client.optimizer.local_normalizing_vec*client.ratio)

                    self.tau_eff += tau_eff_cuda
                    client.optimizer.update_learning_rate(round, n_rounds)
                client_evals = [c.get() for c in client_evals]
                self.aggregate(eligible)
                self.notify_end_round(round + 1, self.model, client_evals if client_evals[0] is not None else None)
                self.rounds += 1
                if self.checkpoint_path is not None:
                    self.save(self.checkpoint_path)
        progress_fl.remove_task(task_rounds)
        progress_client.remove_task(task_local)

    def aggregate(self, eligible: Iterable[Client], weight: float=0) -> None:
        """Aggregate the models of the clients.

        The aggregation is done by averaging the models of the clients. the weight is done through the
        ratio of the fednova model.
        Each client update their local parameters, then aggregate and perform one last update of the parameters.

        Parameters
        ----------
        eligible : Iterable[Client]
            The clients whose models will be aggregated.
        """
        avg_model_sd = OrderedDict()
        clients_sd = [eligible[i].send().state_dict() for i in range(len(eligible))]
        with torch.no_grad():

            for i, client in enumerate(eligible):
                if weight == 0:
                    weight = client.ratio
                param_list = []
                for group in client.optimizer.param_groups:
                    for p in group['params']:
                        param_state = client.optimizer.state[p]
                        scale = self.tau_eff/client.optimizer.local_normalizing_vec
                        param_state['cum_grad'].mul_(weight*scale)
                        param_list.append(param_state['cum_grad'])

            avg_model_sd = OrderedDict()
            clients_sd = [eligible[i].send().state_dict() for i in range(len(eligible))]
            with torch.no_grad():
                for key in self.model.state_dict().keys():
                    if "num_batches_tracked" in key:
                        avg_model_sd[key] = deepcopy(clients_sd[0][key])
                        continue
                    den = 0
                    for i, client_sd in enumerate(clients_sd):
                        d_weight = 1 # we keep this =1 because the ratio has already been calculated in previous steps
                        den += d_weight
                        if key not in avg_model_sd:
                            avg_model_sd[key] = d_weight * client_sd[key]
                        else:
                            avg_model_sd[key] += d_weight * client_sd[key]
                    avg_model_sd[key] /= den
                self.model.load_state_dict(avg_model_sd)

            for i, client in enumerate(eligible):
                for group in client.optimizer.param_groups:
                    lr = group['lr']
                    for p in group['params']:
                        param_state = client.optimizer.state[p]

                        if client.optimizer.gmf != 0:
                            if 'global_momentum_buffer' not in param_state:
                                buf = param_state['global_momentum_buffer'] = torch.clone(param_state['cum_grad']).detach()
                                buf.div_(lr)
                            else:
                                buf = param_state['global_momentum_buffer']
                                buf.mul_(client.optimizer.gmf).add_(1/lr, param_state['cum_grad'])
                            param_state['old_init'].sub_(lr, buf)
                        else:
                            param_state['old_init'].sub_(param_state['cum_grad'])

                        p.data.copy_(param_state['old_init'])
                        param_state['cum_grad'].zero_()

                        # Reinitialize momentum buffer
                        if 'momentum_buffer' in param_state:
                            param_state['momentum_buffer'].zero_()

                client.optimizer.local_counter = 0
                client.optimizer.local_normalizing_vec = 0
                client.optimizer.local_steps = 0


class FedNova(CentralizedFL):
    def __init__(self,
                 n_clients: int,
                 n_rounds: int,
                 n_epochs: int,
                 optimizer_cfg: OptimizerConfigurator,
                 pattern: str,
                 model: Module,
                 loss_fn: Callable,
                 eligible_perc: float=0.5):

        super().__init__(n_clients,
                         n_rounds,
                         n_epochs,
                         model,
                         optimizer_cfg,
                         loss_fn,
                         eligible_perc)
        self.pattern = pattern

    def init_clients(self, data_splitter: DataSplitter, **kwargs):
        assert data_splitter.n_clients == self.n_clients, "Number of clients in data splitter and the FL environment must be the same"
        self.data_assignment = data_splitter.assignments
        total_train_size = sum([i.size for i in data_splitter.client_train_loader])
        self.clients = [FedNovaClient(train_set=data_splitter.client_train_loader[i],
                                      optimizer_cfg=self.optimizer_cfg,
                                      loss_fn=self.loss_fn,
                                      validation_set=data_splitter.client_test_loader[i],
                                      pattern = self.pattern,
                                      total_train_size = total_train_size,
                                      local_epochs=self.n_epochs,
                                      max_epochs = self.n_epochs) for i in range(self.n_clients)]

    def init_server(self, **kwargs):
        self.server = FedNovaServer(self.model, self.clients, self.eligible_perc, weighted=False)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(C={self.n_clients},R={self.n_rounds},E={self.n_epochs}," + \
            f"P={self.eligible_perc},{self.optimizer_cfg},p={self.pattern})"