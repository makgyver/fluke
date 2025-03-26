"""
The module :mod:`fluke.client` provides the base classes for the clients in :mod:`fluke`.
"""
from __future__ import annotations

import sys
from typing import Any, Literal

import torch
from torch import device
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

sys.path.append(".")

from . import DDict, FlukeCache, FlukeENV, ObserverSubject  # NOQA
from .comm import Channel, Message  # NOQA
from .data import FastDataLoader  # NOQA
from .evaluation import Evaluator  # NOQA
from .server import Server  # NOQA
from .utils import (OptimizerConfigurator, cache_obj, clear_cuda_cache,  # NOQA
                    retrieve_obj)
from .utils.model import ModOpt, safe_load_state_dict  # NOQA

__all__ = [
    "Client",
    "PFLClient"
]


class Client(ObserverSubject):
    """Base :class:`Client` class. This class is the base class for all clients in :mod:`fluke`.
    The standard behavior of a client is based on the Federated Averaging algorithm.
    The default behavior of a client includes:

    - Receiving the global model from the server;
    - Training the model locally for a number of epochs using the local training set;
    - Sending the updated local model back to the server;
    - (Optional) Evaluating the model on the local test set before and after the training.

    Attributes:
        hyper_params (DDict): The hyper-parameters of the client. The default hyper-parameters are:

            - ``loss_fn``: The loss function.
            - ``local_epochs``: The number of local epochs.
            - ``fine_tuning_epochs``: The number of fine-tuning epochs, i.e., the number of epochs
              to train the model after the federated learning process.
            - ``clipping``: The clipping value for the gradients.

            When a new client class inherits from this class, it must add all its hyper-parameters
            to this dictionary.

        train_set (FastDataLoader): The local training set.
        test_set (FastDataLoader): The local test set.
        optimizer_cfg (OptimizerConfigurator): The optimizer configurator. This is used to create
          the optimizer and the learning rate scheduler.
        device (torch.device): The device where the client trains the model. By default, it is the
          device defined in :class:`fluke.FlukeENV`.

    Attention:
        **The client should not directly call methods of the server**. The communication between the
        client and the server must be done through the :attr:`channel`.

    Important:
        When inheriting from this class, make sure to handle the caching of the model, optimizer,
        scheduler and any other attribute that should be cached. The caching is done automatically
        by the methods :meth:`_load_from_cache` and :meth:`_save_to_cache`. The method
        :meth:`_load_from_cache` is called before the local update and the method
        :meth:`_save_to_cache` is called after the local update. If the client has additional
        attributes that should be cached, add them to the ``_attr_to_cache`` list.
        If the client has an additional model, optimizer, and scheduler, these object should be
        handled via a :class:`fluke.utils.model.ModOpt` private object. The
        :class:`fluke.utils.model.ModOpt` object is used to store the model, optimizer, and
        scheduler. To access the model, optimizer, and scheduler, define the
        corresponding properties (e.g., :attr:`model`, :attr:`optimizer`, :attr:`scheduler`).

    Caution:
        When inheriting from this class, make sure to put all the specific hyper-parameters in the
        :attr:`hyper_params` attribute. In this way ``fluke`` can properly handle the
        hyper-parameters of the client in the federated learning process.

        For example:

        .. code-block:: python
            :linenos:

            class MyClient(Client):
                # We omit the type hints for brevity
                def __init__(self, index, ..., my_param):
                    super().__init__(index, ...)
                    self.hyper_params.update(my_param=my_param) # This is important
    """

    def __init__(self,
                 index: int,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Module,
                 local_epochs: int = 3,
                 fine_tuning_epochs: int = 0,
                 clipping: float = 0,
                 **kwargs: dict[str, Any]):
        super().__init__()
        self.train_set: FastDataLoader = train_set
        self.test_set: FastDataLoader = test_set
        self.device: device = FlukeENV().get_device()
        self.hyper_params: DDict = DDict(
            loss_fn=loss_fn,
            local_epochs=local_epochs,
            fine_tuning_epochs=fine_tuning_epochs,
            clipping=clipping
        )

        self._index: int = index
        self._modopt: ModOpt = ModOpt()
        self._optimizer_cfg: OptimizerConfigurator = optimizer_cfg
        self._server: Server = None
        self._channel: Channel = None
        self._last_round: int = 0
        # List of additional attributes to cache
        self._attr_to_cache: list[str] = []

    @property
    def model(self) -> Module:
        """The client's local model.

        Warning:
            If the model is stored in the cache, the method retrieves it from the cache but does not
            remove it. Thus, the performance may be affected if this property is used to get the
            model multiple times while the model is in the cache.

        Returns:
            torch.nn.Module: The local model.
        """
        if isinstance(self._modopt, FlukeCache._ObjectRef):
            return retrieve_obj("_modopt", self, pop=False).model
        return self._modopt.model

    @model.setter
    def model(self, model: Module) -> None:
        self._modopt.model = model

    @property
    def optimizer(self) -> Optimizer:
        """The optimizer of the client.

        Warning:
            If the optimizer is stored in the cache, the method retrieves it from the cache but does
            not remove it. Thus, the performance may be affected if this property is used to get the
            optimizer multiple times while the optimizer is in the cache.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        if isinstance(self._modopt, FlukeCache._ObjectRef):
            return retrieve_obj("_modopt", self, pop=False).optimizer
        return self._modopt.optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        self._modopt.optimizer = optimizer

    @property
    def scheduler(self) -> LRScheduler:
        """The learning rate scheduler of the client.

        Warning:
            If the scheduler is stored in the cache, the method retrieves it from the cache but does
            not remove it. Thus, the performance may be affected if this property is used to get the
            scheduler multiple times while the scheduler is in the cache.

        Returns:
            torch.optim.lr_scheduler.LRScheduler: The learning rate scheduler.
        """
        if isinstance(self._modopt, FlukeCache._ObjectRef):
            return retrieve_obj("_modopt", self, pop=False).scheduler
        return self._modopt.scheduler

    @scheduler.setter
    def scheduler(self, scheduler: LRScheduler) -> None:
        self._modopt.scheduler = scheduler

    @property
    def index(self) -> int:
        """The client identifier. This might be useful to identify the client in the federated
        learning process for logging or debugging purposes.

        Returns:
            int: The client identifier.
        """
        return self._index

    @property
    def local_model(self) -> Module:
        """The client's local model.
        This is an alias for `model`.

        Returns:
            torch.nn.Module: The local model.
        """
        return self.model

    @property
    def n_examples(self) -> int:
        """The number of examples in the local training set.

        Returns:
            int: The number of examples in the local training set.
        """
        if isinstance(self.train_set, FastDataLoader):
            return self.train_set.size
        else:
            # Assume that the train_set is a PyTorch DataLoader
            return len(self.train_set.dataset)

    @property
    def channel(self) -> Channel:
        """The communication channel.

        Attention:
            Use this channel to exchange data/information with the server.

        Returns:
            Channel: The communication channel.
        """
        return self._channel

    @property
    def server(self) -> Server:
        """The server to which the client is connected. This reference must only be
        used to send messages through the channel.

        Returns:
            Server: The server.
        """
        return self._server

    def set_server(self, server: Server) -> None:
        """Set the reference to the server. Along with the server, the communication channel is also
        set and the client must use this channel to communicate with the server.

        Args:
            server (Server): The server that orchestrates the federated learning process.
        """
        self._server = server
        self._channel = server.channel

    def receive_model(self) -> None:
        """Receive the global model from the server. This method is responsible for receiving the
        global model from the server and updating the local model accordingly. The model is received
        as a payload of a :class:`fluke.comm.Message` with  ``msg_type="model"`` from the inbox
        of the client itself. The method uses the channel to receive the message.
        """
        msg = self.channel.receive(self, self.server, msg_type="model")
        if self.model is None:
            self.model = msg.payload
        else:
            safe_load_state_dict(self.model, msg.payload.state_dict())

    def send_model(self) -> None:
        """Send the current model to the server. The model is sent as a :class:`fluke.comm.Message`
        with ``msg_type`` "model" to the server. The method uses the channel to send the message.
        """
        self.channel.send(Message(self.model, "model", self, inmemory=True), self.server)

    def local_update(self, current_round: int) -> None:
        """Client's local update procedure.
        Before starting the local training, the client receives the global model from the server.
        Then, the training occurs for a number of ``hyper_params.local_epochs`` epochs using the
        local training set and as loss function the one defined in ``hyper_params.loss_fn``.
        The training happens on the device defined in the client. After the training, the client
        sends the model to the server.

        Args:
            current_round (int): The current round of the federated learning process.
        """
        self._last_round = current_round
        self._load_from_cache()
        self.receive_model()

        if FlukeENV().get_eval_cfg().pre_fit:
            metrics = self.evaluate(FlukeENV().get_evaluator(), self.test_set)
            if metrics:
                self._notify_evaluation(current_round, "pre-fit", metrics)

        self._notify_start_fit(current_round)
        loss = self.fit()
        self._notify_end_fit(current_round, loss)

        if FlukeENV().get_eval_cfg().post_fit:
            metrics = self.evaluate(FlukeENV().get_evaluator(), self.test_set)
            if metrics:
                self._notify_evaluation(current_round, "post-fit", metrics)

        self.send_model()
        self._save_to_cache()

    def _clip_grads(self, model: Module) -> None:
        if self.hyper_params.clipping > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.hyper_params.clipping)

    def fit(self, override_local_epochs: int = 0) -> float:
        """Client's local training procedure.

        Args:
            override_local_epochs (int, optional): Overrides the number of local epochs,
                by default 0 (use the default number of local epochs as in
                ``hyper_params.local_epochs``).

        Returns:
            float: The average loss of the model during the training.
        """
        epochs: int = (override_local_epochs if override_local_epochs > 0
                       else self.hyper_params.local_epochs)

        self.model.train()
        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)

        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(X)
                loss = self.hyper_params.loss_fn(y_hat, y)
                loss.backward()
                self._clip_grads(self.model)
                self.optimizer.step()
                running_loss += loss.item()
            self.scheduler.step()

        running_loss /= (epochs * len(self.train_set))
        self.model.cpu()
        clear_cuda_cache()
        return running_loss

    def evaluate(self,
                 evaluator: Evaluator,
                 test_set: FastDataLoader) -> dict[str, float]:
        """Evaluate the local model on the client's :attr:`test_set`. If the test set is not set or
        the client has not received the global model from the server, the method returns an empty
        dictionary.

        Args:
            evaluator (Evaluator): The evaluator to use for the evaluation.
            test_set (FastDataLoader): The test set to use for the evaluation.

        Returns:
            dict[str, float]: The evaluation results. The keys are the metrics and the values are
            the results.
        """
        model = self.model  # ensure to call the retrieve_obj only once
        if test_set is not None and model is not None:
            evaluation = evaluator.evaluate(self._last_round, model, test_set,
                                            device=self.device)
            return evaluation
        return {}

    def finalize(self) -> None:
        """Finalize the client. This method is called at the end of the federated learning process.
        The default behavior is to receive the global model from the server that is then potentially
        used to evaluate the performance of the model on the local test set.

        Attention:
            When inheriting from this class, make sure to override this method if this behavior is
            not desired.
        """
        self._load_from_cache()
        # This only happens if the federation is stopped with a KeyboardInterrupt
        # and the client was doing the local update
        if self._modopt is None:
            return
        self.receive_model()

        if self.hyper_params.fine_tuning_epochs > 0:
            self.fit(self.hyper_params.fine_tuning_epochs)

        if FlukeENV().get_eval_cfg().pre_fit:
            metrics = self.evaluate(FlukeENV().get_evaluator(), self.test_set)
            if metrics:
                self._notify_evaluation(-1, "pre-fit", metrics)

        self._save_to_cache()

    def state_dict(self) -> dict[str, Any]:
        """Get the client state as a dictionary.

        Returns:
            dict: The client state.
        """
        modopt = retrieve_obj("_modopt", self, pop=False) \
            if isinstance(self._modopt, FlukeCache._ObjectRef) else self._modopt
        return {
            "modopt": modopt.state_dict(),
            "index": self.index,
            "last_round": self._last_round
        }

    def save(self, path: str) -> None:
        """Save the client state to a file.

        Args:
            path (str): The path to the file where the client state will be saved.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str, model: Module) -> dict[str, Any]:
        """Load the client state from a file.

        Args:
            path (str): The path to the file where the client state is saved.
            model (torch.nn.Module): The model to use for the client.

        Returns:
            dict: The loaded lient state.
        """
        state = torch.load(path, weights_only=True)
        if state["modopt"]["model"] is not None:
            self.model = model
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model)
            self._modopt.load_state_dict(state["modopt"])
        else:
            self.model = None

        self._index = state["index"]
        self._last_round = state["last_round"]
        return state

    def __str__(self) -> str:
        hpstr = ", ".join([f"{h}={str(v)}" for h, v in self.hyper_params.items()])
        hpstr = ", " + hpstr if hpstr else ""
        return f"{self.__class__.__name__}[{self._index}](optim={self._optimizer_cfg}, " + \
            f"batch_size={self.train_set.batch_size}{hpstr})"

    def __repr__(self) -> str:
        return str(self)

    def _notify_evaluation(self,
                           round: int,
                           phase: Literal["pre-fit", "post-fit"],
                           metrics: dict[str, float]) -> None:
        for obs in self._observers:
            obs.client_evaluation(round, self.index, phase, metrics)

    def _notify_start_fit(self, round: int) -> None:
        for obs in self._observers:
            obs.start_fit(round, self.index, self.model)

    def _notify_end_fit(self, round: int, loss: float) -> None:
        for obs in self._observers:
            obs.end_fit(round, self.index, self.model, loss)

    def _load_from_cache(self) -> None:
        """Load the model, optimizer, and scheduler from the cache.
        The method retrieves the model, optimizer, and scheduler from the cache and sets them as
        the client's model, optimizer, and scheduler. The method should be called before the
        local update.

        Potential additional attributes that should be loaded from the cache should be added to the
        ``_attr_to_cache`` list.
        """
        if not FlukeENV().is_inmemory():
            for attr_name, attr_value in vars(self).items():
                if isinstance(attr_value, FlukeCache._ObjectRef):
                    new_value = retrieve_obj(attr_name, self)
                    setattr(self, attr_name, new_value)

    def _save_to_cache(self) -> None:
        """Save the model, optimizer, and scheduler to the cache.
        The method should be called after the local update.

        Potential additional attributes that should be saved to the cache should be added to the
        ``_attr_to_cache`` list.
        """
        if not FlukeENV().is_inmemory():
            # Cache the model, optimizer, and scheduler
            for attr_name, attr_value in vars(self).items():
                if not isinstance(attr_value, FlukeCache._ObjectRef) and \
                        isinstance(attr_value, ModOpt):
                    setattr(self, attr_name, cache_obj(attr_value, attr_name, self))

            # Cache additional attributes
            for attr in self._attr_to_cache:
                obj = getattr(self, attr)
                if not isinstance(obj, FlukeCache._ObjectRef):
                    setattr(self, attr, cache_obj(obj, attr, self))


class PFLClient(Client):
    """Personalized Federated Learning client.
    This class is a personalized version of the :class:`Client` class. It is used to implement
    personalized federated learning algorithms. The main difference is that the client has a
    personalized model (i.e., the attribute :attr:`personalized_model`).

    Note:
        The client evaluation is performed using :attr:`personalized_model` instead of the global
        model (i.e., :attr:`fluke.client.Client.model`).
    """

    def __init__(self,
                 index: int,
                 model: Module,
                 train_set: FastDataLoader,
                 test_set: FastDataLoader,
                 optimizer_cfg: OptimizerConfigurator,
                 loss_fn: Module,
                 local_epochs: int = 3,
                 fine_tuning_epochs: int = 0,
                 clipping: float = 0,
                 **kwargs: dict[str, Any]):
        super().__init__(index=index, train_set=train_set, test_set=test_set,
                         optimizer_cfg=optimizer_cfg, loss_fn=loss_fn, local_epochs=local_epochs,
                         fine_tuning_epochs=fine_tuning_epochs, clipping=clipping, **kwargs)
        self._personalized_modopt: ModOpt = ModOpt(model=model)
        self._save_to_cache()

    @property
    def personalized_model(self) -> Module:
        """The personalized model.

        Warning:
            If the model is stored in the cache, the method retrieves it from the cache but does
            not remove it. Thus, the performance may be affected if this property is used to get the
            model multiple times while the model is in the cache.

        Returns:
            torch.nn.Module: The personalized model.
        """
        if isinstance(self._modopt, FlukeCache._ObjectRef):
            return retrieve_obj("_personalized_modopt", self, pop=False).model
        return self._personalized_modopt.model

    @personalized_model.setter
    def personalized_model(self, model: Module | FlukeCache._ObjectRef) -> None:
        self._personalized_modopt.model = model

    @property
    def pers_optimizer(self) -> Optimizer:
        """The optimizer of the personalized model.

        Warning:
            If the optimizer is stored in the cache, the method retrieves it from the cache but does
            not remove it. Thus, the performance may be affected if this property is used to get the
            optimizer multiple times while the optimizer is in the cache.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        if isinstance(self._modopt, FlukeCache._ObjectRef):
            return retrieve_obj("_personalized_modopt", self, pop=False).optimizer
        return self._personalized_modopt.optimizer

    @pers_optimizer.setter
    def pers_optimizer(self, optimizer: Optimizer | FlukeCache._ObjectRef) -> None:
        self._personalized_modopt.optimizer = optimizer

    @property
    def pers_scheduler(self) -> LRScheduler:
        """The learning rate scheduler of the personalized model.

        Warning:
            If the scheduler is stored in the cache, the method retrieves it from the cache but does
            not remove it. Thus, the performance may be affected if this property is used to get the
            scheduler multiple times while the scheduler is in the cache.

        Returns:
            torch.optim.lr_scheduler.LRScheduler: The learning rate scheduler.
        """
        if isinstance(self._modopt, FlukeCache._ObjectRef):
            return retrieve_obj("_personalized_modopt", self, pop=False).scheduler
        return self._personalized_modopt.scheduler

    @pers_scheduler.setter
    def pers_scheduler(self, scheduler: LRScheduler) -> None:
        self._personalized_modopt.scheduler = scheduler

    def evaluate(self, evaluator: Evaluator, test_set: FastDataLoader) -> dict[str, float]:
        """Evaluate the personalized model on the :attr:`test_set`.
        If the test set is not set or the client has no personalized model, the method returns an
        empty dictionary.

        Args:
            evaluator (Evaluator): The evaluator to use for the evaluation.
            test_set (FastDataLoader): The test set to use for the evaluation.

        Returns:
            dict[str, float]: The evaluation results. The keys are the metrics and the values are
            the results.
        """
        if test_set is not None and self.personalized_model is not None:
            return evaluator.evaluate(self._last_round,
                                      self.personalized_model,
                                      test_set,
                                      device=self.device)
        return {}

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        pmodopt = retrieve_obj("_personalized_modopt", self, pop=False) \
            if isinstance(self._personalized_modopt, FlukeCache._ObjectRef) \
            else self._personalized_modopt
        state["personalized_modopt"] = pmodopt.state_dict()
        return state

    def load(self, path: str, model: Module) -> dict[str, Any]:
        state = super().load(path, model)
        self.pers_optimizer, self.pers_scheduler = self._optimizer_cfg(self.personalized_model)
        self._personalized_modopt.load_state_dict(state["personalized_modopt"])
        return state

    @property
    def local_model(self) -> Module:
        """The client's local model.
        This is an alias for :attr:`personalized_model`.

        Returns:
            torch.nn.Module: The local model.
        """
        return self.personalized_model
