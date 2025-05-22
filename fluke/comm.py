"""This module contains the classes for the communication between the clients and the server."""

from __future__ import annotations

import sys
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import torch

sys.path.append(".")

from . import FlukeCache, FlukeENV, ObserverSubject, custom_formatwarning  # NOQA
from .utils import cache_obj, retrieve_obj  # NOQA

warnings.formatwarning = custom_formatwarning

__all__ = ["Message", "Channel", "ChannelObserver"]


class Message:
    """This class represents a message that can be exchanged between clients and the server.
    The message contains a payload, a type and a sender. The payload can be of
    any type. The type of a message is a string that describes the content of the message.
    The sender is the object that sends the message.

    Args:
        payload (Any): The content of the message.
        msg_type (str): The type of the message. Default is "model".
        sender (Optional[Any]): The sender of the message. Default is ``None``.
        inmemory (Optional[bool]): If ``True``, the payload is stored in memory. If ``False``, the
            payload is stored in the cache. If ``None``, the payload is stored according to the
            setting in the :class:`fluke.FlukeENV`. Note that when ``inmemory`` is ``False``,
            the payload is stored in the cache if and only if the environment is not in memory,
            otherwise the payload is stored in memory. Default is ``None``.

    Note:
        The payload can be stored in memory if and only if its type is :class:`torch.nn.Module`,
        :class:`torch.Tensor` or :class:`fluke.FlukeCache.ObjectRef`. In the latter case, the
        reference count of the object is incremented by one.

    Example:
        Let us consider a simple example where a client wants to send prepare a  message to be sent
        to the server with the content "Hello". The client can create a message as follows:

        .. code-block:: python
            :linenos:

            message = Message(payload="Hello", msg_type="greeting", sender=client)
            # Then through a channel (of type Channel) the message can be sent to the server
            # channel.send(message, server)

    See Also:
        :class:`Channel`
    """

    def __init__(
        self,
        payload: Any,
        msg_type: str = "model",
        sender: Optional[Any] = None,
        inmemory: Optional[bool] = None,
    ):
        self.__id: str = str(uuid.uuid4().hex)
        self.__msg_type: str = msg_type
        self.__payload: Any = payload
        self.__sender: Optional[Any] = sender
        self.__size: int = self.__get_size(payload)
        self.__inmemory: bool = inmemory if inmemory is not None else True

        if not FlukeENV().is_inmemory() and not inmemory:
            if isinstance(payload, (torch.nn.Module, FlukeCache.ObjectRef, torch.Tensor)):
                self.__payload = cache_obj(payload, f"{self.__id}")
                self.__inmemory = False

    @property
    def id(self) -> str:
        """Get the unique identifier of the message.

        Returns:
            str: The unique identifier of the message.
        """
        return self.__id

    @property
    def msg_type(self) -> str:
        """Get the type of the message.

        Returns:
            str: The type of the message.
        """
        return self.__msg_type

    @property
    def payload(self) -> Any:
        """Get the payload of the message.

        Note:
            If the payload is stored in the cache, it is loaded in memory when this property is
            called for the first time on the object.

        Returns:
            Any: The payload of the message.
        """
        if not self.__inmemory:
            self.__payload = retrieve_obj(f"{self.__id}")
            self.__inmemory = True

        return self.__payload

    @property
    def sender(self) -> Optional[Any]:
        """Get the sender of the message.

        Returns:
            Optional[Any]: The sender of the message.
        """
        return self.__sender

    def __get_size(self, obj: Any) -> int:
        if obj is None or isinstance(obj, (int, float, bool, np.generic)):
            return 1
        elif isinstance(obj, str):
            return len(obj)
        elif isinstance(obj, (list, tuple, set)):
            return sum([self.__get_size(i) for i in obj])
        elif isinstance(obj, dict):
            return self.__get_size(list(obj.values())) + self.__get_size(list(obj.keys()))
        elif isinstance(obj, np.ndarray):
            return obj.size
        elif isinstance(obj, torch.Tensor):
            return obj.numel()
        elif isinstance(obj, torch.nn.Module):
            return sum(p.numel() for p in obj.parameters())
        elif isinstance(obj, FlukeCache.ObjectRef):
            return 0
        else:
            warnings.warn(
                f"Unknown type {type(obj)} of object {obj} in payload."
                + "Returning object size = 0."
            )
            return 0

    def clone(self, inmemory: Optional[bool] = None) -> Message:
        """Clone the message. The cloned message contains a deepcopy of the payload while
        keeping the same message type and the same reference to the sender.

        Args:
            inmemory (Optional[bool]): If ``True``, the payload is stored in memory. If False, the
                payload is stored in the cache. If None, the payload is stored according to the
                setting in the :class:`fluke.FlukeENV`.

        Returns:
            Message: The cloned message.
        """
        msg = Message(deepcopy(self.__payload), self.msg_type, self.sender, inmemory)
        msg.__size = self.__size
        return msg

    @property
    def size(self) -> int:
        """Get the size of the message. The message size is the size of the payload calculated in
        terms of "floating point" numbers. For example, a message containing a tensor of size
        (10, 10) has a size of 100. A message containing a string of length 10 has a size of 10.
        A message containing an ACK (i.e., with no payload) has a size of 1. In case of unknown
        types, a warning is raised and the size is set to 0.

        Returns:
            int: The size of the message in bytes.

        Example:

            .. code-block:: python
                :linenos:

                message = Message("Hello", "greeting", client)
                print(message.size)  # 5

                message = Message(torch.randn(10, 10), "tensor", client)
                print(message.size)  # 100

                message = Message(None, "ack", client)
                print(message.size)  # 1
        """
        return self.__size

    def __eq__(self, other: Message) -> bool:
        return (
            self.__payload == other.__payload
            and self.msg_type == other.msg_type
            and self.sender == other.sender
        )

    def __str__(self, indent: int = 0) -> str:
        strname = f"Message[{self.id}]"
        indentstr = " " * (indent + len(strname) + 1)
        tostr = f"{strname}(type={self.msg_type},"
        tostr += f"{indentstr}from={self.sender}, "
        tostr += f"{indentstr}payload={self.__payload}, "
        tostr += f"{indentstr}size={self.size}, "
        tostr += f"{indentstr}inmemory={self.__inmemory})"
        return tostr

    def __repr__(self, indent: int = 0) -> str:
        return self.__str__(indent=indent)

    def ram(self) -> bool:
        """Store the payload in memory.
        This method is used when the payload needs to be moved from the cache to memory.

        Returns:
            bool: ``True`` if the payload is moved to memory, ``False`` otherwise.
        """
        if not self.__inmemory:
            self.__payload = retrieve_obj(f"{self.__id}")
            self.__inmemory = True
            return True
        return False

    def cache(self) -> bool:
        """Store the payload in the cache.
        This method is used when the payload needs to be moved from memory to the cache.

        Returns:
            bool: ``True`` if the payload is moved to the cache, ``False`` otherwise.
        """
        if self.__inmemory:
            if isinstance(self.__payload, (torch.nn.Module, FlukeCache.ObjectRef, torch.Tensor)):
                self.__payload = cache_obj(self.__payload, f"{self.__id}")
                self.__inmemory = False
                return True
        return False


class ChannelObserver:
    """Channel observer interface for the Observer pattern.
    This interface is used to observe the communication channel during the federated learning
    process.

    See Also:
        :class:`fluke.utils.ServerObserver`, :class:`fluke.ObserverSubject`

    """

    def message_received(self, by: Any, message: Message) -> None:
        """This method is called when a message is received, i.e., when a message is read from the
        message box of the receiver.

        Args:
            by (Any): The receiver of the message.
            message (Message): The message received.
        """
        pass

    def message_sent(self, to: Any, message: Message) -> None:
        """This method is called when a message is sent, i.e., when a message is written to the
        message box of the receiver.

        Args:
            to (Any): The receiver of the message.
            message (Message): The message sent.
        """
        pass

    def message_broadcasted(self, to: list[Any], message: Message) -> None:
        """This method is called when a message is broadcasted, i.e., when a message is written to
        the message box of all the receivers.

        Args:
            to (list[Any]): The list of receivers of the message.
            message (Message): The message broadcasted.
        """
        pass


class Channel(ObserverSubject):
    """A bi-directional communication channel. It is used to send and receive messages between the
    parties.
    The Channel class implements the Observer pattern. It notifies the observers when a message is
    received. Clients and server are supposed to use a channel to communicate with each other.

    Example:

        Let us consider a simple example where a ``client`` sends a message to the ``server``.

        .. code-block:: python
            :linenos:

            channel = Channel()
            channel.send(Message("Hello", "greeting", server), client)
            mag: Message = channel.receive(server, client, "greeting")
            print(msg.payload) # Hello
    """

    def __init__(self):
        super().__init__()
        self._buffer: dict[Any, list[Message]] = defaultdict(list)

    def __getitem__(self, mbox: Any) -> list[Message]:
        """Get the messages of the given receiver.

        Args:
            mbox (Any): The receiver.

        Returns:
            list[Message]: The list of messages sent to the receiver.
        """
        return self._buffer[mbox]

    @property
    def buffer(self) -> dict[Any, list[Message]]:
        """Get the buffer of the channel. The buffer stores the unread messages in a dictionary.
        The keys are the recipients and the values are the list of messages sent to the recipient.

        Returns:
            dict[Any, list[Message]]: The buffer of the channel.
        """
        return self._buffer

    def send(self, message: Message, mbox: Any) -> None:
        """Send a copy of the message to a receiver.
        To any sent message should correspond a received message. The receiver should call the
        `receive` method of the channel to get the message.

        Args:
            message (Message): The message to be sent.
            mbox (Any): The receiver.

        Example:
            Sending a string message from the ``server`` to a ``client``:

            .. code-block:: python
                :linenos:

                channel = Channel()
                channel.send(Message("Hello", "greeting", server), client)

        """
        msg = message.clone()
        self._buffer[mbox].append(msg)
        self.notify(event="message_sent", message=msg, to=mbox)

    def receive(self, mbox: Any, sender: Any = None, msg_type: str = None) -> Message:
        """Receive (i.e., read) a message from a sender. The message is removed from the message box
        of the receiver. If both ``sender`` and ``msg_type`` are None, the first message in the
        message box is returned. If ``sender`` is None, the first message with the given
        ``msg_type`` is returned. If ``msg_type`` is None, the first message from the given
        ``sender`` is returned.

        Args:
            mbox (Any): The receiver.
            sender (Any): The sender.
            msg_type (str): The type of the message.

        Returns:
            Message: The received message.

        Raises:
            ValueError: message not found in the message box of the receiver with the
                given sender and message type.

        Example:
            Receiving a message from the ``server`` with message type ``greeting``:

            .. code-block:: python
                :linenos:

                channel = Channel()
                message = channel.receive(client, server, "greeting")
        """
        if sender is None and msg_type is None:
            msg = self._buffer[mbox].pop()
            self.notify(event="message_received", message=msg, by=mbox)
            return msg

        for i, msg in enumerate(self._buffer[mbox]):
            if sender is None or msg.sender == sender:  # match sender
                if msg_type is None or msg.msg_type == msg_type:  # match msg_type
                    msg = self._buffer[mbox].pop(i)
                    self.notify(event="message_received", message=msg, by=mbox)
                    return msg

        raise ValueError(f"Message from {sender} with msg type {msg_type} not found in {mbox}")

    def broadcast(self, message: Message, to: list[Any]) -> None:
        """Send a copy of the message to a list of receivers.

        Note:
            This method may have a side effect on the payload of the message.
            The payload is moved to memory after the message is sent to avoid dangling references.
            If the payload was already in memory, nothing is done.

        Important:
            When a message is broadcasted, a copy of the message is sent to each receiver. If the
            message is not stored in memory, the payload is a reference to an object stored in the
            cache, and thus the reference count of the object is incremented by the number of
            receivers. After the message is sent, the sent message is moved to memory to avoid
            dangling references.

        Args:
            message (Message): The message to be sent.
            to (list[Any]): The list of receivers.
        """
        for client in to:
            self.send(message, client)
        message.ram()
        self.notify(event="message_broadcasted", message=message, to=to)

    def clear(self, mbox: Any) -> None:
        """Clear the message box of the given receiver.

        Caution:
            Any unread message will be lost after calling this method.
            Lost messages are not considered in the communication protocol thus they are not
            accounted for in the communication cost.

        Args:
            mbox (Any): The receiver.
        """
        if not FlukeENV().is_inmemory():
            cache = FlukeENV().get_cache()
            for msg in self._buffer[mbox]:
                cache.delete(f"{msg.id}")

        self._buffer[mbox].clear()
