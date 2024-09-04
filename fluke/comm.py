"""This module contains the classes for the communication between the clients and the server."""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import torch

sys.path.append(".")

from . import ObserverSubject  # NOQA

__all__ = [
    'Message',
    'Channel',
    'ChannelObserver'
]


class Message:
    """This class represents a message that can be exchanged between clients and the server. This
    type is immutable. The message contains a payload, a type and a sender. The payload can be of
    any type. The type is a string that describes the content of the message. The sender is the
    object that sends the message.

    Example:
        Let us consider a simple example where a client wants to send preparea  message to be sent
        to the server with the content "Hello". The client can create a message as follows:

        .. code-block:: python
            :linenos:

            message = Message(payload="Hello", msg_type="greeting", sender=client)
            # Then through a channel (of type Channel) the message can be sent to the server
            # channel.send(message, server)

    See Also:
        :class:`Channel`
    """

    def __init__(self,
                 payload: Any,
                 msg_type: str = "model",
                 sender: Optional[Any] = None):
        self.__msg_type: str = msg_type
        self.__payload: Any = payload
        self.__sender: Optional[Any] = sender

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

        Returns:
            Any: The payload of the message.
        """
        return self.__payload

    @property
    def sender(self) -> Optional[Any]:
        """Get the sender of the message.

        Returns:
            Optional[Any]: The sender of the message.
        """
        return self.__sender

    def __get_size(self, obj: Any) -> int:
        if obj is None or isinstance(obj, (int, float, bool)):
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
        elif isinstance(obj, (torch.nn.Module)):
            return sum(p.numel() for p in obj.parameters())
        else:
            warnings.warn(f"Unknown type {type(obj)} of object {obj} in payload." +
                          "Returning object size = 0.")
            return 0

    def clone(self) -> Message:
        """Clone the message. The cloned message containes a deepcopy of the payload while
        keeping the same message type and the same reference to the sender.

        Returns:
            Message: The cloned message.
        """
        return Message(deepcopy(self.payload), self.msg_type, self.sender)

    def get_size(self) -> int:
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
                print(message.get_size())  # 5

                message = Message(torch.randn(10, 10), "tensor", client)
                print(message.get_size())  # 100

                message = Message(None, "ack", client)
                print(message.get_size())  # 1
        """
        return self.__get_size(self.payload)

    def __eq__(self, other: Message) -> bool:
        return self.payload == other.payload and self.msg_type == other.msg_type and \
            self.sender == other.sender

    def __str__(self) -> str:
        return f"Message(type={self.msg_type}, from={self.sender}, payload={self.payload}, " + \
            f"size={self.get_size()})"

    def __repr__(self) -> str:
        return str(self)


class ChannelObserver():
    """Channel observer interface for the Observer pattern.
    This interface is used to observe the communication channel during the federated learning
    process.

    See Also:
        :class:`fluke.utils.ServerObserver`, :class:`fluke.ObserverSubject`

    """

    def message_received(self, message: Message) -> None:
        """This method is called when a message is received, i.e., when a message is read from the
        message box of the receiver.

        Args:
            message (Message): The message received.
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
        self._buffer[mbox].append(message.clone())

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
            self._notify_message_received(msg)
            return msg

        for i, msg in enumerate(self._buffer[mbox]):
            if sender is None or msg.sender == sender:  # match sender
                if msg_type is None or msg.msg_type == msg_type:  # match msg_type
                    msg = self._buffer[mbox].pop(i)
                    self._notify_message_received(msg)
                    return msg

        raise ValueError(f"Message from {sender} with msg type {msg_type} not found in {mbox}")

    def broadcast(self, message: Message, to: list[Any]) -> None:
        """Send a copy of the message to a list of receivers.

        Args:
            message (Message): The message to be sent.
            to (list[Any]): The list of receivers.
        """
        for client in to:
            self.send(message, client)

    def clear(self, mbox: Any) -> None:
        """Clear the message box of the given receiver.

        Caution:
            Any unread message will be lost after calling this method.
            Lost messages are not considered in the communication protocol thus they are not
            accounted for in the communication cost.

        Args:
            mbox (Any): The receiver.
        """
        self._buffer[mbox].clear()

    def _notify_message_received(self, message: Message) -> None:
        """Notify the observers that a message has been received.

        Args:
            message (Message): The message received.
        """
        for observer in self._observers:
            observer.message_received(message)
