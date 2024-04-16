from collections import defaultdict
from typing import Any, Dict, List, Optional
import sys
import numpy as np
import torch
import warnings
sys.path.append(".")

from . import ObserverSubject  # NOQA

__all__ = [
    'Message',
    'Channel'
]


class Message:
    """Message class.

    This class represents a message that can be exchanged between clients and the server.

    Attributes:
        msg_type (str): The type of the message.
        payload (Any): The payload of the message.
        sender (Any): The sender of the message.
    """

    def __init__(self,
                 payload: Any,
                 msg_type: str = "model",
                 sender: Optional[Any] = None):
        self.msg_type: str = msg_type
        self.payload: Any = payload
        self.sender: Optional[Any] = sender

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

    def get_size(self) -> int:
        """Get the size of the message.

        The message size is the size of the payload in "floating point" numbers. For example, a
        message containing a tensor of size (10, 10) has a size of 100. A message containing a
        string of length 10 has a size of 10.
        A message containing an ACK (i.e., with no payload) has a size of 1.

        Returns:
            int: The size of the message in bytes.
        """
        return self.__get_size(self.payload)


class ChannelObserver():
    """Channel observer interface.

    This interface is used to observe the communication channel during the federated learning
    process.

    See Also:
        `ServerObserver`, `ObserverSubject`

    """

    def message_received(self, message: Message):
        pass


class Channel(ObserverSubject):
    """A bi-directional communication channel. It is used to send and receive messages between the
    parties.

    The Channel class implements the Observer pattern. It notifies the observers when a message is
    received. Clients and server are supposed to use a channel to communicate with each other.

    Attributes:
        _buffer (Dict[Any, List[Message]]): The buffer to store the unread messages. The key is the
            recipient and the value is a list of messages.
    """

    def __init__(self):
        super().__init__()
        self._buffer: Dict[Any, List[Message]] = defaultdict(list)

    def send(self, message: Message, mbox: Any):
        """Send a message to a receiver.

        To any sent message should correspond a received message. The receiver should call the
        `receive` method of the channel to get the message.

        Args:
            message (Message): The message to be sent.
            mbox (Any): The receiver.

        Example:
            Sending a string message from the `server` to a `client`:
            >>> channel = Channel()
            >>> channel.send(Message("Hello", "greeting", server), client)
        """
        self._buffer[mbox].append(message)

    def receive(self, mbox: Any, sender: Any = None, msg_type: str = None) -> Message:
        """Receive (read) a message from a sender.

        Args:
            mbox (Any): The receiver.
            sender (Any): The sender.
            msg_type (str): The type of the message.

        Returns:
            Message: The received message.

        Raises:
            ValueError: message not found in the message box of the receiver with the
                given sender and message type.
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

    def broadcast(self, message: Message, to: List[Any]) -> None:
        """Send a message to a list of receivers.

        Args:
            message (Message): The message to be sent.
            to (List[Any]): The list of receivers.
        """
        for client in to:
            self.send(message, client)

    def clear(self, mbox: Any) -> None:
        """Clear the message box of the given receiver.

        Note:
            Any unread message will be lost after calling this method.
            Lost messages are not considered in the communication protocol.

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
