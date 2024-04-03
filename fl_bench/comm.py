import sys
sys.path.append(".")

import pickle
from typing import Any, Dict, List, Optional
from collections import defaultdict

from . import ObserverSubject

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
                 msg_type: str="model",
                 sender: Optional[Any]=None):
        self.msg_type: str = msg_type
        self.payload: Any = payload
        self.sender: Optional[Any] = sender
    
    def get_size(self) -> int:
        """Get the size of the message.

        The message size is the size of the payload in bytes estimated using the `pickle` module.
        A message containing an ACK (i.e., with no payload) has a size of 1 byte.

        Returns:
            int: The size of the message in bytes.
        """
        if self.payload is None:
            return 1
        return sys.getsizeof(pickle.dumps(self.payload))


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

    def _send_action(self, method, kwargs, mbox):
        if callable(method):
            method(**kwargs)
        else:
            getattr(mbox, method)(**kwargs)


    def send(self, message: Message, mbox: Any):
        """Send a message to a receiver.

        If the message is a request to take an action (i.e., `type = '__action__'`), the payload
        should be a tuple with the method to be called and the keyword arguments (as a dictionary) 
        to be passed to the method.

        To any sent message should correspond a received message. The receiver should call the
        `receive` method of the channel to get the message. However, if the message is a request to
        take an action, the channel directly calls the method on the receiver with the given keyword 
        arguments.

        Args:
            message (Message): The message to be sent.
            mbox (Any): The receiver.
        
        Example:
            Sending a string message from the `server` to a `client`:
            >>> channel = Channel()
            >>> channel.send(Message("Hello", "greeting", server), client)

            If the `server` wants to request the clients to run the `train` method which has an 
            argument `epochs`, it can send a message with the following payload:
            >>> message = Message((client.train, {"epochs": 3}), "__action__", server)
            >>> channel.send(message, client)
            This will call the `train` method of the client with the argument `epochs=3`.
        """
        if message.msg_type == "__action__":
            method, kwargs = message.payload
            self._send_action(method, kwargs, mbox)
        else:  
            self._buffer[mbox].append(message)

    def receive(self, mbox: Any, sender: Any=None, msg_type: str=None) -> Message:
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
            self.notify_message_received(msg)
            return msg
        
        for i, msg in enumerate(self._buffer[mbox]):
            if sender is None or msg.sender == sender:  # match sender
                if msg_type is None or msg.msg_type == msg_type: # match msg_type
                    msg = self._buffer[mbox].pop(i)
                    self.notify_message_received(msg)
                    return msg
    
        raise ValueError(f"Message from {sender} with msg type {msg_type} not found in {mbox}")
    
    def broadcast(self, message: Message, to: List[Any]):
        """Send a message to a list of receivers.

        Args:
            message (Message): The message to be sent.
            to (List[Any]): The list of receivers.
        """
        for client in to:
            self.send(message, client)
    
    def notify_message_received(self, message: Message):
        """Notify the observers that a message has been received.

        Args:
            message (Message): The message received.
        """
        for observer in self._observers:
            observer.message_received(message)
