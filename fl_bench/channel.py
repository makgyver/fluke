import sys; sys.path.append(".")

from typing import Any, Dict, List
from collections import defaultdict

from fl_bench import Message, ObserverSubject


class Channel(ObserverSubject):
    def __init__(self):
        super().__init__()
        self._buffer: Dict[Any, Message] = defaultdict(list)

    def _send_action(self, method, kwargs, mbox):
        if callable(method):
            method(**kwargs)
        else:
            getattr(mbox, method)(**kwargs)


    def send(self, message: Message, mbox: Any):
        if message.msg_type == "__action__":
            method, kwargs = message.payload
            self._send_action(method, kwargs, mbox)
        else:  
            self._buffer[mbox].append(message)

    def receive(self, mbox: Any, sender:Any=None, msg_type=None) -> Message:
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
        for client in to:
            self.send(message, client)
    
    def notify_message_received(self, message: Message):
        for observer in self._observers:
            observer.message_received(message)
