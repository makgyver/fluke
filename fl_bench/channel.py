
from collections import defaultdict
import sys; sys.path.append(".")

from typing import Any, Dict, List
from fl_bench import Message


class Channel:
    def __init__(self):
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
            # method(**kwargs)
        else:  
            self._buffer[mbox].append(message)

    def receive(self, mbox: Any, sender:Any=None, msg_type=None) -> Message:
        if sender is None and msg_type is None:
            return self._buffer[mbox].pop()
        
        for i, msg in enumerate(self._buffer[mbox]):
            if sender is None or msg.sender == sender:  # match sender
                if msg_type is None or msg.msg_type == msg_type: # match msg_type
                    return self._buffer[mbox].pop(i)
    
        raise ValueError(f"Message from {sender} with msg type {msg_type} not found in {mbox}")
    
    
    def broadcast(self, message: Message, to: List[Any]):
        for client in to:
            self.send(message, client)
    


# if __name__ == "__main__":
#     class Server:
#         def __init__(self, channel: Channel):
#             self.channel = channel
#             self.clients = [Client(channel, self) for _ in range(10)]

#         def run(self):
#             for msg_type in ['greeting', 'close']:
#                 self.channel.broadcast(Message("#", msg_type=msg_type, sender=self), self.clients)
#                 self.channel.broadcast(Message(("run", {}),msg_type="__action__", sender=self), self.clients)

#                 for client in self.clients:
#                     msg = self.channel.receive(self, sender=client)
#                     print(msg.payload)


#     class Client:
#         def __init__(self, channel: Channel, server: Server):
#             self.channel = channel
#             self.server = server

#         def run(self):
#             msg = self.channel.receive(self)
#             match msg.msg_type:
#                 case 'close':
#                     self.channel.send(Message("bye", msg_type="close", sender=self), self.server)
#                 case 'greeting':
#                     id = msg.payload
#                     channel.send(Message(f"hello from client {id}", msg_type="greating", sender=self), self.server)


#     channel = Channel()
#     server = Server(channel)
#     server.run()