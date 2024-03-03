import sys; sys.path.append(".")

from fl_bench.algorithms import CentralizedFL

from copy import deepcopy

from fl_bench import Message
from fl_bench.client import Client


# https://arxiv.org/abs/2102.07078

class FedRepClient(Client):
    
    def _send_model(self):
        self.channel.send(Message(deepcopy(self.model.fed_E), "model", self), self.server)

    def _receive_model(self) -> None:
        msg = self.channel.receive(self, self.server, msg_type="model")
        self.model.fed_E.load_state_dict(msg.payload.state_dict())
    

class FedRep(CentralizedFL):

    def get_client_class(self) -> Client:
        return FedRepClient
