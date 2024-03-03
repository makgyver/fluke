import sys; sys.path.append(".")

from fl_bench.algorithms import PersonalizedFL

from copy import deepcopy

from fl_bench import Message
from fl_bench.client import PFLClient


# https://arxiv.org/abs/1912.00818
class FedPerClient(PFLClient):
    
    def _send_model(self):
        self.channel.send(Message(deepcopy(self.model.fed_E), "model", self), self.server)

    def _receive_model(self) -> None:
        if self.model is None:
            self.model = self.private_model
        msg = self.channel.receive(self, self.server, msg_type="model")
        self.model.fed_E.load_state_dict(msg.payload.state_dict())
    

class FedPer(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return FedPerClient
