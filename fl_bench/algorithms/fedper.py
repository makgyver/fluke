import sys
sys.path.append(".")
sys.path.append("..")

from copy import deepcopy

from .. import Message
from ..client import PFLClient
from ..algorithms import PersonalizedFL

# https://arxiv.org/abs/1912.00818
class FedPerClient(PFLClient):
    
    def _send_model(self):
        self.channel.send(Message(deepcopy(self.model.get_global()), "model", self), self.server)

    def _receive_model(self) -> None:
        if self.model is None:
            self.model = self.personalized_model
        msg = self.channel.receive(self, self.server, msg_type="model")
        self.model.get_global().load_state_dict(msg.payload.state_dict())
    

class FedPer(PersonalizedFL):

    def get_client_class(self) -> PFLClient:
        return FedPerClient
