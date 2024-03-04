import sys; sys.path.append(".")
from copy import deepcopy

from fl_bench import Message
from fl_bench.client import Client, PFLClient
from fl_bench.algorithms import PersonalizedFL

# FEDerated DISEntanglment Learning
class FedDiselClient(PFLClient):

    def _send_model(self):
        self.channel.send(Message(deepcopy(self.model.fed_E), "model", self), self.server)

    def _receive_model(self) -> None:
        if self.model is None:
            self.model = self.personalized_model
        msg = self.channel.receive(self, self.server, msg_type="model")
        self.model.fed_E.load_state_dict(msg.payload.state_dict())


class FedDisel(PersonalizedFL):
    
    def get_client_class(self) -> Client:
        return FedDiselClient
