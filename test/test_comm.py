from __future__ import annotations

import numpy as np
import torch
import warnings
import sys
sys.path.append(".")
sys.path.append("..")

from fl_bench.comm import Message, Channel, ChannelObserver  # NOQA


def test_message():
    msg = Message(payload="a", msg_type="type_test", sender=None)
    assert msg.msg_type == "type_test"
    assert msg.payload == "a"
    assert msg.sender is None
    assert msg.get_size() == 1
    msg = Message(payload=None, msg_type="type_test", sender=None)
    assert msg.get_size() == 1
    msg = Message(payload=[1, 2, 3, "abc"], msg_type="type_test", sender=None)
    assert msg.get_size() == 6
    msg = Message(payload={"a": 1, "b": 2}, msg_type="type_test", sender=None)
    assert msg.get_size() == 4
    msg = Message(payload=17, msg_type="type_test", sender=None)
    assert msg.get_size() == 1
    msg = Message(payload=17.0, msg_type="type_test", sender=None)
    assert msg.get_size() == 1
    msg = Message(payload=True, msg_type="type_test", sender=None)
    assert msg.get_size() == 1
    msg = Message(payload=np.array([1, 2, 3]), msg_type="type_test", sender=None)
    assert msg.get_size() == 3
    msg = Message(payload=np.array([[1, 2], [3, 4]]), msg_type="type_test", sender=None)
    assert msg.get_size() == 4
    msg = Message(payload=torch.FloatTensor([1, 2, 3]), msg_type="type_test", sender=None)
    assert msg.get_size() == 3
    msg = Message(payload=torch.nn.Linear(10, 10), msg_type="type_test", sender=None)
    assert msg.get_size() == 100 + 10

    class A():
        pass

    warnings.filterwarnings("ignore")
    msg = Message(payload=A(), msg_type="type_test", sender=None)
    assert msg.get_size() == 0


def test_channel():
    class Observer(ChannelObserver):
        def message_received(self, msg):
            self.msg = msg

    channel = Channel()
    assert not channel._buffer
    obs = Observer()
    channel.attach(obs)
    channel.send(Message("a", "type_test", "sender"), "pippo")
    assert len(channel._buffer) == 1
    assert "pippo" in channel._buffer
    msg = channel.receive("pippo")
    assert msg.payload == "a"
    assert msg.msg_type == "type_test"
    assert msg.sender == "sender"
    assert not channel._buffer["pippo"]
    channel.send(Message("a", "type_test", "sender"), "pippo")
    channel.clear("pippo")
    assert not channel._buffer["pippo"]
    assert obs.msg == msg


if __name__ == "__main__":
    test_message()
    test_channel()
