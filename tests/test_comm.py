from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest
import torch

sys.path.append(".")
sys.path.append("..")

from fluke import FlukeENV, FlukeCache  # NOQA
from fluke.comm import Channel, ChannelObserver, Message  # NOQA


def test_message():
    msg = Message(payload="a", msg_type="type_test", sender=None)
    assert msg.msg_type == "type_test"
    assert msg.payload == "a"
    assert msg.sender is None
    assert msg.size == 1
    msg = Message(payload=None, msg_type="type_test", sender=None)
    assert msg.size == 1
    msg = Message(payload=[1, 2, 3, "abc"], msg_type="type_test", sender=None)
    assert msg.size == 6
    msg = Message(payload={"a": 1, "b": 2}, msg_type="type_test", sender=None)
    assert msg.size == 4
    msg = Message(payload=17, msg_type="type_test", sender=None)
    assert msg.size == 1
    msg = Message(payload=17.0, msg_type="type_test", sender=None)
    assert msg.size == 1
    msg = Message(payload=True, msg_type="type_test", sender=None)
    assert msg.size == 1
    msg = Message(payload=np.array([1, 2, 3]), msg_type="type_test", sender=None)
    assert msg.size == 3
    msg = Message(payload=np.array([[1, 2], [3, 4]]), msg_type="type_test", sender=None)
    assert msg.size == 4
    msg = Message(payload=torch.FloatTensor([1, 2, 3]), msg_type="type_test", sender=None)
    assert msg.size == 3
    msg = Message(payload=torch.nn.Linear(10, 10), msg_type="type_test", sender=None)
    assert msg.size == 100 + 10
    msg = Message(payload=FlukeCache.ObjectRef(), msg_type="type_test", sender=None)
    assert msg.size == 0

    class A():
        pass

    warnings.filterwarnings("ignore")
    msg = Message(payload=A(), msg_type="type_test", sender=None)
    assert msg.size == 0

    msg2 = Message(payload="a", msg_type="type_test", sender=None)
    assert msg.id != msg2.id

    msg = Message(payload="prova", msg_type="type_test", sender=None)
    assert str(msg) == repr(msg)

    tensor = torch.FloatTensor([1, 2, 3])
    FlukeENV().set_inmemory(False)
    FlukeENV().open_cache("tmp_comm")
    msg = Message(tensor, "type_test", "sender", inmemory=False)
    assert FlukeENV().get_cache().occupied == 1
    assert not msg.cache()
    msg.ram()
    assert FlukeENV().get_cache().occupied == 0
    assert msg.cache()
    assert FlukeENV().get_cache().occupied == 1
    assert torch.all(msg.payload == tensor)
    assert FlukeENV().get_cache().occupied == 0
    FlukeENV().get_cache().cleanup()
    FlukeENV().close_cache()


def test_channel():

    chobs = ChannelObserver()
    chobs.message_received("b", Message("a", "type_test", "sender"))
    assert isinstance(chobs, ChannelObserver)

    class Observer(ChannelObserver):
        def message_received(self, by, message):
            self.msg = message

        def message_sent(self, to, message):
            return super().message_sent(to, message)

        def message_broadcast(self, msg):
            return super().message_broadcast(msg)

    FlukeENV().open_cache("tmp_comm")
    channel = Channel()
    assert not channel.buffer
    obs = Observer()
    channel.attach(obs)
    msg = Message("a", "type_test", "sender")
    channel.send(msg, "pippo")
    assert len(channel.buffer) == 1
    assert "pippo" in channel._buffer
    assert len(channel["pippo"]) == 1
    assert channel["pippo"][0] == msg
    msg = channel.receive("pippo")
    assert msg.payload == "a"
    assert msg.msg_type == "type_test"
    assert msg.sender == "sender"
    assert not channel.buffer["pippo"]
    channel.send(Message("a", "type_test", "sender"), "pippo")
    channel.clear("pippo")
    assert not channel.buffer["pippo"]
    assert obs.msg == msg

    channel.broadcast(Message("a", "type_test", "sender"), ["pippo", "pluto"])
    assert len(channel._buffer) == 2
    assert "pippo" in channel.buffer
    assert "pluto" in channel.buffer

    channel.send(Message("b", "type_test", "sender"), "pippo")
    msg = channel.receive("pippo", "sender", "type_test")
    assert msg.payload == "a"
    msg = channel.receive("pippo", "sender", "type_test")
    assert msg.payload == "b"

    with pytest.raises(ValueError):
        channel.receive("pippo", "sender", "type_test")

    FlukeENV().get_cache().cleanup()
    FlukeENV().close_cache()


if __name__ == "__main__":
    test_message()
    test_channel()
    # 99% coverage comm.py
