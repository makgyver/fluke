from __future__ import annotations

import os
import shutil
import sys

import torch
from rich.live import Live
from rich.progress import Progress

sys.path.append(".")
sys.path.append("..")

from fluke import DDict, FlukeENV, ObserverSubject, FlukeCache  # NOQA
from fluke.utils import Configuration  # NOQA


def test_env():
    cfg = Configuration("tests/configs/exp.yaml", "tests/configs/alg/fedavg.yaml")
    env = FlukeENV()
    env.set_seed(117)
    assert env.get_seed() == 117
    env.set_device("cpu")
    assert env.get_device() == torch.device("cpu")
    env.set_device("cuda")
    assert env.get_device() == torch.device("cuda")
    env.set_device("mps")
    assert env.get_device() == torch.device("mps")
    env.set_device("auto")
    env.set_eval_cfg(DDict(pre_fit=True))
    assert env.get_eval_cfg().pre_fit
    env.set_save_options(path="temp", save_every=10, global_only=True)
    assert env.get_save_options() == ("temp", 10, True)
    if torch.cuda.is_available():
        assert env.get_device() == torch.device("cuda")
    elif torch.backends.mps.is_available():
        assert env.get_device() == torch.device("mps")
    else:
        assert env.get_device() == torch.device("cpu")
    live = env.get_live_renderer()
    assert live is not None and isinstance(live, Live)
    progress_fl = env.get_progress_bar("FL")
    progress_s = env.get_progress_bar("clients")
    progress_c = env.get_progress_bar("server")
    assert progress_fl is not None and isinstance(progress_fl, Progress)
    assert progress_s is not None and isinstance(progress_s, Progress)
    assert progress_c is not None and isinstance(progress_c, Progress)
    assert progress_fl != progress_s
    assert progress_fl != progress_c
    assert progress_s != progress_c

    env.configure(cfg)
    assert env.get_seed() == 42

    env.force_close()

    env.open_cache("test_env")
    assert env.get_cache() is None

    shutil.rmtree("tmp/tmp", ignore_errors=True)

    env.set_inmemory(False)
    env.open_cache("test_env")
    assert env.get_cache() is not None
    assert os.path.exists(env.get_cache().cache_dir)

    cache = env.get_cache()
    print(list(cache._cache.iterkeys()))
    assert cache.occupied == 0
    ref = cache.push("test1", "this is a test")
    assert isinstance(ref, FlukeCache._ObjectRef)
    assert cache.occupied == 1
    assert cache.get("test1") == "this is a test"
    assert cache.get("test2") is None
    assert cache.get("test2", "default") == "default"
    cache.push("test2", ref)
    assert cache.occupied == 1
    assert cache._cache[ref.id].refs == 2

    test1 = cache.pop("test1")
    assert test1 == "this is a test"
    assert cache.occupied == 1
    assert cache._cache[ref.id].refs == 1

    cache.cleanup()
    assert cache.occupied == 1
    cache.delete("test2")
    assert cache.occupied == 0
    cache.cleanup()
    assert cache.occupied == 0
    FlukeENV().close_cache()


def test_observer():
    subj = ObserverSubject()
    assert subj._observers == []
    subj.attach("test")
    assert subj._observers == ["test"]
    subj.detach("test")
    assert subj._observers == []

    subj.detach("test2")


def test_ddict():
    dd = DDict(**{
        "a": 1,
        "b": 2,
        "c": {
            "d": 3,
            "e": 4
        }
    })

    assert dd.a == 1
    assert dd.b == 2
    assert dd.c.d == 3
    assert dd.c.e == 4

    dd_nota = dd.exclude("a")
    assert "a" not in dd_nota
    assert dd_nota.b == 2

    dd = DDict({"a": 1, "b": 2}, c=3)
    assert dd.a == 1
    assert dd.b == 2
    assert dd.c == 3

    dd = DDict({"a": 1, "b": 2}, c=3)
    dd2 = DDict({"a": 1, "b": 2}, c=3)
    assert dd.match(dd2)
    assert dd2.match(dd, full=True)
    assert dd.diff(dd2) == {}
    d = DDict(a=1, b=2, c=3, e=DDict(a=1, b=2, c=3))
    e = DDict(a=1, b=3, c=4, e=DDict(a=1, b=1))
    assert d.diff(e) == {'b': 3, 'c': 4, 'e': {'b': 1}}
    assert e.diff(d) == {'b': 2, 'c': 3, 'e': {'b': 2, 'c': 3}}
    assert not d.match(e)


if __name__ == "__main__":
    test_env()
    test_observer()
    test_ddict()
    # coverage: 94% __init__.py
