from __future__ import annotations

import sys

import torch
from rich.live import Live
from rich.progress import Progress

sys.path.append(".")
sys.path.append("..")

from fluke import DDict, GlobalSettings, ObserverSubject  # NOQA


def test_settings():
    settings = GlobalSettings()
    settings.set_seed(42)
    assert settings.get_seed() == 42
    settings.set_device("cpu")
    assert settings.get_device() == torch.device("cpu")
    settings.set_device("cuda")
    assert settings.get_device() == torch.device("cuda")
    settings.set_device("mps")
    assert settings.get_device() == torch.device("mps")
    settings.set_device("auto")
    settings.set_eval_cfg(DDict(pre_fit=True))
    assert settings.get_eval_cfg().pre_fit
    if torch.cuda.is_available():
        assert settings.get_device() == torch.device("cuda")
    elif torch.backends.mps.is_available():
        assert settings.get_device() == torch.device("mps")
    else:
        assert settings.get_device() == torch.device("cpu")
    live = settings.get_live_renderer()
    assert live is not None and isinstance(live, Live)
    progress_fl = settings.get_progress_bar("FL")
    progress_s = settings.get_progress_bar("clients")
    progress_c = settings.get_progress_bar("server")
    assert progress_fl is not None and isinstance(progress_fl, Progress)
    assert progress_s is not None and isinstance(progress_s, Progress)
    assert progress_c is not None and isinstance(progress_c, Progress)
    assert progress_fl != progress_s
    assert progress_fl != progress_c
    assert progress_s != progress_c


def test_observer():
    subj = ObserverSubject()
    assert subj._observers == []
    subj.attach("test")
    assert subj._observers == ["test"]
    subj.detach("test")
    assert subj._observers == []


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


if __name__ == "__main__":
    test_settings()
    test_observer()
    test_ddict()
    # coverage: 94% __init__.py
