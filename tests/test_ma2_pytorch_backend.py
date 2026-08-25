"""PyTorchSession orchestration verified against stubbed InferenceCore."""

import contextlib

import numpy as np
import pytest

from video_upscaler.matanyone2 import pytorch_backend as pb
from video_upscaler.matanyone2.backend import BackendSelection


class FakeCore:
    def __init__(self):
        self.calls = []

    def step(self, frame, mask=None, objects=None, first_frame_pred=False):
        self.calls.append(
            {
                "masked": mask is not None,
                "pred": bool(first_frame_pred),
            }
        )
        return "prob"

    def output_prob_to_mask(self, prob):
        assert prob == "prob"
        return FakeMaskTensor()


class FakeMaskTensor:
    shape = (4, 6)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.full((4, 6), 0.5, dtype=np.float32)


class FakeModel:
    cfg = {"k": "v"}

    def __init__(self):
        self.core = FakeCore()


@pytest.fixture
def wired(monkeypatch):
    created = {}

    def model_loader(device):
        return created.setdefault("model", FakeModel())

    def core_factory(model):
        return model.core

    monkeypatch.setattr(pb, "build_session", pb.build_session)  # identity guard
    selection = BackendSelection("pytorch", "fp32", "cpu")
    session = pb.PyTorchSession(
        selection, model_loader=model_loader, core_factory=core_factory
    )
    return session, created


def _frame(h=4, w=6):
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return torch_from(np.ascontiguousarray(img).transpose(2, 0, 1))


def torch_from(arr):
    import torch

    return torch.from_numpy(arr)


def test_start_encodes_mask_then_warmup_predictions(wired):
    session, created = wired
    frame = _frame()
    mask = torch_from(np.zeros((4, 6), np.uint8))

    session.start(frame, mask, warmup=3)

    calls = created["model"].core.calls
    assert calls[0]["masked"] is True
    assert all(c["pred"] for c in calls[1:])
    assert len(calls) == 1 + 1 + 3  # encode + first pred + warmup repeats


def test_step_returns_converted_probabilities(wired):
    session, _ = wired
    session.start(_frame(), torch_from(np.zeros((4, 6), np.uint8)), warmup=0)
    out = session.step(_frame())
    assert isinstance(out, np.ndarray)
    assert out.shape == (4, 6)
    assert float(out.max()) <= 1.0


def test_close_releases_core_but_model_cache_survives(wired):
    session, created = wired
    session.start(_frame(), torch_from(np.zeros((4, 6), np.uint8)), warmup=0)
    session.close()
    assert session._core is None
    assert created["model"].core is not None


def test_first_frame_probability_captured_during_start(wired):
    """start()'s final prediction probability is exposed for output frame 0."""
    session, _ = wired
    session.start(_frame(), torch_from(np.zeros((4, 6), np.uint8)), warmup=2)
    assert isinstance(session.first_prob_np, np.ndarray)
    assert session.first_prob_np.shape == (4, 6)


def test_frame_to_tensor_layout():
    img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    tensor = pb.frame_to_tensor(img)
    assert tensor.shape == (3, 2, 3)
    assert tensor.dtype.is_floating_point
    assert float(tensor.max()) <= 1.0


def test_build_session_uses_default_loaders(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        pb.PyTorchSession, "__init__",
        lambda self, selection, model_loader=None, core_factory=None: seen.update(
            has_default=(model_loader is None and core_factory is None)
        ),
    )
    selection = BackendSelection("pytorch", "fp32", "cpu")
    pb.build_session(selection)
    assert seen["has_default"]
