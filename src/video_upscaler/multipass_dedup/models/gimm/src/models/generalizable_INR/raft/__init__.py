from .raft import RAFT
import argparse
import os
import torch
from .extractor import BasicEncoder


def _resolve_raft_path(model_path):
    """Resolve the RAFT checkpoint path.

    The vendored default ("weights/raft-things.pth") is relative to the
    process cwd and never exists in Clarity; fall back to the configured
    MultiPassDedup weights directory before giving up.
    """
    if os.path.isfile(model_path):
        return model_path
    override = os.environ.get("CLARITY_DEDUP_MODELS_DIR")
    candidates = []
    if override:
        candidates.append(os.path.join(override, "raft-things.pth"))
    try:
        from video_upscaler import config as _clarity_config

        candidates.append(str(_clarity_config.DEDUP_MODELS_DIR / "raft-things.pth"))
    except Exception:
        pass
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return model_path


def initialize_RAFT(model_path="weights/raft-things.pth", device="cuda"):
    """Initializes the RAFT model."""
    args = argparse.ArgumentParser()
    args.raft_model = _resolve_raft_path(model_path)
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False
    model = RAFT(args)
    ckpt = torch.load(args.raft_model, map_location="cpu")

    def convert(param):
        return {k.replace("module.", ""): v for k, v in param.items() if "module" in k}

    ckpt = convert(ckpt)
    model.load_state_dict(ckpt, strict=True)
    print("load raft from " + model_path)

    return model
