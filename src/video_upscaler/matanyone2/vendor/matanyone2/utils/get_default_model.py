"""
A helper function to get a default model for quick testing
"""
from pathlib import Path

from omegaconf import OmegaConf, open_dict

import torch
from ..model.matanyone2 import MatAnyone2

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_default_cfg():
    # ponytail: upstream used hydra's initialize/compose to merge
    # config/eval_matanyone_config.yaml with the `model: base` default.
    # Hydra is deliberately NOT a dependency of the vendored build, so we
    # replicate its merge here: base.yaml keys become the top-level `model`
    # block, and the eval yaml contributes the rest.
    cfg = OmegaConf.load(_CONFIG_DIR / "eval_matanyone_config.yaml")
    cfg.model = OmegaConf.load(_CONFIG_DIR / "model" / "base.yaml")
    cfg.pop("defaults", None)
    cfg.pop("hydra", None)
    return cfg


def get_matanyone2_model(ckpt_path, device=None) -> MatAnyone2:
    cfg = _load_default_cfg()

    with open_dict(cfg):
        cfg['weights'] = ckpt_path

    # Load the network weights
    if device is not None:
        matanyone2 = MatAnyone2(cfg, single_object=True).to(device).eval()
        model_weights = torch.load(cfg.weights, map_location=device)
    else:  # if device is not specified, `.cuda()` by default
        matanyone2 = MatAnyone2(cfg, single_object=True).cuda().eval()
        model_weights = torch.load(cfg.weights)

    matanyone2.load_weights(model_weights)

    return matanyone2