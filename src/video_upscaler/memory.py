"""Memory and VRAM management utilities for Clarity."""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def free_gpu_memory() -> Dict[str, Any]:
    """Release all cached models, predictors, and GPU memory pools.

    Drops MatAnyone2 and SAM models from process-level singletons,
    invokes Python garbage collection, and empties PyTorch CUDA caches.
    """
    # 1. Release MatAnyone2 and SAM cached models
    try:
        from video_upscaler.matanyone2.model import release_model

        release_model()
    except Exception as exc:
        logger.debug("Error releasing MatAnyone2 model: %s", exc)

    try:
        from video_upscaler.matanyone2.sam_segment import release_sam

        release_sam()
    except Exception as exc:
        logger.debug("Error releasing SAM model: %s", exc)

    # 2. Run full garbage collection
    gc.collect()

    # 3. Empty PyTorch CUDA memory cache
    vram_stats: Dict[str, float] = {}
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            vram_stats = {
                "free_gb": round(free_bytes / (1024**3), 2),
                "total_gb": round(total_bytes / (1024**3), 2),
            }
    except Exception as exc:
        logger.debug("Error emptying PyTorch CUDA cache: %s", exc)

    return {
        "success": True,
        "message": "GPU memory and cached models cleared.",
        "vram": vram_stats,
    }
