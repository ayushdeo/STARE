from __future__ import annotations

from pathlib import Path

import numpy as np
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image

from . import config

_MODEL = None
_TRANSFORM = None
_DEVICE = None


def _get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    return _DEVICE


def _get_model():
    """Load UNI2-h via timm's HF-hub integration with the special kwargs.

    This uses timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    so timm handles checkpoint/config parsing. If loading fails, raise an error
    (do NOT fall back to random init).
    """
    global _MODEL, _TRANSFORM
    if _MODEL is not None:
        return _MODEL

    device = _get_device()
    timm_kwargs = dict(
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=config.IMAGE_EMB_DIM,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )

    try:
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to create/load UNI2-h via timm: {e}")

    try:
        cfg = resolve_data_config({}, model=model)
        transform = create_transform(**cfg, use_prefetcher=False)
    except Exception:
        transform = None

    model.to(device)
    model.eval()
    _MODEL = model
    _TRANSFORM = transform
    return _MODEL


def _transform_batch(batch: np.ndarray) -> torch.Tensor:
    batch = batch.astype(np.float32) / 255.0
    x = torch.from_numpy(batch.transpose(0, 3, 1, 2)).contiguous()
    x = x.to(_get_device())
    mean = torch.tensor([0.485, 0.456, 0.406], device=_get_device(), dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=_get_device(), dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


def embed_patches(patches: np.ndarray, batch_size: int | None = None) -> np.ndarray:
    """Embed an array of RGB patches into UNI2-h feature vectors."""
    patches = np.asarray(patches)
    if patches.size == 0:
        return np.empty((0, config.IMAGE_EMB_DIM), dtype=np.float32)

    if patches.ndim == 3:
        patches = patches[np.newaxis, ...]
    if patches.shape[-1] != 3:
        raise ValueError(f"patches must have shape (N, H, W, 3); got {patches.shape}")

    model = _get_model()
    bs = batch_size or config.BATCH_SIZE_EMB
    outs = []
    device = _get_device()

    with torch.no_grad():
        for start in range(0, patches.shape[0], bs):
            batch = patches[start : start + bs]
            x = _transform_batch(batch)
            if device.type == "cuda":
                # Use the public API for autocast to avoid FutureWarning
                with torch.amp.autocast(device_type="cuda"):
                    features = model(x)
            else:
                features = model(x)

            if isinstance(features, tuple):
                features = features[0]

            if features.ndim == 2 and features.shape[-1] == config.IMAGE_EMB_DIM:
                outs.append(features.detach().cpu().numpy())
            else:
                flat = features.detach().cpu().numpy()
                if flat.ndim == 1:
                    flat = flat.reshape(1, -1)
                outs.append(flat.reshape(flat.shape[0], -1))

    if not outs:
        return np.empty((0, config.IMAGE_EMB_DIM), dtype=np.float32)
    out = np.concatenate(outs, axis=0).astype(np.float32)
    return out
