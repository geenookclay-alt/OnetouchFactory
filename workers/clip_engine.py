"""CLIP embedding engine — open_clip ViT-B/32 on Apple Silicon MPS.

Lazy-init singleton. Embedding throughput on M4 MPS: ~50ms/frame batched.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Iterable

import numpy as np


_lock = threading.Lock()
_state: dict = {"model": None, "preprocess": None, "device": None}


def _device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ensure_loaded():
    if _state["model"] is not None:
        return
    with _lock:
        if _state["model"] is not None:
            return
        import torch
        import open_clip
        device = _device()
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model = model.to(device).eval()
        _state["model"] = model
        _state["preprocess"] = preprocess
        _state["device"] = device


def embed_images(images: list) -> np.ndarray:
    """Embed a list of PIL images. Returns (N, 512) float32 array (L2-normalized)."""
    if not images:
        return np.zeros((0, 512), dtype=np.float32)
    _ensure_loaded()
    import torch
    preprocess = _state["preprocess"]
    model = _state["model"]
    device = _state["device"]

    batch = torch.stack([preprocess(img) for img in images]).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().float().numpy().astype(np.float32)


async def embed_images_async(images: list) -> np.ndarray:
    """Async wrapper — runs the sync embed in a thread."""
    return await asyncio.to_thread(embed_images, images)


def embedding_dim() -> int:
    return 512
