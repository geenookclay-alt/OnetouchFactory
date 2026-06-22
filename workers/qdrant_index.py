"""Qdrant local-mode vector index — Korean video frames pool.

Stores per-frame CLIP embeddings with payload metadata. Used by visual_match
to score whether a candidate video matches anything in the Korean pool.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


COLLECTION = os.getenv("QDRANT_COLLECTION", "korean_video_frames")
QDRANT_PATH = os.getenv("QDRANT_PATH",
                        str(Path(__file__).resolve().parent.parent / "qdrant_data"))
VECTOR_DIM = 512


_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        _client = QdrantClient(path=QDRANT_PATH)
        _ensure_collection(_client)
    return _client


def _ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=VECTOR_DIM, distance=qm.Distance.COSINE),
    )


def upsert_frames(items: list[dict]) -> int:
    """Upsert frame embeddings.

    Each item: {vector: np.ndarray | list, video_id, frame_idx,
                channel_id, video_url, channel_handle}.
    Returns number of points written.
    """
    if not items:
        return 0
    client = _get_client()
    points: list[qm.PointStruct] = []
    for it in items:
        vec = it["vector"]
        if isinstance(vec, np.ndarray):
            vec = vec.astype(np.float32).tolist()
        payload = {
            "video_id": it.get("video_id", ""),
            "frame_idx": int(it.get("frame_idx", 0)),
            "channel_id": it.get("channel_id", ""),
            "video_url": it.get("video_url", ""),
            "channel_handle": it.get("channel_handle", ""),
        }
        # Deterministic UUID so same (video_id, frame_idx) doesn't duplicate
        seed = f"{payload['video_id']}:{payload['frame_idx']}"
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
        points.append(qm.PointStruct(id=pid, vector=vec, payload=payload))
    client.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def search(vector, limit: int = 10) -> list[dict]:
    """Search for nearest frames. Returns list of {score, payload}."""
    if isinstance(vector, np.ndarray):
        vector = vector.astype(np.float32).tolist()
    client = _get_client()
    res = client.query_points(
        collection_name=COLLECTION, query=vector, limit=limit, with_payload=True,
    ).points
    return [{"score": float(r.score), "payload": r.payload or {}} for r in res]


def search_batch(vectors: Iterable, limit: int = 10) -> list[list[dict]]:
    """Batch search — one result list per query vector."""
    return [search(v, limit=limit) for v in vectors]


def stats() -> dict:
    client = _get_client()
    info = client.get_collection(COLLECTION)
    return {
        "collection": COLLECTION,
        "points": info.points_count,
        "vectors_dim": VECTOR_DIM,
        "path": QDRANT_PATH,
    }
