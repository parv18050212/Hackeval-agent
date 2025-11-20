"""
qdrant_integration.py
Robust, lazy Qdrant integration for embedding storage and ANN search.

This module intentionally avoids contacting Qdrant at import time so the
application can start even when Qdrant is not running. It will attempt to
connect lazily when the first request to store/search vectors occurs and will
fail gracefully (logging a warning) if Qdrant is unavailable.
"""
from __future__ import annotations

import os
import logging
from typing import Optional, List, Dict, Any

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

_logger = logging.getLogger(__name__)

# Configurable via env vars
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "hackeval_vectors")

# Embedding model
text_model = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_SIZE = 384  # all-MiniLM-L6-v2

# Internal state
_client: Optional[QdrantClient] = None
_collection_initialized: bool = False


def _get_client() -> Optional[QdrantClient]:
    """Return a connected QdrantClient or None if Qdrant is unreachable."""
    global _client
    if _client is not None:
        return _client

    try:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # quick health check
        _client.get_collections()
        _logger.info("Connected to Qdrant at %s:%s", QDRANT_HOST, QDRANT_PORT)
        return _client
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        _logger.warning("Unable to connect to Qdrant at %s:%s — %s", QDRANT_HOST, QDRANT_PORT, exc)
        _client = None
        return None


def create_collection_if_missing() -> None:
    """Create the collection with an HNSW config if it doesn't already exist.

    No-op if the Qdrant client is not available.
    """
    global _collection_initialized
    if _collection_initialized:
        return

    client = _get_client()
    if client is None:
        return

    try:
        # qdrant-client may return a CollectionsResponse object — access via attribute
        cols = client.get_collections()
        names: List[str] = []
        try:
            # If returned object is a mapping
            names = [c.get("name") for c in cols.get("collections", []) if c.get("name")]
        except Exception:
            # Fallback: try attribute access
            try:
                names = [c.name for c in cols.collections]
            except Exception:
                names = []

        if COLLECTION_NAME not in names:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "size": VECTOR_SIZE,
                    "distance": "Cosine",
                },
                hnsw_config={
                    "m": 16,
                    "ef_construct": 128,
                },
            )
            _logger.info("Created Qdrant collection '%s'", COLLECTION_NAME)

        _collection_initialized = True
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        _logger.warning("Failed to ensure Qdrant collection '%s': %s", COLLECTION_NAME, exc)


def ingest_text(text: str, payload: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Encode text and upsert into Qdrant.

    Returns the point id on success, or None if Qdrant is unavailable.
    """
    client = _get_client()
    if client is None:
        _logger.debug("Skipping ingest_text because Qdrant is unavailable")
        return None

    create_collection_if_missing()

    embedding = text_model.encode(text)
    point_id = int(np.random.randint(1, int(1e9)))

    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                {
                    "id": point_id,
                    "vector": embedding.tolist(),
                    "payload": payload or {"text": text},
                }
            ],
        )
        _logger.debug("Upserted point %s to collection %s", point_id, COLLECTION_NAME)
        return point_id
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        _logger.warning("Failed to upsert point to Qdrant: %s", exc)
        return None


def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search for nearest neighbours for the query text.

    Returns a list of result dicts (empty list if Qdrant unavailable).
    """
    client = _get_client()
    if client is None:
        _logger.debug("Skipping search because Qdrant is unavailable")
        return []

    create_collection_if_missing()

    query_vec = text_model.encode(query)
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec.tolist(),
            limit=top_k,
        )
        # results is a list of PointStruct objects; convert to dicts when possible
        out: List[Dict[str, Any]] = []
        for r in results:
            try:
                payload = getattr(r, "payload", None) or getattr(r, "payload", {})
                out.append({
                    "id": getattr(r, "id", None),
                    "score": getattr(r, "score", None),
                    "payload": payload,
                })
            except Exception:
                # best-effort
                out.append({"raw": r})
        return out
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        _logger.warning("Qdrant search failed: %s", exc)
        return []


def ingest_image(image_path: str, payload: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Placeholder image ingestion: replace embedding generation with a real image model.

    Returns point id or None if Qdrant unavailable.
    """
    client = _get_client()
    if client is None:
        _logger.debug("Skipping ingest_image because Qdrant is unavailable")
        return None

    create_collection_if_missing()

    # TODO: replace with a real image embedding (CLIP, ViT, etc.)
    embedding = np.random.rand(VECTOR_SIZE).astype(float)
    point_id = int(np.random.randint(1, int(1e9)))

    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                {
                    "id": point_id,
                    "vector": embedding.tolist(),
                    "payload": payload or {"image_path": image_path},
                }
            ],
        )
        _logger.debug("Upserted image point %s to collection %s", point_id, COLLECTION_NAME)
        return point_id
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        _logger.warning("Failed to upsert image to Qdrant: %s", exc)
        return None
