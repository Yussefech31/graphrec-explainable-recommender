"""FAISS vector retrieval index for high-throughput sub-millisecond candidate generation."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

logger = logging.getLogger(__name__)


class FaissVectorIndex:
    """Sub-millisecond approximate / exact nearest neighbor index powered by FAISS."""

    def __init__(self, dimension: int, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric.lower()
        self.item_ids: list[int] = []
        self.id2pos: dict[int, int] = {}
        self.pos2id: dict[int, int] = {}
        self._index: Any = None
        self._raw_embeddings: np.ndarray | None = None

        if HAS_FAISS:
            # IndexFlatIP handles inner product and cosine similarity (with normalized vectors)
            self._index = faiss.IndexFlatIP(self.dimension)
            logger.info("Initialized Faiss IndexFlatIP (dimension=%d, metric=%s)", dimension, metric)
        else:
            logger.warning("Faiss not available; using NumPy fallback vector index.")

    def add_items(self, item_ids: list[int], embeddings: np.ndarray) -> None:
        """Add item embeddings to the index.

        Args:
            item_ids: List of unique item identifiers.
            embeddings: Float32 NumPy array of shape (N, dimension).
        """
        assert len(item_ids) == len(embeddings), "item_ids and embeddings lengths must match"
        assert embeddings.shape[1] == self.dimension, f"Expected dim {self.dimension}, got {embeddings.shape[1]}"

        embs = embeddings.astype(np.float32)
        if self.metric == "cosine":
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs = embs / norms

        self.item_ids = list(item_ids)
        self.id2pos = {iid: idx for idx, iid in enumerate(item_ids)}
        self.pos2id = {idx: iid for idx, iid in enumerate(item_ids)}
        self._raw_embeddings = embs

        if HAS_FAISS and self._index is not None:
            self._index.reset()
            self._index.add(embs)
            logger.info("Indexed %d items in Faiss index (total count: %d)", len(item_ids), self._index.ntotal)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> tuple[list[tuple[int, float]], float]:
        """Search the nearest items for a query vector.

        Returns:
            Tuple of (list of (item_id, score) pairs, search latency in milliseconds).
        """
        t0 = time.perf_counter()
        query = query_vector.reshape(1, -1).astype(np.float32)

        if self.metric == "cosine":
            q_norm = np.linalg.norm(query)
            if q_norm > 0:
                query = query / q_norm

        exclude_set = exclude_ids or set()
        fetch_k = min(k + len(exclude_set) + 10, len(self.item_ids))

        results: list[tuple[int, float]] = []

        if HAS_FAISS and self._index is not None:
            distances, indices = self._index.search(query, fetch_k)
            for score, pos in zip(distances[0], indices[0]):
                if pos < 0 or pos >= len(self.item_ids):
                    continue
                iid = self.pos2id[pos]
                if iid in exclude_set:
                    continue
                results.append((iid, float(score)))
                if len(results) >= k:
                    break
        elif self._raw_embeddings is not None:
            # NumPy fallback
            scores = np.dot(self._raw_embeddings, query.flatten())
            ranked_pos = np.argsort(-scores)
            for pos in ranked_pos:
                iid = self.pos2id[pos]
                if iid in exclude_set:
                    continue
                results.append((iid, float(scores[pos])))
                if len(results) >= k:
                    break

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return results, latency_ms
