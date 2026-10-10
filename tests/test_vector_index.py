"""Unit tests for Faiss vector candidate retrieval index."""

import numpy as np
import pytest

from src.graphrec.retrieval.vector_index import FaissVectorIndex


def test_faiss_vector_index_add_and_search():
    np.random.seed(42)
    dim = 16
    n_items = 50

    index = FaissVectorIndex(dimension=dim, metric="cosine")
    item_ids = list(range(100, 100 + n_items))
    embeddings = np.random.randn(n_items, dim).astype(np.float32)

    index.add_items(item_ids, embeddings)

    # Search with first item's vector
    query_vec = embeddings[0]
    results, latency_ms = index.search(query_vec, k=5)

    assert len(results) == 5
    assert results[0][0] == 100  # Closest item to embeddings[0] should be item 100
    assert pytest.approx(results[0][1], abs=1e-4) == 1.0  # Cosine self-similarity is 1.0
    assert latency_ms >= 0.0


def test_faiss_vector_index_exclusion():
    np.random.seed(42)
    dim = 8
    index = FaissVectorIndex(dimension=dim, metric="cosine")
    item_ids = [1, 2, 3, 4, 5]
    embeddings = np.eye(5, dim, dtype=np.float32)

    index.add_items(item_ids, embeddings)

    # Query with vector matching item 1, but exclude item 1
    query_vec = embeddings[0]
    results, _ = index.search(query_vec, k=3, exclude_ids={1})

    res_ids = [mid for mid, _ in results]
    assert 1 not in res_ids
    assert len(results) == 3
