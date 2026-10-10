"""Unit tests for recommendation evaluation metrics."""

import pandas as pd
import pytest

from src.graphrec.evaluation.metrics import BenchmarkEvaluator
from src.graphrec.models.base import BaseRecommender


class DummyRecommender(BaseRecommender):
    def __init__(self, fixed_recs):
        super().__init__(name="Dummy")
        self.fixed_recs = fixed_recs

    def fit(self, train_ratings, movies_df):
        return self

    def recommend(self, user_id, k=10, exclude_seen=True):
        return [(mid, 1.0) for mid in self.fixed_recs[:k]]

    def predict_score(self, user_id, movie_id):
        return 1.0


def test_evaluator_metrics():
    test_df = pd.DataFrame([
        {"user_id": 1, "movie_id": 10},
        {"user_id": 2, "movie_id": 999},  # Will miss
    ])
    all_item_ids = [10, 20, 30, 40, 50, 999]
    item_pop = {10: 100, 20: 50, 30: 25, 40: 10, 50: 5, 999: 1}

    evaluator = BenchmarkEvaluator(test_df=test_df, all_item_ids=all_item_ids, item_popularity=item_pop)

    # Dummy model recommends [10, 20, 30]
    dummy = DummyRecommender(fixed_recs=[10, 20, 30])
    results = evaluator.evaluate_model(dummy, k=3)

    # User 1 hits item 10 at rank 1: hit = 1, NDCG = 1.0 / log2(2) = 1.0, MAP = 1.0
    # User 2 misses: hit = 0, NDCG = 0.0, MAP = 0.0
    # Average Hit Rate = 0.5, NDCG = 0.5
    assert pytest.approx(results.hit_rate) == 0.5
    assert pytest.approx(results.ndcg) == 0.5
    assert pytest.approx(results.map_score) == 0.5
    assert results.catalog_coverage_pct > 0.0
    assert results.novelty > 0.0
    assert results.mean_latency_ms >= 0.0
