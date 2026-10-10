"""Quantitative recommendation evaluation metrics and benchmark evaluator."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.graphrec.models.base import BaseRecommender


@dataclass
class EvaluationResults:
    """Holds quantitative evaluation metric scores."""

    model_name: str
    num_test_users: int
    k: int
    hit_rate: float
    ndcg: float
    map_score: float
    catalog_coverage_pct: float
    novelty: float
    mean_latency_ms: float
    p95_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_test_users": self.num_test_users,
            "k": self.k,
            "hit_rate": round(self.hit_rate, 4),
            "ndcg": round(self.ndcg, 4),
            "map_score": round(self.map_score, 4),
            "catalog_coverage_pct": round(self.catalog_coverage_pct, 2),
            "novelty": round(self.novelty, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
        }


class BenchmarkEvaluator:
    """Evaluates recommendation models on standard leave-last-out test interactions."""

    def __init__(self, test_df: pd.DataFrame, all_item_ids: list[int], item_popularity: dict[int, int]):
        """Args:
            test_df: DataFrame with columns ['user_id', 'movie_id'] representing target test interactions.
            all_item_ids: Complete set of catalog items.
            item_popularity: Item interaction counts in training set for Novelty computation.
        """
        self.test_df = test_df
        self.all_item_ids = set(all_item_ids)
        self.total_catalog_size = max(1, len(self.all_item_ids))

        # Target item per test user
        self.user_targets: dict[int, set[int]] = (
            test_df.groupby("user_id")["movie_id"].apply(set).to_dict()
        )

        # Precompute item self-information for Novelty: -log2(P(i))
        total_interactions = max(1, sum(item_popularity.values()))
        self.item_self_info: dict[int, float] = {}
        for iid in self.all_item_ids:
            cnt = item_popularity.get(iid, 1)
            p = cnt / total_interactions
            self.item_self_info[iid] = -math.log2(p)

    def evaluate_model(
        self,
        model: BaseRecommender,
        k: int = 10,
        max_eval_users: int | None = None,
    ) -> EvaluationResults:
        """Run leave-last-out evaluation on model and compute metrics."""
        eval_users = list(self.user_targets.keys())
        if max_eval_users is not None and max_eval_users < len(eval_users):
            eval_users = eval_users[:max_eval_users]

        hits: list[int] = []
        ndcgs: list[float] = []
        maps: list[float] = []
        latencies: list[float] = []
        novelty_scores: list[float] = []
        recommended_items_union: set[int] = set()

        for uid in eval_users:
            target_items = self.user_targets[uid]

            # Measure inference latency
            t0 = time.perf_counter()
            recs = model.recommend(uid, k=k, exclude_seen=True)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            rec_ids = [mid for mid, _ in recs]
            recommended_items_union.update(rec_ids)

            # Hit Rate & NDCG & MAP
            hit = 0
            user_ndcg = 0.0
            user_map = 0.0

            for rank, mid in enumerate(rec_ids, start=1):
                if mid in target_items:
                    hit = 1
                    user_ndcg = 1.0 / math.log2(rank + 1)
                    user_map = 1.0 / rank
                    break

            hits.append(hit)
            ndcgs.append(user_ndcg)
            maps.append(user_map)

            # Novelty
            if rec_ids:
                user_novelty = sum(self.item_self_info.get(mid, 10.0) for mid in rec_ids) / len(rec_ids)
                novelty_scores.append(user_novelty)

        n = max(1, len(eval_users))
        coverage_pct = (len(recommended_items_union) / self.total_catalog_size) * 100.0

        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
        mean_lat = float(np.mean(latencies)) if latencies else 0.0

        return EvaluationResults(
            model_name=model.name,
            num_test_users=len(eval_users),
            k=k,
            hit_rate=sum(hits) / n,
            ndcg=sum(ndcgs) / n,
            map_score=sum(maps) / n,
            catalog_coverage_pct=coverage_pct,
            novelty=float(np.mean(novelty_scores)) if novelty_scores else 0.0,
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
        )
