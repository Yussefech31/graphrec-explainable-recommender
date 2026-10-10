"""Popularity-based baseline recommender."""

from __future__ import annotations

import pandas as pd

from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Recommends globally popular items weighted by Bayesian average rating."""

    def __init__(self, min_ratings: int = 5, m_weight: float = 10.0):
        super().__init__(name="Popularity")
        self.min_ratings = min_ratings
        self.m_weight = m_weight
        self.item_scores: dict[int, float] = {}
        self.ranked_items: list[tuple[int, float]] = []
        self.user_seen_items: dict[int, set[int]] = {}

    def fit(self, train_ratings: pd.DataFrame, movies_df: pd.DataFrame) -> PopularityRecommender:
        """Compute Bayesian weighted rating for each movie."""
        self.user_seen_items = train_ratings.groupby("user_id")["movie_id"].apply(set).to_dict()

        # Compute counts and means
        stats = (
            train_ratings.groupby("movie_id")["rating"]
            .agg(["count", "mean"])
            .reset_index()
        )
        global_mean = train_ratings["rating"].mean() if not train_ratings.empty else 3.5

        # Weighted rating formula: (v / (v + m)) * R + (m / (v + m)) * C
        # where v = count, m = minimum prior weight, R = item mean, C = global mean
        v = stats["count"]
        r = stats["mean"]
        stats["score"] = (v / (v + self.m_weight)) * r + (self.m_weight / (v + self.m_weight)) * global_mean

        stats.sort_values(by="score", ascending=False, inplace=True)
        self.item_scores = dict(zip(stats["movie_id"], stats["score"]))
        self.ranked_items = list(zip(stats["movie_id"], stats["score"]))
        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Return top-K highest scoring popular movies not yet rated by the user."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating recommendations.")

        seen = self.user_seen_items.get(user_id, set()) if exclude_seen else set()
        recommendations: list[tuple[int, float]] = []

        for mid, score in self.ranked_items:
            if mid not in seen:
                recommendations.append((mid, float(score)))
                if len(recommendations) >= k:
                    break

        return recommendations

    def predict_score(self, user_id: int, movie_id: int) -> float:
        """Return the global popularity score of the movie."""
        return float(self.item_scores.get(movie_id, 0.0))
