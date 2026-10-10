"""Base recommender interface defining standard training and inference protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseRecommender(ABC):
    """Abstract base class for all collaborative, neural, and graph recommendation models."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_ratings: pd.DataFrame, movies_df: pd.DataFrame) -> BaseRecommender:
        """Fit model on training interaction data and item metadata."""
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Generate top-K recommended movie IDs and their affinity scores for a given user.

        Returns:
            List of (movie_id, score) tuples sorted by descending affinity score.
        """
        pass

    @abstractmethod
    def predict_score(self, user_id: int, movie_id: int) -> float:
        """Predict the preference score for a single (user_id, movie_id) pair."""
        pass

    def batch_recommend(
        self,
        user_ids: list[int],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> dict[int, list[tuple[int, float]]]:
        """Generate top-K recommendations for a batch of users."""
        results: dict[int, list[tuple[int, float]]] = {}
        for uid in user_ids:
            results[uid] = self.recommend(uid, k=k, exclude_seen=exclude_seen)
        return results

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters."""
        return {"name": self.name}
