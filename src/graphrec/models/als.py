"""Matrix Factorization recommender using low-rank latent factor decomposition."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from .base import BaseRecommender

logger = logging.getLogger(__name__)


class MatrixFactorizationRecommender(BaseRecommender):
    """Latent factor Matrix Factorization decomposing the user-item interaction matrix into dense embeddings."""

    def __init__(self, n_factors: int = 64, n_iter: int = 15, random_state: int = 42):
        super().__init__(name="Matrix Factorization (ALS/SVD)")
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.random_state = random_state

        self.user2idx: dict[int, int] = {}
        self.idx2user: dict[int, int] = {}
        self.item2idx: dict[int, int] = {}
        self.idx2item: dict[int, int] = {}

        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_seen_items: dict[int, set[int]] = {}

    def fit(self, train_ratings: pd.DataFrame, movies_df: pd.DataFrame) -> MatrixFactorizationRecommender:
        """Decompose user-item rating matrix into low-rank latent representations."""
        self.user_seen_items = train_ratings.groupby("user_id")["movie_id"].apply(set).to_dict()

        unique_users = sorted(train_ratings["user_id"].unique())
        unique_items = sorted(train_ratings["movie_id"].unique())

        self.user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx2user = {idx: uid for idx, uid in enumerate(unique_users)}
        self.item2idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx2item = {idx: iid for idx, iid in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)

        rows = train_ratings["user_id"].map(self.user2idx).values
        cols = train_ratings["movie_id"].map(self.item2idx).values
        data = train_ratings["rating"].values

        user_item = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        logger.info(
            "Fitting TruncatedSVD Matrix Factorization (factors=%d, users=%d, items=%d)...",
            self.n_factors,
            n_users,
            n_items,
        )
        effective_factors = min(self.n_factors, n_items - 1, n_users - 1)
        svd = TruncatedSVD(
            n_components=effective_factors,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.user_factors = svd.fit_transform(user_item)
        self.item_factors = svd.components_.T  # Shape: (n_items, effective_factors)

        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Recommend top-K items via inner product of user and item latent factors."""
        if not self.is_fitted or self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model must be fitted before generating recommendations.")

        if user_id not in self.user2idx:
            return []

        u_idx = self.user2idx[user_id]
        u_vec = self.user_factors[u_idx]  # Shape: (factors,)
        scores = self.item_factors @ u_vec  # Shape: (n_items,)

        seen = self.user_seen_items.get(user_id, set())
        ranked_indices = np.argsort(-scores)

        recommendations: list[tuple[int, float]] = []
        for idx in ranked_indices:
            mid = self.idx2item[idx]
            if exclude_seen and mid in seen:
                continue
            recommendations.append((mid, float(scores[idx])))
            if len(recommendations) >= k:
                break

        return recommendations

    def predict_score(self, user_id: int, movie_id: int) -> float:
        """Predict preference score via latent factor dot product."""
        if user_id not in self.user2idx or movie_id not in self.item2idx:
            return 0.0
        u_idx = self.user2idx[user_id]
        i_idx = self.item2idx[movie_id]
        return float(np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))
