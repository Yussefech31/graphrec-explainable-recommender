"""Item-Item Collaborative Filtering recommender using sparse cosine similarity."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommender

logger = logging.getLogger(__name__)


class ItemItemCFRecommender(BaseRecommender):
    """Item-Item Collaborative Filtering using vector cosine similarity over user interaction histories."""

    def __init__(self, top_similar_neighbors: int = 50, min_shared_users: int = 2):
        super().__init__(name="Item-Item CF")
        self.top_similar_neighbors = top_similar_neighbors
        self.min_shared_users = min_shared_users

        self.user2idx: dict[int, int] = {}
        self.idx2user: dict[int, int] = {}
        self.item2idx: dict[int, int] = {}
        self.idx2item: dict[int, int] = {}

        self.item_similarity: np.ndarray | None = None
        self.user_item_matrix: sparse.csr_matrix | None = None
        self.user_seen_items: dict[int, set[int]] = {}

    def fit(self, train_ratings: pd.DataFrame, movies_df: pd.DataFrame) -> ItemItemCFRecommender:
        """Build user-item interaction matrix and compute item-item cosine similarity."""
        self.user_seen_items = train_ratings.groupby("user_id")["movie_id"].apply(set).to_dict()

        unique_users = sorted(train_ratings["user_id"].unique())
        unique_items = sorted(train_ratings["movie_id"].unique())

        self.user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx2user = {idx: uid for idx, uid in enumerate(unique_users)}
        self.item2idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx2item = {idx: iid for idx, iid in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)

        # Build sparse user-item matrix (centered ratings or binary/affinity)
        rows = train_ratings["user_id"].map(self.user2idx).values
        cols = train_ratings["movie_id"].map(self.item2idx).values
        data = train_ratings["rating"].values

        # Item-user matrix (transposed) for item similarity
        item_user = sparse.csr_matrix((data, (cols, rows)), shape=(n_items, n_users))

        self.user_item_matrix = item_user.T.tocsr()

        # Compute cosine similarity between item vectors
        logger.info("Computing Item-Item cosine similarity matrix (%d items)...", n_items)
        sim = cosine_similarity(item_user, dense_output=True)

        # Zero out diagonal
        np.fill_diagonal(sim, 0.0)

        # Retain top-K neighbors per item to reduce noise
        if self.top_similar_neighbors < n_items:
            for i in range(n_items):
                row = sim[i]
                if len(row) > self.top_similar_neighbors:
                    thresh = np.partition(row, -self.top_similar_neighbors)[-self.top_similar_neighbors]
                    row[row < thresh] = 0.0

        self.item_similarity = sim
        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Recommend items based on weighted similarity to user's rated items."""
        if not self.is_fitted or self.item_similarity is None or self.user_item_matrix is None:
            raise RuntimeError("Model must be fitted before generating recommendations.")

        if user_id not in self.user2idx:
            # Cold-start fallback: return empty or handled upstream
            return []

        u_idx = self.user2idx[user_id]
        user_ratings_vec = self.user_item_matrix.getrow(u_idx).toarray().flatten()

        # Score vector = R_u @ S
        scores = user_ratings_vec @ self.item_similarity

        seen = self.user_seen_items.get(user_id, set())
        ranked_indices = np.argsort(-scores)

        recommendations: list[tuple[int, float]] = []
        for idx in ranked_indices:
            mid = self.idx2item[idx]
            if exclude_seen and mid in seen:
                continue
            if scores[idx] <= 0.0:
                break
            recommendations.append((mid, float(scores[idx])))
            if len(recommendations) >= k:
                break

        return recommendations

    def predict_score(self, user_id: int, movie_id: int) -> float:
        """Predict user preference score for a single movie."""
        if user_id not in self.user2idx or movie_id not in self.item2idx:
            return 0.0
        u_idx = self.user2idx[user_id]
        i_idx = self.item2idx[movie_id]
        user_ratings_vec = self.user_item_matrix.getrow(u_idx).toarray().flatten()
        return float(np.dot(user_ratings_vec, self.item_similarity[i_idx]))
