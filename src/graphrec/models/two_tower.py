"""PyTorch Two-Tower Neural Recommender with FAISS retrieval index."""

from __future__ import annotations

import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.graphrec.retrieval.vector_index import FaissVectorIndex

from .base import BaseRecommender

logger = logging.getLogger(__name__)


class TwoTowerDataset(Dataset):
    """PyTorch Dataset generating positive interactions and negative sampled items for BPR loss."""

    def __init__(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        num_items: int,
        user_seen: dict[int, set[int]],
        neg_ratio: int = 1,
    ):
        self.users = user_indices
        self.pos_items = item_indices
        self.num_items = num_items
        self.user_seen = user_seen
        self.neg_ratio = neg_ratio

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[int, int, int]:
        u = int(self.users[idx])
        pos_i = int(self.pos_items[idx])
        seen = self.user_seen.get(u, set())

        # Negative sampling
        neg_i = random.randint(0, self.num_items - 1)
        while neg_i in seen:
            neg_i = random.randint(0, self.num_items - 1)

        return u, pos_i, neg_i


class TwoTowerModule(nn.Module):
    """Dual-encoder neural network architecture."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        # User Tower
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )

        # Item Tower
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def forward_user(self, user_indices: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_embedding(user_indices)
        out = self.user_mlp(u_emb)
        return F.normalize(out, p=2, dim=-1)

    def forward_item(self, item_indices: torch.Tensor) -> torch.Tensor:
        i_emb = self.item_embedding(item_indices)
        out = self.item_mlp(i_emb)
        return F.normalize(out, p=2, dim=-1)

    def forward(
        self,
        user_indices: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u_vec = self.forward_user(user_indices)
        pos_vec = self.forward_item(pos_items)
        neg_vec = self.forward_item(neg_items)

        pos_score = (u_vec * pos_vec).sum(dim=-1)
        neg_score = (u_vec * neg_vec).sum(dim=-1)
        return pos_score, neg_score


class TwoTowerRecommender(BaseRecommender):
    """Production Two-Tower Recommender trained with BPR loss and queried via Faiss."""

    def __init__(
        self,
        embedding_dim: int = 64,
        epochs: int = 8,
        batch_size: int = 512,
        lr: float = 0.003,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        super().__init__(name="Two-Tower Neural (PyTorch + FAISS)")
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device(device)

        self.model: TwoTowerModule | None = None
        self.faiss_index: FaissVectorIndex | None = None

        self.user2idx: dict[int, int] = {}
        self.idx2user: dict[int, int] = {}
        self.item2idx: dict[int, int] = {}
        self.idx2item: dict[int, int] = {}
        self.user_seen_items: dict[int, set[int]] = {}

    def fit(self, train_ratings: pd.DataFrame, movies_df: pd.DataFrame) -> TwoTowerRecommender:
        """Train Two-Tower PyTorch model using Bayesian Personalized Ranking loss."""
        self.user_seen_items = train_ratings.groupby("user_id")["movie_id"].apply(set).to_dict()

        unique_users = sorted(train_ratings["user_id"].unique())
        unique_items = sorted(movies_df["movie_id"].unique())

        self.user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx2user = {idx: uid for idx, uid in enumerate(unique_users)}
        self.item2idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx2item = {idx: iid for idx, iid in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)

        # Prepare interactions for training
        valid_ratings = train_ratings[train_ratings["movie_id"].isin(self.item2idx)].copy()
        u_arr = valid_ratings["user_id"].map(self.user2idx).values
        i_arr = valid_ratings["movie_id"].map(self.item2idx).values

        user_seen_idx = {
            self.user2idx[uid]: {self.item2idx[m] for m in movies if m in self.item2idx}
            for uid, movies in self.user_seen_items.items()
            if uid in self.user2idx
        }

        dataset = TwoTowerDataset(
            user_indices=u_arr,
            item_indices=i_arr,
            num_items=n_items,
            user_seen=user_seen_idx,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        # Instantiate model & optimizer
        self.model = TwoTowerModule(num_users=n_users, num_items=n_items, embedding_dim=self.embedding_dim)
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        logger.info(
            "Training Two-Tower Neural Recommender (users=%d, items=%d, epochs=%d, dim=%d)...",
            n_users,
            n_items,
            self.epochs,
            self.embedding_dim,
        )

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            n_batches = 0
            for u_batch, pos_batch, neg_batch in loader:
                u_batch = u_batch.to(self.device)
                pos_batch = pos_batch.to(self.device)
                neg_batch = neg_batch.to(self.device)

                optimizer.zero_grad()
                pos_score, neg_score = self.model(u_batch, pos_batch, neg_batch)

                # BPR loss: -log(sigmoid(pos_score - neg_score))
                loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            if epoch % 2 == 0 or epoch == self.epochs:
                logger.info("Epoch %d/%d - BPR Loss: %.4f", epoch, self.epochs, avg_loss)

        # Build FAISS candidate retrieval index
        self.model.eval()
        with torch.no_grad():
            all_item_idx = torch.arange(n_items, dtype=torch.long, device=self.device)
            item_embeddings = self.model.forward_item(all_item_idx).cpu().numpy()

        all_movie_ids = [self.idx2item[idx] for idx in range(n_items)]
        self.faiss_index = FaissVectorIndex(dimension=self.embedding_dim, metric="cosine")
        self.faiss_index.add_items(all_movie_ids, item_embeddings)

        self.is_fitted = True
        return self

    def get_user_embedding(self, user_id: int) -> np.ndarray | None:
        """Extract L2-normalized user embedding vector."""
        if not self.is_fitted or self.model is None or user_id not in self.user2idx:
            return None
        self.model.eval()
        with torch.no_grad():
            u_idx = torch.tensor([self.user2idx[user_id]], dtype=torch.long, device=self.device)
            u_emb = self.model.forward_user(u_idx).cpu().numpy().flatten()
        return u_emb

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Retrieve top-K candidates using Faiss index."""
        if not self.is_fitted or self.faiss_index is None:
            raise RuntimeError("Model must be fitted before generating recommendations.")

        u_vec = self.get_user_embedding(user_id)
        if u_vec is None:
            return []

        seen = self.user_seen_items.get(user_id, set()) if exclude_seen else None
        results, _ = self.faiss_index.search(u_vec, k=k, exclude_ids=seen)
        return results

    def predict_score(self, user_id: int, movie_id: int) -> float:
        """Compute inner product affinity score between user and movie."""
        if not self.is_fitted or self.model is None:
            return 0.0
        if user_id not in self.user2idx or movie_id not in self.item2idx:
            return 0.0

        self.model.eval()
        with torch.no_grad():
            u_idx = torch.tensor([self.user2idx[user_id]], dtype=torch.long, device=self.device)
            i_idx = torch.tensor([self.item2idx[movie_id]], dtype=torch.long, device=self.device)
            u_vec = self.model.forward_user(u_idx)
            i_vec = self.model.forward_item(i_idx)
            score = (u_vec * i_vec).sum().item()
        return float(score)
