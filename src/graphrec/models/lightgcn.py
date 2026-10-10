"""LightGCN: Simplified Graph Convolutional Network for Collaborative Filtering."""

from __future__ import annotations

import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from torch.utils.data import DataLoader, Dataset

from src.graphrec.retrieval.vector_index import FaissVectorIndex

from .base import BaseRecommender

logger = logging.getLogger(__name__)


class LightGCNDataset(Dataset):
    """Pairwise BPR training dataset for LightGCN."""

    def __init__(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        num_items: int,
        user_seen: dict[int, set[int]],
    ):
        self.users = user_indices
        self.pos_items = item_indices
        self.num_items = num_items
        self.user_seen = user_seen

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[int, int, int]:
        u = int(self.users[idx])
        pos_i = int(self.pos_items[idx])
        seen = self.user_seen.get(u, set())

        neg_i = random.randint(0, self.num_items - 1)
        while neg_i in seen:
            neg_i = random.randint(0, self.num_items - 1)

        return u, pos_i, neg_i


class LightGCNModule(nn.Module):
    """PyTorch implementation of LightGCN message passing over normalized bipartite adjacency matrix."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Node initial embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def forward_all(self, norm_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-hop graph representations using normalized adjacency matrix."""
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs_list = [all_emb]

        for _ in range(self.num_layers):
            if norm_adj.is_sparse:
                all_emb = torch.sparse.mm(norm_adj, all_emb)
            else:
                all_emb = torch.matmul(norm_adj, all_emb)
            embs_list.append(all_emb)

        # Average across all layer representations: (E^0 + E^1 + ... + E^K) / (K + 1)
        final_embs = torch.stack(embs_list, dim=0).mean(dim=0)
        user_final, item_final = torch.split(final_embs, [self.num_users, self.num_items], dim=0)
        return user_final, item_final

    def compute_bpr_loss(
        self,
        user_final: torch.Tensor,
        item_final: torch.Tensor,
        u_idx: torch.Tensor,
        pos_idx: torch.Tensor,
        neg_idx: torch.Tensor,
        reg_lambda: float = 1e-4,
    ) -> torch.Tensor:
        """Compute BPR ranking loss and embedding L2 regularization."""
        u_vec = user_final[u_idx]
        pos_vec = item_final[pos_idx]
        neg_vec = item_final[neg_idx]

        pos_score = (u_vec * pos_vec).sum(dim=-1)
        neg_score = (u_vec * neg_vec).sum(dim=-1)

        bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

        # Regularization on initial embeddings
        u_init = self.user_embedding(u_idx)
        pos_init = self.item_embedding(pos_idx)
        neg_init = self.item_embedding(neg_idx)
        reg_loss = reg_lambda * (u_init.norm(2).pow(2) + pos_init.norm(2).pow(2) + neg_init.norm(2).pow(2)) / len(u_idx)

        return bpr_loss + reg_loss


class LightGCNRecommender(BaseRecommender):
    """LightGCN graph collaborative filtering recommender with high-order neighborhood aggregation."""

    def __init__(
        self,
        embedding_dim: int = 64,
        num_layers: int = 2,
        epochs: int = 5,
        batch_size: int = 2048,
        lr: float = 0.005,
        reg_lambda: float = 1e-4,
        max_batches_per_epoch: int = 35,
        device: str = "cpu",
    ):
        super().__init__(name="LightGCN (Graph CF)")
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.max_batches_per_epoch = max_batches_per_epoch
        self.device = torch.device(device)

        self.model: LightGCNModule | None = None
        self.norm_adj: torch.Tensor | None = None
        self.faiss_index: FaissVectorIndex | None = None

        self.user2idx: dict[int, int] = {}
        self.idx2user: dict[int, int] = {}
        self.item2idx: dict[int, int] = {}
        self.idx2item: dict[int, int] = {}
        self.user_seen_items: dict[int, set[int]] = {}

        self.user_embeddings: np.ndarray | None = None
        self.item_embeddings: np.ndarray | None = None

    def _build_norm_adj(
        self,
        n_users: int,
        n_items: int,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
    ) -> torch.Tensor:
        """Construct symmetrically normalized bipartite adjacency matrix A = D^(-1/2) * R * D^(-1/2)."""
        n_nodes = n_users + n_items
        rows = np.concatenate([user_indices, item_indices + n_users])
        cols = np.concatenate([item_indices + n_users, user_indices])
        data = np.ones(len(rows), dtype=np.float32)

        adj = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

        # Degree normalization: D^(-1/2) * A * D^(-1/2)
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
        nonzero = deg > 0
        deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])
        d_mat = sparse.diags(deg_inv_sqrt)

        norm_adj_mat = d_mat.dot(adj).dot(d_mat).tocoo()

        # Convert to PyTorch sparse tensor
        indices = torch.tensor(np.vstack((norm_adj_mat.row, norm_adj_mat.col)), dtype=torch.long)
        values = torch.tensor(norm_adj_mat.data, dtype=torch.float32)
        shape = torch.Size(norm_adj_mat.shape)

        return torch.sparse_coo_tensor(indices, values, shape).to(self.device)

    def fit(self, train_ratings: pd.DataFrame, movies_df: pd.DataFrame) -> LightGCNRecommender:
        """Train LightGCN model on user-item interaction graph."""
        self.user_seen_items = train_ratings.groupby("user_id")["movie_id"].apply(set).to_dict()

        unique_users = sorted(train_ratings["user_id"].unique())
        unique_items = sorted(movies_df["movie_id"].unique())

        self.user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx2user = {idx: uid for idx, uid in enumerate(unique_users)}
        self.item2idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx2item = {idx: iid for idx, iid in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)

        valid_ratings = train_ratings[train_ratings["movie_id"].isin(self.item2idx)].copy()
        u_arr = valid_ratings["user_id"].map(self.user2idx).values
        i_arr = valid_ratings["movie_id"].map(self.item2idx).values

        logger.info(
            "Building normalized graph adjacency for LightGCN (nodes=%d, edges=%d)...",
            n_users + n_items,
            len(u_arr),
        )
        self.norm_adj = self._build_norm_adj(n_users, n_items, u_arr, i_arr)

        user_seen_idx = {
            self.user2idx[uid]: {self.item2idx[m] for m in movies if m in self.item2idx}
            for uid, movies in self.user_seen_items.items()
            if uid in self.user2idx
        }

        dataset = LightGCNDataset(
            user_indices=u_arr,
            item_indices=i_arr,
            num_items=n_items,
            user_seen=user_seen_idx,
        )
        # Use batch size of 2048 for high throughput message passing
        effective_batch_size = max(self.batch_size, 2048)
        loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, drop_last=False)

        self.model = LightGCNModule(
            num_users=n_users,
            num_items=n_items,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        logger.info(
            "Training LightGCN (epochs=%d, layers=%d, dim=%d)...",
            self.epochs,
            self.num_layers,
            self.embedding_dim,
        )

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            n_batches = 0

            for u_batch, pos_batch, neg_batch in loader:
                if n_batches >= self.max_batches_per_epoch:
                    break
                u_batch = u_batch.to(self.device)
                pos_batch = pos_batch.to(self.device)
                neg_batch = neg_batch.to(self.device)

                optimizer.zero_grad()
                u_embs, i_embs = self.model.forward_all(self.norm_adj)
                loss = self.model.compute_bpr_loss(
                    u_embs, i_embs, u_batch, pos_batch, neg_batch, reg_lambda=self.reg_lambda
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if epoch % 2 == 0 or epoch == self.epochs:
                logger.info("Epoch %d/%d - LightGCN Loss: %.4f", epoch, self.epochs, total_loss / max(1, n_batches))

        # Final evaluation embeddings
        self.model.eval()
        with torch.no_grad():
            u_final, i_final = self.model.forward_all(self.norm_adj)
            self.user_embeddings = u_final.cpu().numpy()
            self.item_embeddings = i_final.cpu().numpy()

        all_movie_ids = [self.idx2item[idx] for idx in range(n_items)]
        self.faiss_index = FaissVectorIndex(dimension=self.embedding_dim, metric="cosine")
        self.faiss_index.add_items(all_movie_ids, self.item_embeddings)

        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Recommend top-K movies via LightGCN graph embeddings."""
        if not self.is_fitted or self.user_embeddings is None or self.faiss_index is None:
            raise RuntimeError("Model must be fitted before generating recommendations.")

        if user_id not in self.user2idx:
            return []

        u_idx = self.user2idx[user_id]
        u_vec = self.user_embeddings[u_idx]

        seen = self.user_seen_items.get(user_id, set()) if exclude_seen else None
        results, _ = self.faiss_index.search(u_vec, k=k, exclude_ids=seen)
        return results

    def predict_score(self, user_id: int, movie_id: int) -> float:
        """Compute inner product between graph-convolved embeddings."""
        if (
            not self.is_fitted
            or self.user_embeddings is None
            or self.item_embeddings is None
            or user_id not in self.user2idx
            or movie_id not in self.item2idx
        ):
            return 0.0

        u_idx = self.user2idx[user_id]
        i_idx = self.item2idx[movie_id]
        return float(np.dot(self.user_embeddings[u_idx], self.item_embeddings[i_idx]))
