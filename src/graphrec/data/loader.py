"""MovieLens 1M data loader, preprocessor, and temporal split manager."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metadata_enricher import MovieMetadataEnricher

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    """Holds train, validation, and test interaction sets."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    num_users: int
    num_items: int


class MovieLensDataset:
    """High-level abstraction for MovieLens dataset with rich metadata and graph indexing."""

    def __init__(
        self,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        users: pd.DataFrame | None = None,
        min_user_ratings: int = 5,
    ):
        self.movies_raw = movies.copy()
        self.users_raw = users.copy() if users is not None else None

        # Filter users with minimum activity
        user_counts = ratings["user_id"].value_counts()
        active_users = user_counts[user_counts >= min_user_ratings].index
        self.ratings = ratings[ratings["user_id"].isin(active_users)].copy()

        # Sort chronologically
        self.ratings.sort_values(by=["user_id", "timestamp"], inplace=True)
        self.ratings.reset_index(drop=True, inplace=True)

        # Enrich movies metadata
        enricher = MovieMetadataEnricher()
        enriched_records: list[dict[str, Any]] = []
        for _, row in self.movies_raw.iterrows():
            genres_list = str(row["genres"]).split("|") if pd.notna(row["genres"]) else []
            meta = enricher.enrich_movie(int(row["movie_id"]), str(row["title"]), genres_list)
            enriched_records.append(meta)

        self.movies_enriched = pd.DataFrame(enriched_records)
        self.movie_meta_dict: dict[int, dict[str, Any]] = {
            r["movie_id"]: r for r in enriched_records
        }

        # ID Mappings (0-indexed contiguous integer IDs for tensor models)
        self.unique_users = sorted(self.ratings["user_id"].unique())
        self.unique_items = sorted(self.ratings["movie_id"].unique())

        self.user2idx = {uid: idx for idx, uid in enumerate(self.unique_users)}
        self.idx2user = {idx: uid for idx, uid in enumerate(self.unique_users)}
        self.item2idx = {iid: idx for idx, iid in enumerate(self.unique_items)}
        self.idx2item = {idx: iid for idx, iid in enumerate(self.unique_items)}

        self.num_users = len(self.unique_users)
        self.num_items = len(self.unique_items)

        # Add mapped indices to ratings
        self.ratings["user_idx"] = self.ratings["user_id"].map(self.user2idx)
        self.ratings["item_idx"] = self.ratings["movie_id"].map(self.item2idx)

    def leave_last_out_split(self) -> DatasetSplit:
        """Standard RecSys leave-last-out temporal evaluation split.

        For each user:
          - The latest interaction is reserved for test
          - The penultimate interaction is reserved for validation
          - All prior interactions form the training set
        """
        train_rows: list[pd.DataFrame] = []
        val_rows: list[pd.DataFrame] = []
        test_rows: list[pd.DataFrame] = []

        for _, group in self.ratings.groupby("user_id"):
            if len(group) >= 3:
                train_rows.append(group.iloc[:-2])
                val_rows.append(group.iloc[[-2]])
                test_rows.append(group.iloc[[-1]])
            else:
                train_rows.append(group.iloc[:-1])
                test_rows.append(group.iloc[[-1]])

        train_df = pd.concat(train_rows, ignore_index=True)
        val_df = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame()
        test_df = pd.concat(test_rows, ignore_index=True)

        return DatasetSplit(
            train=train_df,
            val=val_df,
            test=test_df,
            num_users=self.num_users,
            num_items=self.num_items,
        )

    def get_user_liked_items(self, user_id: int, threshold: float = 4.0) -> list[int]:
        """Return list of movie IDs liked (rating >= threshold) by user."""
        u_ratings = self.ratings[self.ratings["user_id"] == user_id]
        liked = u_ratings[u_ratings["rating"] >= threshold]
        return liked["movie_id"].tolist()

    def get_movie_meta(self, movie_id: int) -> dict[str, Any]:
        """Retrieve enriched metadata dictionary for a movie."""
        return self.movie_meta_dict.get(
            movie_id,
            {
                "movie_id": movie_id,
                "raw_title": f"Movie #{movie_id}",
                "clean_title": f"Movie #{movie_id}",
                "year": None,
                "decade": "Unknown",
                "genres": [],
                "directors": [],
                "actors": [],
            },
        )

    def get_top_popular_movie_ids(self, n: int = 50) -> list[int]:
        """Return top N most rated movie IDs in descending order."""
        return self.ratings["movie_id"].value_counts().head(n).index.tolist()


def load_movielens_1m(data_dir: Path | str, max_ratings: int | None = None) -> MovieLensDataset:
    """Load MovieLens 1M dataset from local files or generate synthetic fallback if absent."""
    base_path = Path(data_dir)
    candidates = [
        base_path / "ml-1m",
        base_path / "raw" / "ml-1m",
        base_path / "sample",
        base_path,
    ]

    found_dir: Path | None = None
    for cand in candidates:
        if (cand / "ratings.dat").exists() and (cand / "movies.dat").exists():
            found_dir = cand
            break

    if found_dir is not None:
        logger.info("Loading MovieLens 1M from %s", found_dir)
        ratings = pd.read_csv(
            found_dir / "ratings.dat",
            sep="::",
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python",
            encoding="latin-1",
        )
        if max_ratings is not None and len(ratings) > max_ratings:
            ratings = ratings.head(max_ratings).copy()
        movies = pd.read_csv(
            found_dir / "movies.dat",
            sep="::",
            names=["movie_id", "title", "genres"],
            engine="python",
            encoding="latin-1",
        )
        users = None
        if (found_dir / "users.dat").exists():
            users = pd.read_csv(
                found_dir / "users.dat",
                sep="::",
                names=["user_id", "gender", "age", "occupation", "zip_code"],
                engine="python",
                encoding="latin-1",
            )
        return MovieLensDataset(ratings=ratings, movies=movies, users=users)

    # Fallback to realistic synthetic MovieLens snapshot for automated testing/CI
    logger.warning("MovieLens 1M .dat files not found in %s. Creating synthetic dataset.", base_path)
    np.random.seed(42)
    sample_movies = pd.DataFrame([
        {"movie_id": 1, "title": "Toy Story (1995)", "genres": "Animation|Children's|Comedy"},
        {"movie_id": 2, "title": "Jumanji (1995)", "genres": "Adventure|Children's|Fantasy"},
        {"movie_id": 6, "title": "Heat (1995)", "genres": "Action|Crime|Thriller"},
        {"movie_id": 10, "title": "GoldenEye (1995)", "genres": "Action|Adventure|Thriller"},
        {"movie_id": 47, "title": "Seven (1995)", "genres": "Crime|Thriller"},
        {"movie_id": 50, "title": "Usual Suspects, The (1995)", "genres": "Crime|Thriller"},
        {"movie_id": 110, "title": "Braveheart (1995)", "genres": "Action|Drama|War"},
        {"movie_id": 260, "title": "Star Wars: Episode IV - A New Hope (1977)", "genres": "Action|Adventure|Sci-Fi"},
        {"movie_id": 296, "title": "Pulp Fiction (1994)", "genres": "Crime|Drama"},
        {"movie_id": 318, "title": "Shawshank Redemption, The (1994)", "genres": "Drama"},
        {"movie_id": 356, "title": "Forrest Gump (1994)", "genres": "Comedy|Romance|War"},
        {"movie_id": 480, "title": "Jurassic Park (1993)", "genres": "Action|Adventure|Sci-Fi"},
        {"movie_id": 527, "title": "Schindler's List (1993)", "genres": "Drama|War"},
        {"movie_id": 589, "title": "Terminator 2: Judgment Day (1991)", "genres": "Action|Sci-Fi|Thriller"},
        {"movie_id": 593, "title": "Silence of the Lambs, The (1991)", "genres": "Drama|Thriller"},
        {"movie_id": 858, "title": "Godfather, The (1972)", "genres": "Action|Crime|Drama"},
        {"movie_id": 1196, "title": "Star Wars: Episode V - The Empire Strikes Back (1980)", "genres": "Action|Adventure|Sci-Fi"},
        {"movie_id": 1210, "title": "Star Wars: Episode VI - Return of the Jedi (1983)", "genres": "Action|Adventure|Sci-Fi"},
        {"movie_id": 2571, "title": "Matrix, The (1999)", "genres": "Action|Sci-Fi|Thriller"},
        {"movie_id": 2959, "title": "Fight Club (1999)", "genres": "Drama"},
    ])

    sample_ratings_list = []
    user_ids = list(range(1, 31))
    movie_ids = sample_movies["movie_id"].tolist()
    ts = 978300000

    for uid in user_ids:
        # Each user rates 8-15 movies
        n_rated = np.random.randint(8, 16)
        chosen_movies = np.random.choice(movie_ids, size=n_rated, replace=False)
        for i, mid in enumerate(chosen_movies):
            rating = float(np.random.choice([3.0, 4.0, 5.0], p=[0.2, 0.4, 0.4]))
            sample_ratings_list.append({
                "user_id": uid,
                "movie_id": int(mid),
                "rating": rating,
                "timestamp": ts + i * 3600,
            })

    sample_ratings = pd.DataFrame(sample_ratings_list)
    return MovieLensDataset(ratings=sample_ratings, movies=sample_movies)
