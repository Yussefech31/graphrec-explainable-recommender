"""Unit tests for MovieLens data loader and metadata enricher."""

import pandas as pd

from src.graphrec.data.loader import MovieLensDataset, load_movielens_1m
from src.graphrec.data.metadata_enricher import MovieMetadataEnricher


def test_metadata_enricher_extraction():
    enricher = MovieMetadataEnricher()
    clean, year, decade = enricher.extract_year("Toy Story (1995)")
    assert clean == "Toy Story"
    assert year == 1995
    assert decade == "1990s"

    clean_no_yr, yr, dec = enricher.extract_year("Unknown Title")
    assert clean_no_yr == "Unknown Title"
    assert yr is None
    assert dec is None


def test_metadata_enricher_curated():
    enricher = MovieMetadataEnricher()
    meta = enricher.enrich_movie(1, "Toy Story (1995)", ["Animation", "Children's", "Comedy"])
    assert "John Lasseter" in meta["directors"]
    assert "Tom Hanks" in meta["actors"]
    assert meta["decade"] == "1990s"
    assert meta["genres"] == ["Animation", "Children's", "Comedy"]


def test_movielens_dataset_synthetic_fallback(tmp_path):
    dataset = load_movielens_1m(tmp_path / "empty_dir")
    assert dataset.num_users > 0
    assert dataset.num_items > 0
    assert len(dataset.ratings) > 0

    split = dataset.leave_last_out_split()
    assert len(split.train) > 0
    assert len(split.test) == dataset.num_users
    # Check that each test interaction is for a unique user
    assert split.test["user_id"].nunique() == dataset.num_users


def test_user_liked_items():
    ratings = pd.DataFrame([
        {"user_id": 1, "movie_id": 10, "rating": 5.0, "timestamp": 100},
        {"user_id": 1, "movie_id": 20, "rating": 2.0, "timestamp": 200},
        {"user_id": 1, "movie_id": 30, "rating": 4.5, "timestamp": 300},
        {"user_id": 1, "movie_id": 40, "rating": 4.0, "timestamp": 400},
        {"user_id": 1, "movie_id": 50, "rating": 3.0, "timestamp": 500},
    ])
    movies = pd.DataFrame([
        {"movie_id": mid, "title": f"M{mid} (2000)", "genres": "Action"}
        for mid in [10, 20, 30, 40, 50]
    ])
    dataset = MovieLensDataset(ratings=ratings, movies=movies, min_user_ratings=5)
    liked = dataset.get_user_liked_items(1, threshold=4.0)
    assert set(liked) == {10, 30, 40}
