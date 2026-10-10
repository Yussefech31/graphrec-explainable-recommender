"""Unit tests for recommendation models."""

import pytest

from src.graphrec.data.loader import load_movielens_1m
from src.graphrec.models.als import MatrixFactorizationRecommender
from src.graphrec.models.item_cf import ItemItemCFRecommender
from src.graphrec.models.lightgcn import LightGCNRecommender
from src.graphrec.models.popularity import PopularityRecommender
from src.graphrec.models.two_tower import TwoTowerRecommender


@pytest.fixture(scope="module")
def dataset_and_split():
    dataset = load_movielens_1m("data", max_ratings=5000)
    split = dataset.leave_last_out_split()
    return dataset, split


def test_popularity_recommender(dataset_and_split):
    dataset, split = dataset_and_split
    model = PopularityRecommender()
    model.fit(split.train, dataset.movies_raw)

    recs = model.recommend(user_id=1, k=5, exclude_seen=True)
    assert len(recs) == 5
    assert all(score > 0 for _, score in recs)

    score = model.predict_score(user_id=1, movie_id=recs[0][0])
    assert score > 0


def test_item_item_cf_recommender(dataset_and_split):
    dataset, split = dataset_and_split
    model = ItemItemCFRecommender(top_similar_neighbors=20)
    model.fit(split.train, dataset.movies_raw)

    recs = model.recommend(user_id=1, k=5, exclude_seen=True)
    assert len(recs) <= 5
    score = model.predict_score(user_id=1, movie_id=dataset.unique_items[0])
    assert isinstance(score, float)


def test_matrix_factorization_recommender(dataset_and_split):
    dataset, split = dataset_and_split
    model = MatrixFactorizationRecommender(n_factors=16, n_iter=5)
    model.fit(split.train, dataset.movies_raw)

    recs = model.recommend(user_id=1, k=5, exclude_seen=True)
    assert len(recs) == 5
    score = model.predict_score(user_id=1, movie_id=dataset.unique_items[0])
    assert isinstance(score, float)


def test_two_tower_recommender(dataset_and_split):
    dataset, split = dataset_and_split
    model = TwoTowerRecommender(embedding_dim=16, epochs=2, batch_size=256)
    model.fit(split.train, dataset.movies_raw)

    recs = model.recommend(user_id=1, k=5, exclude_seen=True)
    assert len(recs) == 5

    u_emb = model.get_user_embedding(user_id=1)
    assert u_emb is not None
    assert len(u_emb) == 16

    score = model.predict_score(user_id=1, movie_id=dataset.unique_items[0])
    assert isinstance(score, float)


def test_lightgcn_recommender(dataset_and_split):
    dataset, split = dataset_and_split
    model = LightGCNRecommender(embedding_dim=16, num_layers=2, epochs=2, batch_size=256)
    model.fit(split.train, dataset.movies_raw)

    recs = model.recommend(user_id=1, k=5, exclude_seen=True)
    assert len(recs) == 5

    score = model.predict_score(user_id=1, movie_id=dataset.unique_items[0])
    assert isinstance(score, float)
