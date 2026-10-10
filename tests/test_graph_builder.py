"""Unit tests for heterogeneous Knowledge Graph builder."""

import pandas as pd

from src.graphrec.data.graph_builder import KnowledgeGraphBuilder
from src.graphrec.data.loader import MovieLensDataset


def test_knowledge_graph_construction(tmp_path):
    ratings = pd.DataFrame([
        {"user_id": 1, "movie_id": 1, "rating": 5.0, "timestamp": 100},
        {"user_id": 1, "movie_id": 2, "rating": 2.0, "timestamp": 200},
        {"user_id": 1, "movie_id": 3, "rating": 4.5, "timestamp": 300},
        {"user_id": 1, "movie_id": 4, "rating": 4.0, "timestamp": 400},
        {"user_id": 1, "movie_id": 5, "rating": 3.0, "timestamp": 500},
    ])
    movies = pd.DataFrame([
        {"movie_id": 1, "title": "Toy Story (1995)", "genres": "Animation|Comedy"},
        {"movie_id": 2, "title": "Jumanji (1995)", "genres": "Adventure|Fantasy"},
        {"movie_id": 3, "title": "Heat (1995)", "genres": "Action|Crime"},
        {"movie_id": 4, "title": "Seven (1995)", "genres": "Crime|Thriller"},
        {"movie_id": 5, "title": "Casino (1995)", "genres": "Crime|Drama"},
    ])
    dataset = MovieLensDataset(ratings=ratings, movies=movies, min_user_ratings=5)
    builder = KnowledgeGraphBuilder(dataset)
    graph = builder.build_graph(min_like_rating=4.0)

    # Check node types
    assert graph.has_node("u_1")
    assert graph.nodes["u_1"]["node_type"] == "User"

    assert graph.has_node("m_1")
    assert graph.nodes["m_1"]["node_type"] == "Movie"

    assert graph.has_node("g_Animation")
    assert graph.nodes["g_Animation"]["node_type"] == "Genre"

    assert graph.has_node("d_John Lasseter")
    assert graph.nodes["d_John Lasseter"]["node_type"] == "Director"

    # Check edges
    assert graph.has_edge("u_1", "m_1")
    assert graph.get_edge_data("u_1", "m_1")["relation"] == "LIKED"
    assert graph.get_edge_data("u_1", "m_1")["liked"] is True

    assert graph.has_edge("u_1", "m_2")
    assert graph.get_edge_data("u_1", "m_2")["relation"] == "RATED"
    assert graph.get_edge_data("u_1", "m_2")["liked"] is False

    # Check movie entities retrieval
    entities = builder.get_movie_entities(1)
    assert "Animation" in entities["genres"]
    assert "John Lasseter" in entities["directors"]

    # Check user liked movies
    liked_nodes = builder.get_user_liked_movie_nodes(1)
    assert set(liked_nodes) == {"m_1", "m_3", "m_4"}

    # Export cypher test
    cypher_file = tmp_path / "test.cypher"
    builder.export_cypher_script(cypher_file)
    assert cypher_file.exists()
    assert cypher_file.stat().st_size > 0
