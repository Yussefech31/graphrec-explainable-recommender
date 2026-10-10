"""Unit tests for Knowledge Graph path reasoner and explanation formatter."""

import networkx as nx

from src.graphrec.explainability.explanation_formatter import ExplanationFormatter
from src.graphrec.explainability.path_reasoner import (
    ExplanationPath,
    KnowledgeGraphPathReasoner,
)


def test_path_reasoner_finds_paths():
    g = nx.Graph()
    # Add User
    g.add_node("u_1", node_type="User", user_id=1)

    # Add Liked Movie (Heat)
    g.add_node("m_6", node_type="Movie", movie_id=6, title="Heat", year=1995)
    g.add_edge("u_1", "m_6", relation="LIKED", rating=5.0, liked=True)

    # Add Director (Michael Mann)
    g.add_node("d_Michael Mann", node_type="Director", name="Michael Mann")
    g.add_edge("m_6", "d_Michael Mann", relation="DIRECTED_BY")

    # Add Candidate Movie (Casino or Collateral)
    g.add_node("m_100", node_type="Movie", movie_id=100, title="Collateral", year=2004)
    g.add_edge("m_100", "d_Michael Mann", relation="DIRECTED_BY")

    reasoner = KnowledgeGraphPathReasoner(g)
    paths = reasoner.find_explanation_paths(user_id=1, recommended_movie_id=100)

    assert len(paths) >= 1
    top = paths[0]
    assert top.anchor_movie_id == 6
    assert top.anchor_movie_title == "Heat"
    assert top.relation_type == "DIRECTED_BY"
    assert top.connecting_entity_value == "Michael Mann"
    assert top.path_score > 0.0


def test_explanation_formatter_narrative():
    paths = [
        ExplanationPath(
            user_id=1,
            recommended_movie_id=100,
            anchor_movie_id=6,
            anchor_movie_title="Heat",
            anchor_rating=5.0,
            relation_type="DIRECTED_BY",
            connecting_entity_type="Director",
            connecting_entity_value="Michael Mann",
            path_score=1.0,
        )
    ]
    narrative = ExplanationFormatter.format_narrative("Collateral", paths)
    assert "Heat" in narrative
    assert "Michael Mann" in narrative
    assert "5.0★" in narrative

    # Subgraph building
    subgraph = ExplanationFormatter.build_visual_subgraph(
        user_id=1,
        recommended_movie_id=100,
        recommended_title="Collateral",
        paths=paths,
    )
    assert "nodes" in subgraph and "edges" in subgraph
    assert any(n["id"] == "u_1" for n in subgraph["nodes"])
    assert any(n["id"] == "m_100" for n in subgraph["nodes"])
