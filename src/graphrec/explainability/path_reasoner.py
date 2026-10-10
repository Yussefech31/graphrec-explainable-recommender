"""Knowledge Graph meta-path discovery and reasoning for explainable recommendations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class ExplanationPath:
    """Represents a validated reasoning trajectory through the Knowledge Graph."""

    user_id: int
    recommended_movie_id: int
    anchor_movie_id: int
    anchor_movie_title: str
    anchor_rating: float
    relation_type: str  # DIRECTED_BY, STARRING, HAS_GENRE, RELEASED_IN, COLLABORATIVE
    connecting_entity_type: str  # Director, Actor, Genre, Decade, User
    connecting_entity_value: str
    path_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": int(self.user_id),
            "recommended_movie_id": int(self.recommended_movie_id),
            "anchor_movie_id": int(self.anchor_movie_id),
            "anchor_movie_title": str(self.anchor_movie_title),
            "anchor_rating": float(self.anchor_rating),
            "relation_type": str(self.relation_type),
            "connecting_entity_type": str(self.connecting_entity_type),
            "connecting_entity_value": str(self.connecting_entity_value),
            "path_score": round(float(self.path_score), 4),
        }


class KnowledgeGraphPathReasoner:
    """Extracts interpretable meta-paths linking user history to candidate recommendations."""

    RELATION_WEIGHTS = {
        "DIRECTED_BY": 1.0,
        "STARRING": 0.85,
        "HAS_GENRE": 0.50,
        "RELEASED_IN": 0.30,
        "COLLABORATIVE": 0.40,
    }

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def find_explanation_paths(
        self,
        user_id: int,
        recommended_movie_id: int,
        max_paths: int = 5,
    ) -> list[ExplanationPath]:
        """Find the strongest meta-paths connecting user's rated movies to the recommended movie."""
        user_node = f"u_{user_id}"
        rec_node = f"m_{recommended_movie_id}"

        if not self.graph.has_node(user_node) or not self.graph.has_node(rec_node):
            return []

        # 1. Get user's liked movies and their edge attributes
        liked_movies: list[tuple[str, float]] = []
        for nbr in self.graph.neighbors(user_node):
            if self.graph.nodes[nbr].get("node_type") == "Movie":
                edge = self.graph.get_edge_data(user_node, nbr)
                rating = float(edge.get("rating", 3.0))
                if rating >= 3.5:
                    liked_movies.append((nbr, rating))

        # Sort liked movies by rating descending
        liked_movies.sort(key=lambda x: x[1], reverse=True)

        # 2. Get entities connected to recommended movie
        rec_entities: dict[str, tuple[str, str]] = {}
        for nbr in self.graph.neighbors(rec_node):
            edge_data = self.graph.get_edge_data(rec_node, nbr)
            rel = edge_data.get("relation", "")
            node_data = self.graph.nodes[nbr]
            ntype = node_data.get("node_type", "")
            nval = node_data.get("name", str(nbr))
            rec_entities[nbr] = (rel, ntype, nval)

        found_paths: list[ExplanationPath] = []

        # 3. Check for shared entities across liked movies
        for anchor_node, rating in liked_movies:
            anchor_mid = int(self.graph.nodes[anchor_node].get("movie_id", 0))
            anchor_title = self.graph.nodes[anchor_node].get("title", f"Movie #{anchor_mid}")

            for anchor_nbr in self.graph.neighbors(anchor_node):
                if anchor_nbr in rec_entities:
                    rel, ntype, nval = rec_entities[anchor_nbr]
                    rel_weight = self.RELATION_WEIGHTS.get(rel, 0.4)
                    # Score combines relation specificity and user rating enthusiasm
                    score = rel_weight * (rating / 5.0)

                    path = ExplanationPath(
                        user_id=user_id,
                        recommended_movie_id=recommended_movie_id,
                        anchor_movie_id=anchor_mid,
                        anchor_movie_title=anchor_title,
                        anchor_rating=rating,
                        relation_type=rel,
                        connecting_entity_type=ntype,
                        connecting_entity_value=nval,
                        path_score=score,
                    )
                    found_paths.append(path)

        # 4. Sort paths by score descending and deduplicate by (anchor, entity)
        found_paths.sort(key=lambda p: p.path_score, reverse=True)

        deduped: list[ExplanationPath] = []
        seen_keys = set()
        for p in found_paths:
            key = (p.anchor_movie_id, p.connecting_entity_value)
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(p)
                if len(deduped) >= max_paths:
                    break

        return deduped
