"""Knowledge Graph construction and graph indexing using NetworkX."""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import pandas as pd

from .loader import MovieLensDataset

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Builds a heterogeneous knowledge graph connecting Users, Movies, Genres, Directors, Actors, and Decades."""

    def __init__(self, dataset: MovieLensDataset):
        self.dataset = dataset
        self.graph: nx.Graph = nx.Graph()

    def build_graph(
        self,
        ratings_df: pd.DataFrame | None = None,
        min_like_rating: float = 4.0,
    ) -> nx.Graph:
        """Construct the heterogeneous knowledge graph.

        Args:
            ratings_df: Ratings dataframe to include (defaults to all dataset ratings)
            min_like_rating: Threshold above which a rating constitutes a positive 'LIKED' link
        """
        self.graph.clear()
        ratings = ratings_df if ratings_df is not None else self.dataset.ratings

        logger.info("Constructing Knowledge Graph nodes and metadata edges...")

        # 1. Add Movie nodes and item-entity edges
        for _, row in self.dataset.movies_enriched.iterrows():
            mid = int(row["movie_id"])
            movie_node = f"m_{mid}"
            self.graph.add_node(
                movie_node,
                node_type="Movie",
                movie_id=mid,
                title=row["clean_title"],
                raw_title=row["raw_title"],
                year=row["year"],
                decade=row["decade"],
            )

            # Genres
            for genre in row["genres"]:
                genre_node = f"g_{genre}"
                if not self.graph.has_node(genre_node):
                    self.graph.add_node(genre_node, node_type="Genre", name=genre)
                self.graph.add_edge(movie_node, genre_node, relation="HAS_GENRE")

            # Directors
            for director in row["directors"]:
                dir_node = f"d_{director}"
                if not self.graph.has_node(dir_node):
                    self.graph.add_node(dir_node, node_type="Director", name=director)
                self.graph.add_edge(movie_node, dir_node, relation="DIRECTED_BY")

            # Actors
            for actor in row["actors"]:
                act_node = f"a_{actor}"
                if not self.graph.has_node(act_node):
                    self.graph.add_node(act_node, node_type="Actor", name=actor)
                self.graph.add_edge(movie_node, act_node, relation="STARRING")

            # Decade
            if row["decade"] and row["decade"] != "Unknown":
                dec_node = f"dec_{row['decade']}"
                if not self.graph.has_node(dec_node):
                    self.graph.add_node(dec_node, node_type="Decade", name=row["decade"])
                self.graph.add_edge(movie_node, dec_node, relation="RELEASED_IN")

        # 2. Add User nodes and User-Movie interaction edges
        for _, row in ratings.iterrows():
            uid = int(row["user_id"])
            mid = int(row["movie_id"])
            rating = float(row["rating"])
            user_node = f"u_{uid}"
            movie_node = f"m_{mid}"

            if not self.graph.has_node(user_node):
                self.graph.add_node(user_node, node_type="User", user_id=uid)

            if self.graph.has_node(movie_node):
                is_liked = rating >= min_like_rating
                self.graph.add_edge(
                    user_node,
                    movie_node,
                    relation="LIKED" if is_liked else "RATED",
                    rating=rating,
                    liked=is_liked,
                    timestamp=int(row["timestamp"]) if "timestamp" in row else 0,
                )

        logger.info(
            "Graph construction complete: %d nodes, %d edges.",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    def get_movie_entities(self, movie_id: int) -> dict[str, list[str]]:
        """Return all genres, directors, actors, and decade associated with a movie."""
        movie_node = f"m_{movie_id}"
        result: dict[str, list[str]] = {
            "genres": [],
            "directors": [],
            "actors": [],
            "decade": [],
        }
        if not self.graph.has_node(movie_node):
            return result

        for nbr in self.graph.neighbors(movie_node):
            ntype = self.graph.nodes[nbr].get("node_type")
            if ntype == "Genre":
                result["genres"].append(self.graph.nodes[nbr].get("name", nbr))
            elif ntype == "Director":
                result["directors"].append(self.graph.nodes[nbr].get("name", nbr))
            elif ntype == "Actor":
                result["actors"].append(self.graph.nodes[nbr].get("name", nbr))
            elif ntype == "Decade":
                result["decade"].append(self.graph.nodes[nbr].get("name", nbr))

        return result

    def get_user_liked_movie_nodes(self, user_id: int) -> list[str]:
        """Return list of movie node identifiers liked by the user."""
        user_node = f"u_{user_id}"
        if not self.graph.has_node(user_node):
            return []
        liked = []
        for nbr in self.graph.neighbors(user_node):
            if self.graph.nodes[nbr].get("node_type") == "Movie":
                edge_data = self.graph.get_edge_data(user_node, nbr)
                if edge_data.get("liked", False):
                    liked.append(nbr)
        return liked

    def export_cypher_script(self, output_file: Path | str, max_users: int = 500) -> None:
        """Export the knowledge graph as Cypher commands for Neo4j ingestion."""
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [
            "// Neo4j Cypher Schema & Ingestion Script for GraphRec",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Director) REQUIRE d.name IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Actor) REQUIRE a.name IS UNIQUE;",
            "",
        ]

        # Entity nodes
        genres_seen = set()
        directors_seen = set()
        actors_seen = set()

        for _, row in self.dataset.movies_enriched.iterrows():
            mid = int(row["movie_id"])
            title_escaped = str(row["clean_title"]).replace("'", "\\'")
            lines.append(
                f"MERGE (m:Movie {{id: {mid}, title: '{title_escaped}', year: {row['year'] or 0}}});"
            )

            for g in row["genres"]:
                g_esc = g.replace("'", "\\'")
                if g not in genres_seen:
                    lines.append(f"MERGE (:Genre {{name: '{g_esc}'}});")
                    genres_seen.add(g)
                lines.append(
                    f"MATCH (m:Movie {{id: {mid}}}), (g:Genre {{name: '{g_esc}'}}) MERGE (m)-[:HAS_GENRE]->(g);"
                )

            for d in row["directors"]:
                d_esc = d.replace("'", "\\'")
                if d not in directors_seen:
                    lines.append(f"MERGE (:Director {{name: '{d_esc}'}});")
                    directors_seen.add(d)
                lines.append(
                    f"MATCH (m:Movie {{id: {mid}}}), (d:Director {{name: '{d_esc}'}}) MERGE (m)-[:DIRECTED_BY]->(d);"
                )

            for a in row["actors"]:
                a_esc = a.replace("'", "\\'")
                if a not in actors_seen:
                    lines.append(f"MERGE (:Actor {{name: '{a_esc}'}});")
                    actors_seen.add(a)
                lines.append(
                    f"MATCH (m:Movie {{id: {mid}}}), (a:Actor {{name: '{a_esc}'}}) MERGE (m)-[:STARRING]->(a);"
                )

        # Users and ratings
        user_count = 0
        for uid in self.dataset.unique_users[:max_users]:
            lines.append(f"MERGE (:User {{id: {uid}}});")
            user_ratings = self.dataset.ratings[self.dataset.ratings["user_id"] == uid]
            for _, r in user_ratings.iterrows():
                mid = int(r["movie_id"])
                rating = float(r["rating"])
                rel = "LIKED" if rating >= 4.0 else "RATED"
                lines.append(
                    f"MATCH (u:User {{id: {uid}}}), (m:Movie {{id: {mid}}}) "
                    f"MERGE (u)-[:{rel} {{rating: {rating}}}]->(m);"
                )
            user_count += 1

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Exported %d Cypher lines to %s", len(lines), out_path)
