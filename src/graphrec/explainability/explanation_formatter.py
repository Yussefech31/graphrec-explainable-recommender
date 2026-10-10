"""Formats knowledge graph explanation paths into natural language sentences and visual subgraphs."""

from __future__ import annotations

from typing import Any

from .path_reasoner import ExplanationPath


class ExplanationFormatter:
    """Generates human-readable narrative explanations and visual graph schemas."""

    RELATION_TEMPLATES = {
        "DIRECTED_BY": "shares director **{entity}** with",
        "STARRING": "features actor **{entity}**, also starring in",
        "HAS_GENRE": "shares genre **{entity}** with",
        "RELEASED_IN": "comes from the same era (**{entity}**) as",
    }

    @classmethod
    def format_narrative(
        cls,
        recommended_movie_title: str,
        paths: list[ExplanationPath],
    ) -> str:
        """Construct fluent natural language explanation summarizing top knowledge graph paths."""
        if not paths:
            return "Recommended based on overall collaborative patterns similar to your viewing preferences."

        top_path = paths[0]
        rating_star = f"{top_path.anchor_rating:.1f}★"

        if top_path.relation_type == "DIRECTED_BY":
            return (
                f"Recommended because you rated **{top_path.anchor_movie_title}** ({rating_star}), "
                f"which was also directed by **{top_path.connecting_entity_value}**."
            )
        elif top_path.relation_type == "STARRING":
            return (
                f"Recommended because you rated **{top_path.anchor_movie_title}** ({rating_star}), "
                f"which stars **{top_path.connecting_entity_value}**."
            )
        elif top_path.relation_type == "HAS_GENRE":
            # Combine multiple shared genres if available
            genres = [p.connecting_entity_value for p in paths if p.relation_type == "HAS_GENRE"][:2]
            genre_str = ", ".join(genres)
            return (
                f"Recommended because you enjoyed **{top_path.anchor_movie_title}** ({rating_star}), "
                f"sharing genre ({genre_str}) with **{recommended_movie_title}**."
            )
        else:
            return (
                f"Recommended because of strong thematic links to **{top_path.anchor_movie_title}** "
                f"via {top_path.connecting_entity_type.lower()} **{top_path.connecting_entity_value}**."
            )

    @classmethod
    def build_visual_subgraph(
        cls,
        user_id: int,
        recommended_movie_id: int,
        recommended_title: str,
        paths: list[ExplanationPath],
    ) -> dict[str, Any]:
        """Generate node and edge JSON structures for interactive graph visualizers (e.g. PyVis, D3, NetworkX)."""
        nodes: list[dict[str, Any]] = [
            {
                "id": f"u_{user_id}",
                "label": f"User #{user_id}",
                "group": "User",
                "color": "#6366F1",
                "size": 25,
            },
            {
                "id": f"m_{recommended_movie_id}",
                "label": recommended_title,
                "group": "RecommendedMovie",
                "color": "#10B981",
                "size": 30,
            },
        ]

        edges: list[dict[str, Any]] = []
        seen_nodes = {f"u_{user_id}", f"m_{recommended_movie_id}"}
        seen_edges = set()

        for path in paths:
            anchor_node_id = f"m_{path.anchor_movie_id}"
            entity_node_id = f"ent_{path.connecting_entity_type}_{path.connecting_entity_value}"

            # Add anchor movie node
            if anchor_node_id not in seen_nodes:
                nodes.append({
                    "id": anchor_node_id,
                    "label": path.anchor_movie_title,
                    "group": "LikedMovie",
                    "color": "#3B82F6",
                    "size": 22,
                })
                seen_nodes.add(anchor_node_id)

            # Add user -> anchor edge
            e1 = (f"u_{user_id}", anchor_node_id)
            if e1 not in seen_edges:
                edges.append({
                    "from": f"u_{user_id}",
                    "to": anchor_node_id,
                    "label": f"LIKED ({path.anchor_rating:.1f}★)",
                    "color": "#818CF8",
                })
                seen_edges.add(e1)

            # Add entity node
            if entity_node_id not in seen_nodes:
                color_map = {
                    "Director": "#F59E0B",
                    "Actor": "#EC4899",
                    "Genre": "#8B5CF6",
                    "Decade": "#64748B",
                }
                nodes.append({
                    "id": entity_node_id,
                    "label": path.connecting_entity_value,
                    "group": path.connecting_entity_type,
                    "color": color_map.get(path.connecting_entity_type, "#94A3B8"),
                    "size": 18,
                })
                seen_nodes.add(entity_node_id)

            # Add anchor -> entity edge
            e2 = (anchor_node_id, entity_node_id)
            if e2 not in seen_edges:
                edges.append({
                    "from": anchor_node_id,
                    "to": entity_node_id,
                    "label": path.relation_type,
                    "color": "#CBD5E1",
                })
                seen_edges.add(e2)

            # Add recommended -> entity edge
            e3 = (f"m_{recommended_movie_id}", entity_node_id)
            if e3 not in seen_edges:
                edges.append({
                    "from": f"m_{recommended_movie_id}",
                    "to": entity_node_id,
                    "label": path.relation_type,
                    "color": "#CBD5E1",
                })
                seen_edges.add(e3)

        return {"nodes": nodes, "edges": edges}
