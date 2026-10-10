"""Streamlit interactive UI for GraphRec: Explainable Hybrid Recommender Platform."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graphrec.data.graph_builder import KnowledgeGraphBuilder
from src.graphrec.data.loader import load_movielens_1m
from src.graphrec.explainability.explanation_formatter import ExplanationFormatter
from src.graphrec.explainability.path_reasoner import KnowledgeGraphPathReasoner
from src.graphrec.models.als import MatrixFactorizationRecommender
from src.graphrec.models.item_cf import ItemItemCFRecommender
from src.graphrec.models.lightgcn import LightGCNRecommender
from src.graphrec.models.popularity import PopularityRecommender
from src.graphrec.models.two_tower import TwoTowerRecommender

st.set_page_config(
    page_title="GraphRec | Explainable Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading MovieLens 1M and building Knowledge Graph...")
def get_system():
    dataset = load_movielens_1m("data")
    split = dataset.leave_last_out_split()

    builder = KnowledgeGraphBuilder(dataset)
    builder.build_graph(split.train)
    reasoner = KnowledgeGraphPathReasoner(builder.graph)

    # Pre-train models
    models = {
        "Two-Tower (PyTorch + FAISS)": TwoTowerRecommender(embedding_dim=32, epochs=4).fit(split.train, dataset.movies_raw),
        "LightGCN (Graph CF)": LightGCNRecommender(embedding_dim=32, epochs=3).fit(split.train, dataset.movies_raw),
        "Matrix Factorization (ALS)": MatrixFactorizationRecommender(n_factors=32).fit(split.train, dataset.movies_raw),
        "Item-Item CF": ItemItemCFRecommender().fit(split.train, dataset.movies_raw),
        "Popularity Baseline": PopularityRecommender().fit(split.train, dataset.movies_raw),
    }

    return dataset, builder, reasoner, models


def render_subgraph_plot(subgraph_data: dict):
    """Draw matplotlib graph for explanation trajectory."""
    g = nx.Graph()
    colors = []
    labels = {}

    for n in subgraph_data["nodes"]:
        nid = n["id"]
        g.add_node(nid)
        labels[nid] = n["label"]
        colors.append(n.get("color", "#6366F1"))

    for e in subgraph_data["edges"]:
        g.add_edge(e["from"], e["to"], label=e.get("label", ""))

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    pos = nx.spring_layout(g, seed=42, k=1.2)
    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=1200, ax=ax)
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8, font_color="white", ax=ax)
    nx.draw_networkx_edges(g, pos, edge_color="#94A3B8", width=1.5, alpha=0.7, ax=ax)

    edge_labels = {(e["from"], e["to"]): e["label"] for e in subgraph_data["edges"]}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=7, font_color="#E2E8F0", ax=ax)

    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def main():
    dataset, builder, reasoner, models = get_system()

    st.title("🎬 GraphRec: Explainable Hybrid Recommender Platform")
    st.markdown(
        "Combines **PyTorch Two-Tower embeddings**, **sub-millisecond FAISS retrieval**, "
        "and **Knowledge Graph meta-path reasoning** for transparent recommendations."
    )

    # Sidebar
    st.sidebar.header("Configuration")
    sample_users = [1, 2, 6, 8, 10, 15, 23, 48, 100]
    user_id = st.sidebar.selectbox("Select User Profile", options=sample_users, index=0)

    model_choice = st.sidebar.selectbox("Recommender Model", list(models.keys()), index=0)
    top_k = st.sidebar.slider("Top-K Items", min_value=3, max_value=20, value=8)
    explain_enabled = st.sidebar.checkbox("Enable KG Explanation Reasoning", value=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("👤 User Profile & Liked Movies")
        liked = dataset.get_user_liked_items(user_id, threshold=4.0)
        st.write(f"User has rated **{len(dataset.ratings[dataset.ratings['user_id'] == user_id])}** movies, with **{len(liked)}** rated 4★ or 5★.")

        for mid in liked[:8]:
            meta = dataset.get_movie_meta(mid)
            with st.expander(f"⭐ {meta['clean_title']} ({meta['year'] or 'N/A'})"):
                st.caption(f"**Genres:** {', '.join(meta['genres'])}")
                st.caption(f"**Director:** {', '.join(meta['directors'])}")
                st.caption(f"**Actors:** {', '.join(meta['actors'])}")

    with col2:
        st.subheader(f"✨ Top-{top_k} Recommendations ({model_choice})")
        model = models[model_choice]
        recs = model.recommend(user_id=user_id, k=top_k, exclude_seen=True)

        if not recs:
            st.warning("No recommendations available for this user.")
            return

        for rank, (mid, score) in enumerate(recs, start=1):
            meta = dataset.get_movie_meta(mid)
            title = meta["clean_title"]

            st.markdown(f"#### {rank}. {title} ({meta['year'] or 'N/A'}) — Affinity: `{score:.3f}`")
            st.caption(f"**Genres:** {', '.join(meta['genres'])} | **Directors:** {', '.join(meta['directors'])}")

            if explain_enabled:
                paths = reasoner.find_explanation_paths(user_id=user_id, recommended_movie_id=mid, max_paths=3)
                narrative = ExplanationFormatter.format_narrative(title, paths)
                st.info(f"💡 **Why this movie?** {narrative}")

                if paths:
                    with st.expander("🔍 View Knowledge Graph Explanation Path Subgraph"):
                        subgraph = ExplanationFormatter.build_visual_subgraph(
                            user_id=user_id,
                            recommended_movie_id=mid,
                            recommended_title=title,
                            paths=paths,
                        )
                        render_subgraph_plot(subgraph)

            st.markdown("---")


if __name__ == "__main__":
    main()
