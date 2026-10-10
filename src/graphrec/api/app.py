"""FastAPI serving layer for explainable recommendations."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.graphrec.data.graph_builder import KnowledgeGraphBuilder
from src.graphrec.data.loader import MovieLensDataset, load_movielens_1m
from src.graphrec.explainability.explanation_formatter import ExplanationFormatter
from src.graphrec.explainability.path_reasoner import KnowledgeGraphPathReasoner
from src.graphrec.models.als import MatrixFactorizationRecommender
from src.graphrec.models.base import BaseRecommender
from src.graphrec.models.item_cf import ItemItemCFRecommender
from src.graphrec.models.lightgcn import LightGCNRecommender
from src.graphrec.models.popularity import PopularityRecommender
from src.graphrec.models.two_tower import TwoTowerRecommender

from .schemas import (
    ExplanationPathSchema,
    HealthResponse,
    MovieRecommendation,
    MovieSearchItem,
    RecommendationResponse,
)

logger = logging.getLogger(__name__)

# State container
state: dict[str, Any] = {
    "dataset": None,
    "graph_builder": None,
    "path_reasoner": None,
    "models": {},
    "ready": False,
}


def initialize_app_state(
    data_path: Path | str | None = None,
    max_ratings: int | None = None,
) -> None:
    """Load dataset, build knowledge graph, and train/initialize models."""
    if data_path is None:
        data_path = Path(os.environ.get("DATA_DIR", "data"))

    logger.info("Initializing GraphRec application state from %s...", data_path)
    dataset: MovieLensDataset = load_movielens_1m(data_path, max_ratings=max_ratings)
    split = dataset.leave_last_out_split()

    # Build Knowledge Graph
    graph_builder = KnowledgeGraphBuilder(dataset)
    graph_builder.build_graph(split.train)
    path_reasoner = KnowledgeGraphPathReasoner(graph_builder.graph)

    # Initialize models
    models: dict[str, BaseRecommender] = {}

    pop = PopularityRecommender().fit(split.train, dataset.movies_raw)
    models["popularity"] = pop

    item_cf = ItemItemCFRecommender().fit(split.train, dataset.movies_raw)
    models["item_cf"] = item_cf

    als = MatrixFactorizationRecommender(n_factors=32, n_iter=10).fit(split.train, dataset.movies_raw)
    models["als"] = als

    two_tower = TwoTowerRecommender(embedding_dim=32, epochs=4, batch_size=256).fit(split.train, dataset.movies_raw)
    models["two_tower"] = two_tower

    # LightGCN
    lightgcn = LightGCNRecommender(embedding_dim=32, epochs=3, batch_size=256).fit(split.train, dataset.movies_raw)
    models["lightgcn"] = lightgcn

    state["dataset"] = dataset
    state["graph_builder"] = graph_builder
    state["path_reasoner"] = path_reasoner
    state["models"] = models
    state["ready"] = True
    logger.info("GraphRec initialization complete with %d models.", len(models))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to ensure lazy or startup initialization."""
    if not state["ready"]:
        initialize_app_state()
    yield


app = FastAPI(
    title="GraphRec Explainable Recommender API",
    description="High-throughput recommendation serving with FAISS vector retrieval and Knowledge Graph path-based explainability",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return health status, graph statistics, and available models."""
    dataset: MovieLensDataset = state.get("dataset")
    builder: KnowledgeGraphBuilder = state.get("graph_builder")
    models: dict[str, BaseRecommender] = state.get("models", {})

    return HealthResponse(
        status="healthy" if state.get("ready") else "initializing",
        version="0.1.0",
        available_models=list(models.keys()),
        num_users=dataset.num_users if dataset else 0,
        num_movies=dataset.num_items if dataset else 0,
        graph_nodes=builder.graph.number_of_nodes() if builder else 0,
        graph_edges=builder.graph.number_of_edges() if builder else 0,
    )


@app.get("/recommend", response_model=RecommendationResponse)
def get_recommendations(
    user_id: int = Query(..., description="ID of the user to generate recommendations for"),
    k: int = Query(10, ge=1, le=50, description="Number of recommendations to return"),
    model: str = Query("two_tower", description="Model: two_tower, lightgcn, als, item_cf, popularity"),
    explain: bool = Query(True, description="Whether to compute KG reasoning paths and explanations"),
) -> RecommendationResponse:
    """Generate top-K movie recommendations with optional knowledge graph explanations."""
    models: dict[str, BaseRecommender] = state.get("models", {})
    dataset: MovieLensDataset = state.get("dataset")
    reasoner: KnowledgeGraphPathReasoner = state.get("path_reasoner")

    if not state.get("ready") or dataset is None or reasoner is None:
        raise HTTPException(status_code=503, detail="Service initializing or unavailable.")

    selected_model = models.get(model.lower())
    if selected_model is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {list(models.keys())}",
        )

    t0 = time.perf_counter()
    recs = selected_model.recommend(user_id=user_id, k=k, exclude_seen=True)

    # Fallback to popularity if user is cold-start and returned empty
    if not recs:
        recs = models["popularity"].recommend(user_id=user_id, k=k, exclude_seen=True)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    items: list[MovieRecommendation] = []
    for rank, (mid, score) in enumerate(recs, start=1):
        meta = dataset.get_movie_meta(mid)
        title = meta.get("clean_title", f"Movie #{mid}")

        explanation_text = None
        paths_schema: list[ExplanationPathSchema] = []
        subgraph = None

        if explain:
            raw_paths = reasoner.find_explanation_paths(user_id=user_id, recommended_movie_id=mid, max_paths=3)
            paths_schema = [
                ExplanationPathSchema(
                    anchor_movie_id=p.anchor_movie_id,
                    anchor_movie_title=p.anchor_movie_title,
                    anchor_rating=p.anchor_rating,
                    relation_type=p.relation_type,
                    connecting_entity_type=p.connecting_entity_type,
                    connecting_entity_value=p.connecting_entity_value,
                    path_score=p.path_score,
                )
                for p in raw_paths
            ]
            explanation_text = ExplanationFormatter.format_narrative(title, raw_paths)
            subgraph = ExplanationFormatter.build_visual_subgraph(
                user_id=user_id,
                recommended_movie_id=mid,
                recommended_title=title,
                paths=raw_paths,
            )

        items.append(
            MovieRecommendation(
                rank=rank,
                movie_id=mid,
                title=title,
                year=meta.get("year"),
                genres=meta.get("genres", []),
                directors=meta.get("directors", []),
                actors=meta.get("actors", []),
                affinity_score=round(float(score), 4),
                explanation=explanation_text,
                paths=paths_schema,
                subgraph=subgraph,
            )
        )

    return RecommendationResponse(
        user_id=user_id,
        model_name=selected_model.name,
        latency_ms=round(latency_ms, 3),
        count=len(items),
        recommendations=items,
    )


@app.get("/movies/search", response_model=list[MovieSearchItem])
def search_movies(
    q: str = Query(..., min_length=2, description="Search query string"),
    limit: int = Query(15, ge=1, le=50),
) -> list[MovieSearchItem]:
    """Search movies in the catalog by title substring."""
    dataset: MovieLensDataset = state.get("dataset")
    if dataset is None:
        raise HTTPException(status_code=503, detail="Dataset not ready")

    matches: list[MovieSearchItem] = []
    q_lower = q.lower()
    for _, row in dataset.movies_enriched.iterrows():
        clean_title = str(row["clean_title"])
        if q_lower in clean_title.lower():
            matches.append(
                MovieSearchItem(
                    movie_id=int(row["movie_id"]),
                    title=clean_title,
                    genres=row["genres"],
                    year=row["year"],
                )
            )
            if len(matches) >= limit:
                break
    return matches
