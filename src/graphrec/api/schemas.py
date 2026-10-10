"""Pydantic schemas for the GraphRec REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    available_models: list[str]
    num_users: int
    num_movies: int
    graph_nodes: int
    graph_edges: int


class ExplanationPathSchema(BaseModel):
    anchor_movie_id: int
    anchor_movie_title: str
    anchor_rating: float
    relation_type: str
    connecting_entity_type: str
    connecting_entity_value: str
    path_score: float


class MovieRecommendation(BaseModel):
    rank: int
    movie_id: int
    title: str
    year: int | None = None
    genres: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    affinity_score: float
    explanation: str | None = None
    paths: list[ExplanationPathSchema] = Field(default_factory=list)
    subgraph: dict[str, Any] | None = None


class RecommendationResponse(BaseModel):
    user_id: int
    model_name: str
    latency_ms: float
    count: int
    recommendations: list[MovieRecommendation]


class MovieSearchItem(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    year: int | None
