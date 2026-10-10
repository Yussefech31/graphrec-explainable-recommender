"""End-to-end quantitative benchmark across baseline, neural, and graph recommendation models."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from src.graphrec.data.graph_builder import KnowledgeGraphBuilder
from src.graphrec.data.loader import load_movielens_1m
from src.graphrec.evaluation.metrics import BenchmarkEvaluator
from src.graphrec.explainability.explanation_formatter import ExplanationFormatter
from src.graphrec.explainability.path_reasoner import KnowledgeGraphPathReasoner
from src.graphrec.models.als import MatrixFactorizationRecommender
from src.graphrec.models.item_cf import ItemItemCFRecommender
from src.graphrec.models.lightgcn import LightGCNRecommender
from src.graphrec.models.popularity import PopularityRecommender
from src.graphrec.models.two_tower import TwoTowerRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_benchmark_plots(results: list[dict], output_path: Path | str) -> None:
    """Generate 4-panel publication-grade comparison plot."""
    hit_rates = [r["hit_rate"] for r in results]
    ndcgs = [r["ndcg"] for r in results]
    coverages = [r["catalog_coverage_pct"] for r in results]
    latencies = [r["mean_latency_ms"] for r in results]

    # Clean short labels for plotting
    short_labels = [
        "Popularity",
        "Item-Item CF",
        "Matrix Fact.",
        "Two-Tower\n(FAISS)",
        "LightGCN\n(Graph CF)",
    ]

    colors = ["#94A3B8", "#38BDF8", "#818CF8", "#34D399", "#F472B6"]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0F172A")

    for ax in axs.flat:
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="#64748B")

    # 1. Hit Rate@10
    bars1 = axs[0, 0].bar(short_labels, hit_rates, color=colors, width=0.55, edgecolor="white", linewidth=0.5)
    axs[0, 0].set_title("Hit Rate @ 10 (Recall@10)", color="white", fontsize=13, fontweight="bold", pad=12)
    axs[0, 0].set_ylabel("Hit Rate", color="white", fontsize=11)
    for bar in bars1:
        yval = bar.get_height()
        axs[0, 0].text(bar.get_x() + bar.get_width() / 2, yval + 0.002, f"{yval:.4f}", ha="center", va="bottom", color="white", fontsize=9)

    # 2. NDCG@10
    bars2 = axs[0, 1].bar(short_labels, ndcgs, color=colors, width=0.55, edgecolor="white", linewidth=0.5)
    axs[0, 1].set_title("NDCG @ 10 (Ranking Quality)", color="white", fontsize=13, fontweight="bold", pad=12)
    axs[0, 1].set_ylabel("NDCG", color="white", fontsize=11)
    for bar in bars2:
        yval = bar.get_height()
        axs[0, 1].text(bar.get_x() + bar.get_width() / 2, yval + 0.002, f"{yval:.4f}", ha="center", va="bottom", color="white", fontsize=9)

    # 3. Catalog Coverage %
    bars3 = axs[1, 0].bar(short_labels, coverages, color=colors, width=0.55, edgecolor="white", linewidth=0.5)
    axs[1, 0].set_title("Catalog Coverage % (Diversity)", color="white", fontsize=13, fontweight="bold", pad=12)
    axs[1, 0].set_ylabel("Coverage %", color="white", fontsize=11)
    for bar in bars3:
        yval = bar.get_height()
        axs[1, 0].text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", color="white", fontsize=9)

    # 4. Inference Latency (ms)
    bars4 = axs[1, 1].bar(short_labels, latencies, color=colors, width=0.55, edgecolor="white", linewidth=0.5)
    axs[1, 1].set_title("Mean Inference Latency (ms/query)", color="white", fontsize=13, fontweight="bold", pad=12)
    axs[1, 1].set_ylabel("Latency (ms)", color="white", fontsize=11)
    for bar in bars4:
        yval = bar.get_height()
        axs[1, 1].text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2f} ms", ha="center", va="bottom", color="white", fontsize=9)

    plt.suptitle("GraphRec Quantitative Recommender Benchmark (MovieLens 1M)", color="white", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    logger.info("Saved benchmark comparison figure to %s", output_path)


def main():
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("1. Loading MovieLens 1M dataset...")
    dataset = load_movielens_1m("data")
    logger.info("Dataset loaded: %d users, %d movies, %d ratings", dataset.num_users, dataset.num_items, len(dataset.ratings))

    logger.info("2. Performing Leave-Last-Out temporal split...")
    split = dataset.leave_last_out_split()
    logger.info("Split sizes - Train: %d, Val: %d, Test: %d", len(split.train), len(split.val), len(split.test))

    # Build Knowledge Graph
    logger.info("3. Building Heterogeneous Knowledge Graph...")
    graph_builder = KnowledgeGraphBuilder(dataset)
    graph_builder.build_graph(split.train)
    path_reasoner = KnowledgeGraphPathReasoner(graph_builder.graph)

    # Item popularity counts for Novelty metric
    item_pop = split.train["movie_id"].value_counts().to_dict()
    all_item_ids = dataset.unique_items

    evaluator = BenchmarkEvaluator(test_df=split.test, all_item_ids=all_item_ids, item_popularity=item_pop)

    # Models definition
    models = [
        PopularityRecommender(),
        ItemItemCFRecommender(top_similar_neighbors=50),
        MatrixFactorizationRecommender(n_factors=64, n_iter=15),
        TwoTowerRecommender(embedding_dim=64, epochs=6, batch_size=512, lr=0.003),
        LightGCNRecommender(embedding_dim=64, num_layers=2, epochs=5, batch_size=512, lr=0.005),
    ]

    benchmark_records = []
    # Evaluate across 500 test users for statistically solid and fast benchmark
    eval_user_count = min(500, len(split.test))

    for model in models:
        logger.info("Training %s...", model.name)
        model.fit(split.train, dataset.movies_raw)

        logger.info("Evaluating %s across %d test users...", model.name, eval_user_count)
        eval_res = evaluator.evaluate_model(model, k=10, max_eval_users=eval_user_count)
        res_dict = eval_res.to_dict()
        benchmark_records.append(res_dict)
        logger.info(
            "--> %s | HitRate@10: %.4f | NDCG@10: %.4f | Coverage: %.1f%% | Latency: %.2f ms",
            model.name,
            res_dict["hit_rate"],
            res_dict["ndcg"],
            res_dict["catalog_coverage_pct"],
            res_dict["mean_latency_ms"],
        )

    # Save benchmark JSON
    json_path = results_dir / "model_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "MovieLens 1M (1,000,209 ratings, 6,040 users, 3,900 movies)",
                "evaluation_protocol": "Leave-Last-Out temporal split",
                "test_users_evaluated": eval_user_count,
                "top_k": 10,
                "benchmarks": benchmark_records,
            },
            f,
            indent=2,
        )
    logger.info("Saved benchmark results to %s", json_path)

    # Generate Plot
    generate_benchmark_plots(benchmark_records, results_dir / "benchmark_comparison.png")

    # Generate sample explanation justifications for demo and documentation
    sample_explanations = []
    two_tower_model = models[3]
    for sample_uid in [1, 2, 8, 15, 48]:
        recs = two_tower_model.recommend(user_id=sample_uid, k=5, exclude_seen=True)
        user_recs = []
        for rank, (mid, score) in enumerate(recs, start=1):
            meta = dataset.get_movie_meta(mid)
            paths = path_reasoner.find_explanation_paths(user_id=sample_uid, recommended_movie_id=mid, max_paths=3)
            narrative = ExplanationFormatter.format_narrative(meta["clean_title"], paths)
            user_recs.append({
                "rank": rank,
                "movie_id": int(mid),
                "title": str(meta["clean_title"]),
                "year": int(meta["year"]) if meta.get("year") else None,
                "genres": list(meta["genres"]),
                "affinity_score": round(float(score), 4),
                "narrative_explanation": narrative,
                "paths": [p.to_dict() for p in paths],
            })
        sample_explanations.append({
            "user_id": int(sample_uid),
            "recommendations": user_recs,
        })

    with open(results_dir / "sample_explanations.json", "w", encoding="utf-8") as f:
        json.dump(sample_explanations, f, indent=2)
    logger.info("Saved sample explanations to %s", results_dir / "sample_explanations.json")

    # Export Neo4j Cypher script
    cypher_path = results_dir / "neo4j_graphrec_import.cypher"
    graph_builder.export_cypher_script(cypher_path, max_users=100)
    logger.info("Exported Neo4j Cypher script to %s", cypher_path)


if __name__ == "__main__":
    main()
