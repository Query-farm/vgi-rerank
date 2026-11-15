"""Display only SQL-derived candidate scores and rank changes."""

import matplotlib.pyplot as plt
from vgi_notebooks import finish

SOURCE = (
    "Source: eight authored passages and hand-assigned recall order · no measured search benchmark\n"
    "VGI Rerank v0.4.0 + pinned MiniLM model revision a091443 · raw logits, not probabilities"
)


def scores(ranking, CANDIDATE_LIMIT):
    """Plot finite SQL-returned logits and keep zero as a reference only."""
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = ["#397d62" if x == "direct answer" else "#397f91" for x in ranking.fixture_role]
    ax.barh(range(len(ranking)), ranking.score, color=colors)
    ax.set_yticks(range(len(ranking)), [f"{r.id} · {r.title}  [recall {r.recall_rank}]" for r in ranking.itertuples()])
    ax.invert_yaxis()
    span = max(float(ranking.score.max() - ranking.score.min()), 1)
    ax.set_xlim(min(0, float(ranking.score.min())) - span * 0.18, max(0, float(ranking.score.max())) + span * 0.18)
    for i, r in enumerate(ranking.itertuples()):
        ax.text(
            r.score + (span * 0.02 if r.score >= 0 else -span * 0.02),
            i,
            f"{r.score:.2f}",
            va="center",
            ha="left" if r.score >= 0 else "right",
            fontsize=10,
        )
    ax.axvline(0, color="#aab6b0", linewidth=1)
    ax.set_xlabel("Raw relevance logit · higher ranks earlier for this query; not a probability")
    ax.grid(axis="y", visible=False)
    finish(
        fig,
        "Scores order only passages admitted for this query",
        f"Authored recall prefix K={CANDIDATE_LIMIT} · {len(ranking)} scored · NULL and excluded rows omitted",
        SOURCE,
        name="01-rerank-scores",
    )


def movement(ranking, ledger, CANDIDATE_LIMIT):
    """Connect original and reranked positions while marking unscored inputs."""
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    for r in ranking.itertuples():
        color = "#397d62" if r.fixture_role == "direct answer" else "#397f91"
        ax.plot([0, 1], [r.recall_rank, r.rerank_position], "o-", color=color, alpha=0.8, linewidth=2)
        ax.text(-0.04, r.recall_rank, f"{r.id} · {r.title}", ha="right", va="center", fontsize=10)
        ax.text(1.04, r.rerank_position, f"{r.id} · rank {r.rerank_position}", ha="left", va="center", fontsize=10)
    missing = ledger.loc[(ledger.state == "admitted; no usable text")]
    for r in missing.itertuples():
        ax.scatter([0], [r.recall_rank], facecolors="none", edgecolors="#8c9b9d", s=60)
        ax.text(-0.04, r.recall_rank, f"{r.id} · no usable text", ha="right", va="center", fontsize=10, color="#6d7777")
    ax.set_xlim(-0.55, 1.35)
    ax.set_ylim(CANDIDATE_LIMIT + 0.7, 0.3)
    ax.set_xticks([0, 1], ["Original recall position", "Reranked position"])
    ax.set_yticks(range(1, CANDIDATE_LIMIT + 1))
    ax.set_ylabel("Rank · 1 is earliest")
    ax.grid(False)
    finish(
        fig,
        "A passage can move up only after it enters the set",
        f"Authored query · K={CANDIDATE_LIMIT} · green: planted direct answer; open marker: unscored",
        SOURCE,
        name="02-rank-movement",
    )
