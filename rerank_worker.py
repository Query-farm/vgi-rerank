# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.8.4",
#     "fastembed>=0.5",
# ]
# ///
"""VGI worker exposing a local cross-encoder reranker (fastembed/ONNX) to DuckDB/SQL.

Assembles the scalar and table functions in ``vgi_rerank`` into a single ``rerank``
catalog and runs the worker over stdio (DuckDB subprocess) or HTTP (via serve.py).

This is the **precision second stage** of a local RAG stack. Recall (DuckDB VSS ANN
over vgi-embed vectors, and/or BM25 via vgi-tantivy) hands you a top-K candidate
set; ``rerank_score(query, document)`` reorders it with a cross-encoder that reads
the query and document *together*. Because a cross-encoder scores every pair at
query time (no precompute, no cache -- unlike embeddings), only ever run it over a
top-K candidate set, never the whole corpus.

The scores are produced locally with `fastembed` (Qdrant, Apache-2.0), which runs
the model through ONNX Runtime -- **no torch**. The default model
``Xenova/ms-marco-MiniLM-L-6-v2`` (~80 MB, Apache-2.0) is downloaded on first use
and cached (gitignored); see ``vgi_rerank/models.py``.

Usage:
    uv run rerank_worker.py              # serve over stdio (DuckDB subprocess)
    python serve.py --port 8000          # serve over HTTP

    INSTALL vgi FROM community; LOAD vgi;
    ATTACH 'rerank' (TYPE vgi, LOCATION 'uv run rerank_worker.py');

    -- Rerank a top-K candidate set (e.g. from VSS / BM25) by relevance.
    SELECT id, chunk
    FROM candidates
    ORDER BY rerank.rerank_score('how do I reset my password', chunk) DESC
    LIMIT 10;

    SELECT * FROM rerank.supported_models();
"""

from __future__ import annotations

from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, Schema

from vgi_rerank import models
from vgi_rerank.scalars import SCALAR_FUNCTIONS
from vgi_rerank.tables import TABLE_FUNCTIONS

_RERANK_CATALOG = Catalog(
    name="rerank",
    default_schema="main",
    comment="Local cross-encoder reranking (fastembed/ONNX) for second-stage RAG precision.",
    source_url="https://github.com/Query-farm/vgi-rerank",
    tags={
        "vgi.title": "Local Cross-Encoder Reranking",
        "vgi.keywords": (
            "rerank, reranker, cross-encoder, relevance, ranking, retrieval, RAG, "
            "second-stage, top-k, precision, fastembed, ONNX, MS MARCO, MiniLM, "
            "bge-reranker, semantic search, order by, vss, bm25"
        ),
        "vgi.doc_llm": (
            "Score the relevance of a (query, document) pair with a local cross-encoder "
            "reranker (fastembed/ONNX, no torch) and reorder a recall-produced top-K "
            "candidate set. rerank_score(query, document[, model]) returns a DOUBLE "
            "relevance logit (higher = more relevant, meaningful only relative to other "
            "documents for the SAME query); drop it into ORDER BY rerank_score(:q, doc) "
            "DESC LIMIT k. supported_models() lists the available reranker models and "
            "rerank_version() reports worker/backend/default-model identity. This is the "
            "precision SECOND stage of a local RAG stack: run it only over the top-K "
            "candidates that recall (DuckDB VSS / BM25) already produced, never over a "
            "whole corpus."
        ),
        "vgi.doc_md": (
            "# rerank\n\n"
            "Local cross-encoder reranking (fastembed/ONNX -- no torch) as DuckDB SQL "
            "functions: the precision **second stage** of a local RAG stack.\n\n"
            "## Why a cross-encoder\n\n"
            "A cross-encoder reads the query and document *together*, so its score is more "
            "accurate than bi-encoder cosine similarity -- but it cannot be precomputed or "
            "cached, so run it only over a recall-produced top-K candidate set, never over a "
            "whole corpus.\n\n"
            "## Surface\n\n"
            "- Scalars: `rerank_score(query, document)`, "
            "`rerank_score(query, document, model)`, `rerank_version()`.\n"
            "- Table: `supported_models()`.\n\n"
            "## Usage\n\n"
            "```sql\n"
            "SELECT id, chunk\n"
            "FROM candidates\n"
            "ORDER BY rerank.rerank_score('how do I reset my password', chunk) DESC\n"
            "LIMIT 10;\n"
            "```"
        ),
        "vgi.author": "Query.Farm",
        "vgi.copyright": "Copyright 2026 Query Farm LLC - https://query.farm",
        "vgi.license": "MIT",
        "vgi.support_contact": "https://github.com/Query-farm/vgi-rerank/issues",
        "vgi.support_policy_url": "https://github.com/Query-farm/vgi-rerank/blob/main/README.md",
    },
    schemas=[
        Schema(
            name="main",
            comment="Local cross-encoder reranking (fastembed/ONNX) for second-stage RAG precision",
            tags={
                "vgi.title": "Rerank - main schema",
                "vgi.keywords": (
                    "rerank, rerank_score, supported_models, rerank_version, "
                    "cross-encoder, relevance, reranker, retrieval, RAG, top-k, "
                    "order by, semantic search"
                ),
                "vgi.source_url": ("https://github.com/Query-farm/vgi-rerank/blob/main/rerank_worker.py"),
                # VGI123 classifying tags use BARE keys (not vgi.-namespaced).
                "domain": "information-retrieval",
                "category": "reranking",
                "topic": "cross-encoder-relevance",
                "vgi.doc_llm": (
                    "Cross-encoder relevance scoring and reranker discovery: rerank_score "
                    "scores a (query, document) pair to a DOUBLE relevance logit for "
                    "ORDER BY ... DESC LIMIT k reranking of a top-K candidate set, "
                    "supported_models lists the available reranker models with their "
                    "licenses, and rerank_version reports the worker/backend/default-model "
                    "identity."
                ),
                "vgi.doc_md": (
                    "## main\n\n"
                    "The single schema of the `rerank` catalog. It groups the per-row "
                    "relevance scorer `rerank_score` (a scalar with an optional explicit-"
                    "model arity overload), the `supported_models()` discovery table, and "
                    "the `rerank_version()` identity helper -- everything needed to add "
                    "second-stage cross-encoder reranking to a local RAG pipeline."
                ),
                # VGI506 representative example queries for the schema.
                "vgi.example_queries": (
                    "SELECT rerank.main.rerank_score('how do I reset my password', "
                    "'Click the forgot password link to reset it.');\n"
                    "SELECT * FROM rerank.main.supported_models() ORDER BY model;\n"
                    "SELECT rerank.main.rerank_version();"
                ),
            },
            functions=[*SCALAR_FUNCTIONS, *TABLE_FUNCTIONS],
        ),
    ],
)


class RerankWorker(Worker):
    """Worker process hosting the ``rerank`` catalog."""

    catalog = _RERANK_CATALOG

    def run(self, otel_config: Any = None) -> None:
        """Warm the default model, then serve.

        Loading (and, on a cold cache, *downloading*) the ONNX cross-encoder is
        lazy, so without this the first query of every ATTACH pays that multi-second
        cost inline -- a window in which a worker-pool teardown SIGTERM (or a heavily
        loaded host) can kill the run mid-assertion and record a spurious E2E
        failure. Warming at spawn moves that one-time cost ahead of any query,
        keeping the SQL suite deterministic without changing a single output value.
        Best-effort; never fatal.
        """
        models.warm_up()
        super().run(otel_config=otel_config)


def main() -> None:
    """Run the rerank worker process (stdio or, via flags, HTTP)."""
    RerankWorker.main()


if __name__ == "__main__":
    main()
