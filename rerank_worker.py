# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.8.5",
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

import json
from typing import Any

from vgi import Worker
from vgi.catalog import Catalog, Schema, Table

from vgi_rerank import models
from vgi_rerank.scalars import SCALAR_FUNCTIONS
from vgi_rerank.tables import TABLE_FUNCTIONS, SupportedModelsFunction

_RERANK_CATALOG = Catalog(
    name="rerank",
    default_schema="main",
    comment="Local cross-encoder reranking (fastembed/ONNX) for second-stage RAG precision.",
    source_url="https://github.com/Query-farm/vgi-rerank",
    tags={
        "vgi.title": "Local Cross-Encoder Reranking",
        "vgi.keywords": json.dumps(
            [
                "rerank",
                "reranker",
                "cross-encoder",
                "relevance",
                "ranking",
                "retrieval",
                "RAG",
                "second-stage",
                "top-k",
                "precision",
                "fastembed",
                "ONNX",
                "MS MARCO",
                "MiniLM",
                "bge-reranker",
                "semantic search",
                "order by",
                "vss",
                "bm25",
            ]
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
            "# Cross-Encoder Reranking in SQL\n\n"
            "**Rerank retrieval results directly in DuckDB with a local cross-encoder reranker "
            "(fastembed / ONNX Runtime, no torch) -- the precision second stage that turns a "
            "good recall set into a great answer for RAG and semantic search.**\n\n"
            "Retrieval-augmented generation and semantic search almost always run in two stages. "
            "A fast *recall* stage (DuckDB [VSS](https://duckdb.org) approximate nearest-neighbor "
            "search over embeddings, or BM25 full-text search) cheaply pulls a top-K candidate set "
            "out of a large corpus, but it ranks each document in isolation, so the truly best "
            "passage is often buried at rank 7 instead of rank 1. This worker adds the *precision* "
            "second stage: a **cross-encoder** that reads the query and a candidate document "
            "*together* in a single transformer forward pass and emits a relevance score that is "
            "far more accurate than bi-encoder cosine similarity. It is for anyone building local "
            "RAG pipelines, document search, or recommendation ranking who wants higher answer "
            "quality without standing up a separate model-serving service -- the reranking happens "
            "inside your SQL query.\n\n"
            "Scores are computed entirely locally by [fastembed](https://github.com/qdrant/fastembed) "
            "(from Qdrant, Apache-2.0), which runs quantized cross-encoder models on "
            "[ONNX Runtime](https://onnxruntime.ai/) -- there is **no PyTorch dependency** and no "
            "network call at query time. The default model, "
            "[`Xenova/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2) "
            "(~80 MB, Apache-2.0), is fetched once on first use and cached, and the ONNX session is "
            "built a single time per worker process and amortized across every row. Because a "
            "cross-encoder scores each (query, document) pair at query time and cannot be "
            "precomputed or cached the way an embedding can, you should only ever run it over a "
            "recall-produced top-K candidate set -- never over a whole corpus.\n\n"
            "The catalog exposes a small, focused surface. `rerank_score(query, document)` returns "
            "a `DOUBLE` relevance logit (higher = more relevant, meaningful only relative to other "
            "documents for the *same* query), so it drops straight into "
            "`ORDER BY rerank.rerank_score(:q, chunk) DESC LIMIT k`; a three-argument overload "
            "`rerank_score(query, document, model)` lets you pick a specific reranker such as "
            "`BAAI/bge-reranker-base`. The `supported_models()` table function (also available as "
            "the `supported_models` table) lists every reranker the worker can run together with "
            "the license of its weights, and `rerank_version()` reports the worker, backend, and "
            "default-model identity. A NULL or empty query or document always yields a NULL score, "
            "so odd input never crashes a query.\n\n"
            "## Example\n\n"
            "```sql\n"
            "INSTALL vgi FROM community; LOAD vgi;\n"
            "ATTACH 'rerank' (TYPE vgi, LOCATION 'uv run rerank_worker.py');\n\n"
            "-- Reorder a top-K candidate set (e.g. from VSS / BM25) by true relevance.\n"
            "SELECT id, chunk\n"
            "FROM candidates\n"
            "ORDER BY rerank.rerank_score('how do I reset my password', chunk) DESC\n"
            "LIMIT 10;\n"
            "```\n\n"
            "## Learn more\n\n"
            "- fastembed source: <https://github.com/qdrant/fastembed>\n"
            "- fastembed documentation: <https://qdrant.github.io/fastembed/>\n"
            "- ONNX Runtime: <https://onnxruntime.ai/>"
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
                "vgi.keywords": json.dumps(
                    [
                        "rerank",
                        "rerank_score",
                        "supported_models",
                        "rerank_version",
                        "cross-encoder",
                        "relevance",
                        "reranker",
                        "retrieval",
                        "RAG",
                        "top-k",
                        "order by",
                        "semantic search",
                    ]
                ),
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
            # supported_models() takes no arguments and always returns the same
            # rows, so also expose it as a regular table (backed by the function)
            # -- consumers can then SELECT * FROM rerank.main.supported_models
            # without parentheses (VGI311). inline_bind is safe because the
            # function is @bind_fixed_schema (output is exactly its FIXED_SCHEMA).
            tables=[
                Table(
                    name="supported_models",
                    function=SupportedModelsFunction,
                    inline_bind=True,
                    comment="Every cross-encoder reranker model the worker can run, with the license of its weights.",
                    primary_key=(("model",),),
                    not_null=("model", "license"),
                    tags={
                        "vgi.title": "Supported Reranker Models",
                        "vgi.keywords": json.dumps(
                            [
                                "supported_models",
                                "models",
                                "reranker",
                                "cross-encoder",
                                "license",
                                "discovery",
                                "available models",
                                "catalog",
                                "ms-marco",
                                "bge-reranker",
                                "list models",
                            ]
                        ),
                        "domain": "information-retrieval",
                        "category": "reranking",
                        "topic": "model-discovery",
                        "vgi.doc_llm": (
                            "Discovery table listing every cross-encoder reranker model this worker can "
                            "run, one row per model, with the SPDX-style license of its weights. Query it "
                            "to learn which model names are valid as the optional third argument to "
                            "rerank_score(query, document, model) and to confirm licensing before adopting "
                            "a model. The `model` column is the name to pass to rerank_score; the `license` "
                            "column is the weights' license (every advertised model is permissive / "
                            "commercially usable). Identical rows to the supported_models() table function "
                            "-- prefer SELECT * FROM rerank.main.supported_models (no parentheses)."
                        ),
                        "vgi.doc_md": (
                            "# supported_models\n\n"
                            "A discovery table of every cross-encoder reranker the worker can run -- one "
                            "row per model. Backed by the `supported_models()` table function; query it as "
                            "`SELECT * FROM rerank.main.supported_models`.\n\n"
                            "## Columns\n\n"
                            "| column | type | description |\n"
                            "|---|---|---|\n"
                            "| `model` | VARCHAR | Name to pass as the optional third argument to "
                            "`rerank_score(query, document, model)` (primary key). |\n"
                            "| `license` | VARCHAR | SPDX-style license of the model weights. |\n\n"
                            "## Notes\n\n"
                            "Every advertised model is permissive / commercially usable. Reading this "
                            "table loads no model, so it is always cheap."
                        ),
                        "vgi.example_queries": json.dumps(
                            [
                                {
                                    "description": "List every supported reranker model and its license.",
                                    "sql": "SELECT * FROM rerank.main.supported_models ORDER BY model",
                                },
                                {
                                    "description": "Look up the license of the default model.",
                                    "sql": (
                                        "SELECT license FROM rerank.main.supported_models "
                                        "WHERE model = 'Xenova/ms-marco-MiniLM-L-6-v2'"
                                    ),
                                },
                            ]
                        ),
                    },
                ),
            ],
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
