# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python",
#     "fastembed>=0.5",
# ]
#
# [tool.uv.sources]
# vgi-python = { path = "../vgi-python" }
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
    schemas=[
        Schema(
            name="main",
            comment="Local cross-encoder reranking (fastembed/ONNX) for second-stage RAG precision",
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
