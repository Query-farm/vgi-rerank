"""vgi-rerank: local cross-encoder reranking (fastembed/ONNX) as DuckDB SQL functions.

Exposes ``rerank_score(query, document)`` -- a cross-encoder relevance score for a
query/document pair -- plus the ``rerank_version()`` scalar and the
``supported_models()`` discovery table function. It is the precision second stage
of a local RAG stack: recall (DuckDB VSS ANN over vgi-embed vectors, and/or BM25
via vgi-tantivy) hands you a top-K candidate set, and ``rerank_score`` reorders it
with a cross-encoder that reads the query and document *together*.
"""

from __future__ import annotations

__version__ = "0.1.0"
