"""Per-row cross-encoder relevance scoring as DuckDB scalar functions.

``rerank_score(query, document)`` maps a (query, document) pair to one DOUBLE
relevance score, so it drops straight into an ``ORDER BY`` over a candidate set:

    SELECT id, chunk
    FROM candidates                          -- the top-K recall handed you
    ORDER BY rerank.rerank_score('reset password', chunk) DESC
    LIMIT 10;

The score is model-backed; the cross-encoder is loaded once per worker process and
cached (see :mod:`vgi_rerank.models`). The query is usually a constant that DuckDB
broadcasts across the document column.

A note on argument syntax
-------------------------
VGI / DuckDB *scalar* functions take **positional** arguments and resolve overloads
by *arity* -- the ``name := value`` named-argument syntax is a property of table
functions and macros, not scalars. So ``rerank_score`` exposes its optional
``model`` argument as a third arity overload sharing the one name:

    SELECT rerank.rerank_score(:q, chunk)                          FROM c;  -- default model
    SELECT rerank.rerank_score(:q, chunk, 'BAAI/bge-reranker-base') FROM c;  -- pick a model

Returns
-------
``rerank_score`` returns ``DOUBLE`` (a raw relevance logit; higher = more relevant,
meaningful only *relative to other documents for the same query*). It declares an
explicit ``Returns(arrow_type=pa.float64())``.

NULL semantics: a NULL or empty/whitespace-only query OR document yields a NULL
score. Nothing here crashes on odd input.
"""

from __future__ import annotations

from typing import Annotated

import pyarrow as pa
from vgi import Param, Returns, ScalarFunction
from vgi.arguments import ConstParam
from vgi.metadata import FunctionExample

from . import models


def _ex(sql: str, description: str) -> list[FunctionExample]:
    return [FunctionExample(sql=sql, description=description)]


def _score_array(
    query: pa.StringArray,
    document: pa.StringArray,
    *,
    model: str | None,
) -> pa.DoubleArray:
    """Score a parallel (query, document) string-array pair to a DOUBLE array.

    NULL/empty on either side -> NULL score. The live pairs are scored by
    :func:`models.score_pairs`, which batches per distinct query.
    """
    queries = query.to_pylist()
    documents = document.to_pylist()
    scores = models.score_pairs(queries, documents, model=model)
    return pa.array(scores, type=pa.float64())


# ===========================================================================
# rerank_score(query, document) / (query, document, model)
# ===========================================================================


class RerankScore(ScalarFunction):
    """``rerank_score(query, document)`` -- cross-encoder relevance score (default model)."""

    class Meta:
        name = "rerank_score"
        description = (
            "Cross-encoder relevance score (DOUBLE) for a (query, document) pair using the "
            f"default model ({models.DEFAULT_MODEL}). Higher = more relevant; the value is a "
            "raw logit, meaningful only RELATIVE to other documents for the SAME query. Intended "
            "for ORDER BY rerank_score(:q, doc) DESC LIMIT k over a top-K candidate set. "
            "NULL/empty query or document -> NULL."
        )
        categories = ["rerank", "retrieval"]
        examples = _ex(
            "SELECT rerank.rerank_score('how do I reset my password', "
            "'Click the forgot password link to reset it.')",
            "Score one query/document pair with the default model",
        )

    @classmethod
    def compute(
        cls,
        query: Annotated[pa.StringArray, Param(doc="Search query (usually a constant broadcast across the rows)")],
        document: Annotated[pa.StringArray, Param(doc="Candidate document/passage to score against the query")],
    ) -> Annotated[pa.DoubleArray, Returns(arrow_type=pa.float64())]:
        return _score_array(query, document, model=None)


class RerankScoreModel(ScalarFunction):
    """``rerank_score(query, document, model)`` -- relevance score with an explicit model."""

    class Meta:
        name = "rerank_score"
        description = (
            "Cross-encoder relevance score (DOUBLE) for a (query, document) pair with an explicit "
            "model (see supported_models()). Higher = more relevant. NULL/empty -> NULL."
        )
        categories = ["rerank", "retrieval"]
        examples = _ex(
            "SELECT rerank.rerank_score('reset password', chunk, 'BAAI/bge-reranker-base') FROM candidates",
            "Score with a chosen reranker model",
        )

    @classmethod
    def compute(
        cls,
        query: Annotated[pa.StringArray, Param(doc="Search query")],
        document: Annotated[pa.StringArray, Param(doc="Candidate document/passage")],
        model: Annotated[str, ConstParam(doc="Model name; see supported_models()")],
    ) -> Annotated[pa.DoubleArray, Returns(arrow_type=pa.float64())]:
        return _score_array(query, document, model=model or None)


# ===========================================================================
# rerank_version() -- metadata helper
# ===========================================================================


class RerankVersion(ScalarFunction):
    """``rerank_version()`` -- worker + backend + default-model identity string."""

    class Meta:
        name = "rerank_version"
        description = "Version string: worker version, fastembed backend, and default cross-encoder model"
        categories = ["metadata"]
        examples = _ex(
            "SELECT rerank.rerank_version()",
            "Identify the rerank worker / default model",
        )

    @classmethod
    def compute(
        cls,
    ) -> Annotated[pa.StringArray, Returns(arrow_type=pa.string())]:
        return pa.array([_version_string()], type=pa.string())


def _version_string() -> str:
    """Build the rerank_version() string (worker + fastembed backend + default model)."""
    from . import __version__

    try:
        from importlib.metadata import version as _pkg_version

        backend = f"fastembed {_pkg_version('fastembed')}"
    except Exception:  # noqa: BLE001 - version lookup is best-effort
        backend = "fastembed"
    return f"vgi-rerank {__version__} ({backend}; default {models.DEFAULT_MODEL})"


SCALAR_FUNCTIONS: list[type] = [
    RerankScore,
    RerankScoreModel,
    RerankVersion,
]
