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

Returns:
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

_SOURCE_URL = "https://github.com/Query-farm/vgi-rerank/blob/main/vgi_rerank/scalars.py"


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
        """Function metadata."""

        name = "rerank_score"
        description = (
            "Cross-encoder relevance score (DOUBLE) for a (query, document) pair using the "
            f"default model ({models.DEFAULT_MODEL}). Higher = more relevant; the value is a "
            "raw logit, meaningful only RELATIVE to other documents for the SAME query. Intended "
            "for ORDER BY rerank_score(:q, doc) DESC LIMIT k over a top-K candidate set. "
            "NULL/empty query or document -> NULL."
        )
        categories = ["rerank", "retrieval"]
        tags = {
            "vgi.title": "Cross-Encoder Rerank Score",
            "vgi.keywords": (
                "rerank, rerank_score, relevance, cross-encoder, score, ranking, "
                "retrieval, RAG, order by, top-k, logit, semantic search"
            ),
            "vgi.source_url": _SOURCE_URL,
            "vgi.doc_llm": (
                "Per-row cross-encoder relevance scorer. `rerank_score(query, document)` "
                "feeds the query and document through one transformer together (fastembed/"
                f"ONNX, default model {models.DEFAULT_MODEL}) and returns a DOUBLE relevance "
                "logit: higher = more relevant. The value is meaningful only RELATIVE to "
                "other documents scored for the SAME query (it is not normalized and may be "
                "negative), so use it in `ORDER BY rerank_score(:q, doc) DESC LIMIT k` over a "
                "recall-produced top-K candidate set -- never over a whole corpus, since a "
                "cross-encoder cannot be precomputed. A NULL or empty/whitespace-only query "
                "or document yields a NULL score. The query is usually a constant DuckDB "
                "broadcasts across the document column."
            ),
            "vgi.doc_md": (
                "# rerank_score(query, document)\n\n"
                "Per-row cross-encoder relevance score for a `(query, document)` pair using "
                f"the default model (`{models.DEFAULT_MODEL}`).\n\n"
                "## Returns\n\n"
                "`DOUBLE` -- a raw relevance logit. Higher is more relevant; the value is "
                "meaningful only *relative to other documents for the same query* and is not "
                "normalized to any range.\n\n"
                "## Usage\n\n"
                "```sql\n"
                "SELECT id, chunk\n"
                "FROM candidates\n"
                "ORDER BY rerank.rerank_score('reset password', chunk) DESC\n"
                "LIMIT 10;\n"
                "```\n\n"
                "## Notes\n\n"
                "- Run only over a top-K candidate set; cross-encoder scores cannot be "
                "precomputed.\n"
                "- NULL/empty query or document -> NULL score."
            ),
            "vgi.executable_examples": (
                '[{"description": "Score a clearly relevant query/document pair with the '
                'default model.", "sql": "SELECT rerank.main.rerank_score(\'how do I reset '
                "my password', 'Click the forgot password link to reset it.') AS score\"}, "
                '{"description": "Rerank an inline candidate set so the relevant passage '
                'sorts first.", "sql": "SELECT doc FROM (VALUES (\'Reset your password from '
                "the account settings page.'), ('Our office is open from 9am to 5pm.')) AS "
                "t(doc) ORDER BY rerank.main.rerank_score('how do I reset my password', doc) "
                'DESC"}]'
            ),
        }
        examples = _ex(
            "SELECT rerank.rerank_score('how do I reset my password', 'Click the forgot password link to reset it.')",
            "Score one query/document pair with the default model",
        )

    @classmethod
    def compute(
        cls,
        query: Annotated[pa.StringArray, Param(doc="Search query (usually a constant broadcast across the rows)")],
        document: Annotated[pa.StringArray, Param(doc="Candidate document/passage to score against the query")],
    ) -> Annotated[pa.DoubleArray, Returns(arrow_type=pa.float64())]:
        """Score each (query, document) pair with the default model."""
        return _score_array(query, document, model=None)


class RerankScoreModel(ScalarFunction):
    """``rerank_score(query, document, model)`` -- relevance score with an explicit model."""

    class Meta:
        """Function metadata."""

        name = "rerank_score"
        description = (
            "Cross-encoder relevance score (DOUBLE) for a (query, document) pair with an explicit "
            "model (see supported_models()). Higher = more relevant. NULL/empty -> NULL."
        )
        categories = ["rerank", "retrieval"]
        tags = {
            "vgi.title": "Rerank Score With Explicit Model",
            "vgi.keywords": (
                "rerank, rerank_score, model, explicit model, cross-encoder, relevance, "
                "ranking, retrieval, bge-reranker, ms-marco, choose model"
            ),
            "vgi.source_url": _SOURCE_URL,
            "vgi.doc_llm": (
                "Three-argument arity overload of `rerank_score` that scores a "
                "`(query, document)` pair with an EXPLICIT reranker model instead of the "
                "default. `rerank_score(query, document, model)` returns the same DOUBLE "
                "relevance logit as the two-argument form (higher = more relevant, relative "
                "to the same query) but lets you pick any model returned by "
                "`supported_models()` -- e.g. a larger/heavier model for higher quality. "
                "Scalar functions are positional, so the optional model is a separate arity "
                "overload, not a named argument. NULL/empty query or document -> NULL."
            ),
            "vgi.doc_md": (
                "# rerank_score(query, document, model)\n\n"
                "Score a `(query, document)` pair with an **explicit** cross-encoder model "
                "(the three-argument arity overload of `rerank_score`).\n\n"
                "## Arguments\n\n"
                "- `query`, `document` -- the pair to score.\n"
                "- `model` -- a model name from `supported_models()` (e.g. "
                "`Xenova/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-base`). Heavier models "
                "trade speed for quality and download on first use.\n\n"
                "## Returns\n\n"
                "`DOUBLE` -- the same relevance logit as the two-argument form.\n\n"
                "## Notes\n\n"
                "- Scalars are positional; this is an arity overload, not a named arg.\n"
                "- NULL/empty query or document -> NULL score."
            ),
        }
        examples = _ex(
            "SELECT rerank.rerank_score('reset password', "
            "'Reset your password from the account settings page.', "
            "'Xenova/ms-marco-MiniLM-L-6-v2')",
            "Score with an explicitly chosen reranker model",
        )

    @classmethod
    def compute(
        cls,
        query: Annotated[pa.StringArray, Param(doc="Search query")],
        document: Annotated[pa.StringArray, Param(doc="Candidate document/passage")],
        model: Annotated[str, ConstParam(doc="Model name; see supported_models()")],
    ) -> Annotated[pa.DoubleArray, Returns(arrow_type=pa.float64())]:
        """Score each (query, document) pair with an explicit model."""
        return _score_array(query, document, model=model or None)


# ===========================================================================
# rerank_version() -- metadata helper
# ===========================================================================


class RerankVersion(ScalarFunction):
    """``rerank_version()`` -- worker + backend + default-model identity string."""

    class Meta:
        """Function metadata."""

        name = "rerank_version"
        description = "Version string: worker version, fastembed backend, and default cross-encoder model"
        categories = ["metadata"]
        tags = {
            "vgi.title": "Rerank Worker Version String",
            "vgi.keywords": (
                "version, rerank_version, identity, backend, fastembed, default model, "
                "diagnostics, build info, metadata"
            ),
            "vgi.source_url": _SOURCE_URL,
            "vgi.doc_llm": (
                "Return a single human-readable identity string describing this worker: its "
                "vgi-rerank version, the fastembed backend version, and the default cross-"
                "encoder model name -- e.g. "
                "`vgi-rerank 0.1.0 (fastembed 0.5.1; default Xenova/ms-marco-MiniLM-L-6-v2)`. "
                "Takes no arguments and runs without loading any model, so it is the safest "
                "call for health checks, debugging, and confirming which reranker the worker "
                "defaults to before issuing scoring queries."
            ),
            "vgi.doc_md": (
                "# rerank_version()\n\n"
                "A worker/backend/default-model identity string for diagnostics and health "
                "checks.\n\n"
                "## Returns\n\n"
                "`VARCHAR` -- e.g. "
                "`vgi-rerank 0.1.0 (fastembed 0.5.1; default Xenova/ms-marco-MiniLM-L-6-v2)`.\n\n"
                "## Notes\n\n"
                "Takes no arguments and never loads a model, so it is cheap and always "
                "available -- ideal as a liveness probe."
            ),
        }
        examples = _ex(
            "SELECT rerank.rerank_version()",
            "Identify the rerank worker / default model",
        )

    @classmethod
    def compute(
        cls,
    ) -> Annotated[pa.StringArray, Returns(arrow_type=pa.string())]:
        """Return the worker/backend/default-model identity string."""
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
