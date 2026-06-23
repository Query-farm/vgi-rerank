"""Model lifecycle: load a fastembed cross-encoder once, cache it in the worker process.

VGI keeps the worker process alive across queries, so the expensive thing a rerank
worker does -- loading the ONNX cross-encoder (and, on first use ever,
*downloading* it) -- happens once and is amortised over every row of every query.
This module centralises that caching: callers just ask for "the cross-encoder for
model X" and get a ready ``fastembed.rerank.cross_encoder.TextCrossEncoder`` back.

Why a cross-encoder (and why fastembed)
---------------------------------------
A bi-encoder (vgi-embed) maps a query and a document to *independent* vectors, so
their embeddings can be precomputed and indexed (DuckDB VSS / HNSW) -- great for
recall over a whole corpus. A **cross-encoder** instead feeds the query and the
document **together** through one transformer and emits a single relevance score.
That joint attention is more accurate but **cannot be precomputed or cached**:
every (query, document) pair is a fresh forward pass. So a cross-encoder is a
*second-stage reranker* -- you run it only over the top-K candidates that recall
already handed you, never over the whole corpus.

`fastembed` (Qdrant, Apache-2.0) runs these models through ONNX Runtime -- **no
torch** -- so the worker installs light and starts fast once the model is cached.

Default model
-------------
``Xenova/ms-marco-MiniLM-L-6-v2`` -- a small (~80 MB) **Apache-2.0** MS MARCO
cross-encoder, a strong, cheap general-purpose English reranker. Downloaded on
first use to the fastembed cache dir (``~/.cache/...`` by default, or
``VGI_RERANK_CACHE_DIR`` / ``FASTEMBED_CACHE_PATH`` -- see :func:`_cache_dir`).
The cache is gitignored.

Score semantics
---------------
The score is the model's **raw relevance logit**: higher means more relevant. It
is *not* normalised to a fixed range (it can be negative), and values are only
meaningful **relative to each other for the same query** -- which is exactly what
``ORDER BY rerank_score(:q, doc) DESC LIMIT k`` needs. Do not threshold on an
absolute value across different queries or models.

Everything here is lazy: importing this module is cheap; nothing is loaded or
downloaded until the first row needs it (or :func:`warm_up` is called at startup).
A model that cannot be loaded raises a clear, actionable error rather than a deep
library traceback.
"""

from __future__ import annotations

import contextlib
import os
import threading
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastembed.rerank.cross_encoder import TextCrossEncoder


# ---------------------------------------------------------------------------
# Supported models. Keyed by the name users pass to rerank_score(q, d, model).
# Each entry: (license, approximate on-disk size in GB). All are fastembed
# cross-encoders with permissive licenses; the MiniLM default is the smallest.
# (cc-by-nc models that fastembed also ships are intentionally excluded -- this
# worker only advertises models you can use commercially.)
# ---------------------------------------------------------------------------

_SUPPORTED_MODELS: dict[str, tuple[str, float]] = {
    "Xenova/ms-marco-MiniLM-L-6-v2": ("apache-2.0", 0.08),
    "Xenova/ms-marco-MiniLM-L-12-v2": ("apache-2.0", 0.12),
    "BAAI/bge-reranker-base": ("mit", 1.04),
    "jinaai/jina-reranker-v1-tiny-en": ("apache-2.0", 0.13),
    "jinaai/jina-reranker-v1-turbo-en": ("apache-2.0", 0.15),
}

DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

_CACHE_DIR_ENV = "VGI_RERANK_CACHE_DIR"

_lock = threading.Lock()


class ModelNotAvailableError(RuntimeError):
    """A requested reranker model is unknown or could not be loaded/downloaded.

    Carries an actionable hint (the supported model list, or that the first use
    needs network access to download) so the DuckDB-side error tells the user how
    to fix it.
    """


def supported_models() -> list[tuple[str, str]]:
    """Every ``(model, license)`` the worker can produce, sorted by model name."""
    return sorted((name, lic) for name, (lic, _size) in _SUPPORTED_MODELS.items())


def resolve_model(model: str | None) -> str:
    """Normalise a requested model name, defaulting empty/None to the default."""
    name = (model or "").strip() or DEFAULT_MODEL
    if name not in _SUPPORTED_MODELS:
        raise ModelNotAvailableError(
            f"Unknown reranker model {name!r}. Supported models: {', '.join(sorted(_SUPPORTED_MODELS))}."
        )
    return name


def model_license(model: str | None) -> str:
    """The license string for ``model`` (defaulting empty/None to the default)."""
    return _SUPPORTED_MODELS[resolve_model(model)][0]


def _cache_dir() -> str | None:
    """Where fastembed should cache downloaded ONNX models.

    ``VGI_RERANK_CACHE_DIR`` wins; otherwise we honour fastembed's own
    ``FASTEMBED_CACHE_PATH``; otherwise ``None`` lets fastembed pick its default
    (a temp/cache dir under the user's home). The dir is created on demand.
    """
    explicit = os.environ.get(_CACHE_DIR_ENV) or os.environ.get("FASTEMBED_CACHE_PATH")
    if explicit:
        os.makedirs(explicit, exist_ok=True)
        return explicit
    return None


@cache
def _load_model(model_name: str) -> TextCrossEncoder:
    """Load (and cache) a fastembed ``TextCrossEncoder`` by name.

    First-ever use downloads the quantised ONNX model to the fastembed cache; all
    later worker processes that share the cache load it from disk. A download
    failure (e.g. offline on a cold cache) is surfaced as a clear error.
    """
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError as exc:  # pragma: no cover - dependency always present in prod
        raise ModelNotAvailableError("fastembed is not installed. Install it with: uv pip install fastembed") from exc

    try:
        return TextCrossEncoder(model_name=model_name, cache_dir=_cache_dir())
    except Exception as exc:  # noqa: BLE001 - turn any backend failure into an actionable error
        raise ModelNotAvailableError(
            f"Could not load reranker model {model_name!r}. The model is downloaded "
            f"on first use, so this needs network access on a cold cache; afterwards it "
            f"is served from the fastembed cache "
            f"(override with {_CACHE_DIR_ENV}). Original error: {exc}"
        ) from exc


def get_model(model: str | None) -> TextCrossEncoder:
    """Get the cached fastembed cross-encoder for ``model`` (thread-safe first load)."""
    name = resolve_model(model)
    with _lock:
        return _load_model(name)


def score_pairs(
    queries: list[str | None],
    documents: list[str | None],
    *,
    model: str | None,
) -> list[float | None]:
    """Cross-encoder relevance score for each (query, document) pair, in order.

    NULL or empty/whitespace-only query OR document yields a NULL score (the model
    is never handed a blank side). Real pairs are grouped by query so each distinct
    query runs a single batched ``TextCrossEncoder.rerank(query, [docs])`` call --
    the fastembed reranker API is query + many-documents, and the common SQL shape
    (one constant query broadcast across a top-K candidate column) collapses to a
    single forward batch.

    Returns one ``float`` (raw relevance logit; higher = more relevant) or ``None``
    per input position, preserving order.
    """
    n = len(documents)
    scores: list[float | None] = [None] * n

    # Bucket the live (non-empty) positions by their query text so each query is
    # scored against its documents in one batched rerank call.
    by_query: dict[str, list[tuple[int, str]]] = {}
    for i in range(n):
        q = queries[i]
        d = documents[i]
        if q is None or d is None or not q.strip() or not d.strip():
            continue
        by_query.setdefault(q, []).append((i, d))

    if not by_query:
        return scores

    encoder = get_model(model)
    for q, items in by_query.items():
        docs = [d for _i, d in items]
        # fastembed yields one float per document, in input order.
        for (i, _d), s in zip(items, encoder.rerank(q, docs), strict=True):
            scores[i] = float(s)
    return scores


# ---------------------------------------------------------------------------
# Startup warm-up
# ---------------------------------------------------------------------------


def warm_up() -> None:
    """Load (and, if needed, download) the default model once at worker startup.

    Everything in this module is lazy by design, so the *first* query of every
    ATTACH otherwise pays the cross-encoder load -- and on a cold cache the
    multi-second *download* -- inline. Under the end-to-end SQL suite that happens
    while the runner is mid-assertion on the first file: a long window in which a
    worker-pool teardown SIGTERM (or a heavily-loaded host) can kill the run and
    record a spurious failure, even though every score is deterministic.

    Warming here moves that one-time cost to process spawn (before any query), so
    each per-file first query is fast and the vulnerable window shrinks to near
    zero. It only populates the existing cache -- it never changes any output.
    Best-effort: if the model can't be loaded (e.g. offline on a cold cache) it is
    not fatal here -- the function that needs it will raise its own actionable
    error if actually invoked, so a worker still starts cleanly.
    """
    with contextlib.suppress(Exception):
        encoder = _load_model(DEFAULT_MODEL)
        # Touch the rerank path so the ONNX session is built and cached now.
        list(encoder.rerank("warm up", ["warm up document"]))
