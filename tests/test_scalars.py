"""Tests for the rerank scalar functions (compute() array-in / array-out).

Split into two tiers:

* **Pure logic** (no model): NULL/empty masking in ``rerank_score``, the model
  registry (``resolve_model`` / ``model_license`` / unknown-model errors), and
  ``rerank_version()``. These always run.
* **Model-backed** (``@needs_model``): the actual cross-encoder scores. Gated on
  the default fastembed model being loadable, so a bare/offline checkout skips
  them cleanly while a provisioned environment runs them.

The model assertions are deliberately *relative* -- a clearly relevant document
must outscore a clearly irrelevant one for the same query, and the ORDER BY shape
must rank the relevant row first -- never exact float values, which are raw logits
that vary by ONNX build.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from tests.harness import model_available
from vgi_rerank import models
from vgi_rerank.scalars import RerankScore, RerankScoreModel, RerankVersion

needs_model = pytest.mark.skipif(
    not model_available(), reason="fastembed default cross-encoder not available (offline / cold cache)"
)


# --- startup warm-up (best-effort, never fatal) -----------------------------


class TestWarmUp:
    def test_warm_up_is_idempotent_and_never_raises(self) -> None:
        # Called at worker spawn to move the model load/download off the first
        # query. Must be best-effort: safe to call repeatedly and never raise,
        # even with no model present (the function that needs it raises instead).
        models.warm_up()
        models.warm_up()


# --- model registry (pure, always runs) -------------------------------------


class TestModelRegistry:
    def test_default_is_resolved_for_empty_and_none(self) -> None:
        assert models.resolve_model(None) == models.DEFAULT_MODEL
        assert models.resolve_model("") == models.DEFAULT_MODEL
        assert models.resolve_model("   ") == models.DEFAULT_MODEL

    def test_known_model_passes_through(self) -> None:
        assert models.resolve_model("BAAI/bge-reranker-base") == "BAAI/bge-reranker-base"

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(models.ModelNotAvailableError, match="Unknown reranker model"):
            models.resolve_model("no/such-model")

    def test_default_model_is_permissively_licensed(self) -> None:
        # The default must be commercially usable (Apache-2.0).
        assert models.model_license(None) == "apache-2.0"

    def test_supported_models_are_all_permissive_and_sorted(self) -> None:
        rows = models.supported_models()
        assert rows == sorted(rows)
        assert all(lic in {"apache-2.0", "mit"} for _name, lic in rows)


# --- rerank_version (no model) ----------------------------------------------


class TestVersion:
    def test_version_mentions_worker_and_default_model(self) -> None:
        v = RerankVersion.compute().to_pylist()[0]
        assert "vgi-rerank" in v
        assert models.DEFAULT_MODEL in v


# --- NULL/empty masking (no model needed for the NULL rows) -----------------


class TestNullMasking:
    def test_all_null_and_empty_pairs_score_null_without_loading_model(self) -> None:
        # Every pair has a NULL/empty side -> all-NULL output, and the model is
        # never invoked (so this runs even with no model installed).
        q = pa.array([None, "", "   ", "real query"])
        d = pa.array(["real doc", "real doc", "real doc", None])
        out = RerankScore.compute(q, d).to_pylist()
        assert out == [None, None, None, None]


# --- model-backed scoring (gated) -------------------------------------------


@needs_model
class TestRerankScore:
    def test_returns_a_float_per_row(self) -> None:
        q = pa.array(["how do I reset my password"])
        d = pa.array(["Click the forgot password link to reset it."])
        out = RerankScore.compute(q, d).to_pylist()
        assert len(out) == 1
        assert isinstance(out[0], float)

    def test_relevant_doc_outscores_irrelevant_doc(self) -> None:
        # The ordering-sanity assertion the brief calls for: a clearly relevant
        # document must score higher than a clearly irrelevant one for one query.
        query = "how do I reset my password"
        q = pa.array([query, query])
        d = pa.array(
            [
                "To reset your password, click the 'forgot password' link on the login page.",
                "Our office is open from nine to five on weekdays.",
            ]
        )
        relevant, irrelevant = RerankScore.compute(q, d).to_pylist()
        assert relevant > irrelevant

    def test_order_by_pattern_ranks_relevant_first(self) -> None:
        # Mirror the documented SQL: ORDER BY rerank_score(:q, doc) DESC over a
        # tiny candidate set; the relevant candidate must come first.
        query = "best practices for indexing a large table"
        candidates = [
            "Soup recipes for a cold winter evening.",
            "Create an index on the most selective columns to speed up large-table scans.",
            "A history of the Roman aqueducts.",
        ]
        q = pa.array([query] * len(candidates))
        d = pa.array(candidates)
        scores = RerankScore.compute(q, d).to_pylist()
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda r: r[1], reverse=True)
        assert ranked[0][0] == candidates[1]

    def test_null_and_empty_rows_interleave_with_real_ones(self) -> None:
        q = pa.array(["q", None, "q", "q"])
        d = pa.array(["a relevant document about q", "doc", "", "another document"])
        out = RerankScore.compute(q, d).to_pylist()
        assert isinstance(out[0], float)
        assert out[1] is None  # NULL query
        assert out[2] is None  # empty document
        assert isinstance(out[3], float)

    def test_explicit_model_overload(self) -> None:
        q = pa.array(["how do I reset my password"])
        d = pa.array(["Click the forgot password link to reset it."])
        out = RerankScoreModel.compute(q, d, "Xenova/ms-marco-MiniLM-L-6-v2").to_pylist()
        assert isinstance(out[0], float)

    def test_unknown_model_raises(self) -> None:
        q = pa.array(["q"])
        d = pa.array(["d"])
        with pytest.raises(models.ModelNotAvailableError, match="Unknown reranker model"):
            RerankScoreModel.compute(q, d, "no/such-model").to_pylist()

    def test_long_document_does_not_crash(self) -> None:
        q = pa.array(["the cat"])
        d = pa.array(["the cat sat on the mat. " * 2000])
        out = RerankScore.compute(q, d).to_pylist()
        assert isinstance(out[0], float)
