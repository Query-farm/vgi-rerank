"""End-to-end tests through ``vgi.client.Client``, spawning the real worker.

These exercise the full Arrow-IPC round trip the way DuckDB would: the worker
runs as a subprocess and we drive it over stdin/stdout. Gated on the default
fastembed cross-encoder being available, so a bare/offline checkout stays green.
"""

from __future__ import annotations

import os
import shlex
import sys

import pyarrow as pa
import pytest
from vgi import Arguments
from vgi.client import Client

from tests.harness import model_available

_WORKER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rerank_worker.py")

needs_model = pytest.mark.skipif(
    not model_available(), reason="fastembed default cross-encoder not available (offline / cold cache)"
)


def _client() -> Client:
    # Launch the worker with the same interpreter running the tests so it sees the
    # installed deps (rather than going through `uv run`). Client wants a
    # shell-style command string.
    return Client(f"{shlex.quote(sys.executable)} {shlex.quote(_WORKER)}")


@needs_model
def test_rerank_score_end_to_end() -> None:
    # rerank_score(query, document) has no ConstParam, so the two columns bind by
    # position from the input batch and arguments.positional is empty.
    batch = pa.RecordBatch.from_pydict(
        {
            "query": ["how do I reset my password", "how do I reset my password"],
            "document": [
                "Click the 'forgot password' link to reset it.",
                "Our office is open from nine to five on weekdays.",
            ],
        }
    )
    with _client() as client:
        results = list(
            client.scalar_function(
                function_name="rerank_score",
                input=iter([batch]),
                arguments=Arguments(positional=[]),
            )
        )
    scores = results[0]["result"].to_pylist()
    assert len(scores) == 2
    # The relevant document outscores the irrelevant one over the wire too.
    assert scores[0] > scores[1]


@needs_model
def test_rerank_score_with_explicit_model_overload_end_to_end() -> None:
    # The 3-arity overload: the ConstParam model name goes in positional.
    batch = pa.RecordBatch.from_pydict(
        {
            "query": ["reset password"],
            "document": ["Click the forgot password link."],
        }
    )
    with _client() as client:
        results = list(
            client.scalar_function(
                function_name="rerank_score",
                input=iter([batch]),
                arguments=Arguments(positional=[pa.scalar("Xenova/ms-marco-MiniLM-L-6-v2")]),
            )
        )
    assert isinstance(results[0]["result"].to_pylist()[0], float)
