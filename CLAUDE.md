# CLAUDE.md — vgi-rerank

Contributor/agent notes. User-facing docs live in `README.md`; this is the
"how it's built and where the sharp edges are" companion.

## What this is

A [VGI](https://query.farm) worker exposing a **local cross-encoder reranker** to
DuckDB/SQL — the precision second stage of a local RAG stack (companion to
vgi-embed). `rerank_worker.py` assembles every function into one `rerank` catalog
(single `main` schema) and runs it over stdio. Scores are computed locally via
`fastembed` (Qdrant, Apache-2.0) on **ONNX Runtime — no torch**; the default model
is `Xenova/ms-marco-MiniLM-L-6-v2` (~80 MB, **Apache-2.0**).

## Layout

```
rerank_worker.py       repo-root stdio entry; PEP 723 inline deps; warms the model then serves; main()
serve.py               HTTP entry shim
vgi_rerank/
  models.py            loaded-once-and-cached TextCrossEncoder lifecycle + warm_up(); score_pairs(); supported-model registry
  scalars.py           rerank_score (+ explicit-model arity overload), rerank_version
  tables.py            supported_models() discovery table function
  schema_utils.py      pa.Field comment helper
tests/                 pytest: scalars / tables / Client integration (model-gated tests self-skip)
test/sql/*.test        haybarn-unittest sqllogictest — authoritative E2E
Makefile               test / test-unit / test-sql / models / lint / typecheck
```

## Core conventions (read first)

- **Scalars are positional-only** (no `name := value`). `rerank_score`'s optional
  `model` is therefore a third **arity overload** (`rerank_score(query, document)`
  / `rerank_score(query, document, model)`) sharing one name — same idiom as
  vgi-embed's `embed`.
- **`DOUBLE` return is declared explicitly.** `rerank_score` declares
  `Returns(arrow_type=pa.float64())`. (Scalar/LIST/STRUCT/TIMESTAMPTZ returns need
  an explicit `arrow_type`; we set it for clarity and to match the sibling.)
- **`supported_models()` is a table function** (rows out), so it lives in
  `tables.py` with the `@init_single_worker` / `@bind_fixed_schema` pattern.
- **NULL/empty → NULL.** `score_pairs` masks any pair with a NULL or
  whitespace-only query *or* document to a NULL score *before* calling the model,
  and splices NULLs back by index — so the model is only ever handed real text,
  and an all-empty input batch never loads the model at all.

## Sharp edges

1. **Expensive init: load the model ONCE.** `models._load_model` is `@cache`'d and
   guarded by a lock; the whole point is that VGI keeps the worker alive so the
   ONNX session is built once and amortised over every row. Never construct a
   `TextCrossEncoder` per call.
2. **First use downloads the model.** `fastembed` fetches a quantised ONNX model on
   first use and caches it (gitignored). On a **cold cache offline**, scoring
   raises an actionable `ModelNotAvailableError`. `warm_up()` (called at worker
   spawn from `RerankWorker.run`) moves that download/load off the first query;
   it's best-effort and never fatal. `make models` pre-warms for dev/CI.
3. **The fastembed reranker API is `rerank(query, [docs])`** — one query, many
   documents — not a flat list of pairs. `score_pairs` buckets the live rows by
   their query text and runs one batched call per distinct query. The common SQL
   shape (one constant query broadcast across a candidate column) collapses to a
   single forward batch; mixed queries in one batch still work (one call each).
4. **`haybarn-unittest` skips `require vgi`** — use explicit `statement ok` /
   `LOAD vgi;` in `.test` files (the ones here do), and `require-env
   VGI_RERANK_WORKER` + `ATTACH ... '${VGI_RERANK_WORKER}'`.
5. **Determinism in assertions.** Scores are deterministic per model, but they are
   **raw logits** whose exact values vary by ONNX build/platform — and they are
   **not** normalised to any range. So never assert exact floats; assert the
   planted **relevant > irrelevant** ordering and that `ORDER BY rerank_score(:q,
   doc) DESC` ranks the relevant row first.
6. **Model-gated tests.** Unit/integration tests that actually score are guarded by
   `@needs_model` (default model loadable) so a bare/offline checkout stays green;
   the pure-logic tests (model registry, NULL masking, `rerank_version`) always
   run.
7. **Top-K only — by design.** A cross-encoder scores every pair at query time; it
   cannot be cached like an embedding. This worker is *only* meant to rerank a
   recall-produced top-K candidate set, never a whole corpus. That is the
   documented usage and the reason the E2E test asserts the `ORDER BY ... LIMIT`
   shape.

## Testing

```sh
uv run --no-sync pytest -q     # unit (model-gated tests self-skip on a cold/offline checkout)
make models                    # pre-warm the fastembed cache for local dev
make test-sql                  # E2E: haybarn-unittest over test/sql/*  (authoritative)
make test                      # both
uv run --no-sync ruff check . && uv run --no-sync mypy vgi_rerank/
```

`make test-sql` exports `VGI_RERANK_WORKER="uv run --python 3.13 rerank_worker.py"`
and runs `haybarn-unittest --test-dir . "test/sql/*"` (install once:
`uv tool install haybarn-unittest`). **The SQL suite is authoritative** — it
exercises the real RPC + model path. CI runs unit + lint plus a gated `e2e` job
(installs worker deps from PyPI, warms the model, launches the worker from the
prepared venv).
```
