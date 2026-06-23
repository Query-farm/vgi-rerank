# vgi-rerank

Local **cross-encoder reranking** as a DuckDB SQL function — the precision
second stage of retrieval-augmented generation (RAG), entirely **on your machine**.

A [VGI](https://query.farm) worker that scores how relevant a *document* is to a
*query* with a cross-encoder, so a SQL query can do precise **second-stage
reranking in-engine** over a top-K candidate set. It runs MS MARCO / bge reranker
models through [`fastembed`](https://github.com/qdrant/fastembed) (Qdrant,
Apache-2.0), which uses **ONNX Runtime — no torch**, so it installs light and
starts fast. No API keys, no network at query time.

`vgi-rerank` is the companion to [`vgi-embed`](https://query.farm): together with
DuckDB VSS (ANN recall) and BM25 (keyword recall), they complete a fully local
RAG retrieval stack —

> **vgi-embed** vectors → **DuckDB VSS** ANN + **BM25** keyword → top-K candidates → **vgi-rerank** precision

```sql
INSTALL vgi FROM community; LOAD vgi;
ATTACH 'rerank' (TYPE vgi, LOCATION 'uv run rerank_worker.py');

-- Score one query/document pair (higher = more relevant).
SELECT rerank.rerank_score('how do I reset my password',
                           'Click the forgot password link to reset it.');

-- THE pattern: rerank a top-K candidate set (e.g. from VSS / BM25) by relevance.
SELECT id, chunk
FROM candidates                       -- the ~100 rows recall handed you
ORDER BY rerank.rerank_score('how do I reset my password', chunk) DESC
LIMIT 10;

-- What models are available?
SELECT * FROM rerank.supported_models();
```

## How a reranker differs from an embedder (and why it's top-K only)

A **bi-encoder** (vgi-embed) maps a query and a document to *independent* vectors.
Those vectors can be precomputed once, indexed (DuckDB VSS / HNSW), and reused for
every future query — that's what makes embedding-based **recall** over a whole
corpus cheap.

A **cross-encoder** (this worker) instead feeds the query and the document
**together** through one transformer and emits a single relevance score. That
joint attention is markedly more accurate — but it **cannot be precomputed or
cached**: every `(query, document)` pair is a fresh forward pass at query time.

> **The load-bearing caveat:** because each pair is scored live, a cross-encoder is
> only viable as a **second-stage reranker over a top-K candidate set** (rerank the
> ~100 candidates recall handed you — *never the whole corpus*). Embeddings you
> compute once; rerank scores you pay for on every query. Run recall first (VSS
> ANN and/or BM25), `LIMIT` to a candidate set, then `ORDER BY rerank_score(...)`.

## The model

| | |
|---|---|
| **Default model** | `Xenova/ms-marco-MiniLM-L-6-v2` |
| **Size** | ~80 MB (quantised ONNX) |
| **Model license** | **Apache-2.0** (commercial use permitted) — see the [model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) |
| **Runtime** | `fastembed` (Apache-2.0) on ONNX Runtime — no torch |

The model is **downloaded on first use** and cached on disk; later runs load it
locally. The cache directory is gitignored. Override it with `VGI_RERANK_CACHE_DIR`
(or fastembed's own `FASTEMBED_CACHE_PATH`). Pre-warm it once with `make models`.

Other supported models (pass as the third argument to `rerank_score`, or query
`supported_models()`) — all permissively licensed:

| model | license | size |
|---|---|---|
| `Xenova/ms-marco-MiniLM-L-6-v2` (default) | Apache-2.0 | ~80 MB |
| `Xenova/ms-marco-MiniLM-L-12-v2` | Apache-2.0 | ~120 MB |
| `BAAI/bge-reranker-base` | MIT | ~1 GB |
| `jinaai/jina-reranker-v1-tiny-en` | Apache-2.0 | ~130 MB |
| `jinaai/jina-reranker-v1-turbo-en` | Apache-2.0 | ~150 MB |

(fastembed also ships `jina-reranker-v2-base-multilingual`, but it is **CC-BY-NC**
— non-commercial — so this worker deliberately does **not** advertise it.)

## Functions

### Scalars

| function | signature | notes |
|---|---|---|
| `rerank_score(query, document)` | `(VARCHAR, VARCHAR) → DOUBLE` | cross-encoder relevance score, default model |
| `rerank_score(query, document, model)` | `(VARCHAR, VARCHAR, VARCHAR) → DOUBLE` | explicit model (arity overload) |
| `rerank_version()` | `→ VARCHAR` | worker + backend + default-model identity |

### Table function

| function | columns |
|---|---|
| `supported_models()` | `(model VARCHAR, license VARCHAR)` |

`rerank_score` exposes its optional `model` via an **arity overload** because VGI
scalar functions are positional-only (`name := value` is a table-function feature).

### Score semantics

The score is the model's **raw relevance logit**: higher means more relevant. It
is **not normalised** to a fixed range (it can be negative), and a value is only
meaningful **relative to other documents for the same query** — which is exactly
what `ORDER BY rerank_score(:q, doc) DESC LIMIT k` needs. Do **not** threshold on
an absolute value across different queries or models. (If you need a 0–1 score,
apply a sigmoid in SQL: `1 / (1 + exp(-rerank_score(...)))`.)

### NULL / robustness semantics

A NULL or empty/whitespace-only query **or** document → a NULL score. Nothing
crashes on odd input. NULL scores sort last under `ORDER BY ... DESC`.

## Using with the local RAG stack

Recall first (cheap, over the whole corpus), then rerank the candidates (precise,
top-K only):

```sql
INSTALL vss; LOAD vss;

-- Stage 1 — recall: ANN over precomputed vgi-embed vectors (top 100 candidates).
WITH candidates AS (
  SELECT id, body
  FROM docs
  ORDER BY array_cosine_distance(v, embed.embed_query('reset my password')::FLOAT[384])
  LIMIT 100
)
-- Stage 2 — precision: cross-encoder rerank the 100, keep the best 10.
SELECT id, body
FROM candidates
ORDER BY rerank.rerank_score('reset my password', body) DESC
LIMIT 10;
```

## Performance & deployment

- **CPU latency is real.** Each row is a transformer forward pass. The MiniLM-L-6
  default is the cheapest; the bge-reranker is more accurate but ~1 GB and slower.
  Keep candidate sets in the tens-to-low-hundreds.
- **GPU is a deployment upsell.** ONNX Runtime can use a GPU execution provider;
  that is an environment/packaging concern outside this worker's defaults.
- The value here is the **in-engine, offline packaging** and the local RAG suite —
  the cross-encoder models themselves are commodity, permissively licensed
  checkpoints.

## Development

```bash
uv sync --extra dev
make models                  # pre-warm the fastembed cache (downloads the default model)
uv run --no-sync pytest -q   # unit (model-gated tests self-skip on a cold/offline checkout)
make test-sql                # E2E: haybarn-unittest over test/sql/* (authoritative)
make test                    # both
uv run --no-sync ruff check . && uv run --no-sync mypy vgi_rerank/
```

`make test-sql` exports `VGI_RERANK_WORKER="uv run --python 3.13 rerank_worker.py"`
and runs `haybarn-unittest --test-dir . "test/sql/*"` (install once:
`uv tool install haybarn-unittest`, then put `~/.local/bin` on `PATH`).

## License

Worker code: **MIT** (see [LICENSE](LICENSE)). The default model
`Xenova/ms-marco-MiniLM-L-6-v2` is Apache-2.0-licensed; `BAAI/bge-reranker-base`
is MIT; `fastembed` is Apache-2.0; ONNX Runtime is MIT. The `vgi` DuckDB extension
and `vgi-python` are licensed separately by Query Farm.
