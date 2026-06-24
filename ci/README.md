# CI: the vgi-rerank worker integration suite

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs the unit tests
and this repo's sqllogictest suite (`test/sql/*.test`) against the vgi-rerank
VGI worker through the **real DuckDB `vgi` extension** on every push / PR.

## How it works (no C++ build)

Rather than building the vgi DuckDB extension from source, CI drives a
**prebuilt** standalone `haybarn-unittest` (the DuckDB/Haybarn sqllogictest
runner, published in Haybarn's releases) and installs the **signed** `vgi`
extension from the Haybarn community channel:

1. **Install the worker** — `uv sync --frozen --extra http` into a venv.
   `rerank_worker.py` is a self-contained PEP 723 worker the extension can spawn
   via stdio, or that the harness can boot over HTTP / an AF_UNIX socket.
2. **Download the runner** — the matching `haybarn_unittest-*` asset per
   platform from the latest Haybarn release.
3. **Preprocess** — the standalone runner links none of the extensions the
   tests gate on, so [`preprocess-require.awk`](preprocess-require.awk) rewrites
   each `require <ext>` into an explicit signed `INSTALL <ext> FROM
   {community,core}; LOAD <ext>;`. These tests skip `require vgi` (haybarn
   silently SKIPs it) and `LOAD vgi;` directly, so the awk also injects an
   `INSTALL vgi FROM community;` right before each bare `LOAD vgi;`. `require-env`
   and everything else pass through untouched.
4. **Run** — [`run-integration.sh`](run-integration.sh) stages the preprocessed
   tree, resolves `VGI_RERANK_WORKER` (the ATTACH `LOCATION`) per the
   `$TRANSPORT` it's run with (see below), warms the extension cache once, then
   runs the suite in a single `haybarn-unittest` invocation. Any failed
   assertion exits non-zero and fails the job.

## Transport matrix (subprocess | http | unix)

The same `test/sql/*.test` suite is run over all three VGI transports — the
extension picks the transport from the `LOCATION` string the `.test` files
`ATTACH`, and `run-integration.sh` builds that string from `$TRANSPORT`:

| `TRANSPORT`  | `VGI_RERANK_WORKER` (LOCATION)        | How the worker is reached |
|--------------|---------------------------------------|---------------------------|
| `subprocess` | `.venv/bin/python rerank_worker.py`   | extension spawns the worker per query; Arrow IPC over stdin/stdout (default) |
| `http`       | `http://127.0.0.1:<port>`             | harness boots `rerank_worker.py --http --port 0 --port-file <f>`, waits for the port-file, then ATTACHes that URL |
| `unix`       | `unix:///tmp/rerank-<pid>.sock`       | harness boots `rerank_worker.py --unix <sock>`, waits for the socket to appear, then ATTACHes it |

The CI `integration` job is a `transport: [subprocess, http, unix]` × `os`
matrix; each leg runs `ci/run-integration.sh` with `TRANSPORT=<t>`. Run a single
transport locally with e.g. `TRANSPORT=http ci/run-integration.sh`.

### Port / readiness discovery

- **http**: the worker writes its auto-selected port to `--port-file`
  atomically (tmp + rename), so the harness watches for that file to appear and
  reads the port from it — it does **not** parse stdout. Boot line:
  `rerank_worker.py --http --port 0 --port-file <f>`.
- **unix**: the worker binds the AF_UNIX socket and prints `UNIX:<abs-path>`;
  the harness polls for the socket file (`test -S`) to appear. Boot line:
  `rerank_worker.py --unix <sock>`.

Both out-of-band server processes are booted with cwd = the repo root (so they
resolve the worker's relative resources / model cache) and are trap-killed on
exit.

### HTTP transport needs the `httpfs` extension (resolved, not gated)

The vgi extension implements HTTP transport on top of DuckDB's **httpfs**
extension, so an `http://` ATTACH binds with

> `Binder Error: VGI HTTP transport requires the httpfs extension. Install it with: INSTALL httpfs; LOAD httpfs;`

unless httpfs is loaded first. This is a **dependency**, not a protocol
limitation, so we resolve it rather than gate: the http leg of
`run-integration.sh` injects a signed `INSTALL httpfs FROM core; LOAD httpfs;`
into each staged `.test` (right after the awk-injected `LOAD vgi;`). The
`.test` files themselves stay transport-agnostic.

The http leg also needs the worker's `http` extra (waitress): `pyproject.toml`
ships an `http` extra (`vgi-python[http]`), the PEP 723 header lists
`vgi-python[http]`, and CI runs `uv sync --frozen --extra http`.

> **Sharp edge — the runner silently SKIPs HTTP errors.** The haybarn/DuckDB
> sqllogictest runner's default skip list skips any statement whose error
> message contains `"HTTP"` or `"Unable to connect"`. Without the httpfs load,
> *every* HTTP-leg test SKIPs (the httpfs binder error contains "HTTP") and the
> suite reports "All tests were skipped" — a green-looking **fake pass**, not a
> real one. `run-integration.sh` therefore fails the leg unless the runner
> reports `All tests passed (N assertions …)` with N > 0 and reports zero
> skips.

### Per-transport status

- **subprocess**: GREEN — 21 assertions.
- **http**: GREEN — 25 assertions (21 + the injected httpfs INSTALL/LOAD
  statements).
- **unix**: GREEN — 21 assertions. No extra deps; `--unix` is built into the
  worker's `Worker.main()`.

The suite here is stateless scalar scoring + a static discovery table function
(`supported_models()`), so none of the inherent HTTP limitations (streaming
partition-local state, etc.) apply — nothing needed gating.

## Run it locally

```bash
uv sync --python 3.13 --extra http          # install the worker + deps (http extra for the http leg)
# point HAYBARN_UNITTEST at a haybarn-unittest binary (or a local DuckDB
# `unittest` built with the vgi extension). WORKER_CMD is the stdio command that
# runs the worker; the harness uses it directly for subprocess and boots it with
# --http / --unix for the other transports.
HAYBARN_UNITTEST=/path/to/haybarn-unittest \
WORKER_CMD="uv run --python 3.13 rerank_worker.py" \
  TRANSPORT=subprocess ci/run-integration.sh    # or TRANSPORT=http / TRANSPORT=unix
```

`TRANSPORT` defaults to `subprocess`, and `WORKER_CMD` defaults to
`uv run --python 3.13 <repo>/rerank_worker.py`, so a bare
`HAYBARN_UNITTEST=… ci/run-integration.sh` runs the subprocess leg.
