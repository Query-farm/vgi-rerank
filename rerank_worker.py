# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http]>=0.46.1",
#     "fastembed>=0.5",
# ]
# ///
"""Repo-root stdio entrypoint for the rerank worker (thin PEP 723 shim).

The worker's catalog, ``RerankWorker`` class, and ``main()`` live in the
wheel-importable ``vgi_rerank.worker`` module so the released package ships them.
This file stays a runnable script (``uv run rerank_worker.py`` / ``python
rerank_worker.py``) for the Makefile, ``ci/run-integration.sh``, and the
``vgi.client``-based tests, re-exporting the package symbols unchanged.

See ``vgi_rerank/worker.py`` for the catalog assembly and usage docs.
"""

from __future__ import annotations

from vgi_rerank.worker import (
    _RERANK_CATALOG,
    RerankWorker,
    _RerankCatalogInterface,
    main,
)

__all__ = ["RerankWorker", "_RERANK_CATALOG", "_RerankCatalogInterface", "main"]


if __name__ == "__main__":
    main()
