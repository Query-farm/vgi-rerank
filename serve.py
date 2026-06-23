# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "vgi-python[http,oauth]>=0.8.3",
#     "fastembed>=0.5",
# ]
# ///
"""HTTP entrypoint for the rerank worker.

Forces the worker's CLI into HTTP mode (``Worker.main()`` serves stdio by
default) so callers only pass ``--host``/``--port``.
"""

import sys

from rerank_worker import RerankWorker

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--http" not in argv:
        argv = ["--http", *argv]
    sys.argv = [sys.argv[0], *argv]
    RerankWorker.main()
