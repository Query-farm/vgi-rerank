# Reranking SQL notebooks

[Reranking cannot recover a passage it never receives](01_candidate_budget_before_reranking.ipynb) applies a candidate budget before cross-encoder scoring, traces rank changes, preserves NULL-score diagnostics, and demonstrates how admitting an excluded answer changes the selected text.

## Run from this directory

Install [uv](https://docs.astral.sh/uv/) and Git. The separate notebook environment uses Python 3.13 or 3.14; the worker explicitly uses Python 3.13. First use requires network access for the published Haybarn/VGI extension, Git worker, runtime dependencies and about 92 MB of pinned model/tokenizer assets. Inference then runs locally. Copy this entire directory; no worker source checkout or local build is required.

```bash
uv sync --frozen
uv run --frozen python execute.py
```

Visible SQL uses the exact remote Git revision `9d9707722e04c59199c508c70fd7dfca4b5d11e4` through `uvx`, with SDK 0.51.0 / RPC 0.53.1, fastembed 0.9.1, ONNX Runtime 1.31.1 and huggingface-hub 1.33.0. These runtime versions match the worker's tested lock. The independent notebook lock pins Haybarn 2.0.0rc11.

[model-manifest.json](model-manifest.json) pins `Xenova/ms-marco-MiniLM-L-6-v2` to revision `a09144355adeed5f58c8ed011d209bf8ee5a1fec`, with sizes and SHA-256 hashes for the actual ONNX/config/tokenizer files. [model_assets.py](model_assets.py) downloads and verifies these bytes into ignored `validation/model-cache`, atomically replaces completed files, sets the supported worker cache directory, and enables `HF_HUB_OFFLINE=1` before worker startup. Cleanup restores the prior environment. This helper performs no scoring and changes no worker functionality. The selected `onnx/model.onnx` file is 90,992,115 bytes. Changing worker/runtime/model pins requires rerunning defaults and the experiment; exact logits may still vary slightly by hardware/runtime.

For interactive use, register this interpreter and select it in Jupyter:

```bash
uv run --frozen python -m ipykernel install --user --name vgi-rerank-notebooks --display-name 'Python (vgi-rerank notebooks)'
```

Restart and run all after changing `CANDIDATE_LIMIT`. Python setup, plots, prose-generation source and cleanup are folded; parameters and SQL remain visible. The runner uses a temporary kernel, sets this directory as cwd, saves only after success, and requires a separate destination for overrides. Unknown parameters and the source directory as destination are rejected.

## Focused experiment

```bash
uv run --frozen python execute.py --parameters '{"CANDIDATE_LIMIT":8}' --output-dir validation/more-candidates
```

The default admits four candidates, including a missing passage. The experiment admits all eight, including the planted direct answer and a second unusable passage. Compare the selected text, raw logits and rank movement, then restore 4. SQL applies the limit before the real worker call and materializes scores once; it does not score an entire external corpus.

[Input provenance](fixtures/provenance.json) records the hash of eight invented help passages and their deliberately hand-assigned recall order. The query and author role labels are synthetic. No observed search logs, personal records or event timestamps are involved. This is not a BM25/vector-retrieval implementation or an accuracy benchmark.

## Checks and interpretation

Twelve SQL checks cover identity, candidate limits, score ordering, finite values, contiguous ranks, selection membership, model availability and NULL/empty semantics. Input and model hashes are verified separately. Missing text yields NULL, which is distinct from a negative raw logit; excluded candidates have not been scored at all. Raw logits are not calibrated probabilities and should not be compared as universal cutoffs across queries/models. The two-result selection is a demonstration choice.

Source validation: 20 pytest tests with the actual model available (no skips), 20 stdio / 24 HTTP / 20 Unix SQL assertions, Ruff/format, mypy, isolated pydoclint, and metadata quality 100/L2 with zero findings. Saved default outputs are embedded in the main notebook. Logs, chart images, model caches and experiment copies remain ignored under `validation/`. The runner copies all assets, fixtures, model manifests and runtime helpers with experiment outputs. [STANDARD.md](STANDARD.md) records presentation conventions.
