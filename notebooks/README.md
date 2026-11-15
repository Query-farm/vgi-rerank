# Cross-Encoder Reranking examples

[Reranking cannot recover a passage it never receives](01_candidate_budget_before_reranking.ipynb) applies a candidate budget before local cross-encoder scoring, preserves excluded and NULL-score rows, and widens the budget to admit a planted answer. The worker's model registry validates the selected model and labels the returned passages with its reported license.

## Run locally or in Colab

Open the notebook's Colab badge and run all cells; setup installs the base shared package and retrieves five hash-verified companions. For a local copy of this directory, install [uv](https://docs.astral.sh/uv/) and Git, then:

```bash
uv sync --locked
uv run --locked vgi-notebooks 01_candidate_budget_before_reranking.ipynb
```

The separate notebook lock includes the shared `[runner]` extra. Colab installs only the base package, preserving its kernel and Jupyter client. For an interactive local kernel:

```bash
uv run --locked python -m ipykernel install --user --name vgi-rerank-notebooks --display-name 'Python (VGI Rerank examples)'
```

The visible, literal worker command is:

```sql
ATTACH 'rerank' (TYPE vgi, LOCATION 'uvx --python 3.13 --from git+https://github.com/Query-farm/vgi-rerank@v0.4.0 vgi-rerank-worker');
```

This short released Git launcher installs the worker and its declared dependencies in an isolated environment. It needs no local worker checkout, binary build or transitive dependency overrides. First use needs network access for Git, Python packages, the published Haybarn/VGI extension and about 92 MB of public model/tokenizer assets. Inference then runs locally. The notebook connection uses a fresh empty SQL-secret directory with persistent secrets disabled, checked before attachment and at cleanup.

[model-manifest.json](model-manifest.json) pins `Xenova/ms-marco-MiniLM-L-6-v2` to revision `a09144355adeed5f58c8ed011d209bf8ee5a1fec`, with sizes and SHA-256 hashes for five files. [model_assets.py](model_assets.py) downloads and verifies these bytes into ignored `validation/model-cache`, atomically publishes completed files and sets the worker's supported cache directory with `HF_HUB_OFFLINE=1`. Cleanup restores the prior environment. It performs no scoring and changes no worker functionality. The ONNX file is 90,992,115 bytes. An unavailable or corrupt download fails; there is no substitute model or fabricated score.

## Focused experiment

```bash
uv run --locked vgi-notebooks 01_candidate_budget_before_reranking.ipynb \
  --parameters '{"CANDIDATE_LIMIT":8}' --output-dir artifacts/more-candidates
```

Default K=4 admits four authored candidates, including one missing passage. K=8 also admits the planted direct answer and a second unusable passage. The final selection keeps at most two scored passages. Compare chosen text, logits and rank movement; restore 4 for the baseline. The runner requires a separate destination for overrides, rejects unknown parameters and saves defaults only after successful execution.

[Input provenance](fixtures/provenance.json) records the hash of eight invented help passages and their hand-assigned recall order. No observed search logs, personal records or event timestamps are involved. This is neither a retrieval implementation nor an accuracy/latency benchmark. SQL materializes the local prefix before scoring once, so the bound applies to model input rows. Missing text yields NULL; excluded candidates were not scored; negative logits are valid scores. These logits order this query/model's candidates and are not calibrated probabilities or universal thresholds.

## Coverage and validation

The complete actual v0.4.0 catalog has **two logical named surfaces**: `rerank_score` (two arity overloads) and `supported_models` (table function with a same-named table alias). There are no views, COPY handlers or authenticated/write surfaces. The notebook meaningfully exercises both. The registry advertises five models; only the explicitly pinned MiniLM model is downloaded and evaluated. Catalog coverage does not imply that every advertised model/backend combination was tested.

[coverage.json](coverage.json) binds catalog exports, visible SQL, interpreted results and successful execution receipts. The released worker source, entrypoint and root dependency files are byte-identical to current public source `8fe51fc838f9481405426ce72d46a07a2e7e53ef`: **2/2 released-runtime and 2/2 reviewed current-interface coverage**, with **0 exact-current runtime executions**. The proper title/subtitle come from the actual released catalog metadata. [Source compatibility](coverage/source-compatibility.json) retains that evidence and the resolved worker runtime versions; rerunning the Git tag may resolve newer compatible dependencies, so upgrading any pin requires validation.

Fourteen SQL invariants cover identity, prefix limits, NULL behavior, finite scores, ordering, selected membership and model provenance. Defaults and the separate K=8 experiment are executed with real model inference; a fresh notebook-only bootstrap is checked in a Python 3.12 kernel with eight libraries already imported, without changing their versions. This verifies a Colab-compatible setup, not manual operation of hosted Colab. SVG chart geometry/text remains vector output and committed content-addressed images support GitHub. Model caches, logs, PNG review images and experiment copies are ignored.

Worker source is unchanged. Its existing [green source CI](https://github.com/Query-farm/vgi-rerank/actions/runs/43579516861) is reused under the shared notebook-only workflow; no source rebuild or repeated transport suite is performed. [STANDARD.md](STANDARD.md) links the maintained shared standard.
