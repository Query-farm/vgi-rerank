"""Discovery table function for the rerank worker.

``supported_models()`` expands to **many rows** (one per known cross-encoder), so
it is a **table function** -- the form that accepts DuckDB ``name := value``
arguments (this one takes none, but the table-function shape is still its right
home). The per-row scoring function is a *scalar* and lives in
:mod:`vgi_rerank.scalars`.

    SELECT * FROM rerank.supported_models() ORDER BY model;
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pyarrow as pa
from vgi.metadata import FunctionExample
from vgi.table_function import (
    BindParams,
    ProcessParams,
    TableCardinality,
    TableFunctionGenerator,
    bind_fixed_schema,
    init_single_worker,
)
from vgi_rpc.rpc import OutputCollector

from . import models
from .schema_utils import field


@dataclass(kw_only=True)
class _NoArgs:
    """A discovery table function that takes no arguments."""


_SUPPORTED_MODELS_SCHEMA = pa.schema(
    [
        field("model", pa.string(), "Model name to pass to rerank_score(query, document, model).", nullable=False),
        field("license", pa.string(), "SPDX-style license of the model weights (all permissive here).", nullable=False),
    ]
)


@init_single_worker
@bind_fixed_schema
class SupportedModelsFunction(TableFunctionGenerator[_NoArgs]):
    """Every ``(model, license)`` cross-encoder the worker can run, one per row.

    ``model`` is the value you pass as the optional third argument to
    ``rerank_score(query, document, model)``; ``license`` is the license of the
    model weights (every advertised model is commercially usable).
    """

    FIXED_SCHEMA: ClassVar[pa.Schema] = _SUPPORTED_MODELS_SCHEMA

    class Meta:
        name = "supported_models"
        description = "Every (model, license) cross-encoder the rerank worker supports"
        categories = ["rerank", "metadata"]
        examples = [
            FunctionExample(
                sql="SELECT * FROM rerank.supported_models() ORDER BY model",
                description="List the supported reranker models and their licenses",
            ),
            FunctionExample(
                sql="SELECT license FROM rerank.supported_models() WHERE model = 'Xenova/ms-marco-MiniLM-L-6-v2'",
                description="License of the default model",
            ),
        ]

    @classmethod
    def cardinality(cls, params: BindParams[_NoArgs]) -> TableCardinality:
        n = len(models.supported_models())
        return TableCardinality(estimate=n, max=n)

    @classmethod
    def process(cls, params: ProcessParams[_NoArgs], state: None, out: OutputCollector) -> None:
        rows = models.supported_models()
        out.emit(
            pa.RecordBatch.from_pydict(
                {
                    "model": [r[0] for r in rows],
                    "license": [r[1] for r in rows],
                },
                schema=params.output_schema,
            )
        )
        out.finish()


TABLE_FUNCTIONS: list[type] = [
    SupportedModelsFunction,
]
