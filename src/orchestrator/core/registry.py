# # src/orchestrator/core/registry.py
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Dict

# from ..adapters.workbench_adapter import WorkbenchAdapter


# @dataclass
# class PipelineSpec:
#     pipeline_id: str
#     module_path: str


# class Registry:
#     def __init__(self):
#         # Add more pipelines later (graph_rag, multimodal_rag, hybrid_rag)
#         self._pipelines: Dict[str, WorkbenchAdapter] = {}

#         specs = [
#             PipelineSpec(pipeline_id="simple_rag", module_path="src.simple_rag.workbench"),
#         ]

#         for spec in specs:
#             self._pipelines[spec.pipeline_id] = WorkbenchAdapter(
#                 pipeline_id=spec.pipeline_id,
#                 module_path=spec.module_path,
#             )

#     def list_pipeline_ids(self) -> list[str]:
#         return sorted(self._pipelines.keys())

#     def get(self, pipeline_id: str) -> WorkbenchAdapter:
#         if pipeline_id not in self._pipelines:
#             raise KeyError(f"Unknown pipeline_id: {pipeline_id}")
#         return self._pipelines[pipeline_id]


# registry = Registry()


###----------------###

# src/orchestrator/core/registry.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

from ..adapters.workbench_adapter import WorkbenchAdapter
from ..adapters.http_adapter import HttpAdapter
import os


@dataclass
class PipelineSpec:
    pipeline_id: str
    module_path: str


class Registry:
    def __init__(self):
        # self._pipelines: Dict[str, WorkbenchAdapter] = {}
        #
        # specs = [
        #     PipelineSpec(pipeline_id="simple_rag", module_path="src.simple_rag.workbench"),
        #     # add more later
        # ]
        #
        # for spec in specs:
        #     self._pipelines[spec.pipeline_id] = WorkbenchAdapter(
        #         pipeline_id=spec.pipeline_id,
        #         module_path=spec.module_path,
        #     )

        self._pipelines: Dict[str, HttpAdapter] = {}

        # 1. Simple RAG
        simple_url = os.getenv("SIMPLE_RAG_URL", "http://simple_rag:9000")
        self._pipelines["simple_rag"] = HttpAdapter(
            pipeline_id="simple_rag",
            base_url=simple_url,
            display_name="Simple RAG",
            capabilities={"upload": True}
        )

        # 2. Graph RAG
        graph_url = os.getenv("GRAPH_RAG_URL", "http://graph_rag:2000")
        self._pipelines["graph_rag"] = HttpAdapter(
            pipeline_id="graph_rag",
            base_url=graph_url,
            display_name="Graph RAG",
            capabilities={"upload": True}
        )

        # 3. MultiModal RAG
        mm_url = os.getenv("MULTIMODAL_RAG_URL", "http://multimodal_rag:5001")
        self._pipelines["multimodal_rag"] = HttpAdapter(
            pipeline_id="multimodal_rag",
            base_url=mm_url,
            display_name="MultiModal RAG",
            capabilities={"upload": True, "multimodal": True}
        )

        # 4. Agentic RAG
        agentic_url = os.getenv("AGENTIC_RAG_URL", "http://agentic_rag:8001")
        self._pipelines["agentic_rag"] = HttpAdapter(
            pipeline_id="agentic_rag",
            base_url=agentic_url,
            display_name="Agentic RAG",
            capabilities={"upload": True, "tools": True}
        )

    def list_pipeline_ids(self) -> list[str]:
        return sorted(self._pipelines.keys())

    # Generalized return type to support base adapter
    def get(self, pipeline_id: str) -> HttpAdapter:
        if pipeline_id not in self._pipelines:
            raise KeyError(f"Unknown pipeline_id: {pipeline_id}")
        return self._pipelines[pipeline_id]

    # Generalized return type to support base adapter
    def get_adapter(self, pipeline_id: str) -> Optional[HttpAdapter]:
        return self._pipelines.get(pipeline_id)


registry = Registry()
