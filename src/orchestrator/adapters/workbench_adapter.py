# # src/orchestrator/adapters/workbench_adapter.py
# import importlib
# import logging

# logger = logging.getLogger(__name__)


# class WorkbenchAdapter:
#     """
#     Loads a pipeline workbench module dynamically.
#     Expects module to expose:
#       - async workbench_ingest(session_id, file_paths, options) -> dict
#       - async workbench_chat(session_id, message, options) -> dict
#     """

#     def __init__(self, pipeline_id: str, module_path: str):
#         self.pipeline_id = pipeline_id
#         self.module_path = module_path
#         self._mod = self._import_workbench()

#     def _import_workbench(self):
#         try:
#             logger.info(f"WorkbenchAdapter({self.pipeline_id}): importing {self.module_path}")
#             return importlib.import_module(self.module_path)
#         except Exception as e:
#             raise RuntimeError(f"Failed to import {self.module_path}: {e}") from e

#     async def ingest(self, session_id: str, file_paths: list[str], options: dict):
#         return await self._mod.workbench_ingest(session_id=session_id, file_paths=file_paths, options=options)

#     async def chat(self, session_id: str, message: str, options: dict):
#         return await self._mod.workbench_chat(session_id=session_id, message=message, options=options)


##-----##

# src/orchestrator/adapters/workbench_adapter.py
import importlib
import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class WorkbenchAdapter:
    """
    Loads a pipeline workbench module dynamically.

    Expects module to expose:
      - workbench_ingest(session_id, file_paths, options) -> dict or awaitable
      - workbench_chat(session_id, message, options) -> dict or awaitable
    """

    def __init__(self, pipeline_id: str, module_path: str):
        self.pipeline_id = pipeline_id
        self.module_path = module_path
        self._mod = None  # lazy import

    def _ensure_imported(self):
        if self._mod is not None:
            return

        logger.info("WorkbenchAdapter(%s): importing %s", self.pipeline_id, self.module_path)
        self._mod = importlib.import_module(self.module_path)

        # Hard check: must expose chat at minimum
        if not hasattr(self._mod, "workbench_chat"):
            raise RuntimeError(
                f"{self.module_path} missing required function workbench_chat(session_id, message, options)"
            )

    async def _maybe_await(self, value):
        if asyncio.iscoroutine(value):
            return await value
        return value

    def _normalize_chat_result(self, result: Any) -> Dict[str, Any]:
        # If pipeline returned plain string, treat as answer
        if not isinstance(result, dict):
            return {"answer": str(result), "sources": [], "chunks": [], "debug": {}}

        return {
            "answer": str(result.get("answer", "")),
            "sources": result.get("sources", []) or [],
            "chunks": result.get("chunks", []) or [],
            "debug": result.get("debug", {}) or {},
        }

    async def ingest(self, session_id: str, file_paths: list[str], options: dict):
        try:
            self._ensure_imported()
            if not hasattr(self._mod, "workbench_ingest"):
                # Not implemented is OK; orchestrator still works
                return {
                    "status": "ok",
                    "ingested_files": len(file_paths),
                    "details": "Files saved; workbench_ingest not implemented in pipeline",
                }

            res = self._mod.workbench_ingest(session_id=session_id, file_paths=file_paths, options=options)
            res = await self._maybe_await(res)
            return res if isinstance(res, dict) else {"status": "ok", "detail": str(res)}

        except Exception as e:
            logger.exception("WorkbenchAdapter(%s): ingest failed", self.pipeline_id)
            return {"status": "error", "error": str(e)}

    async def chat(self, session_id: Optional[str], message: str, options: dict):
        try:
            self._ensure_imported()
            res = self._mod.workbench_chat(session_id=session_id, message=message, options=options)
            res = await self._maybe_await(res)
            normalized = self._normalize_chat_result(res)
            normalized["debug"].setdefault("pipeline_id", self.pipeline_id)
            return normalized

        except Exception as e:
            logger.exception("WorkbenchAdapter(%s): chat failed", self.pipeline_id)
            return {
                "answer": f"(adapter error) {e}",
                "sources": [],
                "chunks": [],
                "debug": {"pipeline_id": self.pipeline_id, "error": str(e)},
            }
