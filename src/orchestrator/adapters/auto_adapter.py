
# src/orchestrator/adapters/auto_adapter.py
import importlib
import inspect
import asyncio
import logging
from .base import BaseAdapter
from .mock_adapter import MockAdapter

logger = logging.getLogger(__name__)


class AutoAdapter(BaseAdapter):
    def __init__(self, pipeline_id: str, display_name: str = None, capabilities: dict = None):
        super().__init__(pipeline_id, display_name, capabilities)
        # locate chat callable
        self._callable = self._locate_callable()
        # locate ingest callable
        self._ingest = self._locate_ingest_callable()
        self._mock = MockAdapter(pipeline_id, display_name=display_name, capabilities=capabilities)

    def _locate_callable(self):
        module_patterns = [
            f"src.{self.pipeline_id}.workbench",
            f"src.{self.pipeline_id}.app.api.main",
            f"src.{self.pipeline_id}.run",
            f"src.{self.pipeline_id}",
        ]
        func_names = ["workbench_chat", "answer_question", "chat", "run", "handle_query"]

        for mod_name in module_patterns:
            logger.debug("AutoAdapter(%s): trying module '%s' for chat", self.pipeline_id, mod_name)
            try:
                mod = importlib.import_module(mod_name)
            except Exception as e:
                logger.debug("AutoAdapter(%s): import failed for %s: %s", self.pipeline_id, mod_name, e)
                continue
            for fn in func_names:
                if hasattr(mod, fn):
                    fn_obj = getattr(mod, fn)
                    if callable(fn_obj):
                        logger.info("AutoAdapter(%s): located chat callable '%s' in %s", self.pipeline_id, fn, mod_name)
                        return fn_obj
            for name, obj in inspect.getmembers(mod, inspect.isfunction):
                if name in func_names:
                    logger.info("AutoAdapter(%s): located chat callable '%s' in %s (inspected)", self.pipeline_id, name, mod_name)
                    return obj
        logger.warning("AutoAdapter(%s): no chat callable found; will fallback to MockAdapter", self.pipeline_id)
        return None

    def _locate_ingest_callable(self):
        module_patterns = [
            f"src.{self.pipeline_id}.workbench",
            f"src.{self.pipeline_id}.app.api.main",
            f"src.{self.pipeline_id}.run",
            f"src.{self.pipeline_id}",
        ]
        ingest_names = ["workbench_ingest", "ingest_documents", "ingest", "index", "index_documents"]

        for mod_name in module_patterns:
            logger.debug("AutoAdapter(%s): trying module '%s' for ingest", self.pipeline_id, mod_name)
            try:
                mod = importlib.import_module(mod_name)
            except Exception as e:
                logger.debug("AutoAdapter(%s): import failed for %s: %s", self.pipeline_id, mod_name, e)
                continue
            for fn in ingest_names:
                if hasattr(mod, fn):
                    fn_obj = getattr(mod, fn)
                    if callable(fn_obj):
                        logger.info("AutoAdapter(%s): located ingest callable '%s' in %s", self.pipeline_id, fn, mod_name)
                        return fn_obj
            for name, obj in inspect.getmembers(mod, inspect.isfunction):
                if name in ingest_names:
                    logger.info("AutoAdapter(%s): located ingest callable '%s' in %s (inspected)", self.pipeline_id, name, mod_name)
                    return obj
        logger.warning("AutoAdapter(%s): no ingest callable found; will fallback to MockAdapter", self.pipeline_id)
        return None

    async def chat(self, session_id: str, message: str, options: dict):
        if self._callable:
            logger.debug("AutoAdapter(%s): calling chat callable %s", self.pipeline_id, getattr(self._callable, "__name__", self._callable))
            try:
                res = self._callable(session_id, message, options)
                if asyncio.iscoroutine(res):
                    res = await res
                if isinstance(res, dict):
                    return res
                return {"answer": str(res), "sources": [], "chunks": [], "debug": {}}
            except Exception as e:
                logger.exception("AutoAdapter(%s): chat callable raised exception", self.pipeline_id)
                return {"answer": "(adapter error) " + str(e), "sources": [], "chunks": [], "debug": {"error": str(e)}}
        logger.info("AutoAdapter(%s): using MockAdapter.chat", self.pipeline_id)
        return await self._mock.chat(session_id, message, options)

    async def ingest(self, session_id: str, file_paths: list, options: dict):
        if self._ingest:
            logger.debug("AutoAdapter(%s): calling ingest callable %s", self.pipeline_id, getattr(self._ingest, "__name__", self._ingest))
            try:
                res = self._ingest(session_id, file_paths, options)
                if asyncio.iscoroutine(res):
                    res = await res
                if isinstance(res, dict):
                    return res
                return {"status": "ok", "detail": str(res)}
            except Exception as e:
                logger.exception("AutoAdapter(%s): ingest callable raised exception", self.pipeline_id)
                return {"status": "error", "error": str(e)}
        logger.info("AutoAdapter(%s): using MockAdapter.ingest", self.pipeline_id)
        return await self._mock.ingest(session_id, file_paths, options)
