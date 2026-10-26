# src/orchestrator/adapters/base.py
from typing import Dict, Any

class BaseAdapter:
    def __init__(self, pipeline_id: str, display_name: str = None, capabilities: Dict[str, Any] = None):
        self.pipeline_id = pipeline_id
        self.display_name = display_name or pipeline_id
        self.capabilities = capabilities or {}
        self.description = ""

    async def chat(self, session_id: str, message: str, options: Dict[str, Any]):
        """
        Return either a dict with keys: answer, sources, chunks, debug
        or a string (treated as answer).
        """
        raise NotImplementedError
