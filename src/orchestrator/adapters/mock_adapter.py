# src/orchestrator/adapters/mock_adapter.py
import asyncio
from .base import BaseAdapter
import random

SAMPLE = [
    "Mock: I found a passage that looks relevant and summarized it for you.",
    "Mock: Short answer generated from stored documents.",
    "Mock: I matched documents A and B; main point: ..."
]

class MockAdapter(BaseAdapter):
    def __init__(self, pipeline_id: str, display_name: str = None, capabilities: dict = None, latency_ms: int = 400):
        super().__init__(pipeline_id, display_name, capabilities)
        self.latency_ms = latency_ms

    async def chat(self, session_id: str, message: str, options: dict):
        await asyncio.sleep(self.latency_ms / 1000.0)
        text = random.choice(SAMPLE)
        return {
            "answer": f"{text} (echo: {message})",
            "sources": [{"title": "mock_doc.pdf", "page": 2, "snippet": "Mock snippet..."}],
            "chunks": [],
            "debug": {"mock": True, "latency_ms": self.latency_ms},
        }

    # inside MockAdapter
    async def ingest(self, session_id: str, file_paths: list, options: dict):
        await asyncio.sleep(0.2)
        return {"status": "ok", "ingested_files": len(file_paths), "mock": True}
