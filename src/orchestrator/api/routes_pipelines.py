# src/orchestrator/api/routes_pipelines.py
from fastapi import APIRouter, HTTPException
from ..core.registry import registry
from ..schemas.pipeline_schema import (
    ChatRequest, ChatResponse,
    IngestRequest, IngestResponse,
)

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


@router.get("")
def list_pipelines():
    return {"pipelines": registry.list_pipeline_ids()}


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    try:
        adapter = registry.get(req.pipeline_id)
        return await adapter.ingest(req.session_id, req.file_paths, req.options)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        adapter = registry.get(req.pipeline_id)
        return await adapter.chat(req.session_id, req.message, req.options)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
