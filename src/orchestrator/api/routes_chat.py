# # src/orchestrator/api/routes_chat.py
# from typing import Optional, Dict, Any
# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from ..core.registry import registry
# from ..core import sessions as sessions_mod
# import time

# router = APIRouter()


# class ChatRequest(BaseModel):
#     session_id: Optional[str] = None
#     message: str
#     options: Optional[Dict[str, Any]] = {}


# @router.post("/pipelines/{pipeline_id}/chat")
# async def pipeline_chat(pipeline_id: str, body: ChatRequest):
#     adapter = registry.get_adapter(pipeline_id)
#     if adapter is None:
#         raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")

#     # ensure session exists (optional: create if missing)
#     if body.session_id:
#         sessions_mod.touch_session(body.session_id)

#     start = time.time()
#     try:
#         result = await adapter.chat(body.session_id, body.message, body.options or {})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Adapter error: {e}")

#     total_ms = int((time.time() - start) * 1000)
#     # normalize result shape
#     normalized = {
#         "answer": result.get("answer") if isinstance(result, dict) else str(result),
#         "sources": result.get("sources", []) if isinstance(result, dict) else [],
#         "chunks": result.get("chunks", []) if isinstance(result, dict) else [],
#         "debug": result.get("debug", {}) if isinstance(result, dict) else {},
#     }
#     normalized["debug"].setdefault("pipeline_id", pipeline_id)
#     normalized["debug"].setdefault("total_ms", total_ms)
#     return normalized


###---------###

# src/orchestrator/api/routes_chat.py
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..core.registry import registry
from ..core import sessions as sessions_mod
import time
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    options: Optional[Dict[str, Any]] = {}


@router.post("/pipelines/{pipeline_id}/chat")
async def pipeline_chat(pipeline_id: str, body: ChatRequest):
    logger.info(f"📨 ORCHESTRATOR: Chat request - pipeline: {pipeline_id}, body.session_id: {body.session_id}")
    adapter = registry.get_adapter(pipeline_id)

    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")

    # Ensure session exists (create if missing)
    if body.session_id:
        sessions_mod.touch_session(body.session_id)
        session_id = body.session_id
        logger.info(f"✅ Using existing session: {session_id}")
    else:
        session_id = sessions_mod.create_session()["session_id"]
        logger.info(f"🆕 Created new session: {session_id}")

    logger.info(f"🔀 Calling adapter.chat with session_id: {session_id}, options: {body.options}")
    start = time.time()
    result = await adapter.chat(session_id, body.message, body.options or {})
    total_ms = int((time.time() - start) * 1000)

    # Normalize output
    normalized = {
        "session_id": session_id,
        "answer": result.get("answer") if isinstance(result, dict) else str(result),
        "sources": result.get("sources", []) if isinstance(result, dict) else [],
        "chunks": result.get("chunks", []) if isinstance(result, dict) else [],
        "debug": result.get("debug", {}) if isinstance(result, dict) else {},
    }
    normalized["debug"].setdefault("pipeline_id", pipeline_id)
    normalized["debug"].setdefault("total_ms", total_ms)
    return normalized
