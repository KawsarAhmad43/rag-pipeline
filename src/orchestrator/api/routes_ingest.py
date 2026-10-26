# src/orchestrator/api/routes_ingest.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional
import asyncio

from ..core.registry import registry
from ..core.sessions import create_session, touch_session
from ..core.storage import save_upload

router = APIRouter()


@router.post("/pipelines/{pipeline_id}/ingest")
async def pipeline_ingest(
    pipeline_id: str,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    options: Optional[str] = Form(None),
):
    """
    Ingest uploaded files for a pipeline.
    - Stores files under data/workbench/uploads/{pipeline_id}/{session_id}/
    - Calls adapter.ingest(session_id, file_paths, options) if available.
    Returns JSON with uploaded paths and adapter result.
    """
    adapter = registry.get_adapter(pipeline_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")

    # ensure session
    if not session_id:
        session = create_session()
        session_id = session["session_id"]
    else:
        touch_session(session_id)

    saved_paths = []
    for upload in files:
        saved = await save_upload(pipeline_id, session_id, upload)
        saved_paths.append(saved)

    ingest_result = None
    if hasattr(adapter, "ingest") and callable(getattr(adapter, "ingest")):
        try:
            res = adapter.ingest(session_id, saved_paths, options or {})
            if asyncio.iscoroutine(res):
                ingest_result = await res
            else:
                ingest_result = res
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Adapter ingest error: {e}")
    else:
        ingest_result = {
            "status": "ok",
            "ingested_files": len(saved_paths),
            "details": "Files saved; adapter.ingest not implemented",
        }

    return {
        "status": "ok",
        "session_id": session_id,
        "uploaded_files": saved_paths,
        "ingest_result": ingest_result,
    }
