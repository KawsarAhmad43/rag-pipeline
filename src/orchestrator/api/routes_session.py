# src/orchestrator/api/routes_sessions.py

from fastapi import APIRouter, HTTPException
from ..core.sessions import create_session, touch_session
from ..core.registry import registry
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/sessions")
async def create_session_route():
    """
    Create a new session_id for the workbench.
    """
    return create_session()


@router.delete("/sessions/{session_id}")
async def delete_session_route(session_id: str):
    """
    Delete/reset a session across all RAG pipelines.
    This will clear vector databases and session data for all RAG types.
    """
    results = {}
    errors = []

    # Get all pipeline IDs
    pipeline_ids = registry.list_pipeline_ids()

    for pipeline_id in pipeline_ids:
        try:
            adapter = registry.get_adapter(pipeline_id)
            if adapter and hasattr(adapter, 'reset_session'):
                result = await adapter.reset_session(session_id)
                results[pipeline_id] = {"status": "success", "result": result}
                logger.info(f"Session {session_id} reset for {pipeline_id}")
            else:
                results[pipeline_id] = {"status": "skipped", "reason": "No reset method available"}
        except Exception as e:
            error_msg = f"Failed to reset {pipeline_id}: {str(e)}"
            errors.append(error_msg)
            results[pipeline_id] = {"status": "error", "error": str(e)}
            logger.error(error_msg)

    # Return summary
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    return {
        "status": "completed" if not errors else "partial",
        "session_id": session_id,
        "results": results,
        "summary": {
            "total": len(pipeline_ids),
            "success": success_count,
            "failed": len(errors),
            "errors": errors
        }
    }
