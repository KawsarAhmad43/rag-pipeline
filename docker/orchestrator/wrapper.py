import sys
import os
import logging
import asyncio
import aiohttp
from typing import List, Dict, Any

# Ensure root is in path
sys.path.insert(0, os.getcwd())

# Import original modules
from src.orchestrator.adapters.base import BaseAdapter
from src.orchestrator.core.registry import Registry
from src.orchestrator.main import app
from fastapi import File, UploadFile, Form, HTTPException
import uvicorn
import requests
from typing import Optional, Literal

# --- Define HTTP Adapter ---
class HttpAdapter(BaseAdapter):
    def __init__(self, pipeline_id: str, display_name: str = None, capabilities: dict = None, service_url: str = None):
        super().__init__(pipeline_id, display_name, capabilities)
        self.service_url = service_url.rstrip("/")

    async def chat(self, session_id: str, message: str, options: dict) -> Dict[str, Any]:
        # Map specific endpoints for each pipeline
        endpoint_map = {
            "simple_rag": "/simple/chat/",
            "multimodal_rag": "/chat",
            "graph_rag": "/graph/chat/",
            "agentic_rag": "/api/v1/chat"
        }

        path = endpoint_map.get(self.pipeline_id, "/api/chat")
        url = f"{self.service_url}{path}"

        # Adjust payloads to match specific service expectations
        if self.pipeline_id == "simple_rag":
            payload = {"text": message, "pdf_context": options.get("pdf_context", False)}
        elif self.pipeline_id == "graph_rag":
            payload = {"prompt": message}
        elif self.pipeline_id == "multimodal_rag":
            payload = {"user_input": message, "session_id": session_id, "generation_mode": options.get("mode", "local")}
        elif self.pipeline_id == "agentic_rag":
             payload = {"user_input": message, "session_id": session_id}
        else:
            payload = {"session_id": session_id, "message": message, "options": options}

        async with aiohttp.ClientSession() as session:
            try:
                # Handle services that require Form Data (like Multimodal)
                if self.pipeline_id == "multimodal_rag":
                    form_data = aiohttp.FormData()
                    form_data.add_field("user_input", message)
                    form_data.add_field("session_id", session_id)
                    form_data.add_field("generation_mode", options.get("mode", "local"))

                    async with session.post(url, data=form_data) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return {"answer": data.get("response", ""), "sources": data.get("sources", [])}
                        else:
                            text = await resp.text()
                            logging.error(f"Multimodal Chat Error ({resp.status}): {text}")
                            return {"answer": f"Multimodal Error ({resp.status}): {text}", "sources": []}

                # Standard JSON services
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Normalize response for orchestrator
                        if self.pipeline_id == "simple_rag":
                            return {"answer": data.get("response", ""), "sources": []}
                        elif self.pipeline_id == "graph_rag":
                             return {"answer": data.get("response", ""), "sources": []}
                        elif self.pipeline_id == "agentic_rag":
                             return {"answer": data.get("response", ""), "sources": []}
                        elif "response" in data:
                             return {"answer": data["response"], "sources": data.get("sources", [])}
                        return data
                    else:
                        text = await resp.text()
                        logging.error(f"{self.pipeline_id} Chat Error ({resp.status}): {text}")
                        return {"answer": f"Error from {self.pipeline_id} ({resp.status}): {text}", "sources": []}
            except Exception as e:
                logging.exception(f"Connection error to {self.pipeline_id}")
                return {"answer": f"Connection error to {self.pipeline_id}: {str(e)}", "sources": [], "debug": {"url": url, "error": str(e)}}

    async def status(self, session_id: str) -> Dict[str, Any]:
        """Check status of the session/pipeline"""
        # Map specific status endpoints
        # Agentic RAG has a specific session status endpoint
        # Multimodal has one too (handled via proxy route or here if updated)
        # Simple/Graph might just check health or existence of file

        # --- NEW UNIFIED STATUS LOGIC ---
        # Goal: Always allow interaction unless service is down.

        # 1. Agentic RAG Status
        if self.pipeline_id == "agentic_rag":
            url = f"{self.service_url}/api/v1/session/status/{session_id}"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # Map 'has_document' to ready, BUT force True to allow chat
                            return {"ready": True, "details": data}
                        return {"ready": True, "error": f"Status check failed {resp.status}"}
            except Exception:
                return {"ready": True, "error": "Connection failed"}

        # 2. MultiModal RAG (Handled by proxy route, but good to have fallback here)
        if self.pipeline_id == "multimodal_rag":
             return {"ready": True}

        # 3. Simple & Graph RAG (No explicit status endpoint commonly used)
        # Assume always ready if container is up
        return {"ready": True}
        # --- END NEW STATUS LOGIC ---

    async def ingest(self, session_id: str, file_paths: list, options: dict) -> Dict[str, Any]:
        # --- NEW UNIFIED PROXY LOGIC ---
        if not file_paths:
            return {"status": "error", "message": "No files provided"}

        # Map services to their specific ingestion endpoints
        endpoint_map = {
            "simple_rag": "/simple/upload-pdf/",
            "graph_rag": "/graph/upload-pdf/",
            "agentic_rag": "/api/v1/upload_pdf",
            "multimodal_rag": "/initialize_session"
        }

        path = endpoint_map.get(self.pipeline_id)
        if not path:
            return {"status": "error", "message": f"Ingestion not supported for {self.pipeline_id}"}

        url = f"{self.service_url}{path}"
        file_path = file_paths[0]  # Standardize on first file for now
        filename = os.path.basename(file_path)

        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()

            # Map field names based on service requirements
            if self.pipeline_id == "multimodal_rag":
                data.add_field("session_id", session_id)
                file_field = "pdf_file"
            elif self.pipeline_id == "agentic_rag":
                data.add_field("session_id", session_id)
                file_field = "file"
            else: # simple_rag and graph_rag
                file_field = "file"

            try:
                with open(file_path, "rb") as f:
                    file_content = f.read()

                data.add_field(file_field, file_content, filename=filename, content_type="application/pdf")

                logging.info(f"Proxying file {filename} to {self.pipeline_id} at {url}")
                async with session.post(url, data=data) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        text = await resp.text()
                        logging.error(f"{self.pipeline_id} Ingest Error ({resp.status}): {text}")
                        return {"status": "error", "error": f"Backend error: {text}"}
            except Exception as e:
                logging.exception(f"Failed to proxy file to {self.pipeline_id}")
                return {"status": "error", "error": str(e)}

# --- New Proxy Routes ---

@app.get("/api/pipelines/{pipeline_id}/status/{session_id}")
async def proxy_pipeline_status(pipeline_id: str, session_id: str):
    # This route intercepts status calls correctly
    from src.orchestrator.core.registry import registry
    adapter = registry.get(pipeline_id)
    if not adapter:
        # Fallback if not found in registry (e.g. env var defined but not loaded yet?)
        # But allow specific handle for known ones if needed
        # For now, just generic error or verify registry
        return {"ready": False, "error": f"Pipeline {pipeline_id} not found"}

    if hasattr(adapter, "status"):
        return await adapter.status(session_id)
    return {"ready": True, "message": "No status check implemented"}


def inject_remote_adapters():
    # Helper to map pipeline_id to environment variable URL
    # format: <PIPELINE_ID>_URL
    from src.orchestrator.core.registry import registry

    # We want to OVERRIDE local adapters with HttpAdapters if URL is present
    pipelines = {
        "simple_rag": os.getenv("SIMPLE_RAG_URL"),
        "graph_rag": os.getenv("GRAPH_RAG_URL"),
        "multimodal_rag": os.getenv("MULTIMODAL_RAG_URL"),
        "agentic_rag": os.getenv("AGENTIC_RAG_URL")
    }

    for pid, url in pipelines.items():
        if url:
             print(f"Injecting HttpAdapter for {pid} at {url}")
             adapter = HttpAdapter(pipeline_id=pid, service_url=url)
             # Force register/overwrite in the internal dict
             registry._pipelines[pid] = adapter

# --- New Proxy Routes for Multimodal Voice ---

@app.post("/api/pipelines/multimodal_rag/chat")
async def proxy_chat(
    user_input: str = Form(...),
    session_id: str = Form(...),
    generation_mode: str = Form("local"),
    attachment: Optional[UploadFile] = File(None)
):
    url = os.getenv("MULTIMODAL_RAG_URL")
    if not url:
        raise HTTPException(status_code=503, detail="Multimodal RAG service not configured")

    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field("user_input", user_input)
        data.add_field("session_id", session_id)
        data.add_field("generation_mode", generation_mode)

        if attachment:
            content = await attachment.read()
            data.add_field("attachment", content, filename=attachment.filename, content_type=attachment.content_type)

        async with session.post(f"{url}/chat", data=data) as resp:
            if resp.status == 200:
                result = await resp.json()
                # Normalize for orchestrator expectation
                return {
                    "answer": result.get("response") or result.get("answer") or "",
                    "sources": result.get("sources") or result.get("context") or [],
                    "debug": {"backend_url": f"{url}/chat", "session_id": session_id, "raw_response": result}
                }
            else:
                text = await resp.text()
                raise HTTPException(status_code=resp.status, detail=f"Multimodal error: {text}")


@app.post("/api/pipelines/multimodal_rag/voice_chat")
async def proxy_voice_chat(
    session_id: str = Form(...),
    generation_mode: str = Form("local"),
    voice: str = Form("af_sarah"),
    audio_file: UploadFile = File(...)
):
    url = os.getenv("MULTIMODAL_RAG_URL")
    if not url:
        raise HTTPException(status_code=503, detail="Multimodal RAG service not configured")

    # Forward the multipart request
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field("session_id", session_id)
        data.add_field("generation_mode", generation_mode)
        data.add_field("voice", voice)

        content = await audio_file.read()
        data.add_field("audio_file", content, filename=audio_file.filename, content_type=audio_file.content_type)

        async with session.post(f"{url}/voice_chat", data=data) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                text = await resp.text()
                raise HTTPException(status_code=resp.status, detail=f"Multimodal error: {text}")

@app.post("/api/pipelines/multimodal_rag/tts")
async def proxy_tts(
    user_input: str = Form(...),
    session_id: str = Form(...),
    generation_mode: str = Form("local"),
    voice: str = Form("af_sarah")
):
    url = os.getenv("MULTIMODAL_RAG_URL")
    if not url:
        raise HTTPException(status_code=503, detail="Multimodal RAG service not configured")

    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field("user_input", user_input)
        data.add_field("session_id", session_id)
        data.add_field("generation_mode", generation_mode)
        data.add_field("voice", voice)

        async with session.post(f"{url}/tts", data=data) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                text = await resp.text()
                raise HTTPException(status_code=resp.status, detail=f"Multimodal error: {text}")

@app.get("/api/pipelines/multimodal_rag/status/{session_id}")
async def proxy_status(session_id: str):
    url = os.getenv("MULTIMODAL_RAG_URL")
    if not url:
        raise HTTPException(status_code=503, detail="Multimodal RAG service not configured")

    target_url = f"{url}/status/{session_id}"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(target_url) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    return {"ready": False, "error": f"Backend status {resp.status}"}
        except Exception as e:
            return {"ready": False, "error": str(e)}

@app.delete("/api/sessions/{session_id}")
async def proxy_delete_session(session_id: str):
    """Proxy session deletion to sub-services (currently MultiModal)."""
    # 1. Clear from MultiModal RAG
    url = os.getenv("MULTIMODAL_RAG_URL")
    if url:
        async with aiohttp.ClientSession() as session:
            try:
                # Note: multimodal backend uses same path structure /api/sessions/{session_id}
                async with session.delete(f"{url}/api/sessions/{session_id}") as resp:
                    logger.info(f"MultiModal session reset ({session_id}): {resp.status}")
            except Exception as e:
                logger.error(f"Failed to reset MultiModal session {session_id}: {e}")

    # 2. Cleanup Orchestrator local storage if any
    # (Orchestrator usually stores files in data/workbench/uploads/{pipeline}/{session_id})
    try:
        import shutil
        upload_base = "/app/data/workbench/uploads"
        # Since we don't necessarily know the pipeline here, we check all
        if os.path.exists(upload_base):
            for pipeline_dir in os.listdir(upload_base):
                sess_dir = os.path.join(upload_base, pipeline_dir, session_id)
                if os.path.exists(sess_dir):
                    shutil.rmtree(sess_dir)
                    logger.info(f"Cleaned up orchestrator files for {session_id} in {pipeline_dir}")
    except Exception as e:
        logger.error(f"Orchestrator cleanup error: {e}")

    return {"status": "ok", "message": f"Session {session_id} reset signals sent."}

if __name__ == "__main__":
    # Inject adapters
    inject_remote_adapters()

    # Run server
    print("Starting Orchestrator (Wrapper Mode)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
