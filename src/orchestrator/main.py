# src/orchestrator/main.py
from pathlib import Path
import sys

import logging
logging.basicConfig(level=logging.DEBUG)

# ensure repo root is on sys.path so imports like "src.simple_rag" work
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, Form

from .api.routes_pipelines import router as pipelines_router
from .api.routes_session import router as sessions_router
from .api.routes_chat import router as chat_router
from .api.routes_ingest import router as ingest_router


app = FastAPI(title="RAG Orchestrator")

# Very permissive CORS for dev. Lock down for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Multimodal RAG proxy routes (must be before more generic routers to take precedence)
import os
import aiohttp
from typing import Optional
from fastapi import HTTPException

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

# API routers under /api
app.include_router(pipelines_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(ingest_router, prefix="/api")


# Optionally serve your static frontend from the same process.
static_dir = ROOT / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
