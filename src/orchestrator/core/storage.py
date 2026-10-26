# src/orchestrator/core/storage.py
from pathlib import Path
from uuid import uuid4
import os
from fastapi import UploadFile

# data/workbench/uploads/<pipeline>/<session>/
BASE = Path(__file__).resolve().parents[3] / "data" / "workbench" / "uploads"
BASE.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name or "")
    # basic cleanup
    name = name.replace(" ", "_")
    return name or f"file_{uuid4().hex}"


async def save_upload(pipeline_id: str, session_id: str, upload: UploadFile) -> str:
    dest_dir = BASE / pipeline_id / session_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = _sanitize_filename(upload.filename)
    dest_path = dest_dir / filename
    content = await upload.read()
    dest_path.write_bytes(content)
    return str(dest_path)
