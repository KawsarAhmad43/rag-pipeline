# src/orchestrator/core/sessions.py
from uuid import uuid4
from datetime import datetime

_sessions = {}  # in-memory session store; replace with redis/db for prod


def create_session():
    sid = str(uuid4())
    _sessions[sid] = {"created_at": datetime.utcnow().isoformat()}
    return {"session_id": sid}


def touch_session(session_id: str):
    if session_id not in _sessions:
        _sessions[session_id] = {"created_at": datetime.utcnow().isoformat()}
    return _sessions[session_id]
