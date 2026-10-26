# src/orchestrator/schemas/pipeline_schemas.py
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict


class ChatRequest(BaseModel):
    pipeline_id: str = Field(..., examples=["simple_rag"])
    session_id: str = Field(..., examples=["abc123"])
    message: str
    options: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    pipeline_id: str = Field(..., examples=["simple_rag"])
    session_id: str = Field(..., examples=["abc123"])
    file_paths: List[str]
    options: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    status: str
    ingested_files: int
    details: dict = Field(default_factory=dict)
