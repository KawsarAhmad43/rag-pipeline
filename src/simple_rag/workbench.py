# src/simple_rag/workbench.py
from __future__ import annotations

import os
import sys
import asyncio
import importlib
from pathlib import Path
from typing import Any


# Repo root: .../rag_pipelines
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "workbench"


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _session_paths(pipeline_id: str, session_id: str) -> dict[str, Path]:
    uploads = DATA_ROOT / "uploads" / pipeline_id / session_id
    vectordb = DATA_ROOT / "vectorstores" / pipeline_id / session_id
    chatmem = DATA_ROOT / "chatmemory" / pipeline_id / session_id
    logs = DATA_ROOT / "logs" / pipeline_id / session_id
    _ensure_dirs(uploads, vectordb, chatmem, logs)
    return {
        "uploads": uploads,
        "vectordb": vectordb,
        "chatmem": chatmem,
        "logs": logs,
    }


def _bind_simple_rag_config_env(session_id: str) -> dict[str, str]:
    """
    Override Simple RAG config paths PER SESSION without editing its files.
    """
    paths = _session_paths("simple_rag", session_id)

    # These env var names already exist in src/simple_rag/config.py
    os.environ["SIMPLE_VECTOR_DB_PATH"] = str(paths["vectordb"])
    os.environ["SIMPLE_CHAT_MEMORY_PATH"] = str(paths["chatmem"])
    os.environ["SIMPLE_LOG_PATH"] = str(paths["logs"])

    return {k: str(v) for k, v in paths.items()}


def _alias_simple_rag_config_module() -> None:
    """
    Simple RAG internals do: `from config import config`
    In orchestrator context, Python resolves `config` to repo_root/config/ (wrong).
    Fix: alias module name `config` -> `src.simple_rag.config` in sys.modules.
    """
    sr_config_mod = importlib.import_module("src.simple_rag.config")
    sys.modules["config"] = sr_config_mod  # critical alias


def _lazy_import_simple_rag_internals():
    """
    Import internal modules only when called (not at orchestrator startup).
    Returns references we need for ingest/chat.
    """
    _alias_simple_rag_config_module()

    # Now safe to import internals that use `from config import config`
    pdf_processor = importlib.import_module("src.simple_rag.utils.pdf_processor")
    db_mod = importlib.import_module("src.simple_rag.utils.database")
    agent_mod = importlib.import_module("src.simple_rag.agents.agent")
    reranker_mod = importlib.import_module("src.simple_rag.app.reranker.reranker")
    embed_mod = importlib.import_module("src.simple_rag.app.embedding.embedding")

    return {
        "process_pdf": pdf_processor.process_pdf,
        "db": db_mod.db,
        "search_documents": agent_mod.search_documents,
        "create_context_from_docs": agent_mod.create_context_from_docs,
        "get_agent_response": agent_mod.get_agent_response,
        "rank_documents": reranker_mod.rank_documents,
        "get_embedding_model": embed_mod.get_embedding_model,
    }


async def workbench_ingest(session_id: str, file_paths: list[str], options: dict[str, Any]):
    """
    Real ingest:
      - session-scoped persistence paths (vector db + chat memory)
      - parse PDFs -> chunks
      - embed chunks
      - upsert into Chroma via existing Simple RAG db.store_document()
    """
    session_paths = _bind_simple_rag_config_env(session_id)
    mods = _lazy_import_simple_rag_internals()

    ingested = []
    total_chunks = 0

    for fp in file_paths:
        p = Path(fp)
        if not p.exists():
            continue

        # Copy into session uploads (optional but helpful for traceability)
        upload_dst = Path(session_paths["uploads"]) / p.name
        if p.resolve() != upload_dst.resolve():
            try:
                upload_dst.write_bytes(p.read_bytes())
            except Exception:
                # If copy fails, still try reading original
                upload_dst = p

        # Process PDF
        try:
            with open(upload_dst, "rb") as f:
                docs, embeddings = mods["process_pdf"](f)
        except Exception as e:
            ingested.append({"file": str(p), "status": "failed", "error": str(e)})
            continue

        # Store into vector db (Simple RAG's existing DB wrapper)
        try:
            for d, emb in zip(docs, embeddings):
                # store_document(text, embedding)
                mods["db"].store_document(d.page_content, emb)
            total_chunks += len(docs)
            ingested.append({"file": str(p), "status": "ok", "chunks": len(docs)})
        except Exception as e:
            ingested.append({"file": str(p), "status": "failed", "error": str(e)})

        # yield to event loop
        await asyncio.sleep(0)

    return {
        "status": "ok",
        "ingested_files": len(ingested),
        "details": {
            "session_id": session_id,
            "paths": session_paths,
            "files": ingested,
            "total_chunks": total_chunks,
        },
    }


async def workbench_chat(session_id: str, message: str, options: dict[str, Any]):
    """
    Real chat:
      - session-scoped persistence
      - embed query -> retrieve -> rerank -> build context
      - call Simple RAG agent to generate final answer
      - return sources + debug
    """
    _bind_simple_rag_config_env(session_id)
    mods = _lazy_import_simple_rag_internals()

    query = (message or "").strip()
    if not query:
        return {"answer": "Empty message.", "sources": [], "debug": {"session_id": session_id}}

    use_pdf = bool(options.get("use_pdf", True))

    sources = []
    context = None

    if use_pdf:
        docs = mods["search_documents"](query, context_type="pdf")
        if docs:
            ranked = mods["rank_documents"](query, docs)
            context = mods["create_context_from_docs"](ranked)

            # Build “sources” payload for UI
            for d in ranked:
                meta = getattr(d, "metadata", {}) or {}
                sources.append({
                    "title": meta.get("source", meta.get("file", "uploaded.pdf")),
                    "page": meta.get("page"),
                    "chunk_id": meta.get("chunk_id"),
                    "snippet": (d.page_content[:240] + "...") if len(d.page_content) > 240 else d.page_content,
                })

    # Final answer via existing Simple RAG agent (LLM)
    answer = mods["get_agent_response"](query, context)

    await asyncio.sleep(0)

    return {
        "answer": answer,
        "sources": sources,
        "debug": {
            "session_id": session_id,
            "use_pdf": use_pdf,
            "context_used": bool(context),
        },
    }
