from app.embedding.embedding import get_embedding_model
from app.llm.llm import get_llm
from app.reranker.reranker import get_reranker, rank_documents
from agents.agent import get_agent_response, search_documents, create_context_from_docs
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles  # Import StaticFiles
from fastapi.responses import FileResponse  # Import FileResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from utils.database import db
from utils.logging import logger
from utils.vector_search import vector_search
from utils.pdf_processor import process_pdf
import os
import psutil
import torch
import gc
import numpy as np
from pydantic import BaseModel
from typing import Optional
import traceback
import re
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event for pre-loading models"""
    logger.info("🚀 Pre-loading models on startup...")
    try:
        initialize_models()
        logger.info("✅ Models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load models on startup: {e}")
        # We don't raise here to allow the server to start even if models fail
        # (though mostly it will fail anyway during initialization)
    yield
    logger.info("🛑 Shutting down Simple RAG API...")

class QueryRequest(BaseModel):
    text: str
    pdf_context: bool = False
    pdf_analyzer: bool = False
    web_search: bool = False

# app = FastAPI(
#     title="Simple RAG API",
#     description="API for the Simple RAG application with /simple/ prefix",
#     version="1.0.0"
# )

app = FastAPI(
    title="Simple RAG API",
    description="API for the Simple RAG application with /simple/ prefix",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/frontend/static"), name="static")

# Serve index.html at root
@app.get("/")
async def serve_index():
    logger.info("Serving index.html at root")
    return FileResponse("app/frontend/static/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Initializing FastAPI application")

llm = None
embedding_model = None
reranker = None

def initialize_models():
    global llm, embedding_model, reranker
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Memory cleared before initial model load/check")

        if embedding_model is None:
            logger.debug("Attempting to get embedding model")
            embedding_model = get_embedding_model()
            logger.info("Embedding model initialized/retrieved successfully")

        if reranker is None:
            logger.debug("Attempting to get reranker model")
            reranker_model, _ = get_reranker()
            reranker = reranker_model
            logger.info("Reranker model initialized/retrieved successfully")

        if llm is None:
            logger.debug("Attempting to get LLM")
            llm = get_llm()
            logger.info("LLM initialized/retrieved successfully")

    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}. Memory usage: CPU {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        if torch.cuda.is_available():
            logger.error(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB used, {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB max allocated")
        raise RuntimeError(f"Model initialization failed: {str(e)}")

@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    try:
        if llm is None or embedding_model is None or reranker is None:
            initialize_models()
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Simple RAG API",
        version="1.0.0",
        description="API for the Simple RAG application with /simple/ prefix",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.get("/docs", include_in_schema=False)
async def get_documentation():
    logger.info("Serving Swagger UI at /docs")
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Simple RAG API Docs"
    )

@app.post("/simple/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), session_id: str = Form(None)):
    try:
        if embedding_model is None:
            initialize_models()

        logger.info(f"📥 Simple RAG: Received PDF upload request: {file.filename}")

        # Define directory to save PDFs
        pdf_storage_dir = "data/uploads"
        os.makedirs(pdf_storage_dir, exist_ok=True)

        # Save the uploaded PDF to disk
        pdf_path = os.path.join(pdf_storage_dir, file.filename)
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"Saved uploaded PDF to {pdf_path}")

        # Reset file pointer to the beginning
        file.file.seek(0)
        docs, embeddings = process_pdf(file.file, filename=file.filename)
        if docs:
            logger.info(f"📤 Storing {len(docs)} chunks into Simple RAG knowledge base")
            for i, doc in enumerate(docs):
                db.store_document(doc.page_content, embeddings[i], metadata=doc.metadata)
                if i == 0:  # Log first chunk as sample
                    logger.info(f"📝 Sample chunk: {doc.page_content[:150]}...")
                    logger.info(f"📝 Sample metadata: {doc.metadata}")
            logger.info(f"✅ Simple RAG: Uploaded and stored PDF: {file.filename}, {len(docs)} chunks indexed")
            return {
                "message": f"PDF processed and stored successfully - {len(docs)} chunks extracted",
                "chunk_count": len(docs),
                "status": "success"
            }
        else:
            logger.warning(f"No content extracted from PDF: {file.filename}")
            return {
                "message": "No content extracted from PDF",
                "chunk_count": 0,
                "status": "warning"
            }
    except Exception as e:
        logger.error(f"Error processing PDF upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/simple/chat/")
async def query(request: QueryRequest):
    try:
        if llm is None or embedding_model is None:
            initialize_models()

        text = request.text.strip()
        pdf_context = request.pdf_context
        pdf_analyzer = request.pdf_analyzer
        web_search = request.web_search

        if not text:
            raise HTTPException(status_code=400, detail="Query text cannot be empty")

        logger.info(f"📥 Simple RAG: Received query - '{text[:50]}...', pdf_context: {pdf_context}, pdf_analyzer: {pdf_analyzer}, web_search: {web_search}")

        # Check for meta-queries
        meta_query_pattern = re.compile(r"(previous|last)\s+query", re.IGNORECASE)
        if meta_query_pattern.search(text):
            logger.debug(f"Detected meta-query: '{text}'")
            memory_response = db.search_chat_memory(text, embedding_model, context_type="not_pdf")
            if memory_response:
                logger.info(f"Returning meta-query response: {memory_response[:50]}...")
                return {"response": memory_response, "context": "memory"}
            else:
                logger.info("No previous query found for meta-query")
                return {"response": "No previous query found in chat memory.", "context": "memory"}

        context_type = "simple_rag_pdf" if pdf_context else "simple_rag_general"

        # Check chat memory for Simple RAG (RAG type specific memory)
        memory_response = db.search_chat_memory(text, embedding_model, context_type=context_type)
        if memory_response:
            logger.info(f"✅ Query matched existing Simple RAG chat memory: {memory_response[:50]}...")
            return {"response": memory_response, "context": "memory"}

        # Retrieve 5-10 documents for better context coverage (matching Agentic RAG)
        search_k = 10 if any(keyword in text.lower() for keyword in ["summarize", "summary", "overview", "about"]) else 5
        documents = search_documents(text, context_type="pdf", k=search_k) if pdf_context else []

        # Enhanced logging for document retrieval
        logger.info(f"📊 Simple RAG document retrieval - pdf_context: {pdf_context}, k: {search_k}, found {len(documents)} documents")
        if documents:
            for i, doc in enumerate(documents[:3]):  # Log first 3 for brevity
                doc_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                logger.info(f"📄 Document {i+1} preview: {doc_content[:200]}...")

        if pdf_context and documents:
            ranked_docs = rank_documents(text, documents)
            context = create_context_from_docs(ranked_docs)
            logger.info(f"✅ Using PDF context with {len(ranked_docs)} ranked documents")
            logger.debug(f"Retrieved context for PDF query: {context[:200]}...")
        else:
            context = "No relevant PDF content found." if pdf_context else "No context provided."
            logger.warning(f"⚠️ No relevant context for query - pdf_context: {pdf_context}, docs found: {len(documents)}")

        # Handle different tool modes
        if web_search:
            # Web Search tool enabled - fetch information from web
            from agents.agent import get_web_search_response
            response = get_web_search_response(text, context if pdf_context and documents else None)
            logger.info(f"🌐 Web Search: Generated response with web information")
        elif pdf_analyzer and pdf_context and documents:
            # PDF Analyzer tool enabled - advanced PDF analysis
            from agents.agent import get_pdf_analyzer_response
            response = get_pdf_analyzer_response(text, ranked_docs)
            logger.info(f"📊 PDF Analyzer: Generated advanced analysis for query")
        else:
            # Standard RAG response
            response = get_agent_response(text, context)

        if not response or response.strip() == "":
            response = "Sorry, I didn't get that. Can you rephrase?"

        if response and not response.startswith("Error:") and not response.startswith("I'm sorry, I encountered an error") and "No relevant information found" not in response:
            db.store_chat_memory(text, response, embedding_model, context_type=context_type)
            logger.debug(f"Stored chat memory with context_type: {context_type}")

        logger.info(f"Generated response for query: '{text[:30]}...': {response[:50]}...")
        return {"response": response, "context": context_type}

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "response": f"I'm sorry, something went wrong while processing your query. Please try again.",
            "context": "error"
        }

@app.get("/simple/chat-memory/")
async def get_chat_memory():
    try:
        logger.info("Retrieving all chat memory entries")
        # Retrieve all chat memory entries
        results = db.chat_memory.get(
            include=["documents", "metadatas"]
        )

        if not results["ids"]:
            logger.debug("Chat memory is empty")
            return {"chat_memory": [], "message": "No chat memory entries found."}

        # Sort entries by ID (ascending for oldest first)
        sorted_indices = sorted(range(len(results["ids"])), key=lambda i: int(results["ids"][i]))
        chat_memory = []

        for idx in sorted_indices:
            document = results["documents"][idx]
            metadata = results["metadatas"][idx]
            parts = document.split("\nA: ")
            if len(parts) > 1:
                query = parts[0].replace("Q: ", "").strip()
                response = parts[1].strip()
                context_type = metadata.get("context_type", "unknown")
                chat_memory.append({
                    "id": results["ids"][idx],
                    "query": query,
                    "response": response,
                    "context_type": context_type
                })
            else:
                logger.warning(f"Malformed chat memory document with ID {results['ids'][idx]}: {document[:50]}...")

        logger.debug(f"Retrieved {len(chat_memory)} chat memory entries")
        return {"chat_memory": chat_memory, "message": f"Retrieved {len(chat_memory)} chat memory entries."}

    except Exception as e:
        logger.error(f"Error retrieving chat memory: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "chat_memory": [],
            "message": f"Error retrieving chat memory: {str(e)}"
        }
