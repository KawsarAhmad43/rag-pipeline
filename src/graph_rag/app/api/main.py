from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import psutil
import torch
import gc
import traceback
import re
import io
from contextlib import asynccontextmanager
from app.embedding.embedding import get_embedding_model
from app.llm.llm import get_llm
from app.reranker.reranker import get_reranker
from agents.agent import get_agent_response, search_documents
from app.evaluation.ragas_evaluator import evaluate_ragas, get_ragas_evaluation_history
from utils.database import db
from utils.logging import logger
from utils.pdf_processor import process_pdf
from utils.guard_rails import guard_rails
from app.graph.graph import graph_db
import hashlib
from datetime import datetime

# QueryRequest now only takes a prompt. paper_id is removed.
class QueryRequest(BaseModel):
    prompt: str

llm = None
embedding_model = None
reranker = None

def initialize_models():
    global llm, embedding_model, reranker
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if embedding_model is None:
            embedding_model = get_embedding_model()
        if reranker is None:
            reranker_model, _ = get_reranker()
            reranker = reranker_model
        if llm is None:
            llm = get_llm()
    except Exception as e:
        raise RuntimeError(f"Model initialization failed: {str(e)}")

def is_greeting(query: str) -> bool:
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings", "how are you"]
    query_lower = query.lower().strip()
    # Check if the query starts with a greeting or is entirely a greeting
    query_words = query_lower.split()
    if not query_words:
        return False
    first_word = query_words[0]
    # Only consider it a greeting if the first word is a greeting and the query is short or ends with a question
    is_greeting_start = first_word in greetings
    is_short_query = len(query_words) <= 3  # Arbitrary threshold for short queries
    is_question = query_lower.endswith("?")
    return is_greeting_start and (is_short_query or (not is_question and len(query_lower) < 20))

def sanitize_response(response: str) -> str:
    """Remove citation markers like (doc_id=..., chunk_id=...) from the response."""
    return re.sub(r'\s*\(doc_id=[^)]+, chunk_id=[^)]+\)\s*', '', response).strip()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Initializing models...")
    initialize_models()
    yield
    logger.info("Application shutdown.")
    global llm, embedding_model, reranker
    del llm
    del embedding_model
    del reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="Graph RAG API",
    description="API for the Graph RAG application",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/frontend/static"), name="static")
logger.info("Initializing FastAPI application")

@app.get("/health")
async def health_check():
    try:
        if llm is None or embedding_model is None or reranker is None:
            raise RuntimeError("Models are not initialized.")
        graph_db.health_check()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Graph RAG API",
        version="1.0.0",
        description="API for the Graph RAG application",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.get("/docs", include_in_schema=False)
async def get_documentation():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Graph RAG API Docs")

@app.get("/graph/")
async def get_index():
    return FileResponse("app/frontend/static/index.html")

@app.get("/graph/upload/")
async def get_upload():
    return FileResponse("app/frontend/static/pdf_upload.html")

@app.post("/graph/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        # Define upload directory and create if it doesn't exist
        upload_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'uploaded_docs'))
        os.makedirs(upload_dir, exist_ok=True)

        # Save the uploaded file
        pdf_hash = hashlib.md5(content).hexdigest()
        file_path = os.path.join(upload_dir, f"{pdf_hash}.pdf")
        with open(file_path, 'wb') as f:
            f.write(content)
        logger.info(f"PDF saved to {file_path}")

        pdf_hash = hashlib.md5(content).hexdigest()
        cypher_check = f"MATCH (p:PDF {{hash: '{pdf_hash}'}}) RETURN p.doc_id LIMIT 1"
        existing = graph_db.query_graph(cypher_check)
        if existing:
            result_str = existing[0]
            if "'doc_id': " in result_str:
                doc_id_start = result_str.find("'doc_id': '") + len("'doc_id': '")
                doc_id_end = result_str.find("'", doc_id_start)
                doc_id = result_str[doc_id_start:doc_id_end]
                return {"message": "PDF already exists", "doc_id": doc_id}

        doc_id = f"PDF_{pdf_hash[:8]}"
        with graph_db.get_session() as session:
            session.run(
                """
                MERGE (p:PDF {doc_id: $doc_id, hash: $hash})
                SET p.title = $title, p.uploaded_at = $uploaded_at, p.file_path = $file_path
                """,
                doc_id=doc_id,
                title=file.filename,
                hash=pdf_hash,
                uploaded_at=str(datetime.now()),
                file_path=file_path
            )

        file_stream = io.BytesIO(content)
        docs, embeddings = process_pdf(file_stream, doc_id=doc_id)
        if not docs or not embeddings:
            return {"message": "No content extracted from PDF"}

        for doc, embedding in zip(docs, embeddings):
            metadata = doc.metadata if doc.metadata else {"source": "pdf", "filename": file.filename}
            db.store_document(doc.page_content, embedding, metadata)

        return {
            "message": f"PDF processed and stored successfully - {len(docs)} chunks extracted",
            "doc_id": doc_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/graph/chat/")
async def query(request: QueryRequest):
    try:
        text = request.prompt.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Query text cannot be empty")

        # Guard rails check
        is_safe, reason = guard_rails.check_query(text)
        if not is_safe:
            safe_response = guard_rails.get_safe_response(reason)
            return {
                "response": safe_response,
                "final_response": safe_response,
                "context": "guard_rails",
                "source_paper": None,
                "search_metadata": {"type": "guard_rails_blocked", "reason": reason},
            }

        if not db.has_knowledge_base():
            return {
                "response": "No Knowledge base available. Please upload a document first.",
                "final_response": "No Knowledge base available. Please upload a document first.",
                "context": "system",
                "source_paper": None,
                "search_metadata": {"type": "no_knowledge_base"},
            }

        if is_greeting(text):
            response = "Hello! I'm your Graph RAG assistant. How can I help you today?"
            db.store_chat(text, response)
            return {
                "response": response,
                "final_response": response,
                "context": "system",
                "source_paper": None,
                "search_metadata": {"type": "greeting"},
            }

        memory_response, distances = db.search_chat_memory(text, k=1)
        SIMILARITY_THRESHOLD = 0.8
        if memory_response and distances and distances[0] < SIMILARITY_THRESHOLD:
            full_content = memory_response[0].page_content
            answer_part = full_content.split("\nA: ")[1] if "\nA: " in full_content else full_content

            # Check memory response safety
            memory_safe, memory_reason = guard_rails.check_response(answer_part)
            if not memory_safe:
                safe_response = guard_rails.get_safe_response(memory_reason)
                return {
                    "response": safe_response,
                    "final_response": safe_response,
                    "context": "guard_rails_memory",
                    "source_paper": None,
                    "search_metadata": {"type": "guard_rails_memory_blocked", "reason": memory_reason},
                }

            return {
                "response": answer_part,
                "final_response": sanitize_response(answer_part),
                "context": "memory",
                "source_paper": None,
                "search_metadata": {"type": "memory", "similarity": 1 - distances[0]},
            }

        # Call search_documents without paper_id. Detection is now automatic.
        docs, context, source_papers = search_documents(text)

        primary_paper_title = source_papers[0].get("title") if source_papers else None
        primary_paper_id = source_papers[0].get("id") if source_papers else None
        response = get_agent_response(text, context, doc_id=primary_paper_id)

        # Check response safety
        response_safe, response_reason = guard_rails.check_response(response)
        if not response_safe:
            safe_response = guard_rails.get_safe_response(response_reason)
            return {
                "response": safe_response,
                "final_response": safe_response,
                "context": "guard_rails_response",
                "source_paper": None,
                "search_metadata": {"type": "guard_rails_response_blocked", "reason": response_reason},
            }

        # Check if query is out of context
        if not docs or response == "I cannot find enough information in the uploaded documents to answer that.":
            return {
                "response": "I cannot find enough information in the uploaded documents to answer that.",
                "final_response": "I cannot find enough information in the uploaded documents to answer that.",
                "context": "no_pdf",
                "source_paper": "out_of_knowledgebase",
                "search_metadata": {
                    "search_type": "null",
                    "results_count": 0,
                    "source_papers": [
                        {
                            "id": None,
                            "title": "out_of_knowledgebase",
                            "score": 0
                        }
                    ]
                }
            }

        # Only store chat memory if the query is relevant (docs exist and response is not the fallback message)
        if docs and response != "I cannot find enough information in the uploaded documents to answer that.":
            query_embedding = embedding_model.embed_query(text)
            metadata = {"context_type": "pdf", "paper_id": primary_paper_id, "paper_title": primary_paper_title}
            db.store_chat_memory(text, response, query_embedding, "pdf", metadata)

        search_metadata = {
            "search_type": "scoped_vector_graph",
            "results_count": len(docs),
            "source_papers": source_papers or [],
        }

        return {
            "response": response,
            "final_response": sanitize_response(response),
            "context": "pdf",
            "source_paper": primary_paper_title,
            "search_metadata": search_metadata,
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        error_response = "I'm sorry, an unexpected error occurred while processing your request."
        return {
            "response": error_response,
            "final_response": error_response,
            "context": "error",
            "source_paper": None,
            "search_metadata": {"type": "error", "error": str(e)},
        }

@app.get("/graph/chat-memory/")
async def get_chat_memory():
    try:
        results = db.get_chat_memory()
        if not results.get("ids"):
            return {"chat_memory": [], "message": "No chat memory entries found."}

        valid_entries = []
        for i, doc_id in enumerate(results["ids"]):
            try:
                entry = {
                    "id": doc_id,
                    "document": results["documents"][i],
                    "metadata": results["metadatas"][i]
                }
                valid_entries.append(entry)
            except (IndexError, TypeError):
                logger.warning(f"Skipping malformed chat memory entry with id {doc_id}")
                continue

        # Sort by id assuming they are numeric strings, handle non-numeric cases
        valid_entries.sort(key=lambda x: int(x["id"]) if x["id"].isdigit() else 0)

        chat_memory = []
        for entry in valid_entries:
            document = entry["document"]
            metadata = entry["metadata"]
            parts = document.split("\nA: ")
            if len(parts) > 1:
                query = parts[0].replace("Q: ", "").strip()
                response = parts[1].strip()
                context_type = metadata.get("context_type", "unknown") if metadata else "unknown"
                chat_memory.append({
                    "id": entry["id"],
                    "query": query,
                    "response": response,
                    "context_type": context_type
                })
        return {"chat_memory": chat_memory, "message": f"Retrieved {len(chat_memory)} chat memory entries."}
    except Exception as e:
        return {"chat_memory": [], "message": f"Error retrieving chat memory: {str(e)}"}

@app.delete("/graph/chat-memory/")
async def clear_chat_memory():
    try:
        db.clear_chat_memory()
        return {"message": "Chat memory cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear chat memory: {str(e)}")


@app.get("/graph/uploaded-pdfs/")
async def get_uploaded_pdfs():
    try:
        with graph_db.get_session() as session:
            result = session.run("MATCH (p:PDF) RETURN p.doc_id, p.title, p.uploaded_at")
            pdfs = [{"doc_id": record["p.doc_id"], "title": record["p.title"], "uploaded_at": record["p.uploaded_at"]} for record in result]
        return {"pdfs": pdfs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch PDFs: {str(e)}")




@app.post("/evaluate/ragas/")
async def evaluate_ragas_endpoint(request: QueryRequest):
    try:
        text = request.prompt.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Query text cannot be empty")

        # Use existing search and response generation
        docs, context, source_papers = search_documents(text)
        primary_paper_id = source_papers[0].get("id") if source_papers else None
        response = get_agent_response(text, context, doc_id=primary_paper_id)

        # Evaluate with RAGAs
        evaluation_results = evaluate_ragas(text, response, context, doc_id=primary_paper_id)

        return {
            "evaluation_results": evaluation_results,
            "message": "RAGAs evaluation completed successfully"
        }
    except Exception as e:
        logger.error(f"Error in RAGAs evaluation endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to evaluate with RAGAs: {str(e)}")

@app.get("/evaluate/ragas/history")
async def get_ragas_history():
    try:
        history = get_ragas_evaluation_history()
        return {"evaluation_history": history}
    except Exception as e:
        logger.error(f"Error retrieving RAGAs history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve RAGAs history: {str(e)}")

@app.get("/ragas/")
async def get_ragas_dashboard():
    return FileResponse("app/frontend/static/ragas_dashboard.html")
