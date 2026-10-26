from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import tempfile
import os
import shutil
import uuid
from typing import Dict, Any, Optional
import json
from pathlib import Path
import asyncio # For async operations
from starlette.concurrency import run_in_threadpool # To run synchronous functions in a thread pool

# Import local modules
from config import ModelPaths
from models import ModelLoader
from rag import RAGPipeline
from document_tools import DocumentTools
from agents import AgentSystem

app = FastAPI(title="Local Multi-Agent RAG API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session management
sessions: Dict[str, Dict[str, Any]] = {}

class MessageRequest(BaseModel):
    message: str
    session_id: str

class SessionResponse(BaseModel):
    session_id: str
    message: str

def get_session(session_id: str) -> Dict[str, Any]:
    """Get or create a session for a user"""
    if session_id not in sessions:
        sessions[session_id] = {
            "rag_pipeline": None,
            "agent_system": None,
            "document_path": None,
            "models_loaded": False,
            "llm": None,
            "embeddings": None
        }
    return sessions[session_id]

async def initialize_models(session_id: str) -> bool:
    """Initialize models for a session (now asynchronous)"""
    session = get_session(session_id)

    if session["models_loaded"]:
        print("Models already loaded for session")
        return True

    try:
        print("Starting model initialization...")
        # Initialize model paths and check availability (synchronous, can be fast)
        paths = ModelPaths()
        available_models = await run_in_threadpool(paths.check_availability) # check_availability is blocking

        print(f"Available models: {available_models}")

        if not available_models:
            print("No models available")
            return False

        # Get best models for each type (prioritizing smaller/more efficient models)
        best_models = await run_in_threadpool(paths.get_best_models, available_models)
        print(f"Selected best models: {best_models}")

        # Initialize model loader (synchronous init)
        loader = ModelLoader()

        # Load best LLM (prioritizing smaller models)
        print("Loading best available LLM...")
        llm_path = best_models.get("llm")
        if not llm_path:
            print("No suitable LLM found")
            return False

        llm, tokenizer = await loader.load_gemma_llm(llm_path)
        if llm is None:
            print("Failed to load LLM")
            return False
        print("LLM loaded successfully")

        # Load best embedding model (prioritizing smaller models)
        print("Loading best available embedding model...")
        embedding_path = best_models.get("embedding")
        if not embedding_path:
            print("No suitable embedding model found")
            return False

        embeddings = await loader.load_embedding_model(embedding_path)
        if embeddings is None:
            print("Failed to load embedding model")
            return False
        print("Embedding model loaded successfully")

        # Store models in session
        session["llm"] = llm
        session["embeddings"] = embeddings
        session["models_loaded"] = True

        print("Models stored in session successfully")
        return True
    except Exception as e:
        print(f"Error initializing models: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.post("/create-session", response_model=SessionResponse)
async def create_session():
    """Create a new session for a user"""
    session_id = str(uuid.uuid4())
    get_session(session_id)
    return {"session_id": session_id, "message": "Session created successfully"}

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...), session_id: str = Form(None)):
    """Upload a PDF document for processing"""
    print(f"Upload request received - Session ID: {session_id}, File: {file.filename}")

    if not session_id:
        # Create a new session if none provided
        session_id = str(uuid.uuid4())
        get_session(session_id)
        print(f"Created new session: {session_id}")

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    session = get_session(session_id)
    print(f"Session retrieved: {session_id}")

    # Initialize models if not already done (now async)
    print("Initializing models...")
    if not await initialize_models(session_id):
        print("Model initialization failed")
        raise HTTPException(status_code=500, detail="Failed to initialize models")
    print("Models initialized successfully")

    # Create a temporary file (synchronous I/O, wrap in thread pool)
    temp_file_path = None
    try:
        # Create temporary file properly
        temp_file = await run_in_threadpool(tempfile.NamedTemporaryFile, delete=False, suffix='.pdf')
        temp_file_path = temp_file.name
        temp_file.close()  # Close the file handle
        print(f"Created temporary file: {temp_file_path}")

        # Write uploaded file to temporary file (synchronous I/O, wrap in thread pool)
        # Handle SpooledTemporaryFile properly without async context manager
        def copy_file_sync():
            with open(temp_file_path, 'wb') as f_temp:
                shutil.copyfileobj(file.file, f_temp)

        await run_in_threadpool(copy_file_sync)
        print(f"File content copied to: {temp_file_path}")

        # Store document path in session
        session["document_path"] = temp_file_path

        # Initialize RAG pipeline (now async)
        print("Initializing RAG pipeline...")
        rag = RAGPipeline(session["embeddings"], session["llm"])
        # These RAG pipeline methods are now async
        await rag.load_documents(temp_file_path)
        await rag.create_vectorstore()
        rag.create_rag_chain() # This method remains synchronous as it just sets up the runnable
        print("RAG pipeline initialized successfully")

        # Store RAG pipeline in session
        session["rag_pipeline"] = rag

        # Initialize document tools (synchronous init, but fast)
        print("Initializing document tools...")
        document_tools = await run_in_threadpool(DocumentTools)
        print("Document tools initialized successfully")

        # Initialize agent system (now async, as it compiles graphs)
        print("Initializing agent system...")
        # AgentSystem.__init__ is where graphs are compiled, which can be heavy.
        # It needs to be called in a way that doesn't block.
        # However, the graph compilation itself isn't async, so the __init__ call is blocking.
        # We assume compilation is fast enough, or ensure heavy parts are async.
        # Given the previous fix, the models are loaded. The graph compilation itself is typically fast.
        agent_system = AgentSystem(session["llm"], rag, document_tools)
        session["agent_system"] = agent_system
        print("Agent system initialized successfully")

        return {
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "session_id": session_id
        }
    except Exception as e:
        # Clean up temp file if error occurs
        if temp_file_path and await run_in_threadpool(os.path.exists, temp_file_path):
            await run_in_threadpool(os.unlink, temp_file_path)
        print(f"Error uploading document: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.post("/process-message")
async def process_message(request: MessageRequest):
    """Process a user message through the multi-agent system"""
    session = get_session(request.session_id)

    if not session["models_loaded"]:
        raise HTTPException(status_code=400, detail="Models not initialized. Please upload a document first.")

    if not session["document_path"] or not await run_in_threadpool(os.path.exists, session["document_path"]):
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a PDF document first.")

    if not session["agent_system"]:
        raise HTTPException(status_code=400, detail="Agent system not initialized. Please upload a document first.")

    try:
        print(f"Processing message: {request.message}")
        # Process the message through the agent system (now async)
        result = await session["agent_system"].invoke(request.message)
        print(f"Agent system result: {result}")

        return {
            "response": result,
            "session_id": request.session_id
        }
    except Exception as e:
        print(f"Error processing message: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/stream-message")
async def stream_message(request: MessageRequest):
    """Stream the processing of a user message through the multi-agent system"""
    session = get_session(request.session_id)

    if not session["models_loaded"]:
        raise HTTPException(status_code=400, detail="Models not initialized. Please upload a document first.")

    if not session["document_path"] or not await run_in_threadpool(os.path.exists, session["document_path"]):
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a PDF document first.")

    if not session["agent_system"]:
        raise HTTPException(status_code=400, detail="Agent system not initialized. Please upload a document first.")

    async def generate():
        try:
            # For now, we'll return the result as a single stream
            # In a more sophisticated implementation, you could stream the agent steps
            result = await session["agent_system"].invoke(request.message) # Await agent system invoke
            yield f"data: {json.dumps({'response': result})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/session-status/{session_id}")
async def get_session_status(session_id: str):
    """Get the status of a session"""
    session = get_session(session_id)
    return {
        "session_id": session_id,
        "models_loaded": session["models_loaded"],
        "document_uploaded": session["document_path"] is not None and await run_in_threadpool(os.path.exists, session["document_path"]),
        "agent_system_ready": session["agent_system"] is not None
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and clean up resources"""
    if session_id in sessions:
        session = sessions[session_id]

        # Clean up temporary file (synchronous I/O, wrap in thread pool)
        if session["document_path"] and await run_in_threadpool(os.path.exists, session["document_path"]):
            try:
                await run_in_threadpool(os.unlink, session["document_path"])
            except Exception as e:
                print(f"Error unlinking temporary file {session['document_path']}: {e}")

        # Remove session
        del sessions[session_id]

        return {"message": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "active_sessions": len(sessions)}

@app.get("/test-models")
async def test_models():
    """Test if models are available"""
    try:
        paths = ModelPaths()
        # check_availability is blocking, wrap in thread pool
        available_models = await run_in_threadpool(paths.check_availability)
        return {
            "status": "success",
            "available_models": available_models,
            "models_count": len(available_models) if available_models else 0
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "available_models": {},
            "models_count": 0
        }

if __name__ == "__main__":
    import uvicorn
    # Use 'api:app' and reload for development
    uvicorn.run(app, host="0.0.0.0", port=8004)
