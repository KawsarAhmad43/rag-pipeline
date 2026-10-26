from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import shutil
import uuid
from typing import Dict, Any
import json

# Import local modules
from config import ModelPaths
from models import ModelLoader
from rag import RAGPipeline

app = FastAPI(title="Simple Local RAG API", version="1.0.0")

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

def get_session(session_id: str) -> Dict[str, Any]:
    """Get or create a session for a user"""
    if session_id not in sessions:
        sessions[session_id] = {
            "rag_pipeline": None,
            "document_path": None,
            "models_loaded": False
        }
    return sessions[session_id]

def initialize_models(session_id: str) -> bool:
    """Initialize models for a session"""
    session = get_session(session_id)

    if session["models_loaded"]:
        return True

    try:
        print("Starting model initialization...")
        # Initialize model paths and check availability
        paths = ModelPaths()
        available_models = paths.check_availability()
        print(f"Available models: {available_models}")

        if not available_models:
            print("No models available")
            return False

        # Initialize model loader
        loader = ModelLoader()

        # Load Gemma LLM
        print("Loading Gemma LLM...")
        llm, tokenizer = loader.load_gemma_llm(available_models.get("Gemma LLM", ""))
        if llm is None:
            print("Failed to load Gemma LLM")
            return False
        print("Gemma LLM loaded successfully")

        # Load embedding model
        print("Loading embedding model...")
        embeddings = loader.load_embedding_model(available_models.get("Colnomic Embed 3B", ""))
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

@app.post("/create-session")
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

    # Initialize models if not already done
    print("Initializing models...")
    if not initialize_models(session_id):
        print("Model initialization failed")
        raise HTTPException(status_code=500, detail="Failed to initialize models")
    print("Models initialized successfully")

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    try:
        # Write uploaded file to temporary file
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()

        # Store document path in session
        session["document_path"] = temp_file.name

        # Initialize RAG pipeline
        print("Initializing RAG pipeline...")
        rag = RAGPipeline(session["embeddings"], session["llm"])
        rag.load_documents(temp_file.name)
        rag.create_vectorstore()
        rag.create_rag_chain()
        print("RAG pipeline initialized successfully")

        # Store RAG pipeline in session
        session["rag_pipeline"] = rag

        return {
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "session_id": session_id
        }
    except Exception as e:
        # Clean up temp file if error occurs
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        print(f"Error uploading document: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.post("/process-message")
async def process_message(request: MessageRequest):
    """Process a user message through the RAG system"""
    session = get_session(request.session_id)

    if not session["models_loaded"]:
        raise HTTPException(status_code=400, detail="Models not initialized. Please upload a document first.")

    if not session["document_path"] or not os.path.exists(session["document_path"]):
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a PDF document first.")

    if not session["rag_pipeline"]:
        raise HTTPException(status_code=400, detail="RAG pipeline not initialized. Please upload a document first.")

    try:
        print(f"Processing message: {request.message}")
        # Process the message through the RAG pipeline
        result = session["rag_pipeline"].invoke(request.message)
        print(f"RAG pipeline result: {result}")

        return {
            "response": result,
            "session_id": request.session_id
        }
    except Exception as e:
        print(f"Error processing message: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/session-status/{session_id}")
async def get_session_status(session_id: str):
    """Get the status of a session"""
    session = get_session(session_id)
    return {
        "session_id": session_id,
        "models_loaded": session["models_loaded"],
        "document_uploaded": session["document_path"] is not None and os.path.exists(session["document_path"]),
        "rag_pipeline_ready": session["rag_pipeline"] is not None
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and clean up resources"""
    if session_id in sessions:
        session = sessions[session_id]

        # Clean up temporary file
        if session["document_path"] and os.path.exists(session["document_path"]):
            try:
                os.unlink(session["document_path"])
            except:
                pass

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
        available_models = paths.check_availability()
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
    uvicorn.run(app, host="0.0.0.0", port=8005)
