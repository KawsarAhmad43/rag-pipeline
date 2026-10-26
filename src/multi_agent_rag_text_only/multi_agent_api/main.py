from pathlib import Path
from pipeline import create_super_graph, set_current_document, get_current_document, create_simple_rag_chain, is_simple_query
from config import set_current_model, get_current_model, get_current_provider, get_available_models
from langchain_core.messages import HumanMessage
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import tempfile
import os
import shutil
from typing import List, Dict, Any
import json

app = FastAPI(title="Multi-Agent RAG API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_DIR = Path.cwd() / "llama_context_project"
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

class MessageRequest(BaseModel):
    message: str

class ModelRequest(BaseModel):
    model: str

class StreamResponse(BaseModel):
    step: str
    content: str

@app.get("/models")
async def get_models():
    """Get available models"""
    return {"models": get_available_models()}

@app.get("/current-model")
async def get_current_model_info():
    """Get current model information"""
    return {
        "model": get_current_model(),
        "provider": get_current_provider()
    }

@app.post("/set-model")
async def set_model(request: ModelRequest):
    """Set the current model"""
    try:
        success = set_current_model(request.model)
        if success:
            return {
                "message": f"Model changed to {request.model}",
                "model": get_current_model(),
                "provider": get_current_provider()
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model: {request.model}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting model: {str(e)}")

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF document for processing"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    try:
        # Write uploaded file to temporary file
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()

        # Set the document path for processing
        set_current_document(temp_file.name)

        return {"message": "Document uploaded successfully", "filename": file.filename}
    except Exception as e:
        # Clean up temp file if error occurs
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.post("/process-message")
async def process_message(request: MessageRequest):
    """Process a user message through the multi-agent system"""
    try:
        # Check if a document is uploaded
        current_doc = get_current_document()
        if not current_doc or not os.path.exists(current_doc):
            raise HTTPException(
                status_code=400,
                detail="No document uploaded. Please upload a PDF document first."
            )

        # Check if this is a simple query that can use the simple RAG chain
        if is_simple_query(request.message):
            try:
                simple_chain = create_simple_rag_chain()
                if simple_chain:
                    result = simple_chain.invoke({"question": request.message})
                    return {
                        "response": [
                            jsonable_encoder({
                                "messages": [HumanMessage(content=f"Simple RAG Response: {result}")]
                            })
                        ]
                    }
            except Exception as e:
                # Fall back to multi-agent system if simple chain fails
                pass

        # Use multi-agent system for complex queries
        super_graph = create_super_graph()

        # Stream the response with reduced recursion limit
        response_stream = []
        try:
            for s in super_graph.stream(
                {
                    "messages": [
                        HumanMessage(content=request.message)
                    ],
                },
                {"recursion_limit": 8},  # Reduced from 30 to 8
            ):
                if "__end__" not in s:
                    response_stream.append(jsonable_encoder(s))
        except Exception as stream_error:
            # If we hit recursion limit or other errors, return what we have
            if "recursion limit" in str(stream_error).lower():
                if response_stream:
                    response_stream.append(jsonable_encoder({
                        "messages": [HumanMessage(content="Processing completed (recursion limit reached).")]
                    }))
                else:
                    raise HTTPException(
                        status_code=500,
                        detail="Processing failed due to recursion limit. Please try a simpler query."
                    )
            else:
                raise stream_error

        if not response_stream:
            raise HTTPException(
                status_code=500,
                detail="No response generated. Please try again."
            )

        return {"response": response_stream}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/stream-message")
async def stream_message(request: MessageRequest):
    """Stream the processing of a user message through the multi-agent system"""
    try:
        # Check if a document is uploaded
        current_doc = get_current_document()
        if not current_doc or not os.path.exists(current_doc):
            raise HTTPException(
                status_code=400,
                detail="No document uploaded. Please upload a PDF document first."
            )

        super_graph = create_super_graph()

        def generate():
            try:
                for s in super_graph.stream(
                    {
                        "messages": [
                            HumanMessage(content=request.message)
                        ],
                    },
                    {"recursion_limit": 8},  # Reduced from 30 to 8
                ):
                    if "__end__" not in s:
                        yield f"data: {json.dumps(jsonable_encoder(s))}\n\n"
            except Exception as stream_error:
                # Handle recursion limit gracefully
                if "recursion limit" in str(stream_error).lower():
                    yield f"data: {json.dumps(jsonable_encoder({'error': 'Processing completed (recursion limit reached).'}))}\n\n"
                else:
                    yield f"data: {json.dumps(jsonable_encoder({'error': str(stream_error)}))}\n\n"

        return StreamingResponse(generate(), media_type="text/plain")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
