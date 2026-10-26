from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import torch
import tempfile
import shutil
import mimetypes
from chatbot import Chatbot
from langchain.memory import ConversationBufferWindowMemory
from pdf_handler import PDFHandler
from reranker import Reranker
from vector_db import VectorDB
import gc
from transformers.utils.import_utils import is_flash_attn_2_available
from contextlib import asynccontextmanager
from duckduckgo_search import DDGS
from spech_to_text import get_stt_model
from text_to_spech import get_tts_model

# Global variables for chatbot components
reranker = None
stt_model = None
tts_model = None

# User-specific components (will be stored per session)
user_components = {}  # session_id -> {chatbot, vector_db, memory, pdf_handler}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    initialize_chatbot()
    yield
    # Shutdown
    cleanup_resources()

def initialize_chatbot():
    """Initialize the chatbot and all its components"""
    global chatbot, vector_db, reranker, memory, pdf_handler, stt_model, tts_model

    print("🤖 Initializing chatbot components...")

    # Device and dtype configuration
    CPU_DEVICE = "cpu"
    CPU_DTYPE = torch.float32
    GPU_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    GPU_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    GPU_ATTN_IMPL = "flash_attention_2" if torch.cuda.is_available() and is_flash_attn_2_available() else None

    if torch.cuda.is_available():
        print(f"Image-to-text model will load on {GPU_DEVICE} with {GPU_DTYPE}")
        print(f"Text-to-text model (Gemma) will load on {CPU_DEVICE} with {CPU_DTYPE}")
    else:
        print("CUDA is NOT available. All models will run on CPU.")

    # Initialize reranker (shared across all users)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    reranker_model_path = os.path.join(base_dir, "models", "bge-reranker-v2-m3")
    if not os.path.exists(reranker_model_path):
        print(f"⚠️ Reranker model not found at: {reranker_model_path}")
        print("Using dummy reranker for testing...")
        reranker = None
    else:
        reranker = Reranker(reranker_model_path, device=CPU_DEVICE)
        print("✅ Reranker initialized successfully!")

    # Initialize speech models
    print("🎤 Initializing speech models...")
    try:
        stt_model = get_stt_model()
        if stt_model.is_model_loaded():
            print("✅ Speech-to-Text model loaded successfully!")
        else:
            print("⚠️ Speech-to-Text model failed to load")
    except Exception as e:
        print(f"❌ Error loading Speech-to-Text model: {e}")

    try:
        tts_model = get_tts_model()
        if tts_model.is_model_loaded():
            print("✅ Text-to-Speech model loaded successfully!")
        else:
            print("⚠️ Text-to-Speech model failed to load")
    except Exception as e:
        print(f"❌ Error loading Text-to-Speech model: {e}")

def get_user_components(session_id: str):
    """Get or create user-specific components"""
    global user_components, reranker

    if session_id not in user_components:
        print(f"🔄 Initializing components for session: {session_id}")

        # Initialize user-specific components
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

        # Device and dtype configuration
        CPU_DEVICE = "cpu"
        CPU_DTYPE = torch.float32
        GPU_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        GPU_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        GPU_ATTN_IMPL = "flash_attention_2" if torch.cuda.is_available() and is_flash_attn_2_available() else None

        # Initialize memory for this user
        memory = ConversationBufferWindowMemory(k=5)

        # Initialize empty vector DB (will be populated when user uploads documents)
        embedding_model_path = os.path.join(base_dir, "models", "colnomic-embed-multimodal-3b")
        if os.path.exists(embedding_model_path):
            # Create user-specific vector DB directory
            user_db_path = os.path.join(base_dir, "chroma_db", f"user_{session_id}")
            vector_db = VectorDB(
                embedding_model_path,
                persist_directory=user_db_path,
                device=CPU_DEVICE,
                torch_dtype=CPU_DTYPE
            )
            print(f"✅ Vector DB initialized for session: {session_id}")
        else:
            print(f"⚠️ Embedding model not found, using dummy vector DB for session: {session_id}")
            vector_db = None

        # Initialize chatbot for this user
        gemma_model_path = os.path.join(base_dir, "models", "gemma-3-4b-it")
        if os.path.exists(gemma_model_path):
            chatbot = Chatbot(
                generation_model_path=gemma_model_path,
                vector_db=vector_db,
                reranker=reranker,
                memory=memory,
                gemma_device=CPU_DEVICE,
                gemma_torch_dtype=CPU_DTYPE,
                vision_device=GPU_DEVICE,
                vision_torch_dtype=GPU_DTYPE,
                vision_attn_implementation=GPU_ATTN_IMPL
            )
            print(f"✅ Chatbot initialized for session: {session_id}")
        else:
            print(f"⚠️ Gemma model not found, chatbot not available for session: {session_id}")
            chatbot = None

        # Store user components
        user_components[session_id] = {
            "chatbot": chatbot,
            "vector_db": vector_db,
            "memory": memory,
            "pdf_handler": None
        }

    return user_components[session_id]

def cleanup_resources():
    """Cleanup on shutdown"""
    global user_components, reranker, stt_model, tts_model

    print("🧹 Cleaning up resources...")

    # Cleanup all user components
    for session_id, components in user_components.items():
        print(f"🧹 Cleaning up session: {session_id}")
        if components.get("chatbot"):
            components["chatbot"].cleanup()
        if components.get("vector_db"):
            components["vector_db"].cleanup()

    if reranker:
        reranker.cleanup()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("✅ Cleanup complete.")

# Initialize FastAPI app with lifespan
app = FastAPI(title="Multimodal Chatbot API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    max_results: Optional[int] = 3

class SearchResponse(BaseModel):
    response: str
    session_id: str
    search_results: List[dict]

class ImageAnalysisResponse(BaseModel):
    description: str
    session_id: str

class SessionResponse(BaseModel):
    session_id: str
    history: List[dict]

class SpeechToTextResponse(BaseModel):
    text: str
    success: bool
    error: Optional[str] = None

class TextToSpeechRequest(BaseModel):
    text: str
    voice: Optional[str] = "af_heart"
    speed: Optional[float] = 1.0
    lang: Optional[str] = "en-us"

class TextToSpeechResponse(BaseModel):
    success: bool
    audio_file: Optional[str] = None
    error: Optional[str] = None

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    chunks_count: int
    error: Optional[str] = None

def is_image_file(filename: str) -> bool:
    """Check if file is an image based on extension"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    file_ext = os.path.splitext(filename.lower())[1]
    return file_ext in image_extensions

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Multimodal Chatbot API is running!", "status": "healthy"}

@app.post("/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text(file: UploadFile = File(...)):
    """Convert speech audio to text"""
    global stt_model

    if not stt_model or not stt_model.is_model_loaded():
        raise HTTPException(status_code=503, detail="Speech-to-Text model not initialized")

    # Check if file is an audio file (more flexible validation)
    is_audio = False

    # Method 1: Check content type
    if file.content_type and file.content_type.startswith("audio/"):
        is_audio = True

    # Method 2: Check file extension
    if not is_audio and file.filename:
        audio_extensions = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext in audio_extensions:
            is_audio = True

    # Method 3: Accept files from st.audio_input() (which might not have proper content type)
    if not is_audio and file.filename and "recording" in file.filename.lower():
        is_audio = True

    if not is_audio:
        raise HTTPException(
            status_code=400,
            detail=f"File must be an audio file. Received: filename={file.filename}, content_type={file.content_type}"
        )

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # Transcribe audio
        transcribed_text = stt_model.transcribe_audio_file(temp_path)

        # Clean up temp file
        os.unlink(temp_path)

        if transcribed_text:
            return SpeechToTextResponse(text=transcribed_text, success=True)
        else:
            return SpeechToTextResponse(text="", success=False, error="Failed to transcribe audio")

    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing speech: {str(e)}")

@app.post("/text-to-speech", response_model=TextToSpeechResponse)
async def text_to_speech(request: TextToSpeechRequest):
    """Convert text to speech"""
    global tts_model

    if not tts_model or not tts_model.is_model_loaded():
        raise HTTPException(status_code=503, detail="Text-to-Speech model not initialized")

    try:
        # Generate speech and save to temporary file
        temp_audio_path = tts_model.synthesize_to_temp_file(
            request.text,
            voice=request.voice,
            speed=request.speed,
            lang=request.lang
        )

        if temp_audio_path:
            return TextToSpeechResponse(success=True, audio_file=temp_audio_path)
        else:
            return TextToSpeechResponse(success=False, error="Failed to generate speech")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve generated audio files"""
    # Security check: only allow .wav files
    if not filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are allowed")

    # Create the full path to the audio file
    audio_path = os.path.join(tempfile.gettempdir(), filename)

    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(audio_path, media_type="audio/wav")

@app.get("/speech-status")
async def get_speech_status():
    """Get status of speech models"""
    global stt_model, tts_model

    return {
        "speech_to_text": {
            "loaded": stt_model.is_model_loaded() if stt_model else False,
            "model_path": stt_model.model_path if stt_model else None
        },
        "text_to_speech": {
            "loaded": tts_model.is_model_loaded() if tts_model else False,
            "model_path": tts_model.model_path if tts_model else None,
            "available_voices": tts_model.get_available_voices() if tts_model else []
        }
    }

@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form("default")
):
    """Upload and process a document for a specific user session"""

    # Check if file is a supported document type
    supported_types = {'.pdf', '.txt', '.docx', '.doc'}
    file_ext = os.path.splitext(file.filename.lower())[1]

    if file_ext not in supported_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported types: {', '.join(supported_types)}"
        )

    try:
        # Get user components
        user_components = get_user_components(session_id)
        vector_db = user_components["vector_db"]

        if not vector_db:
            raise HTTPException(status_code=503, detail="Vector database not available")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # Process document based on type
        if file_ext == '.pdf':
            pdf_handler = PDFHandler(temp_path)
            chunks = pdf_handler.get_chunks()
        elif file_ext == '.txt':
            with open(temp_path, 'r', encoding='utf-8') as f:
                text = f.read()
            # Simple text chunking (you might want to improve this)
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000) if text[i:i+1000].strip()]
        else:
            # For docx/doc files, you might need additional processing
            chunks = [f"Document content from {file.filename}"]

        # Build vector index with the new chunks
        vector_db.build_index(chunks)

        # Update user components to track document upload
        user_components["pdf_handler"] = True

        # Clean up temp file
        os.unlink(temp_path)

        return DocumentUploadResponse(
            success=True,
            message=f"Document '{file.filename}' processed successfully",
            chunks_count=len(chunks)
        )

    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.get("/session-status/{session_id}")
async def get_session_status(session_id: str):
    """Get status of a specific user session"""
    try:
        user_components = get_user_components(session_id)
        vector_db = user_components["vector_db"]

        # Check if vector DB has documents
        has_documents = False
        if vector_db:
            try:
                # Try a simple search to see if there are any documents
                docs = vector_db.search("test", k=1)
                has_documents = len(docs) > 0
            except:
                has_documents = False

        return {
            "session_id": session_id,
            "chatbot_available": user_components["chatbot"] is not None,
            "vector_db_available": user_components["vector_db"] is not None,
            "memory_available": user_components["memory"] is not None,
            "document_uploaded": has_documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session status: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle text chat requests"""

    try:
        # Get user-specific components
        user_components = get_user_components(request.session_id)
        chatbot = user_components["chatbot"]

        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot not initialized. Check if models are available.")

        # Check if the message requests a web search
        if request.message.lower().startswith("web:"):
            query = request.message[4:].strip()
            if not query:
                raise HTTPException(status_code=400, detail="Web search query is empty.")

            # Perform web search
            ddg = DDGS()
            search_results = ddg.text(keywords=query, max_results=5)

            # Pass search results to chatbot
            response = chatbot.get_response(query, web_results=search_results)
            return ChatResponse(response=response, session_id=request.session_id)
        else:
            # Normal chat without web search
            response = chatbot.get_response(request.message)
            return ChatResponse(response=response, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_web(request: SearchRequest):
    """Handle web search requests"""

    try:
        # Get user-specific components
        user_components = get_user_components(request.session_id)
        chatbot = user_components["chatbot"]

        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot not initialized. Check if models are available.")

        # Perform DuckDuckGo search
        ddg = DDGS()
        search_results = ddg.text(keywords=request.query, max_results=request.max_results)

        # Get response from chatbot with search results
        response = chatbot.get_response(request.query, web_results=search_results)

        return SearchResponse(
            response=response,
            session_id=request.session_id,
            search_results=search_results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing web search: {str(e)}")

@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form("Describe this image"),
    session_id: str = Form("default")
):
    """Handle image analysis requests"""

    try:
        # Get user-specific components
        user_components = get_user_components(session_id)
        chatbot = user_components["chatbot"]

        if not chatbot:
            raise HTTPException(status_code=503, detail="Chatbot not initialized. Check if models are available.")

        # Check if file is provided
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        # Check if file is an image using multiple methods
        is_image = False

        # Method 1: Check content type
        if file.content_type and file.content_type.startswith("image/"):
            is_image = True

        # Method 2: Check file extension
        if not is_image and file.filename:
            is_image = is_image_file(file.filename)

        # Method 3: Check MIME type using mimetypes
        if not is_image and file.filename:
            mime_type, _ = mimetypes.guess_type(file.filename)
            if mime_type and mime_type.startswith("image/"):
                is_image = True

        if not is_image:
            raise HTTPException(
                status_code=400,
                detail=f"File must be an image. Received: filename={file.filename}, content_type={file.content_type}"
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        print(f"Saved image to: {temp_path}")
        print(f"File size: {os.path.getsize(temp_path)} bytes")

        # Analyze image
        result = chatbot.describe_image(temp_path, prompt)

        # Handle different return types from describe_image
        if isinstance(result, list):
            # If it's a list, extract the text from the first item
            if result and isinstance(result[0], dict) and 'generated_text' in result[0]:
                description = result[0]['generated_text']
            elif result and isinstance(result[0], str):
                description = result[0]
            else:
                description = str(result)
        elif isinstance(result, dict):
            # If it's a dict, look for generated_text
            if 'generated_text' in result:
                description = result['generated_text']
            else:
                description = str(result)
        elif isinstance(result, str):
            description = result
        else:
            description = str(result)

        print(f"Image analysis result type: {type(result)}")
        print(f"Image analysis result: {result}")
        print(f"Final description: {description}")

        # Clean up temp file
        os.unlink(temp_path)

        return ImageAnalysisResponse(description=description, session_id=session_id)

    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        print(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@app.get("/session/{session_id}", response_model=SessionResponse)
async def get_session_history(session_id: str):
    """Get conversation history for a session"""

    try:
        # Get user-specific components
        user_components = get_user_components(session_id)
        memory = user_components["memory"]

        if not memory:
            raise HTTPException(status_code=503, detail="Memory not initialized")

        history_data = memory.load_memory_variables({})
        history = history_data.get("history", "")

        # Convert history string to list of dicts for better frontend handling
        history_list = []
        if history:
            # Simple parsing - in a real app you might want more sophisticated parsing
            lines = history.split('\n')
            for line in lines:
                if line.strip():
                    history_list.append({"message": line.strip()})

        return SessionResponse(session_id=session_id, history=history_list)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session history: {str(e)}")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""

    try:
        # Get user-specific components
        user_components = get_user_components(session_id)
        memory = user_components["memory"]

        if not memory:
            raise HTTPException(status_code=503, detail="Memory not initialized")

        # Clear the memory
        memory.clear()
        return {"message": f"Session {session_id} cleared successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6001)
