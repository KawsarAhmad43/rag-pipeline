
import os
import sys
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import shutil
from uuid import UUID
import google.generativeai as genai
from pathlib import Path
import logging
from typing import Optional, Literal


project_root = Path(__file__).resolve().parent.parent
print(project_root)
sys.path.insert(0, str(project_root))

from config import config
from src.multimodal_rag.main_voice  import AgenticChatSystem
from data_load.doc_load import PDFHandler
from vector_store.vector_db import ChromaDBHandler
from retrieval.reranker import Reranker
from generation.response_generator import ResponseGenerator
from transformers.utils.import_utils import is_flash_attn_2_available
from duckduckgo_search import DDGS
# from fastapi import Form  # Removed redundant


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Get host and port from .env, with fallbacks
host = os.getenv('MultiModal_RAG_HOST', '0.0.0.0')
port = os.getenv('MultiModal_RAG_PORT', '5001')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Multimodal RAG API...")

    global global_multimodal_generator

    try:
        if config.GOOGLE_API_KEY:
            logger.info("🧠 Initializing global multimodal generator...")
            global_multimodal_generator = ResponseGenerator()
            genai.configure(api_key=config.GOOGLE_API_KEY)
            logger.info("✅ Global multimodal generator initialized.")
        else:
            logger.warning("GOOGLE_API_KEY not found. Multimodal capabilities will be limited.")
    except Exception as e:
        logger.critical(f"Failed to initialize global multimodal generator: {e}", exc_info=True)
        global_multimodal_generator = None

    yield

    # Shutdown
    logger.info("🛑 Shutting down Multimodal RAG API...")

# Update FastAPI application initialization
app = FastAPI(
    title="Multimodal RAG Chatbot API",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

class AgentManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentManager, cls).__new__(cls)
            cls._instance.agents = {}
            logger.info("AgentManager initialized.")
        return cls._instance
    def get_agent(self, session_id: str) -> Optional[AgenticChatSystem]:
        return self.agents.get(session_id)
    def set_agent(self, session_id: str, agent: AgenticChatSystem):
        self.agents[session_id] = agent

agent_manager = AgentManager()

# class ChatRequest(BaseModel):
#     user_input: str
#     session_id: str
#     generation_mode: Literal["api", "local"] = config.DEFAULT_GENERATION_MODE

class ChatResponse(BaseModel):
    response: str
    image: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    pdf_path: str
    session_id: str
    chunk_count: int = 0

class AvailableSessionsResponse(BaseModel):
    sessions: list[str]
    message: str

class TTSRequest(BaseModel):  # NEW
    user_input: str
    session_id: str
    generation_mode: Literal["api", "local"] = config.DEFAULT_GENERATION_MODE
    voice: Optional[str] = None  # NEW optional voice

class TTSResponse(BaseModel):  # NEW
    text: str
    audio_path: str
    voice_used: str

# Global response generator for direct multimodal calls
global_multimodal_generator: Optional[ResponseGenerator] = None

@app.post("/initialize_session", response_model=UploadResponse)
async def initialize_session(session_id: str = Form(...), pdf_file: Optional[UploadFile] = File(None)):
    pdf_save_path: Optional[Path] = None
    logger.info(f"🔵 Multimodal RAG: initialize_session called with session_id={session_id}")
    logger.info(f"🔵 Multimodal RAG: pdf_file received: {pdf_file.filename if pdf_file else 'None'}")

    if pdf_file:
        allowed_extensions = ('.pdf', '.txt')
        if not pdf_file.filename.lower().endswith(allowed_extensions):
            raise HTTPException(status_code=400, detail=f"Only {', '.join(allowed_extensions)} files are allowed.")
        pdf_save_path = config.DATA_PATH / f"{session_id}_{pdf_file.filename}"
        logger.info(f"🔵 Multimodal RAG: Will save file to: {pdf_save_path}")
        try:
            content = await pdf_file.read()
            logger.info(f"🔵 Multimodal RAG: Read {len(content)} bytes from upload")
            with open(pdf_save_path, "wb") as buffer:
                buffer.write(content)
            logger.info(f"📁 Multimodal RAG: PDF saved to: {pdf_save_path}")
            logger.info(f"📊 Multimodal RAG: File size: {pdf_save_path.stat().st_size} bytes")
            logger.info(f"✅ Multimodal RAG: File exists: {pdf_save_path.exists()}")
        except Exception as e:
            logger.error(f"❌ Multimodal RAG: Failed to save PDF: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded PDF: {str(e)}")
        finally:
            pdf_file.file.close()

    logger.info(f"🔵 Multimodal RAG: Initializing AgenticChatSystem for session {session_id}...")
    logger.info(f"🔵 Multimodal RAG: PDF path being passed: {pdf_save_path}")
    try:
        agent = AgenticChatSystem(session_id=session_id, pdf_file_path=str(pdf_save_path) if pdf_save_path else None)
        agent_manager.set_agent(session_id, agent)
        logger.info(f"AgenticChatSystem successfully initialized for session {session_id}.")

        # Verify the index was built
        try:
            chroma_instance = agent.db_handler._get_chroma_instance()
            doc_count = chroma_instance._collection.count()
            logger.info(f"✅ Session initialized with {doc_count} indexed documents")
            if doc_count == 0 and pdf_save_path:
                logger.error(f"❌ CRITICAL: PDF was provided but collection has 0 documents!")
                logger.error(f"   File path: {pdf_save_path}")
                logger.error(f"   File exists: {pdf_save_path.exists()}")
                logger.error(f"   Chunk count from agent: {agent.chunk_count}")
        except Exception as e:
            logger.warning(f"Could not verify index: {e}")

        return UploadResponse(
            message=f"Session '{session_id}' initialized.",
            pdf_path=str(pdf_save_path) if pdf_save_path else "N/A",
            session_id=session_id,
            chunk_count=agent.chunk_count
        )
    except Exception as init_error:
        if pdf_save_path and pdf_save_path.exists():
            os.remove(pdf_save_path)
        logger.error(f"CRITICAL: Failed to initialize AgenticChatSystem: {init_error}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize chatbot: {str(init_error)}. Check server logs.")

@app.post("/chat", response_model=ChatResponse)
async def unified_chat(
    user_input: str = Form(""),
    session_id: str = Form(...),
    generation_mode: str = Form(config.DEFAULT_GENERATION_MODE),
    attachment: Optional[UploadFile] = File(None)
):
    global global_multimodal_generator

    logger.info(f"Received unified chat request for session {session_id}, mode: {generation_mode}")

    if not user_input and not attachment:
        raise HTTPException(status_code=400, detail="Either user_input or attachment must be provided.")

    if not user_input and attachment:
        user_input = "Describe this image or document in detail."

    # Initialize/Get Agent first
    agent = agent_manager.get_agent(session_id)
    if not agent:
        logger.warning(f"Session {session_id} not in memory. Lazy-loading from disk.")
        try:
            agent = AgenticChatSystem(session_id=session_id, pdf_file_path=None)
            agent_manager.set_agent(session_id, agent)
        except Exception as e:
            logger.error(f"Could not load existing session {session_id}: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail="Chat session not found or failed to load.")

    response_data = None
    image_data = None

    # Handle Attachment if present
    uploaded_file_obj = None
    temp_file_path = None

    try:
        if attachment:
            allowed_extensions = ('.jpg', '.jpeg', '.png', '.pdf')
            if not attachment.filename.lower().endswith(allowed_extensions):
                raise HTTPException(status_code=400, detail=f"Only {', '.join(allowed_extensions)} files are allowed.")

            temp_file_path = config.DATA_PATH / f"temp_chat_{session_id}_{attachment.filename}"
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(attachment.file, buffer)

            if generation_mode == "api":
                if not config.GOOGLE_API_KEY:
                     raise HTTPException(status_code=503, detail="API multimodal model is not available.")

                mime_type = attachment.content_type or ("application/pdf" if attachment.filename.endswith(".pdf") else "image/jpeg")
                # Upload to Gemini (returns a file handle)
                uploaded_file_obj = genai.upload_file(path=str(temp_file_path), mime_type=mime_type)
            else:
                 # Local mode uses path str
                 uploaded_file_obj = str(temp_file_path)

        # NEW LOGIC: Extract PDF text for local mode context injection
        pdf_text = None
        if attachment and attachment.filename.lower().endswith('.pdf') and generation_mode == "local":
            try:
                pdf_handler = PDFHandler(str(temp_file_path))
                pdf_text = pdf_handler.extract_text()
                logger.info(f"Extracted {len(pdf_text)} chars from attachment PDF for local context.")
            except Exception as e:
                logger.warning(f"Failed to extract text from attachment PDF: {e}")

        # Get Response from Agent (which now handles RAG + Attachment)
        response = agent.get_response(
            user_input=user_input,
            generation_mode=generation_mode,
            uploaded_genai_file=uploaded_file_obj,
            pdf_text=pdf_text # Pass extracted text
        )

        return ChatResponse(response=response, image=None)

    except Exception as e:
        logger.error(f"Error processing unified chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing your chat request: {str(e)}")
    finally:
        if temp_file_path and temp_file_path.exists() and generation_mode != "api":
             pass # Keep local file for a bit? Or remove? Local model reads it so we might need to keep until generated.
             # Actually LocalResponseGenerator opens it. We can delete after response returns.
             if temp_file_path.exists():
                os.remove(temp_file_path)
        if attachment:
            attachment.file.close()

@app.get("/get_available_sessions", response_model=AvailableSessionsResponse)
async def get_available_sessions():
    return AvailableSessionsResponse(sessions=list(agent_manager.agents.keys()), message="Found sessions.")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/status/{session_id}")
async def get_session_status(session_id: str):
    """Check if session is ready (i.e., has indexed documents)."""
    agent = agent_manager.get_agent(session_id)
    doc_count = 0
    ready = False

    if agent:
         try:
            if hasattr(agent, 'db_handler'):
                 doc_count = agent.db_handler.get_doc_count()
                 # Ready if we have documents OR if it's a new session without documents yet
                 # but initialized (though usually we want docs for RAG).
                 # To satisfy "notifying when vector store is ready", we'll check doc_count.
                 ready = doc_count > 0
         except Exception as e:
             logger.error(f"Error checking status for {session_id}: {e}")
             return {"ready": False, "doc_count": 0, "session_id": session_id, "error": str(e)}

    return {"ready": ready, "doc_count": doc_count, "session_id": session_id}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete session from memory and cleanup disk files (PDFs)."""
    logger.info(f"🗑️ Reset request for session {session_id}")

    # 1. Remove from memory
    # agent = agent_manager.get_agent(session_id)
    if session_id in agent_manager.agents:
        del agent_manager.agents[session_id]
        logger.info(f"Removed session {session_id} from memory")

    # 2. Cleanup disk files (PDFs)
    # config.DATA_PATH: /home/aiteam/Desktop/rag_pipelines/data
    try:
        data_dir = config.DATA_PATH
        # Remove primary PDF associated with session_id
        # Files are saved as {session_id}_{filename}
        for file in data_dir.glob(f"{session_id}_*"):
            if file.is_file():
                file.unlink()
                logger.info(f"Deleted file: {file}")

        # Also check for temp chat files
        for file in data_dir.glob(f"temp_chat_{session_id}_*"):
            file.unlink()

        # Optional: cleanup chromadb if session-specific
        # (Though currently it's a shared local persistent DB per session_id)

    except Exception as e:
        logger.error(f"Error during disk cleanup for session {session_id}: {e}")
        # Non-fatal, session is at least out of memory

    return {"status": "ok", "message": f"Session {session_id} reset successfully."}


#  ---------------- NEW TTS Endpoint ----------------
from fastapi import Form

@app.post("/tts", response_model=TTSResponse)
async def tts_endpoint(
    user_input: str = Form(...),
    session_id: str = Form(...),
    generation_mode: str = Form(config.DEFAULT_GENERATION_MODE),
    voice: str = Form("af_sarah")
):
    agent = agent_manager.get_agent(session_id)
    if not agent:
        try:
            agent = AgenticChatSystem(session_id=session_id, pdf_file_path=None)
            agent_manager.set_agent(session_id, agent)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Chat session not found or failed to load.")

    audio_out_path = Path(project_root / "voice_notebook" / f"{session_id}_tts.wav")
    audio_out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = agent.get_response(
            user_input=user_input,
            generation_mode=generation_mode,
            speak=True,
            voice=voice,
            audio_out_path=audio_out_path
        )

        if isinstance(response, dict):
            return TTSResponse(
                text=response.get("text", ""),
                audio_path=response.get("audio_path", str(audio_out_path)),
                voice_used=response.get("voice_used", voice)
            )
        else:
            return TTSResponse(
                text=str(response),
                audio_path=str(audio_out_path),
                voice_used=voice
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")
@app.post("/voice_chat", response_model=TTSResponse)
async def voice_chat_endpoint(
    session_id: str = Form(...),
    generation_mode: str = Form(config.DEFAULT_GENERATION_MODE),
    voice: str = Form("af_sarah"),
    audio_file: UploadFile = File(...)
):
    agent = agent_manager.get_agent(session_id)
    if not agent:
        try:
            agent = AgenticChatSystem(session_id=session_id, pdf_file_path=None)
            agent_manager.set_agent(session_id, agent)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Chat session not found or failed to load.")

    # Save incoming audio to a temporary file
    temp_audio_path = config.DATA_PATH / f"voice_input_{session_id}_{audio_file.filename}"
    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        # 1. Transcribe
        transcribed_text = agent.transcribe_audio(str(temp_audio_path))

        if transcribed_text.startswith("Error:"):
             raise HTTPException(status_code=500, detail=transcribed_text)

        # 2. Get Response with TTS
        audio_out_path = Path(project_root / "voice_notebook" / f"{session_id}_voice_response.wav")
        response = agent.get_response(
            user_input=transcribed_text,
            generation_mode=generation_mode,
            speak=True,
            voice=voice,
            audio_out_path=audio_out_path
        )

        return TTSResponse(
            text=response.get("text", "") if isinstance(response, dict) else str(response),
            audio_path=response.get("audio_path", str(audio_out_path)) if isinstance(response, dict) else str(audio_out_path),
            voice_used=response.get("voice_used", voice) if isinstance(response, dict) else voice
        )
    except Exception as e:
        logger.error(f"Voice chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voice interaction failed: {str(e)}")
    finally:
        if temp_audio_path.exists():
            os.remove(temp_audio_path)
        audio_file.file.close()


@app.delete("/reset_session/{session_id}")
async def reset_session(session_id: str):
    """
    Reset/clear session data and vector database for a specific session.
    This will remove the agent from memory and delete associated vector DB files.
    """
    try:
        logger.info(f"🗑️ Resetting session: {session_id}")

        # Remove agent from agent manager
        if agent_manager.has_agent(session_id):
            agent_manager.remove_agent(session_id)
            logger.info(f"✅ Agent removed from memory for session {session_id}")

        # Delete vector database directory
        import shutil
        chroma_db_path = config.VECTOR_DB_BASE_PATH / session_id if hasattr(config, "VECTOR_DB_BASE_PATH") else Path.home() / "Desktop" / "rag_pipelines" / "chroma_db" / session_id

        if chroma_db_path.exists():
            shutil.rmtree(chroma_db_path)
            logger.info(f"✅ Vector DB deleted: {chroma_db_path}")

        # Delete uploaded PDF files for this session
        data_path = config.DATA_PATH
        for file in data_path.glob(f"{session_id}_*"):
            file.unlink()
            logger.info(f"✅ Deleted file: {file}")

        return {
            "status": "success",
            "message": f"Session {session_id} reset successfully",
            "session_id": session_id
        }

    except Exception as e:
        logger.error(f"❌ Error resetting session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset session: {str(e)}")
