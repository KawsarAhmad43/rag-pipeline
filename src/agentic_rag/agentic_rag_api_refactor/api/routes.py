from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from typing import List
import os
import tempfile
import uuid

from models.api_models import (
    ChatRequest,
    ChatResponse,
    UploadResponse,
    DirectAIRequest,
    DirectAIResponse,
    SessionStatus,
    ConversationHistory,
    AIProvider
)
from services.chat_service import ChatService
from services.file_service import FileService
from core.session_manager import SessionManager
from core.ai_providers import AIProviderFactory

# Import speech modules
from spech_to_text import get_stt_model
from text_to_spech import get_tts_model

# Initialize global session manager
session_manager = SessionManager()

# Initialize services
chat_service = ChatService(session_manager)
file_service = FileService(session_manager)

# Initialize speech models
stt_model = get_stt_model()
tts_model = get_tts_model()

# Create router
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat request"""
    return await chat_service.process_chat_request(request)

@router.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    ai_provider: AIProvider = Form(AIProvider.GEMINI)
):
    """Upload and process PDF file"""
    return await file_service.upload_pdf(file, session_id, ai_provider)

@router.post("/ai_direct", response_model=DirectAIResponse)
async def ai_direct(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    session_id: str = Form(...),
    ai_provider: AIProvider = Form(AIProvider.GEMINI)
):
    """Process direct AI request with file"""
    try:
        # Save temporary file
        file_path = await file_service.save_temporary_file(file, session_id)

        try:
            # Process with AI
            response_text = await chat_service.process_direct_ai_request(
                file_path, prompt, session_id, ai_provider
            )

            return DirectAIResponse(
                response=response_text,
                ai_provider_used=ai_provider,
                session_id=session_id
            )
        finally:
            # Clean up temporary file
            file_service.cleanup_temporary_file(file_path)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing AI request: {str(e)}"
        )

@router.delete("/clear_session/{session_id}")
async def clear_session(session_id: str):
    """Clear session data and memory"""
    success = await chat_service.clear_session(session_id)
    if success:
        return {"message": f"Session {session_id} cleared successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to clear session")

@router.get("/session_status/{session_id}", response_model=SessionStatus)
async def get_session_status(session_id: str):
    """Get session status and agent type"""
    session_info = await chat_service.get_session_status(session_id)
    return SessionStatus(
        session_id=session_info["session_id"],
        agent_type=session_info["agent_type"],
        has_document=session_info["has_document"],
        can_chat=session_info["can_chat"],
        ai_provider=session_info.get("ai_provider")
    )

@router.get("/conversation_history/{session_id}", response_model=ConversationHistory)
async def get_conversation_history(session_id: str):
    """Get conversation history for debugging"""
    history_info = await chat_service.get_conversation_history(session_id)
    return ConversationHistory(
        session_id=history_info["session_id"],
        agent_type=history_info["agent_type"],
        history=str(history_info["history"]),
        user_info=history_info["user_info"],
        has_agent=history_info["has_agent"],
        ai_provider=history_info.get("ai_provider")
    )

@router.get("/available_providers")
async def get_available_providers():
    """Get list of available AI providers"""
    try:
        available_providers = AIProviderFactory.get_available_providers()
        return {
            "providers": [provider.value for provider in available_providers],
            "count": len(available_providers)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting available providers: {str(e)}"
        )

@router.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    try:
        sessions = session_manager.list_sessions()
        return {
            "sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing sessions: {str(e)}"
        )

@router.post("/switch_provider/{session_id}")
async def switch_ai_provider(session_id: str, ai_provider: AIProvider):
    """Switch AI provider for an existing session"""
    try:
        if not session_manager.has_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        success = session_manager.update_session_provider(session_id, ai_provider)
        if success:
            return {
                "message": f"AI provider switched to {ai_provider.value} for session {session_id}",
                "session_id": session_id,
                "new_provider": ai_provider.value
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to switch AI provider")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error switching AI provider: {str(e)}"
        )
@router.post("/web_search")
async def web_search(query: str, session_id: str):
    """Perform web search using DuckDuckGo"""
    try:
        from utils.tools import WebSearchManager

        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        results = WebSearchManager.web_search(query)

        return {
            "query": query,
            "results": results,
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing web search: {str(e)}"
        )

# Test speech models endpoint
@router.get("/test-speech-models")
async def test_speech_models():
    """Test if speech models are working properly"""
    try:
        # Test STT model
        stt_working = False
        stt_error = None
        if stt_model and stt_model.is_model_loaded():
            try:
                # Create a simple test audio file (silence)
                import numpy as np
                import soundfile as sf

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    # Create 1 second of silence at 16kHz
                    sample_rate = 16000
                    duration = 1.0
                    samples = np.zeros(int(sample_rate * duration), dtype=np.int16)
                    sf.write(temp_file.name, samples, sample_rate)

                    # Try to transcribe (should return empty string for silence)
                    result = stt_model.transcribe_audio_file(temp_file.name)
                    stt_working = True

                    # Clean up
                    os.unlink(temp_file.name)
            except Exception as e:
                stt_error = str(e)
        else:
            stt_error = "STT model not loaded"

        # Test TTS model
        tts_working = False
        tts_error = None
        if tts_model and tts_model.is_model_loaded():
            try:
                # Try to synthesize a simple text
                test_text = "Hello"
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    success = tts_model.synthesize_to_file(test_text, temp_file.name, "af_heart", 1.0, "en-us")
                    tts_working = success

                    # Clean up
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
            except Exception as e:
                tts_error = str(e)
        else:
            tts_error = "TTS model not loaded"

        return {
            "speech_to_text": {
                "working": stt_working,
                "error": stt_error
            },
            "text_to_speech": {
                "working": tts_working,
                "error": tts_error
            }
        }

    except Exception as e:
        return {
            "error": str(e),
            "speech_to_text": {"working": False, "error": str(e)},
            "text_to_speech": {"working": False, "error": str(e)}
        }

# Speech status endpoint
@router.get("/speech-status")
async def get_speech_status():
    """Get status of speech models"""
    try:
        stt_info = stt_model.get_model_info() if stt_model else {"is_loaded": False, "error": "Model not initialized"}
        tts_info = tts_model.get_model_info() if tts_model else {"is_loaded": False, "error": "Model not initialized"}

        return {
            "speech_to_text": {
                "loaded": stt_info.get("is_loaded", False),
                "model_path": stt_info.get("model_path", "Not set"),
                "device": stt_info.get("device", "Unknown"),
                "error": stt_info.get("error", None)
            },
            "text_to_speech": {
                "loaded": tts_info.get("is_loaded", False),
                "available_voices": tts_info.get("available_voices", []),
                "error": tts_info.get("error", None)
            }
        }
    except Exception as e:
        return {
            "speech_to_text": {"loaded": False, "error": str(e)},
            "text_to_speech": {"loaded": False, "error": str(e)}
        }

# Speech-to-Text endpoint
@router.post("/speech-to-text")
async def speech_to_text(file: UploadFile = File(...)):
    """Convert speech to text using Whisper"""
    try:
        # Validate file type - handle None content_type
        if file.content_type is None:
            # If content_type is None, check filename extension
            if not file.filename or not any(file.filename.lower().endswith(ext) for ext in ['.wav', '.mp3', '.m4a', '.ogg']):
                raise HTTPException(status_code=400, detail="File must be an audio file (WAV, MP3, M4A, OGG)")
        elif not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")

        # Check if STT model is loaded
        if not stt_model.is_model_loaded():
            raise HTTPException(status_code=503, detail="Speech-to-Text model not loaded")

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Check file size
            file_size = os.path.getsize(temp_file_path)
            print(f"Audio file size: {file_size} bytes")

            if file_size < 100:  # Very small file, likely empty or corrupted
                raise HTTPException(status_code=400, detail="Audio file is too small or empty")

            # Transcribe audio
            transcribed_text = stt_model.transcribe_audio_file(temp_file_path)

            if transcribed_text is None:
                raise HTTPException(status_code=500, detail="Failed to transcribe audio - model returned None")

            if not transcribed_text.strip():
                raise HTTPException(status_code=500, detail="Transcription result is empty")

            return {
                "success": True,
                "text": transcribed_text,
                "file_name": file.filename,
                "file_size": file_size
            }

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in speech-to-text: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in speech-to-text: {str(e)}"
        )

# Text-to-Speech endpoint
@router.post("/text-to-speech")
async def text_to_speech(
    text: str = Form(...),
    voice: str = Form("af_heart"),
    speed: float = Form(1.0),
    lang: str = Form("en-us")
):
    """Convert text to speech using Kokoro"""
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Generate unique filename
        audio_filename = f"tts_{uuid.uuid4().hex}.wav"
        audio_dir = "audio_files"

        # Create audio directory if it doesn't exist
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)

        audio_path = os.path.join(audio_dir, audio_filename)

        # Synthesize speech
        success = tts_model.synthesize_to_file(text, audio_path, voice, speed, lang)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to synthesize speech")

        return {
            "success": True,
            "audio_file": audio_path,
            "filename": audio_filename,
            "text": text,
            "voice": voice,
            "speed": speed,
            "lang": lang
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in text-to-speech: {str(e)}"
        )

# Serve audio files
@router.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve generated audio files"""
    try:
        audio_dir = "audio_files"
        file_path = os.path.join(audio_dir, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")

        return FileResponse(
            file_path,
            media_type="audio/wav",
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error serving audio file: {str(e)}"
        )

# Session status endpoint (enhanced with speech info)
@router.get("/session-status/{session_id}", response_model=SessionStatus)
async def get_session_status(session_id: str):
    """Get session status and agent type"""
    session_info = await chat_service.get_session_status(session_id)
    return SessionStatus(
        session_id=session_info["session_id"],
        agent_type=session_info["agent_type"],
        has_document=session_info["has_document"],
        can_chat=session_info["can_chat"],
        ai_provider=session_info.get("ai_provider")
    )
