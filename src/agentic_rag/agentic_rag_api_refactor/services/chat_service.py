import os
from typing import Optional
from fastapi import HTTPException

from config.settings import settings
from models.api_models import ChatRequest, ChatResponse, AIProvider
from core.session_manager import SessionManager
from core.ai_providers import AIProviderFactory
from utils.validation import validate_uuid

class ChatService:
    """Service for handling chat operations"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    async def process_chat_request(self, request: ChatRequest) -> ChatResponse:
        """Process a chat request and return response"""
        # Validate session ID
        if not validate_uuid(request.session_id):
            raise HTTPException(status_code=400, detail="Invalid session ID")

        # Validate input
        if not request.user_input.strip():
            raise HTTPException(status_code=400, detail="Please provide some input")

        try:
            # Check if session exists
            if not self.session_manager.has_session(request.session_id):
                # Create new session with specified AI provider
                success = self.session_manager.create_session(
                    session_id=request.session_id,
                    ai_provider=request.ai_provider
                )
                if not success:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to initialize chat session"
                    )
            else:
                # Check if AI provider changed
                current_provider = self.session_manager.get_session_provider(request.session_id)
                if current_provider != request.ai_provider:
                    print(f"Switching AI provider from {current_provider} to {request.ai_provider}")
                    success = self.session_manager.update_session_provider(
                        request.session_id,
                        request.ai_provider
                    )
                    if not success:
                        raise HTTPException(
                            status_code=500,
                            detail="Failed to update AI provider"
                        )

            # Get RAG system and process request
            rag_system = self.session_manager.get_session(request.session_id)
            if not rag_system:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to get chat session"
                )

            # Get response from RAG system
            response_text = rag_system.get_response(request.user_input)

            return ChatResponse(
                response=response_text,
                ai_provider_used=request.ai_provider,
                session_id=request.session_id
            )

        except HTTPException:
            raise
        except Exception as e:
            print(f"Error in chat service: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request: {str(e)}"
            )

    async def process_direct_ai_request(
        self,
        file_path: str,
        prompt: str,
        session_id: str,
        ai_provider: AIProvider
    ) -> str:
        """Process direct AI request with file"""
        try:
            # Validate session ID
            if not validate_uuid(session_id):
                raise HTTPException(status_code=400, detail="Invalid session ID")

            # Create AI provider
            provider = AIProviderFactory.create_provider(ai_provider)

            # Generate response with file
            response_text = provider.generate_with_image(prompt, file_path)

            # Save to session memory if session exists
            if self.session_manager.has_session(session_id):
                rag_system = self.session_manager.get_session(session_id)
                if rag_system:
                    try:
                        memory = rag_system.get_memory()
                        memory.save_context(
                            {"input": f"Prompt: {prompt} (File: {os.path.basename(file_path)})"},
                            {"output": response_text}
                        )
                    except Exception as mem_error:
                        print(f"Error saving to memory: {mem_error}")

            return response_text

        except HTTPException:
            raise
        except Exception as e:
            print(f"Error in direct AI request: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing AI request: {str(e)}"
            )

    async def clear_session(self, session_id: str) -> bool:
        """Clear a chat session"""
        if not validate_uuid(session_id):
            raise HTTPException(status_code=400, detail="Invalid session ID")

        try:
            return self.session_manager.clear_session(session_id)
        except Exception as e:
            print(f"Error clearing session: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error clearing session: {str(e)}"
            )

    async def get_session_status(self, session_id: str) -> dict:
        """Get session status"""
        if not validate_uuid(session_id):
            raise HTTPException(status_code=400, detail="Invalid session ID")

        try:
            return self.session_manager.get_session_info(session_id)
        except Exception as e:
            print(f"Error getting session status: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error getting session status: {str(e)}"
            )

    async def get_conversation_history(self, session_id: str) -> dict:
        """Get conversation history for a session"""
        if not validate_uuid(session_id):
            raise HTTPException(status_code=400, detail="Invalid session ID")

        try:
            return self.session_manager.get_conversation_history(session_id)
        except Exception as e:
            print(f"Error getting conversation history: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error retrieving conversation history: {str(e)}"
            )
