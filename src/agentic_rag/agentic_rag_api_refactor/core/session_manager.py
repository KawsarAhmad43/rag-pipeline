from typing import Dict, Optional
from langchain.memory import ConversationBufferMemory

from models.api_models import AIProvider
from core.rag_system import RAGSystem
from utils.tools import ToolsManager

class SessionManager:
    """Manage user sessions and their associated RAG systems"""

    def __init__(self):
        self.rag_sessions: Dict[str, RAGSystem] = {}
        self.memory_store: Dict[str, ConversationBufferMemory] = {}
        self.session_providers: Dict[str, AIProvider] = {}
        self.tools_manager = ToolsManager()

    def create_session(
        self,
        session_id: str,
        ai_provider: AIProvider,
        pdf_file_path: Optional[str] = None
    ) -> dict:
        """Create a new session with the specified AI provider
        Returns dict with success status and chunk_count
        """
        try:
            # Get existing memory if available
            existing_memory = self.memory_store.get(session_id)

            # Create RAG system
            rag_system = RAGSystem(
                ai_provider=ai_provider,
                session_id=session_id,
                pdf_file_path=pdf_file_path,
                existing_memory=existing_memory
            )

            # Store session
            self.rag_sessions[session_id] = rag_system
            self.memory_store[session_id] = rag_system.get_memory()
            self.session_providers[session_id] = ai_provider

            session_type = "document-based" if pdf_file_path else "simple"
            chunk_count = rag_system.chunk_count if pdf_file_path else 0
            print(f"✅ {session_type} session created for {session_id} using {ai_provider.value}, {chunk_count} chunks indexed")
            return {"success": True, "chunk_count": chunk_count}

        except Exception as e:
            print(f"Error creating session {session_id}: {str(e)}")
            return {"success": False, "chunk_count": 0}

    def get_session(self, session_id: str) -> Optional[RAGSystem]:
        """Get RAG system for a session"""
        return self.rag_sessions.get(session_id)

    def has_session(self, session_id: str) -> bool:
        """Check if session exists"""
        return session_id in self.rag_sessions

    def get_session_provider(self, session_id: str) -> Optional[AIProvider]:
        """Get AI provider for a session"""
        return self.session_providers.get(session_id)

    def update_session_provider(self, session_id: str, ai_provider: AIProvider) -> bool:
        """Update AI provider for an existing session"""
        try:
            existing_rag = self.rag_sessions.get(session_id)
            if not existing_rag:
                return False

            # Get current state
            existing_memory = existing_rag.get_memory()
            pdf_file_path = existing_rag.pdf_file_path

            # Clean up old session
            existing_rag.cleanup()

            # Create new session with updated provider
            return self.create_session(session_id, ai_provider, pdf_file_path)

        except Exception as e:
            print(f"Error updating session provider for {session_id}: {str(e)}")
            return False

    def clear_session(self, session_id: str) -> bool:
        """Clear a session and its resources"""
        try:
            # Clean up RAG system
            if session_id in self.rag_sessions:
                self.rag_sessions[session_id].cleanup()
                del self.rag_sessions[session_id]

            # Clear memory
            if session_id in self.memory_store:
                del self.memory_store[session_id]

            # Clear provider info
            if session_id in self.session_providers:
                del self.session_providers[session_id]

            # Clear user info
            self.tools_manager.clear_user_info(session_id)

            print(f"✅ Session {session_id} cleared successfully")
            return True

        except Exception as e:
            print(f"Error clearing session {session_id}: {str(e)}")
            return False

    def get_session_info(self, session_id: str) -> Dict:
        """Get information about a session"""
        rag_system = self.rag_sessions.get(session_id)
        ai_provider = self.session_providers.get(session_id)

        if not rag_system:
            return {
                "session_id": session_id,
                "exists": False,
                "agent_type": "none",
                "has_document": False,
                "can_chat": False,
                "ai_provider": None
            }

        has_document = rag_system.pdf_file_path is not None
        agent_type = "document-based" if has_document else "simple"

        return {
            "session_id": session_id,
            "exists": True,
            "agent_type": agent_type,
            "has_document": has_document,
            "can_chat": True,
            "ai_provider": ai_provider.value if ai_provider else None
        }

    def get_conversation_history(self, session_id: str) -> Dict:
        """Get conversation history for a session"""
        rag_system = self.rag_sessions.get(session_id)
        ai_provider = self.session_providers.get(session_id)

        if not rag_system:
            return {
                "session_id": session_id,
                "agent_type": "none",
                "chat_history": "No session found",
                "user_info": {},
                "has_agent": False,
                "ai_provider": None
            }

        try:
            memory = rag_system.get_memory()
            history_vars = memory.load_memory_variables({})
            history = history_vars.get("chat_history", "")

            # Get user info
            user_info = self.tools_manager.get_user_info_for_session(session_id)

            has_document = rag_system.pdf_file_path is not None
            agent_type = "document-based" if has_document else "simple"

            return {
                "session_id": session_id,
                "agent_type": agent_type,
                "chat_history": history,
                "user_info": user_info,
                "has_agent": True,
                "ai_provider": ai_provider.value if ai_provider else None
            }

        except Exception as e:
            return {
                "session_id": session_id,
                "agent_type": "error",
                "chat_history": f"Error retrieving history: {str(e)}",
                "user_info": {},
                "has_agent": False,
                "ai_provider": ai_provider.value if ai_provider else None
            }

    def list_sessions(self) -> Dict[str, Dict]:
        """List all active sessions"""
        return {
            session_id: self.get_session_info(session_id)
            for session_id in self.rag_sessions.keys()
        }

    def cleanup_all(self):
        """Cleanup all sessions"""
        for session_id in list(self.rag_sessions.keys()):
            self.clear_session(session_id)
        print("All sessions cleaned up")
