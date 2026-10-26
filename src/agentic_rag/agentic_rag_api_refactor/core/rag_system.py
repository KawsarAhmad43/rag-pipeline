import os
import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import Field

from config.settings import settings
from models.api_models import AIProvider
from utils.pdf_handler import PDFHandler
from utils.vector_db import VectorDB
from utils.reranker import Reranker
from core.ai_providers import AIProviderFactory, BaseAIProvider

# Import TavilySearch and handle potential import error
try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
    print("✅ TavilySearch import successful.")
except ImportError:
    print("⚠️ langchain-tavily not installed. Web search will be disabled.")
    TAVILY_AVAILABLE = False


class RAGSystem:
    """
    Robust RAG system with a direct, reliable web search implementation.
    """
    def __init__(
        self,
        ai_provider: AIProvider,
        session_id: str,
        pdf_file_path: Optional[str] = None,
        existing_memory: Optional[ConversationBufferMemory] = None
    ):
        self.ai_provider_type = ai_provider
        self.session_id = session_id
        self.pdf_file_path = pdf_file_path

        # Initialize AI provider
        self.ai_provider = AIProviderFactory.create_provider(ai_provider)

        # Initialize memory
        self.memory = existing_memory or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )

        # Initialize shared Agentic RAG vector DB (always, not just when PDF uploaded)
        # This ensures queries can access the shared knowledge base
        chroma_db_path = os.path.join(settings.ALL_DB_DIR, "agentic_rag_collection")
        os.makedirs(chroma_db_path, exist_ok=True)

        self.vector_db = VectorDB(
            embedding_model_path=settings.EMBEDDING_MODEL_PATH,
            persist_directory=chroma_db_path
        )
        print(f"✅ Agentic RAG: Vector DB initialized for shared knowledge base")

        # Initialize RAG components if PDF provided
        self.reranker = None
        self.chunk_count = 0
        if pdf_file_path:
            self._initialize_rag_components()

        # Initialize the direct web search tool
        self._initialize_web_search_tool()

    def _initialize_rag_components(self):
        """Initialize RAG components for document-based chat"""
        try:
            print(f"🔍 Agentic RAG: Processing PDF file: {self.pdf_file_path}")

            pdf_handler = PDFHandler(self.pdf_file_path)
            chunks = pdf_handler.get_chunks()
            self.chunk_count = len(chunks)
            print(f"✅ Agentic RAG: Extracted {self.chunk_count} chunks from PDF")

            # Use the already initialized shared vector DB and add chunks to it
            self.vector_db.build_index(chunks)
            print(f"✅ Agentic RAG: Indexed {self.chunk_count} chunks into shared Agentic RAG knowledge base")

            # Try to load reranker, but don't fail if it doesn't work
            if os.path.exists(settings.RERANKER_MODEL_PATH):
                try:
                    self.reranker = Reranker(settings.RERANKER_MODEL_PATH)
                    print(f"✅ Agentic RAG: Reranker loaded successfully")
                except Exception as reranker_error:
                    print(f"⚠️ Agentic RAG: Reranker failed to load, continuing without it: {reranker_error}")
                    self.reranker = None
            else:
                print(f"ℹ️ Agentic RAG: Reranker model not found at {settings.RERANKER_MODEL_PATH}")

        except Exception as e:
            print(f"❌ Agentic RAG: Error initializing RAG components: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _initialize_web_search_tool(self):
        """Initializes the Tavily search tool for direct use."""
        self.web_search_tool = None
        if not TAVILY_AVAILABLE:
            print("❌ TavilySearch not available due to import error.")
            return

        if not settings.TAVILY_API_KEY:
            print("❌ TAVILY_API_KEY not configured. Web search is disabled.")
            return

        try:
            self.web_search_tool = TavilySearch(
                max_results=5,
                api_key=settings.TAVILY_API_KEY
            )
            print("✅ Tavily search tool initialized successfully.")
        except Exception as e:
            print(f"❌ Error initializing Tavily search tool: {e}")

    def _is_simple_datetime_query(self, user_input: str) -> bool:
        """Checks for simple date/time queries that don't need web search."""
        user_input_lower = user_input.lower()
        patterns = [
            r"what.{0,10}(is|s)?.{0,10}(today|date|time)",
            r"what.{0,10}day.{0,10}is",
            r"current.{0,10}(date|time)",
            r"^(date|time)$",
        ]
        for pattern in patterns:
            if re.search(pattern, user_input_lower):
                return True
        return False

    def _should_use_web_search(self, user_input: str) -> bool:
        """Determines if a query needs real-time web search."""
        if self._is_simple_datetime_query(user_input):
            return False

        user_input_lower = user_input.lower()

        # High-intent search triggers (require web search)
        real_time_triggers = [
            "latest", "current", "recent", "news", "price", "stock", "weather",
            "happening", "update", "bitcoin", "cryptocurrency", "today", "now"
        ]

        # Collaborative search triggers (only if not found in PDF)
        general_triggers = ["search", "web search", "look up", "find", "who is", "what is"]

        # 1. Check for real-time triggers first
        for trigger in real_time_triggers:
            if trigger in user_input_lower:
                print(f"🔍 Web search triggered by real-time keyword: '{trigger}'")
                return True

        # 2. For general triggers, only use web search if NO PDF context is available
        # This prevents "Hi" or basic questions from triggering web search when we have a document
        if not self.vector_db:
            for trigger in general_triggers:
                if trigger in user_input_lower:
                    # Avoid triggering on very short queries like "Hi" or "What is"
                    if len(user_input_lower.split()) > 2:
                        print(f"🔍 Web search triggered by general keyword: '{trigger}' (No PDF available)")
                        return True

        return False

    def get_response(self, user_input: str) -> str:
        """
        Gets a response, intelligently deciding between web search and regular chat.
        """
        try:
            print(f"\n🔄 Processing user input: '{user_input}'")

            if self.web_search_tool and self._should_use_web_search(user_input):
                response = self._get_web_search_response(user_input)
            else:
                response = self._get_regular_response(user_input)

            # Save context to memory
            self.memory.save_context({"input": user_input}, {"output": response})
            return response

        except Exception as e:
            print(f"❌ Critical error in get_response: {e}")
            return "I'm sorry, a critical error occurred. Please try again."

    def _get_web_search_response(self, user_input: str) -> str:
        """
        Performs a direct web search and uses the LLM to synthesize a final answer.
        This is a reliable Retrieve-Augment-Generate (RAG) flow.
        """
        try:
            print("🌐 Using direct web search RAG flow...")

            # 1. Retrieve: Call the Tavily tool directly
            print(f"   - Searching for: '{user_input}'")
            search_results = self.web_search_tool.invoke(user_input)

            if not search_results:
                print("   - ⚠️ No search results returned from Tavily.")
                return "I couldn't find any information on that topic. Could you try rephrasing your search?"

            print(f"   - ✅ Search results received.")

            # 2. Augment: Create a new, clear prompt for the LLM
            current_datetime = datetime.now().strftime("%A, %B %d, %Y")

            prompt = f"""
You are a helpful AI assistant. Your task is to answer the user's question based on the provided real-time web search results.

Current Date: {current_datetime}

User's Original Question: "{user_input}"

Web Search Results:
---
{search_results}
---

Instructions:
- Synthesize the information from the web search results into a clear and concise answer.
- Directly answer the user's question.
- Do NOT mention that you performed a search or mention "based on the search results."
- If the results contain a direct answer, provide it.
- If the results are mixed, provide a summary.
- If the results do not answer the question, state that you couldn't find a specific answer.
"""

            # 3. Generate: Call the LLM to get the final, synthesized response
            print("   - 🧠 Synthesizing final answer with LLM...")
            final_response = self.ai_provider.generate_text(prompt)
            print("   - ✅ Final answer generated.")

            return final_response

        except Exception as e:
            print(f"❌ Error in direct web search RAG flow: {e}")
            return "I'm sorry, but I encountered an error while trying to find the latest information. Please try again shortly."

    def _get_regular_response(self, user_input: str) -> str:
        """Handles regular conversation or document-based Q&A with deep context awareness."""
        try:
            print("💬 Using regular response flow...")
            context = ""

            # 1. Broad query detection for Summary/Overview
            is_global_query = any(keyword in user_input.lower() for keyword in ["summarize", "summary", "overview", "about", "describe document"])

            # Add document context if available
            if self.vector_db:
                search_k = 10 if is_global_query else 5
                docs = self.vector_db.search(user_input, k=search_k)

                if docs:
                    context += "[DOCUMENT CONTEXT]\n"
                    for i, d in enumerate(docs, 1):
                        source = d.metadata.get("source", "Document")
                        page = d.metadata.get("page", "?")
                        context += f"Segment {i} (Source: {source}, Page: {page}):\n{d.page_content}\n---\n"
                elif is_global_query:
                    # Fallback for summaries if vector search is thin: try a generic query
                    backup_docs = self.vector_db.search("introduction overview summary", k=5)
                    if backup_docs:
                         context += "[GENERIC DOCUMENT OVERVIEW]\n"
                         for i, d in enumerate(backup_docs, 1):
                             context += f"Overview Segment {i}:\n{d.page_content}\n---\n"

            # 2. History retrieval
            history = self.memory.load_memory_variables({}).get("chat_history", [])

            # 3. Comprehensive Professional Prompt
            current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
            prompt = f"""
You are a highly skilled AI assistant. Your goal is to provide exceptional, accurate, and helpful responses.

[SYSTEM INFO]
Current Time: {current_datetime}
Document Loaded: {"Yes" if self.vector_db else "No"}

[GUIDELINES]
1. Prioritize Sources: If [DOCUMENT CONTEXT] is present, use it to provide a detailed and factual answer.
2. Global Queries: If the user asks for a summary or general info, synthesize all provided context segments into a clear overview.
3. Conversational Fallback: If context is missing or irrelevant, use your pre-trained knowledge to remain helpful. Do NOT give a hard refusal; instead, offer general information while noting it is not from the specific document.
4. Professional Style: Use professional language and Markdown formatting (bolding, bullet points) for readability.

[CONVERSATION HISTORY]
{history if history else "No previous messages."}

[CONTEXT DATA]
{context if context else "No document data found for this specific query."}

[USER QUESTION]
{user_input}

Assistant Response:
"""
            return self.ai_provider.generate_text(prompt)

        except Exception as e:
            print(f"❌ Error in regular response: {e}")
            return "I'm sorry, I had trouble processing that. Could you please rephrase?"

    def get_memory(self) -> ConversationBufferMemory:
        return self.memory

    def cleanup(self):
        if self.vector_db:
            self.vector_db.cleanup()
        print(f"RAG system cleaned up for session: {self.session_id}")
