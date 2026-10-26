import os
import sys
from pathlib import Path
import logging
from typing import Optional, Union

import argparse

# ---------------- Project root for module imports ----------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
print(project_root)

# ---------------- Local imports ----------------
from config import config
from data_load.doc_load import PDFHandler
from vector_store.vector_db import ChromaDBHandler
from retrieval.reranker import Reranker
from generation.response_generation_voice import ResponseGenerator
# TTS voices (for validation / discoverability)
from multimodal_rag.speech.TTS import VOICE_NAMES  # uses your local Kokoro pack
from multimodal_rag.speech.STT import get_stt_engine

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AgenticChatSystem:
    """
    End-to-end agent: ingest (optional) → retrieve → rerank → generate → (optional) TTS.
    """
    def __init__(self, session_id: str, pdf_file_path: Optional[str] = None):
        self.session_id = session_id
        logging.info(f"🤖 Initializing Multimodal RAG Chat System for session: {session_id}...")

        # Use single shared knowledge base for Multimodal RAG (not session-specific)
        base_path = Path(config.VECTOR_DB_BASE_PATH) if getattr(config, "VECTOR_DB_BASE_PATH", None) \
            else Path.home() / "Desktop" / "rag_pipelines" / "chroma_db"

        shared_multimodal_path = base_path / "multimodal_rag_collection"

        self.db_handler = ChromaDBHandler(
            session_specific_persist_directory=str(shared_multimodal_path),
            collection_name="multimodal_rag_collection"
        )
        logging.info(f"✅ Multimodal RAG: Using shared knowledge base at {shared_multimodal_path}")

        self.default_audio_out = project_root / "voice_notebook" / "output.wav"
        self.default_audio_out.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists

        self.reranker = Reranker()
        self.generator = ResponseGenerator()
        self.stt_engine = None

        self.chunk_count = 0
        # Optional ingestion on startup
        if pdf_file_path:
            abs_pdf_path = os.path.abspath(pdf_file_path)
            logging.info(f"🔍 Multimodal RAG: Checking ingestion for: {abs_pdf_path}")
            logging.info(f"🔍 Multimodal RAG: File exists check: {os.path.exists(abs_pdf_path)}")
            if os.path.exists(abs_pdf_path):
                logging.info(f"📄 Multimodal RAG: Loading and indexing document: {abs_pdf_path}...")
                logging.info(f"📄 Multimodal RAG: File size: {os.path.getsize(abs_pdf_path)} bytes")
                try:
                    pdf_handler = PDFHandler(abs_pdf_path)
                    chunks = pdf_handler.get_chunks()
                    logging.info(f"🔍 Multimodal RAG: Got {len(chunks) if chunks else 0} chunks from PDFHandler")
                    if chunks:
                        self.db_handler.build_index(chunks)
                        self.chunk_count = len(chunks)
                        logging.info(f"✅ Multimodal RAG: Indexed {len(chunks)} text chunks from {abs_pdf_path}")
                    else:
                        logging.warning(f"⚠️ Multimodal RAG: No chunks generated from {abs_pdf_path}")
                except Exception as e:
                    logging.error(f"❌ Multimodal RAG: Error during document processing/indexing: {e}", exc_info=True)
            else:
                logging.error(f"❌ Multimodal RAG: Ingestion file not found on disk: {abs_pdf_path}")
        else:
            logging.info(f"ℹ️ Multimodal RAG: No PDF path provided. Loading existing DB for session {session_id}.")

        logging.info(f"✅ Agentic Chat System initialized successfully for session: {session_id}!")

    def transcribe_audio(self, audio_path: str) -> str:
        """Lazy-load STT engine and transcribe audio."""
        if self.stt_engine is None:
            self.stt_engine = get_stt_engine()
        return self.stt_engine.transcribe(audio_path)

    def get_response(
        self,
        user_input: str,
        generation_mode: str,
        *,
        speak: bool = False,
        voice: Optional[str] = None,
        audio_out_path: Optional[Union[str, Path]] = None,
        top_k: Optional[int] = None,
        uploaded_genai_file = None,  # NEW: accept genai file object or similar handle
        pdf_text: Optional[str] = None  # NEW: accept extracted text
    ) -> Union[str, dict]:
        """
        Get response to user query using the specified generation mode.

        Args:
            user_input: The user's query or input.
            generation_mode: 'api' or 'local'.
            speak: If True, synthesize the final answer to speech.
            voice: Voice to use (must be one of VOICE_NAMES) when speak=True.
            audio_out_path: Where to write the WAV. If None, uses voice_notebook/output.wav.
            top_k: Optional override for retrieval K (else uses config.TOP_K).
            uploaded_genai_file: Optional file object for multimodal requests.
            pdf_text: Optional text extracted from PDF attachment.

        Returns:
            str if speak=False
            dict {text, audio_path, voice_used} if speak=True
        """
        if not user_input or not user_input.strip():
            # If there's an image but no text, provide default text
            if uploaded_genai_file:
                user_input = "Describe this content."
            else:
                return "Please provide a valid query."

        if speak:
            v = voice or "af_sarah"
            if v not in VOICE_NAMES:
                return f"Invalid voice '{v}'. Choose one of: {VOICE_NAMES}"
            voice = v

        try:

            # --- Intelligent Global Query Detection ---
            is_global_query = any(kw in user_input.lower() for kw in ["summarize", "summary", "overview", "about", "describe document"])

            k = top_k if (isinstance(top_k, int) and top_k > 0) else getattr(config, "TOP_K", 5)
            if is_global_query:
                k = max(k, 12) # Fetch more segments for a summary
                logging.info(f"🔎 Global query detected ('{user_input[:20]}...'), increasing k to {k}")

            # Always try to retrieve docs, even if multimodal
            try:
                retrieved_documents = self.db_handler.search(user_input, k=k)
                if not retrieved_documents and is_global_query:
                     # Fallback for empty summaries: search for general terms
                     logging.info("⚠️ Vector search empty for global query. Retrying with generic terms...")
                     retrieved_documents = self.db_handler.search("introduction summary overview", k=8)
            except Exception as e:
                logging.warning(f"Retrieval error: {e}")
                retrieved_documents = None

            context_text = "No specific document context found."
            if retrieved_documents:
                doc_texts = [doc.page_content for doc in retrieved_documents]
                reranked_results = self.reranker.rerank(user_input, doc_texts)

                # Loosen threshold for global queries or use top-ranked docs
                score_threshold = 0.0001 if is_global_query else 0.001
                relevant_docs = [doc for doc in reranked_results if doc['score'] > score_threshold]

                if relevant_docs:
                    # Increased limit from 3 to 5 for better context coverage (matching Agentic RAG)
                    limit = 8 if is_global_query else 5
                    top_docs = relevant_docs[:limit]
                    context_text = "\n---\n".join([doc["text"] for doc in top_docs])
                    logging.info(f"Final context built from {len(top_docs)} segments.")
                else:
                    if reranked_results:
                        context_text = reranked_results[0]["text"]
                        logging.info("Using top-ranked document as fallback context.")


            out = self.generator.generate(
                user_input,
                context_text,
                self.session_id,
                mode=generation_mode,
                uploaded_genai_file=uploaded_genai_file,  # Pass the file
                speak=speak,
                voice=voice,
                audio_out_path=str(audio_out_path or self.default_audio_out),
                pdf_text=pdf_text
            )

            if isinstance(out, dict):
                logging.info(f"Generated (voice={out.get('voice_used')}): {out.get('text','')[:100]}...")
                logging.info(f"Audio saved to: {out.get('audio_path')}")
                return out
            else:
                logging.info(f"Generated text: {out[:100]}...")
                return out

        except Exception as e:
            logging.error(f"Error getting RAG response: {e}", exc_info=True)
            return f"Sorry, I encountered an error: {str(e)}"

    def available_voices(self):
        """Expose the Kokoro voice list (useful for UI)."""
        return VOICE_NAMES

    def cleanup(self):
        """Cleanup all resources."""
        logging.info("Cleaning up Agentic Chat System resources...")
        if hasattr(self, 'db_handler'):
            self.db_handler.cleanup()
        if hasattr(self, 'reranker'):
            del self.reranker
        if hasattr(self, 'generator'):
            self.generator.cleanup()
        if self.stt_engine:
            del self.stt_engine
