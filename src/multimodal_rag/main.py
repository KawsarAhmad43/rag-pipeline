import os
import sys
from pathlib import Path
import logging
from typing import Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
print(project_root)

from config import config
from data_load.doc_load import PDFHandler
from vector_store.vector_db import ChromaDBHandler
from retrieval.reranker import Reranker

from generation.response_generator import ResponseGenerator


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AgenticChatSystem:
    def __init__(self, session_id: str, pdf_file_path: Optional[str] = None):
        self.session_id = session_id

        logging.info(f"🤖 Initializing Multimodal RAG Chat System for session: {session_id}...")

        # Use single shared knowledge base for Multimodal RAG (not session-specific)
        base_path = Path(config.VECTOR_DB_BASE_PATH) if config.VECTOR_DB_BASE_PATH else Path.home() / "Desktop" / "rag_pipelines" / "chroma_db"

        shared_multimodal_path = base_path / "multimodal_rag_collection"

        self.db_handler = ChromaDBHandler(
            session_specific_persist_directory=str(shared_multimodal_path),
            collection_name="multimodal_rag_collection"
        )
        logging.info(f"✅ Multimodal RAG: Using shared knowledge base at {shared_multimodal_path}")

        self.chunk_count = 0
        if pdf_file_path and os.path.exists(pdf_file_path):
            logging.info(f"Loading and indexing PDF: {pdf_file_path}...")
            try:
                pdf_handler = PDFHandler(pdf_file_path)
                chunks = pdf_handler.get_chunks()
                if chunks:
                    self.db_handler.build_index(chunks)
                    self.chunk_count = len(chunks)
                    logging.info(f"✅ Multimodal RAG: Indexed {len(chunks)} text chunks into shared knowledge base")
            except Exception as e:
                logging.error(f"Error during PDF processing/indexing: {e}", exc_info=True)
        else:
            logging.info(f"No new PDF provided. Using existing shared Multimodal RAG knowledge base.")

        self.reranker = Reranker()
        self.generator = ResponseGenerator()

        logging.info(f"✅ Multimodal RAG Chat System initialized successfully for session: {session_id}!")


    def get_response(self, user_input: str, generation_mode: str) -> str:
        """
        Get response to user query using the specified generation mode.

        Args:
            user_input (str): The user's query or input.
            generation_mode (str): Either 'api' or 'local' for response generation.

        Returns:
            str: The generated response.
        """
        if not user_input or not user_input.strip():
            return "Please provide a valid query."

        try:

            retrieved_documents = self.db_handler.search(user_input, k=config.TOP_K)
            context_text = "No specific context found related to your question."
            if retrieved_documents:

                doc_texts = [doc.page_content for doc in retrieved_documents]

                reranked_results = self.reranker.rerank(user_input, doc_texts)

                score_threshold = 0.001
                relevant_docs = [doc for doc in reranked_results if doc['score'] > score_threshold]

                if relevant_docs:

                    top_docs = relevant_docs[:3]
                    context_text = "\n\n".join([doc["text"] for doc in top_docs])
                    logging.info(f"Final context built from {len(top_docs)} high-quality documents.")
                    logging.debug(f"Final context for LLM: '{context_text[:300]}...'")
                else:

                    if reranked_results:
                        context_text = reranked_results[0]["text"]
                        logging.info("Using top-ranked document despite low threshold score.")
                    else:
                        logging.warning("No documents retrieved at all.")


            response = self.generator.generate(
                user_input,
                context_text,
                self.session_id,
                mode=generation_mode
            )
            logging.info(f"Generated response for query '{user_input[:50]}...': {response[:100]}...")
            return response

        except Exception as e:
            logging.error(f"Error getting RAG response: {e}", exc_info=True)
            return f"Sorry, I encountered an error: {str(e)}"

    def cleanup(self):
        """Cleanup all resources."""
        logging.info("Cleaning up Agentic Chat System resources...")
        if hasattr(self, 'db_handler'):
            self.db_handler.cleanup()
        if hasattr(self, 'reranker'):
            del self.reranker
        if hasattr(self, 'generator'):
            self.generator.cleanup()
