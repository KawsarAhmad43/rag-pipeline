import os
from chromadb import PersistentClient
from config import config
import logging
import numpy as np
import re

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        # Ensure the directories exist
        os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)
        os.makedirs(config.CHAT_MEMORY_PATH, exist_ok=True)

        logger.info(f"Initializing vector database at {config.VECTOR_DB_PATH}")
        try:
            self.client = PersistentClient(path=config.VECTOR_DB_PATH)
            self.vector_db = self.client.get_or_create_collection(
                name="rag_collection",
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(f"Initializing chat memory at {config.CHAT_MEMORY_PATH}")
            self.chat_memory_client = PersistentClient(path=config.CHAT_MEMORY_PATH)
            self.chat_memory = self.chat_memory_client.get_or_create_collection(
                name="chat_memory",
                metadata={"hnsw:space": "cosine"}
            )
            self.max_chat_history = 100  # Increased for better memory retention

            logger.info("Database initialization complete")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def store_document(self, text, embedding, metadata=None):
        """Store document in Simple RAG's single shared knowledge base with metadata"""
        try:
            if not text or not embedding:
                logger.warning("Attempted to store empty document or embedding")
                return

            # Check if embedding contains NaN or inf values
            if isinstance(embedding, list):
                embedding_np = np.array(embedding)
                if not np.all(np.isfinite(embedding_np)):
                    logger.warning("Embedding contains NaN or inf values - fixing before storage")
                    embedding = np.nan_to_num(embedding_np).tolist()

            # Prepare metadata with source information
            doc_metadata = {"context_type": "pdf"}
            if metadata:
                doc_metadata.update(metadata)

            # Use Simple RAG's single shared collection
            self.vector_db.add(
                embeddings=[embedding],
                documents=[text],
                ids=[str(len(self.vector_db.get()["ids"]) + 1)],
                metadatas=[doc_metadata]
            )
            logger.debug(f"Stored document in Simple RAG knowledge base: {text[:50]}... with metadata: {doc_metadata}")
        except ValueError as ve:
            if "dimension" in str(ve).lower():
                logger.warning(f"Embedding dimension mismatch detected. Resetting rag_collection.")
                self.client.delete_collection(name="rag_collection")
                self.vector_db = self.client.get_or_create_collection(
                    name="rag_collection",
                    metadata={"hnsw:space": "cosine"}
                )
                # Prepare metadata for retry
                doc_metadata = {"context_type": "pdf"}
                if metadata:
                    doc_metadata.update(metadata)
                # Retry adding
                self.vector_db.add(
                    embeddings=[embedding],
                    documents=[text],
                    ids=[str(len(self.vector_db.get()["ids"]) + 1)],
                    metadatas=[doc_metadata]
                )
                logger.debug(f"Stored document after resetting collection: {text[:50]}...")
            else:
                logger.error(f"Error storing document: {str(ve)}")
                raise
        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            raise

    def store_chat_memory(self, query, response, embedding_model, context_type="not_pdf"):
        generic_error_phrases = [
            "i'm sorry, something went wrong",
            "i'm sorry, i encountered an error",
            "couldn't generate a response",
            "no relevant information found",
            "error:"
        ]

        if any(phrase in response.lower() for phrase in generic_error_phrases):
            logger.warning(f"Skipping storage of generic error response for query: {query[:50]}...")
            return

        try:
            if not query or not response:
                logger.warning("Attempted to store empty query or response")
                return

            embedding = embedding_model.embed_query(query)

            current_memory = self.chat_memory.get()
            memory_id = str(len(current_memory.get("ids", [])) + 1)

            if len(current_memory.get("ids", [])) >= self.max_chat_history:
                oldest_id = current_memory["ids"][0]
                self.chat_memory.delete(ids=[oldest_id])
                logger.debug(f"Removed oldest chat memory entry with ID {oldest_id}")

            self.chat_memory.add(
                embeddings=[embedding],
                documents=[f"Q: {query}\nA: {response}"],
                ids=[memory_id],
                metadatas=[{"context_type": context_type}]
            )
            logger.debug(f"Stored chat memory with ID {memory_id}: Q: {query[:50]}...")
        except ValueError as ve:
            if "dimension" in str(ve).lower():
                logger.warning("Embedding dimension mismatch detected. Resetting chat_memory collection.")
                self.chat_memory_client.delete_collection(name="chat_memory")
                self.chat_memory = self.chat_memory_client.get_or_create_collection(
                    name="chat_memory",
                    metadata={"hnsw:space": "cosine"}
                )
                # Retry adding
                self.chat_memory.add(
                    embeddings=[embedding],
                    documents=[f"Q: {query}\nA: {response}"],
                    ids=[memory_id],
                    metadatas=[{"context_type": context_type}]
                )
                logger.debug(f"Stored chat memory after resetting collection: Q: {query[:50]}...")
            else:
                logger.error(f"Error storing chat memory: {str(ve)}")
                raise
        except Exception as e:
            logger.error(f"Error storing chat memory: {str(e)}")
            raise

    def search_chat_memory(self, query, embedding_model, context_type="not_pdf"):
        try:
            current_memory = self.chat_memory.get()
            if not current_memory.get("ids"):
                logger.debug("Chat memory is empty, no search performed")
                return None

            # Check for meta-queries about previous interactions
            meta_query_pattern = re.compile(r"(previous|last)\s+query", re.IGNORECASE)
            if meta_query_pattern.search(query):
                logger.debug(f"Detected meta-query for previous query: {query[:50]}...")
                # Retrieve all chat memory entries and sort by ID to get the most recent
                results = self.chat_memory.get(
                    include=["documents", "metadatas", "ids"],
                    where={"context_type": context_type}
                )
                if not results["ids"]:
                    logger.debug("No previous query found in chat memory")
                    return "No previous query found in chat memory."

                # Sort IDs to find the most recent entry (highest ID)
                sorted_indices = sorted(range(len(results["ids"])), key=lambda i: int(results["ids"][i]), reverse=True)
                for idx in sorted_indices:
                    document = results["documents"][idx]
                    parts = document.split("\nA: ")
                    if len(parts) > 1:
                        query_part = parts[0].replace("Q: ", "").strip()
                        # Skip if the query is a meta-query itself to avoid self-reference
                        if not meta_query_pattern.search(query_part):
                            logger.debug(f"Returning previous query: {query_part[:50]}...")
                            return f"The previous query was: {query_part}"
                logger.debug("No valid previous query found (all entries are meta-queries)")
                return "No previous query found in chat memory."

            # Standard similarity-based search
            embedding = embedding_model.embed_query(query)

            results = self.chat_memory.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["documents", "distances", "metadatas"],
                where={"context_type": context_type}
            )

            if not results["documents"] or not results["documents"][0]:
                logger.debug(f"No chat memory found for query: {query[:50]}... with context_type: {context_type}")
                return None

            distance = results["distances"][0][0] if results["distances"] and results["distances"][0] else 1.0
            logger.debug(f"Chat memory search for query: '{query[:50]}...', distance: {distance}, context_type: {context_type}")

            if distance > 0.1:
                logger.debug(f"Chat memory found but similarity too low: {distance}")
                return None

            document = results["documents"][0][0]
            parts = document.split("\nA: ")
            if len(parts) > 1:
                logger.debug(f"Returning chat memory response: {parts[1][:50]}...")
                return parts[1].strip()

            logger.warning(f"Malformed chat memory document: {document[:50]}...")
            return None
        except Exception as e:
            logger.error(f"Error searching chat memory: {str(e)}")
            return None

db = Database()
