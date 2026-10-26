import chromadb
import logging
from config import config
from utils.logging import logger

class Database:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
        self.vector_collection = self.client.get_or_create_collection("documents")
        self.chat_memory = self.client.get_or_create_collection("chat_memory")
        logger.info("ChromaDB initialized with collections: documents, chat_memory")

    def store_document(self, text: str, embedding: list, metadata: dict = None):
        """Store a document with its embedding in ChromaDB."""
        try:
            doc_id = str(abs(hash(text)))  # Use abs to avoid negative hashes
            self.vector_collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata or {}]
            )
            logger.debug(f"Stored document with ID {doc_id}")
        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            raise

    def store_chat_memory(self, query: str, response: str, embedding: list, context_type: str, metadata: dict = None):
        """Store chat history in ChromaDB with optional metadata."""
        try:
            chat_id = str(abs(hash(query + response)))
            full_metadata = {"context_type": context_type}
            if metadata:
                full_metadata.update(metadata)
            self.chat_memory.add(
                ids=[chat_id],
                embeddings=[embedding],
                documents=[f"Q: {query}\nA: {response}"],
                metadatas=[full_metadata]
            )
            logger.debug(f"Stored chat memory with ID {chat_id}")
        except Exception as e:
            logger.error(f"Error storing chat memory: {str(e)}")
            raise

    def store_chat(self, query: str, response: str):
        """Store chat without embedding - simplified version."""
        try:
            chat_id = str(abs(hash(query + response)))
            # Use a simple embedding or generate one
            from app.embedding.embedding import get_embedding_model
            embedding_model = get_embedding_model()
            query_embedding = embedding_model.embed_query(query)

            self.chat_memory.add(
                ids=[chat_id],
                embeddings=[query_embedding],
                documents=[f"Q: {query}\nA: {response}"],
                metadatas={"context_type": "general"}
            )
            logger.debug(f"Stored chat with ID {chat_id}")
        except Exception as e:
            logger.error(f"Error storing chat: {str(e)}")

    def search_chat_memory(self, query: str, k: int = 5):
        """Search chat memory for similar queries."""
        try:
            from app.embedding.embedding import get_embedding_model
            embedding_model = get_embedding_model()
            query_embedding = embedding_model.embed_query(query)

            results = self.chat_memory.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            if results["documents"] and results["documents"][0]:
                from langchain.schema import Document
                documents = []
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    documents.append(Document(page_content=doc, metadata=meta))
                logger.debug(f"Found {len(documents)} chat memory matches for query: {query[:50]}...")
                return documents, results["distances"][0]
            return [], []
        except Exception as e:
            logger.error(f"Error searching chat memory: {str(e)}")
            return [], []

    def get_chat_memory(self):
        """Retrieve all chat memory entries."""
        try:
            return self.chat_memory.get(include=["documents", "metadatas"])
        except Exception as e:
            logger.error(f"Error retrieving chat memory: {str(e)}")
            return {"ids": [], "documents": [], "metadatas": []}

    def has_knowledge_base(self) -> bool:
        """Check if any documents exist in the vector collection."""
        try:
            count = self.vector_collection.count()
            logger.debug(f"Vector DB count: {count}")
            return count > 0
        except Exception as e:
            logger.error(f"Error checking knowledge base: {str(e)}")
            return False

    def clear_chat_memory(self):
        """Delete all entries in the chat_memory collection."""
        try:
            results = self.chat_memory.get()
            ids = results['ids']
            if ids:
                self.chat_memory.delete(ids=ids)
                logger.info(f"Cleared {len(ids)} chat memory entries successfully")
            else:
                logger.info("Chat memory was already empty")
        except Exception as e:
            logger.error(f"Error clearing chat memory: {str(e)}")
            raise


db = Database()
