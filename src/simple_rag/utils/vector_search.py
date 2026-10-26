import os
from functools import lru_cache
from chromadb import PersistentClient
from config import config
import logging
from langchain.schema import Document  # Add this import

logger = logging.getLogger(__name__)

class VectorSearch:
    def __init__(self):
        os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)
        self.client = PersistentClient(path=config.VECTOR_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="rag_collection", metadata={"hnsw:space": "cosine"}
        )

    def search(self, embedding, k=5, context_type="pdf"):
        try:
            # Use the single shared collection for Simple RAG
            collection = self.collection
            collection_name = "rag_collection"

            count = collection.count()
            logger.info(f"🔍 Searching in Simple RAG collection '{collection_name}' - Total documents: {count}, Context type: {context_type}")

            if count == 0:
                logger.warning(f"⚠️ Simple RAG vector database is EMPTY")
                return []

            results = collection.query(
                query_embeddings=[embedding],
                n_results=min(k, count),
                include=["documents", "distances", "metadatas"],
                where={"context_type": context_type}
            )

            logger.info(f"✅ Query returned {len(results['documents'][0]) if results['documents'] and results['documents'][0] else 0} results")

            if results["documents"] and results["documents"][0]:
                documents = [
                    Document(page_content=doc, metadata=meta)
                    for doc, meta in zip(results["documents"][0], results["metadatas"][0])
                ]
                logger.debug(f"Found {len(documents)} documents in vector search")
                return documents
            else:
                logger.debug("No documents found in vector search")
                return []
        except ValueError as ve:
            if "dimension" in str(ve).lower():
                logger.warning("Embedding dimension mismatch detected. Resetting vector_db collection.")
                self.client.delete_collection(name="rag_collection")
                self.collection = self.client.get_or_create_collection(
                    name="rag_collection",
                    metadata={"hnsw:space": "cosine"}
                )
                # Retry query after reset
                return self.search(embedding, k=k, context_type=context_type)
            else:
                logger.error(f"Error in vector search: {str(ve)}")
                raise
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []

vector_search = VectorSearch()
