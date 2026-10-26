import os
import sys
# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(project_root)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from typing import List, Optional
import gc
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from embedder.embedder import ColnomicEmbeddings
import logging
from config import config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ChromaDBHandler:
    """
    ChromaDB handler with on-demand model loading and proper cleanup.
    """

    def __init__(self, session_specific_persist_directory: str = None, collection_name: str = None):


        if collection_name is None:
            collection_name = str(config.CHROMA_COLLECTION_NAME)
        if not isinstance(collection_name, str):
            raise ValueError(f"Collection name must be a string, got {type(collection_name)}")

        self.collection_name = collection_name

        self.persist_directory = session_specific_persist_directory or str(config.VECTOR_DB_BASE_PATH)
        os.makedirs(self.persist_directory, exist_ok=True)

        self.embedding_model_path = str(config.EMBEDDING_MODEL_PATH)
        self.embedding_function = None
        self._chroma_instance = None
        logging.info(f"Using collection: {self.collection_name}")

    def _get_embedding_function(self):
        """Get or create the embedding function."""
        if self.embedding_function is None:
            logging.info("🧠 Initializing embedding function...")
            try:
                self.embedding_function = ColnomicEmbeddings(
                    model_path=self.embedding_model_path,
                    device="cpu",
                    torch_dtype=torch.float32
                )
            except Exception as e:
                logging.critical(f"Failed to initialize embedding function: {e}", exc_info=True)
                raise RuntimeError(f"ChromaDBHandler: Could not initialize embedding function: {e}")
        return self.embedding_function

    def _get_chroma_instance(self) -> Chroma:
        """Get or create the Chroma instance."""
        if self._chroma_instance is None:
            logging.info(f"Loading ChromaDB from {self.persist_directory}...")
            try:
                embedding_function = self._get_embedding_function()

                self._chroma_instance = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=embedding_function
                )

                count = self._chroma_instance._collection.count()
                if count == 0:
                    logging.warning(f"⚠️ Loaded collection has 0 documents. May need to re-index.")
                else:
                    logging.info(f"✅ Loaded collection with {count} existing documents")

                logging.info(f"ChromaDB instance loaded for {self.persist_directory}.")
            except Exception as e:
                logging.error(f"Failed to load ChromaDB instance: {e}", exc_info=True)
                raise RuntimeError(f"ChromaDBHandler: Could not initialize Chroma instance: {e}")
        return self._chroma_instance

    def build_index(self, texts: List[str], metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None):
        """Build vector index from texts."""
        if not texts:
            logging.warning("No texts provided to build index. Skipping.")
            return

        logging.info(f"Building/rebuilding vector index in {self.persist_directory} for {len(texts)} texts.")

        try:
            embedding_function = self._get_embedding_function()

            self._chroma_instance = Chroma.from_texts(
                texts=texts,
                embedding=embedding_function,
                persist_directory=self.persist_directory,
                metadatas=metadatas,
                ids=ids
            )

            # Explicitly persist if method exists (older langchain-community versions)
            if hasattr(self._chroma_instance, 'persist'):
                self._chroma_instance.persist()

            # Verify persistence
            doc_count = self._chroma_instance._collection.count()
            logging.info(f"✅ Vector DB persisted with {doc_count} documents at {self.persist_directory}")

            if doc_count == 0:
                logging.error("⚠️ WARNING: Collection shows 0 documents after build_index!")
            elif doc_count != len(texts):
                logging.warning(f"⚠️ Document count mismatch: expected {len(texts)}, got {doc_count}")

        except Exception as e:
            logging.error(f"Error building vector index: {e}", exc_info=True)
            raise

    def search(self, query: str, k: int = 5):

        logging.info("🔍 Performing similarity search...")
        try:
            chroma_index = self._get_chroma_instance()
            results = chroma_index.similarity_search(query, k=k)
            return results

        except Exception as e:
            logging.error(f"Error during similarity search: {e}")
            raise

    def add_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None):
        """Add documents to existing index."""
        if not texts:
            logging.warning("No texts provided to add documents. Skipping.")
            return

        logging.info(f"Adding {len(texts)} documents to index.")

        try:
            chroma = self._get_chroma_instance()
            chroma.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            logging.info(f"Documents added. Total docs: {chroma._collection.count()}")

        except Exception as e:
            logging.error(f"Error adding documents: {e}", exc_info=True)
            raise

    def get_doc_count(self) -> int:
        """Get the total number of documents in the collection."""
        try:
            chroma = self._get_chroma_instance()
            return chroma._collection.count()
        except Exception as e:
            logging.error(f"Error getting document count: {e}")
            return 0

    def cleanup(self):
        """Cleanup all resources."""
        if self.embedding_function:
            self.embedding_function.cleanup()
            self.embedding_function = None
        if self._chroma_instance:
            self._chroma_instance = None
