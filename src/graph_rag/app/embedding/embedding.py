from langchain_community.embeddings import HuggingFaceEmbeddings
from config import config
import logging
import torch
import psutil
import gc
from utils.logging import logger
from utils.system import clean_gpu_cache, get_memory_usage


class EmbeddingModel:
    def __init__(self):
        # Force CPU for embedding model to avoid OOM errors when running alongside large LLMs.
        # This is the most reliable way to solve the CUBLAS_STATUS_ALLOC_FAILED error.
        device = "cpu"
        logger.info(f"Initializing embedding model on device: {device}")

        # Clear cache before loading
        clean_gpu_cache()

        self.model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": device, "trust_remote_code": False},
            encode_kwargs={"normalize_embeddings": True}  # Recommended for many models
        )
        cpu_mem = psutil.Process().memory_info().rss / 1024 / 1024
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        logger.info(f"Embedding model loaded: {config.EMBEDDING_MODEL} on {device}. Memory usage: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single text query."""
        try:
            embedding = self.model.embed_query(text)
            logger.debug(f"Generated embedding for text: '{text[:50]}...'")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for query: {str(e)}")
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of documents."""
        try:
            embeddings = self.model.embed_documents(texts)
            logger.debug(f"Generated {len(embeddings)} embeddings for a batch of documents.")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings for documents: {str(e)}")
            raise

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
