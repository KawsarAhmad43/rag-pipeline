import os
import torch
import numpy as np
from typing import List, Optional
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# --- UPDATED IMPORTS FOR LANGCHAIN AND CHROMA ---
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
# ------------------------------------------------

# Import necessary components from colpali_engine
try:
    from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
    BIQWEN_AVAILABLE = True
except ImportError:
    BIQWEN_AVAILABLE = False
    print("⚠️ colpali_engine not available, will use fallback embeddings")


# --- Custom Nomic BiQwen Embeddings Class for LangChain ---
class NomicBiQwenEmbeddings(Embeddings):
    """
    Custom Embeddings class to wrap the BiQwen2_5 model from colpali_engine
    for use with LangChain.
    """
    model: BiQwen2_5
    processor: BiQwen2_5_Processor
    device: str

    def __init__(self, model_path: str, device: str = "cpu", torch_dtype=torch.float32, attn_implementation: Optional[str] = None):
        super().__init__()
        self.device = device
        print(f"Loading Nomic BiQwen model from {model_path} for embeddings...")
        self.model = BiQwen2_5.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            attn_implementation=attn_implementation,
            local_files_only=True
        ).eval()
        self.processor = BiQwen2_5_Processor.from_pretrained(model_path, local_files_only=True)
        self.device = self.model.device
        print(f"Nomic BiQwen model loaded on {self.device}.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            batch_queries = self.processor.process_queries(texts).to(self.device)
            embeddings = self.model(**batch_queries).cpu().numpy()
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# --- Simple TF-IDF Based Embeddings Fallback ---
class SimpleTfidfEmbeddings(Embeddings):
    """
    Simple TF-IDF based embeddings that work offline without any model downloads.
    This is a fallback when advanced models fail to load.
    """
    def __init__(self, embedding_dim: int = 384):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vectorizer = None
        self.corpus = []
        print(f"Initialized SimpleTfidfEmbeddings with dimension {embedding_dim}")

    def _ensure_vectorizer(self, texts: List[str]):
        """Ensure vectorizer is fitted on the corpus"""
        if self.vectorizer is None:
            # Add texts to corpus
            self.corpus.extend(texts)
            # Create and fit vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=self.embedding_dim,
                ngram_range=(1, 2),
                strip_accents='unicode',
                lowercase=True
            )
            self.vectorizer.fit(self.corpus)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents using TF-IDF"""
        self._ensure_vectorizer(texts)

        # Transform texts to TF-IDF vectors
        vectors = self.vectorizer.transform(texts).toarray()

        # Normalize vectors
        vectors = normalize(vectors, norm='l2')

        # Pad or truncate to exact embedding_dim
        result = []
        for vec in vectors:
            if len(vec) < self.embedding_dim:
                # Pad with zeros
                padded = np.zeros(self.embedding_dim)
                padded[:len(vec)] = vec
                result.append(padded.tolist())
            else:
                # Truncate
                result.append(vec[:self.embedding_dim].tolist())

        return result

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if self.vectorizer is None:
            # If vectorizer not fitted yet, use the query itself
            self._ensure_vectorizer([text])

        return self.embed_documents([text])[0]


# --- Memory Efficient VectorDB Class ---
class VectorDB:
    """Vector database for document retrieval using Chroma and Nomic BiQwen embeddings"""

    def __init__(self, embedding_model_path, persist_directory="chroma_db", device="cpu", torch_dtype=torch.float32):
        """
        Initializes the VectorDB and KEEPS the embedding model loaded in memory.
        """
        self.embedding_model_path = embedding_model_path
        self.persist_directory = persist_directory
        self.device = device
        self.torch_dtype = torch_dtype

        # Load the embedding model once and keep it in memory
        print("🧠 Loading embedding model into memory (will stay loaded)...")

        # Try to load BiQwen model first, fallback to HuggingFace embeddings if it fails
        try:
            if BIQWEN_AVAILABLE:
                self.embedding_model = NomicBiQwenEmbeddings(
                    model_path=self.embedding_model_path,
                    device=self.device,
                    torch_dtype=self.torch_dtype
                )
                print("✅ BiQwen embedding model loaded and ready for use")
            else:
                raise ImportError("BiQwen not available")
        except Exception as e:
            print(f"⚠️ Failed to load BiQwen model: {e}")
            print("🔄 Falling back to SimpleTfidfEmbeddings (offline, no download required)")
            # Use our custom TF-IDF embeddings that work offline
            self.embedding_model = SimpleTfidfEmbeddings(embedding_dim=384)
            print("✅ SimpleTfidfEmbeddings loaded and ready for use")

    def build_index(self, chunks: List[str]):
        """
        Builds the Chroma index using the already loaded embedding model.
        """
        print("🔍 Building vector index...")
        chroma_index = Chroma.from_texts(
            texts=chunks,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory,
        )
        chroma_index.persist()
        print(f"✅ Vector DB created and persisted at {self.persist_directory}")

    def search(self, query: str, k: int = 5):
        """
        Performs similarity search using the already loaded embedding model.
        """
        print("🔍 Performing similarity search...")
        # Load the persisted ChromaDB instance
        chroma_index = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )
        results = chroma_index.similarity_search(query, k=k)
        return results

    def cleanup(self):
        """
        Cleanup method to unload the model when needed.
        """
        if hasattr(self, 'embedding_model') and self.embedding_model is not None:
            print("🧹 Unloading embedding model from memory...")
            del self.embedding_model
            self.embedding_model = None
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
