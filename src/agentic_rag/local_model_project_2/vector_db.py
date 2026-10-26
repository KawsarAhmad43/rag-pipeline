import os
import torch
from typing import List, Optional
import gc
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor

class NomicBiQwenEmbeddings(Embeddings):
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

    def cleanup(self):
        """Unloads the embedding model from memory."""
        print("🧹 Unloading embedding model from memory...")
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
        gc.collect()
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()

class VectorDB:
    def __init__(self, embedding_model_path, persist_directory="chroma_db", device="cpu", torch_dtype=torch.float32):
        self.embedding_model_path = embedding_model_path
        self.persist_directory = persist_directory
        self.device = device
        self.torch_dtype = torch_dtype
        print("🧠 Loading embedding model into memory...")
        self.embedding_model = NomicBiQwenEmbeddings(
            model_path=self.embedding_model_path,
            device=self.device,
            torch_dtype=self.torch_dtype
        )

    def build_index(self, chunks: List[str]):
        print("🔍 Building vector index...")
        try:
            chroma_index = Chroma.from_texts(
                texts=chunks,
                embedding=self.embedding_model,
                persist_directory=self.persist_directory,
            )
            print(f"✅ Vector DB created and persisted at {self.persist_directory}")
        except Exception as e:
            print(f"Error building vector index: {e}")
            raise

    def search(self, query: str, k: int = 5):
        print("🔍 Performing similarity search...")
        try:
            chroma_index = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )
            results = chroma_index.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error during similarity search: {e}")
            raise

    def cleanup(self):
        """Unloads the embedding model."""
        if self.embedding_model is not None:
            self.embedding_model.cleanup()
            self.embedding_model = None
