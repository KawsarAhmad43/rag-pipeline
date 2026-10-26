from sentence_transformers import CrossEncoder
import logging
from typing import List, Dict
import os
import torch
import gc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_gpu_memory_reranker():
    """Centralized GPU memory cleanup function for reranker."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

class Reranker:
    def __init__(self):
        from config import config

        # MONKEYPATCH: Bypass transformers version check
        import transformers.utils.versions
        def dummy_require_version(*args, **kwargs): return None
        transformers.utils.versions.require_version = dummy_require_version

        model_path = str(config.RERANKER_MODEL_PATH).strip()
        logging.info(f"Loading reranker from {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Reranker model not found at: {model_path}")

        clean_gpu_memory_reranker()

        try:
            # CPU Fallback for stability in restricted environments
            self.device = torch.device("cpu")
            logging.info("⚙️ Using CPU for Reranker for maximum stability.")

            # CrossEncoder initialization
            self.model = CrossEncoder(
                model_path,
                device=self.device,
                trust_remote_code=True
            )
            self.model.eval()
            logging.info(f"Reranker loaded successfully on {self.device}")
        except Exception as e:
            logging.critical(f"Failed to load reranker model: {e}", exc_info=True)
            # Try without automodel_args as fallback for older sentence-transformers
            try:
                self.model = CrossEncoder(model_path, device=self.device)
                self.model.eval()
                logging.info("Reranker loaded successfully (fallback mode)")
            except Exception:
                raise RuntimeError(f"Could not load Reranker model: {e}")

    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Dict]:
        if not documents:
            logging.warning("No documents provided to reranker.")
            return []

        if not query or not query.strip():
            logging.warning("Empty query provided to reranker.")
            return [{"text": doc, "score": 0.0} for doc in documents]

        try:
            pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(pairs, show_progress_bar=False)

            scored_docs = list(zip(scores, documents))
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            ranked_results = [{"text": doc, "score": float(score)} for score, doc in scored_docs]

            logging.info(f"Reranker scores for query '{query[:50]}...':")
            for i, result in enumerate(ranked_results[:5]):
                logging.info(f"  Rank {i+1}: Score={result['score']:.4f} | Text='{result['text'][:100].replace(chr(10), ' ')}...'")

            return ranked_results[:top_k] if top_k else ranked_results

        except Exception as e:
            logging.error(f"Error in reranking: {e}", exc_info=True)
            return [{"text": doc, "score": 0.0} for doc in documents]

    def __del__(self):
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            clean_gpu_memory_reranker()
        except Exception as e:
            logging.warning(f"Error during reranker cleanup: {e}")
