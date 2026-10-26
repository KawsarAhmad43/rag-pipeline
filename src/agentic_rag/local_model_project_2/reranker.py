from sentence_transformers import CrossEncoder
import gc

class Reranker:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        print("🧠 Loading reranker model into memory...")
        self.model = CrossEncoder(model_path, device=device)
        print("✅ Reranker model loaded.")

    def rerank(self, query: str, documents: list) -> list:
        print("🔍 Reranking documents...")
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        sorted_pairs = sorted(zip(scores, documents), reverse=True)
        return [doc for score, doc in sorted_pairs]

    def cleanup(self):
        """Unloads the reranker model from memory."""
        print("🧹 Unloading reranker model from memory...")
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        gc.collect()
