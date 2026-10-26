from sentence_transformers import CrossEncoder

class Reranker:
    """Rerank documents based on relevance to query"""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = CrossEncoder(model_path, device=device)

    def rerank(self, query: str, documents: list) -> list:
        """Rerank documents by relevance to query"""
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        sorted_pairs = sorted(zip(scores, documents), reverse=True)
        return [doc for score, doc in sorted_pairs]
