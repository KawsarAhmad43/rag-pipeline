import logging
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from langchain.schema import Document
import re

logger = logging.getLogger(__name__)

class BM25Search:
    def __init__(self):
        self.bm25 = None
        self.documents: List[Document] = []
        self.corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenization: lowercase and alphanumeric only
        return re.findall(r'\w+', text.lower())

    def index_documents(self, documents: List[Document]):
        """Index a list of LangChain documents for BM25 search."""
        if not documents:
            logger.warning("No documents provided for BM25 indexing.")
            return

        self.documents = documents
        self.corpus = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)
        logger.info(f"Indexed {len(documents)} documents for BM25 search.")

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Perform BM25 search and return top k documents with scores in metadata."""
        if not self.bm25 or not self.documents:
            logger.warning("BM25 index is empty. Call index_documents first.")
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k indices
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for i in top_n:
            if scores[i] > 0:
                doc = self.documents[i]
                # Avoid mutating the original cached document metadata
                metadata = dict(doc.metadata or {})
                metadata["bm25_score"] = float(scores[i])
                # Normalize BM25 score roughly for initial sorting?
                # (Actually we just want the docs, reranker handles the rest)
                results.append(Document(page_content=doc.page_content, metadata=metadata))

        return results

# Global helper for quick search if needed
def keyword_search_on_docs(query: str, docs: List[Document], k: int = 5) -> List[Document]:
    searcher = BM25Search()
    searcher.index_documents(docs)
    return searcher.search(query, k=k)
