from utils.database import db
import logging
from langchain.schema import Document
from typing import List, Dict, Tuple, Optional
import hashlib
from collections import defaultdict
from utils.hybrid_search import BM25Search
from config import config

logger = logging.getLogger(__name__)


class ScopedVectorSearch:
    def __init__(self, vector_db, graph_db=None):
        self.vector_db = vector_db
        self.graph_db = graph_db

    def search(self, query_embedding, where: dict = None, k: int = 5) -> List[Document]:
        try:
            where = where or None
            results = self.vector_db.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
                where=where,
            )
            documents: List[Document] = []
            if results.get("documents") and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0], results["metadatas"][0], results["distances"][0]
                ):
                    metadata = dict(meta or {})
                    if "chunk_id" not in metadata:
                        metadata["chunk_id"] = hashlib.md5(doc.encode("utf-8")).hexdigest()[:8]
                    metadata["similarity_score"] = 1 - float(dist)
                    documents.append(Document(page_content=doc, metadata=metadata))
            return documents
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    def _backfill_doc_meta_from_graph(self, docs: List[Document]) -> None:
        """
        Ensure each document has doc_id and title metadata.
        If missing, fetch via Neo4j using chunk_id -> Chunk -> PDF.
        No-op if graph_db is not available.
        """
        if not self.graph_db or not docs:
            return
        # Build a set of chunk_ids that lack doc_id/title
        missing = []
        for d in docs:
            m = d.metadata or {}
            if not m.get("doc_id") or not m.get("title"):
                cid = m.get("chunk_id")
                if cid:
                    missing.append(cid)
        if not missing:
            return

        cypher = """
        UNWIND $chunk_ids AS cid
        MATCH (c:Chunk {id: cid})-[:PART_OF]->(p:PDF)
        RETURN cid AS chunk_id, p.doc_id AS doc_id, p.title AS title
        """
        try:
            with self.graph_db.get_session() as session:
                rows = session.run(cypher, chunk_ids=list(set(missing)))
                info = {r["chunk_id"]: {"doc_id": r["doc_id"], "title": r["title"]} for r in rows}
            for d in docs:
                m = d.metadata or {}
                cid = m.get("chunk_id")
                if cid and cid in info:
                    if not m.get("doc_id"):
                        m["doc_id"] = info[cid]["doc_id"]
                    if not m.get("title"):
                        m["title"] = info[cid]["title"] or "Untitled"
                    d.metadata = m
        except Exception as e:
            logger.error(f"Error backfilling doc metadata from graph: {e}")

    def detect_source_papers(
        self,
        query_embedding,
        min_similarity: float = 0.50,
        k: int = 20,
        top_n: int = 3,
        where_filter: Optional[dict] = None,
    ) -> List[Dict]:
        """
        Detects multiple relevant source papers for a query, returning a list of dicts
        sorted by relevance score. Optionally accept a where_filter to broaden search.
        """
        try:
            where = where_filter if where_filter is not None else {"source": {"$eq": "pdf"}}
            docs = self.search(query_embedding, where=where, k=k)
            if not docs:
                return []

            # Make sure doc_id/title exist
            self._backfill_doc_meta_from_graph(docs)

            agg = defaultdict(lambda: {"title": "Untitled", "score": 0.0, "count": 0})
            for d in docs:
                doc_id = d.metadata.get("doc_id")
                if not doc_id:
                    continue
                title = d.metadata.get("title", "Untitled")
                sim = float(d.metadata.get("similarity_score", 0.0))
                agg[doc_id]["title"] = title
                agg[doc_id]["score"] += sim
                agg[doc_id]["count"] += 1

            items = []
            for doc_id, stats in agg.items():
                avg_sim = stats["score"] / max(stats["count"], 1)
                if avg_sim >= min_similarity:
                    items.append({"id": doc_id, "title": stats["title"], "score": stats["score"]})

            items.sort(key=lambda x: x["score"], reverse=True)
            return items[:top_n]
        except Exception as e:
            logger.error(f"Error detecting multiple source papers: {e}")
            return []

    def search_within_papers(
        self, query_embedding, paper_ids: List[str], k_per_paper: int = 10, query_text: str = None
    ) -> List[Document]:
        merged: List[Document] = []
        seen_ids = set()

        try:
            # 1. Vector Search
            for pid in paper_ids:
                if not pid:
                    continue
                results = self.search(query_embedding, where={"doc_id": {"$eq": pid}}, k=k_per_paper)
                for d in results:
                    cid = d.metadata.get("chunk_id")
                    if cid and cid not in seen_ids:
                        merged.append(d)
                        seen_ids.add(cid)

            # 2. Keyword Search (Sidecar)
            if config.HYBRID_SEARCH_ENABLED and query_text:
                logger.info(f"Performing keyword search sidecar for query: {query_text[:50]}...")
                # To do BM25 effectively, we need to fetch candidates or all chunks for these papers.
                # For efficiency in this "sidecar" pattern, we can fetch more chunks using vector or
                # fetch all chunks for these papers from the graph/DB if the set is small.
                # Here we'll fetch all chunks for these papers to ensure exact matches are found.
                all_chunks = self.get_all_chunks_for_papers(paper_ids)
                if all_chunks:
                    bm25_searcher = BM25Search()
                    bm25_searcher.index_documents(all_chunks)
                    bm25_results = bm25_searcher.search(query_text, k=k_per_paper * len(paper_ids))

                    for d in bm25_results:
                        cid = d.metadata.get("chunk_id")
                        if cid and cid not in seen_ids:
                            d.metadata["search_source"] = "keyword_sidecar"
                            merged.append(d)
                            seen_ids.add(cid)
                        elif cid:
                            # If already found by vector, mark as hybrid
                            for existing in merged:
                                if existing.metadata.get("chunk_id") == cid:
                                    existing.metadata["search_source"] = "hybrid"
                                    existing.metadata["bm25_score"] = d.metadata.get("bm25_score")
                                    break

            # Backfill missing doc_id/title to ensure downstream stability
            self._backfill_doc_meta_from_graph(merged)

            # Sort by similarity score (if exists) or fallback
            merged.sort(key=lambda d: float(d.metadata.get("similarity_score", 0.0)), reverse=True)
            return merged[:50] # Increased limit for hybrid before reranking
        except Exception as e:
            logger.error(f"Error in search_within_papers: {e}")
            return merged

    def get_all_chunks_for_papers(self, paper_ids: List[str]) -> List[Document]:
        """Fetch all chunks for a list of papers from the vector DB or graph."""
        all_docs = []
        try:
            for pid in paper_ids:
                # Query ChromaDB for all documents with this doc_id
                results = self.vector_db.get(
                    where={"doc_id": {"$eq": pid}},
                    include=["documents", "metadatas"]
                )
                if results["documents"]:
                    for doc, meta in zip(results["documents"], results["metadatas"]):
                        metadata = dict(meta or {})
                        if "chunk_id" not in metadata:
                            metadata["chunk_id"] = hashlib.md5(doc.encode("utf-8")).hexdigest()[:8]
                        all_docs.append(Document(page_content=doc, metadata=metadata))
            return all_docs
        except Exception as e:
            logger.error(f"Error fetching all chunks for papers: {e}")
            return []

    def get_chunk_neighborhood_compact(self, chunk_id: str) -> Dict:
        """
        Gets a compact, valid neighborhood of a chunk using a Cypher query designed to
        avoid aggregation errors.
        """
        if not self.graph_db:
            return {}
        cypher = """
        MATCH (c:Chunk {id: $chunk_id})
        OPTIONAL MATCH (c)-[:PART_OF]->(p:PDF)
        OPTIONAL MATCH (c)<-[:HAS_CHUNK]-(s:Section)
        OPTIONAL MATCH (c)<-[:IN_CHUNK]-(:Mention)<-[:HAS_MENTION]-(e:Entity)
        WITH c, p, s, collect(DISTINCT e) AS ents
        RETURN {
          source_chunk: { id: c.id, doc_id: c.doc_id, chunk_index: c.chunk_index, content: c.content },
          pdf: CASE WHEN p IS NULL THEN null ELSE {doc_id: p.doc_id, title: p.title} END,
          section: CASE WHEN s IS NULL THEN null ELSE {name: s.name} END,
          entities: [e IN ents | {id: e.id, name: e.name, type: e.type}]
        } AS neighborhood
        """
        try:
            with self.graph_db.get_session() as session:
                rec = session.run(cypher, chunk_id=chunk_id).single()
                return rec["neighborhood"] if rec else {}
        except Exception as e:
            logger.error(f"Error in get_chunk_neighborhood_compact: {e}")
            return {}


# Global instance
try:
    from app.graph.graph import graph_db
    vector_search = ScopedVectorSearch(db.vector_collection, graph_db)
except ImportError:
    logger.warning("Could not import graph_db, vector_search will have limited capabilities.")
    vector_search = ScopedVectorSearch(db.vector_collection, None)
