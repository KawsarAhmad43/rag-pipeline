import hashlib
from typing import Dict, List, Optional, Tuple

from langchain.prompts import PromptTemplate
from langchain.schema import Document

from app.llm.llm import get_llm
from app.reranker.reranker import rank_documents
from app.embedding.embedding import get_embedding_model
from app.graph.graph import graph_db
from utils.vector_search import vector_search
from utils.logging import logger

def build_context_from_scoped_results(
    chunks: List[Document],
    neighborhoods: List[Dict],
    subgraph_nodes: List[Dict],
    subgraph_edges: List[Dict],
) -> Tuple[str, List[Document]]:
    fused_docs: List[Document] = []
    nb_by_chunk: Dict[str, Dict] = {nb.get("source_chunk", {}).get("id"): nb for nb in neighborhoods if nb.get("source_chunk", {}).get("id")}
    subgraph_node_ids = {node["id"] for node in subgraph_nodes}

    for d in chunks:
        meta = d.metadata or {}
        chunk_id = meta.get("chunk_id")
        doc_id = meta.get("doc_id")
        title = meta.get("title", "Untitled")
        sim = meta.get("similarity_score", 0.0)
        base_text = d.page_content.strip()

        nb = nb_by_chunk.get(chunk_id, {})
        entities = [f"{e.get('name')}({e.get('type')})" for e in nb.get("entities", []) if e]
        related_ids = [rc.get("id") for rc in nb.get("related_chunks", []) if rc]
        section_name = nb.get("section", {}).get("name") if nb.get("section") else None

        enhanced = base_text
        if section_name:
            enhanced += f"\n[Section: {section_name}]"
        if entities:
            enhanced += "\n[Entities: " + ", ".join(entities[:8]) + "]"
        if related_ids:
            enhanced += "\n[Related chunks: " + ", ".join(related_ids[:6]) + "]"
        # Include subgraph information if chunk is part of multi-hop
        if chunk_id in subgraph_node_ids:
            node_details = [n for n in subgraph_nodes if n["id"] == chunk_id]
            edge_details = [e for e in subgraph_edges if e["source"] == chunk_id or e["target"] == chunk_id]
            if node_details:
                enhanced += f"\n[Subgraph Node: {node_details[0].get('content', '')[:100]}]"
            if edge_details:
                enhanced += f"\n[Subgraph Edges: {', '.join([f'{e['source']}--{e['type']}->{e['target']}' for e in edge_details])}]"

        fused_docs.append(
            Document(
                page_content=enhanced,
                metadata={
                    "doc_id": doc_id,
                    "title": title,
                    "chunk_id": chunk_id,
                    "similarity_score": sim,
                    "search_source": "scoped_vector+graph_nb+subgraph",
                    "priority_score": float(sim),
                },
            )
        )

    ctx_lines = [
        f"[Result {i+1}] (doc_id={fd.metadata.get('doc_id')}, chunk_id={fd.metadata.get('chunk_id')}, score={fd.metadata.get('similarity_score'):.3f})\n{fd.page_content}"
        for i, fd in enumerate(fused_docs)
    ]
    return "\n---\n".join(ctx_lines), fused_docs

def search_documents(
    query: str,
    context_type: str = "pdf",
    paper_id: Optional[str] = None,
) -> Tuple[List[Document], str, Optional[List[Dict]]]:
    """
    Dynamic, robust scoped search with multi-hop support:
    1) If paper_id provided, use it.
    2) Else try multi-paper detection (min_similarity=0.50), k=24.
    3) If empty, retry with relaxed threshold and broader where (no 'source' filter), k=50.
    4) If still empty, take top global chunks (k=5) and infer paper_ids from their doc_id.
    Then scope retrieval to those paper_ids, build neighborhoods, perform multi-hop traversal,
    construct subgraph, and pass all data to LLM.
    """
    try:
        emb_model = get_embedding_model()
        q_emb = emb_model.embed_query(query)

        # Stage 1: explicit override
        if paper_id:
            title = graph_db.get_document_title(paper_id) or "Untitled"
            source_papers = [{"id": paper_id, "title": title, "score": 1.0}]
        else:
            # Stage 2: normal detection
            source_papers = vector_search.detect_source_papers(
                q_emb, min_similarity=0.50, k=24, top_n=3, where_filter={"source": {"$eq": "pdf"}}
            )

            # Stage 3: relaxed detection if none
            if not source_papers:
                logger.info("Primary detection returned no papers. Retrying with relaxed threshold and broader search.")
                source_papers = vector_search.detect_source_papers(
                    q_emb, min_similarity=0.35, k=50, top_n=3, where_filter=None  # no where filter
                )

            # Stage 4: derive from top global chunks if still empty
            if not source_papers:
                logger.info("Relaxed detection returned no papers. Inferring from top global chunks.")
                global_top = vector_search.search(q_emb, where=None, k=5)
                vector_search._backfill_doc_meta_from_graph(global_top)
                inferred = []
                seen = set()
                for d in global_top:
                    doc_id = d.metadata.get("doc_id")
                    title = d.metadata.get("title", "Untitled")
                    if doc_id and doc_id not in seen:
                        seen.add(doc_id)
                        inferred.append({"id": doc_id, "title": title, "score": d.metadata.get("similarity_score", 0.0)})
                source_papers = inferred

        if not source_papers:
            logger.info("No source papers could be detected or inferred.")
            return [], "No relevant information was found in the knowledge base for this query.", None

        paper_ids = [p["id"] for p in source_papers if p.get("id")]

        # Initial chunk retrieval (Hybrid: Vector + Keyword)
        top_chunks = vector_search.search_within_papers(q_emb, paper_ids, k_per_paper=10, query_text=query)
        if not top_chunks:
            return [], "No relevant chunks found within the detected paper(s).", source_papers

        # Get chunk IDs for neighborhood and multi-hop traversal
        chunk_ids = list({d.metadata.get("chunk_id") for d in top_chunks if d.metadata.get("chunk_id")})
        neighborhoods = []
        for cid in chunk_ids[:12]:
            nb = vector_search.get_chunk_neighborhood_compact(cid)
            if nb:
                neighborhoods.append(nb)

        # Multi-hop traversal and subgraph construction
        subgraph_nodes = []
        subgraph_edges = []
        visited_nodes = set()
        max_hops = 2  # Adjustable based on query complexity or config

        def traverse_multi_hop(start_chunk_id: str, hop: int = 0):
            if hop > max_hops or start_chunk_id in visited_nodes:
                return
            visited_nodes.add(start_chunk_id)

            # Fetch neighborhood
            nb = vector_search.get_chunk_neighborhood_compact(start_chunk_id)
            if not nb or "source_chunk" not in nb:
                return

            chunk_data = nb["source_chunk"]
            if chunk_data and chunk_data["id"] not in [n["id"] for n in subgraph_nodes]:
                subgraph_nodes.append({
                    "id": chunk_data["id"],
                    "content": chunk_data.get("content", "")[:1000],
                    "doc_id": chunk_data.get("doc_id"),
                    "chunk_index": chunk_data.get("chunk_index")
                })

            # Traverse relationships (e.g., NEXT, HAS_MENTION, SAME_AS)
            cypher = """
            MATCH (c:Chunk {id: $chunk_id})
            OPTIONAL MATCH (c)-[r:NEXT|HAS_MENTION|SAME_AS]->(n)
            WHERE n:Chunk OR n:Entity
            RETURN n.id AS target_id, type(r) AS rel_type, n.content AS content
            LIMIT 5
            """
            with graph_db.get_session() as session:
                results = session.run(cypher, chunk_id=start_chunk_id)
                for record in results:
                    target_id = record["target_id"]
                    rel_type = record["rel_type"]
                    content = record.get("content", "")[:1000]
                    if target_id and target_id not in [n["id"] for n in subgraph_nodes]:
                        subgraph_nodes.append({
                            "id": target_id,
                            "content": content,
                            "doc_id": chunk_data.get("doc_id"),
                            "chunk_index": chunk_data.get("chunk_index", -1)
                        })
                    if target_id and [e for e in subgraph_edges if e["source"] == start_chunk_id and e["target"] == target_id]:
                        subgraph_edges.append({
                            "source": start_chunk_id,
                            "target": target_id,
                            "type": rel_type
                        })
                    if target_id and hop < max_hops:
                        traverse_multi_hop(target_id, hop + 1)

        # Start multi-hop traversal from initial chunks
        for chunk_id in chunk_ids[:5]:  # Limit to top 5 chunks for efficiency
            traverse_multi_hop(chunk_id)

        # Build context with subgraph data
        context, fused_docs = build_context_from_scoped_results(top_chunks, neighborhoods, subgraph_nodes, subgraph_edges)
        final_docs = rank_documents(query, fused_docs)[:5] if len(fused_docs) > 1 else fused_docs

        final_ctx_lines = [
            f"[Result {i+1}] (doc_id={fd.metadata.get('doc_id')}, chunk_id={fd.metadata.get('chunk_id')}, score={fd.metadata.get('similarity_score'):.3f})\n{fd.page_content}"
            for i, fd in enumerate(final_docs)
        ]
        final_context = "\n---\n".join(final_ctx_lines) if final_ctx_lines else "No relevant information was found after filtering."
        return final_docs, final_context, source_papers
    except Exception as e:
        logger.error(f"Error in search_documents: {e}", exc_info=True)
        return [], "An error occurred during the search process.", None

def get_agent_response(query: str, context: str = None, doc_id: Optional[str] = None):
    llm = get_llm()
    if context is None or context.strip() == "":
        _, context, _ = search_documents(query, paper_id=doc_id)

    # First, check relevance
    relevance_prompt = PromptTemplate.from_template(
        """
Based on the context and query, is there any relevant information in the context to answer the query? Answer 'yes' or 'no' only.

Context:
{context}

Query: {query}
"""
    )
    relevance_resp = llm.invoke(relevance_prompt.format(context=context, query=query))
    relevance = (relevance_resp.content.strip() if hasattr(relevance_resp, "content") else str(relevance_resp).strip()).lower()

    if 'yes' in relevance:
        # If relevant, generate answer without fallback option
        answer_prompt = PromptTemplate.from_template(
            """
You are an AI assistant answering questions based ONLY on the provided knowledge base context.
- Provide a direct and concise answer using the relevant information from the context, even if partial.
- Always include citations using (doc_id=..., chunk_id=...) for each piece of information used.
- Do not use any external knowledge or add explanations beyond what is in the context.
- Use the subgraph information (nodes and edges) to perform multi-hop reasoning if required.

Context:
{context}

Question: {query}

Answer:
"""
        )
        resp = llm.invoke(answer_prompt.format(context=context, query=query))
    else:
        # If not relevant, use fallback
        resp = "I cannot find enough information in the uploaded documents to answer that."

    return str(resp).strip() if isinstance(resp, str) else resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
