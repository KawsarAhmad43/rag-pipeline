from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import logging
import torch
import gc
import os
import re
import hashlib
import tempfile
import pdfplumber
import unicodedata
import string
from app.embedding.embedding import get_embedding_model
from app.llm.llm import get_llm
from app.graph.graph import graph_db
from utils.graph_processor import extract_graph_from_text
from utils.database import db
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize extracted PDF text while preserving bullet points and structure."""
    if not text:
        return ""

    text = unicodedata.normalize('NFKD', text)
    allowed_chars = set(string.printable)
    allowed_chars.update(['\u2019', '\u2018', '\u201c', '\u201d', '•', '▪', '▫', '►', '▸', '‣', '⁃'])

    cleaned_chars = []
    for char in text:
        if char in allowed_chars or char.isprintable():
            cleaned_chars.append(char)
        elif char.isspace():
            cleaned_chars.append(' ')
        else:
            if unicodedata.category(char).startswith('M'):
                continue
            elif unicodedata.category(char).startswith('C'):
                continue
            else:
                cleaned_chars.append(' ')

    text = ''.join(cleaned_chars)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Page \d+.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\s*of\s*\d+.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])\s*([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()

def extract_title_from_text(text: str) -> str:
    """Extract title from text using heuristics."""
    if not text:
        return "Untitled Document"

    text = clean_text(text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return "Untitled Document"

    for line in lines[:10]:
        line = line.strip()
        if (len(line) < 10 or
            line.lower().startswith(('http', 'www', 'page ', 'abstract', 'introduction')) or
            re.match(r'^\d+[\.\)]\s', line) or
            re.match(r'^[A-Z\s]+$', line) and len(line) > 50):
            continue
        if 10 <= len(line) <= 200 and not line.endswith(':'):
            title = re.sub(r'^[^\w]*', '', line)
            title = re.sub(r'[^\w\s\-\(\)\.]*$', '', title)
            if title:
                return title
    for line in lines:
        if len(line) > 5:
            return line[:100]
    return "Untitled Document"

def safe_extract_text_with_pdfplumber(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber with PyPDFLoader as fallback."""
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    logger.debug(f"Extracted text from page {page_num} using pdfplumber")
                    lines = page_text.split('\n')
                    processed_lines = []
                    for line in lines:
                        if line.isupper() and len(line) > 5:
                            processed_lines.append(f"[HEADING] {line}")
                        else:
                            processed_lines.append(line)
                    page_text = '\n'.join(processed_lines)
                    page_text = clean_text(page_text)
                    full_text += page_text + "\n\n"
        if not full_text.strip():
            logger.warning(f"No text extracted with pdfplumber from {pdf_path}, trying PyPDFLoader")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            full_text = "\n\n".join(page.page_content for page in pages)
            full_text = clean_text(full_text)
            if full_text.strip():
                logger.debug(f"Successfully extracted text using PyPDFLoader from {pdf_path}")
            else:
                logger.warning(f"No text extracted with PyPDFLoader from {pdf_path}")
    except Exception as e:
        logger.error(f"Error opening PDF with pdfplumber: {str(e)}")
        try:
            logger.info(f"Falling back to PyPDFLoader for {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            full_text = "\n\n".join(page.page_content for page in pages)
            full_text = clean_text(full_text)
            if full_text.strip():
                logger.debug(f"Successfully extracted text using PyPDFLoader from {pdf_path}")
            else:
                logger.warning(f"No text extracted with PyPDFLoader from {pdf_path}")
        except Exception as e:
            logger.error(f"Error opening PDF with PyPDFLoader: {str(e)}")
            return ""
    return full_text.strip()

def semantic_chunk_text(text: str, embedding_model) -> List[str]:
    """Perform semantic chunking based on sentence embeddings and statistical coherence."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text]  # Fallback to full text if no sentences

    embeddings = embedding_model.embed_documents(sentences)
    embeddings = np.array(embeddings)

    chunks = []
    current_chunk = [sentences[0]]
    current_embedding = embeddings[0].reshape(1, -1)

    for i in range(1, len(sentences)):
        new_embedding = embeddings[i].reshape(1, -1)
        similarity = cosine_similarity(current_embedding, new_embedding)[0][0]
        avg_embedding = np.mean([current_embedding, new_embedding], axis=0)
        if similarity < 0.85 or len(" ".join(current_chunk)) > 1000:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_embedding = new_embedding
        else:
            current_chunk.append(sentences[i])
            current_embedding = avg_embedding

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def process_pdf(file_stream, doc_id=None):
    temp_file_path = None
    try:
        embedding_model = get_embedding_model()
        llm = get_llm()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = file_stream.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        full_text = safe_extract_text_with_pdfplumber(temp_file_path)
        if not full_text.strip():
            logger.warning(f"No text extracted from PDF at {temp_file_path}")
            return [], []

        title = extract_title_from_text(full_text)
        if not doc_id:
            doc_id = f"PDF_{hashlib.md5(content).hexdigest()[:8]}"
            logger.debug(f"Generated doc_id: {doc_id}")

        # Store full document embedding
        full_embedding = embedding_model.embed_documents([full_text])[0]
        full_doc = Document(page_content=full_text, metadata={"doc_id": doc_id, "title": title, "source": "pdf", "is_full": True})
        db.store_document(full_text, full_embedding, full_doc.metadata)
        graph_db.add_node("FullDocument", doc_id, {"content": full_text[:1000], "title": title, "doc_id": doc_id})
        with graph_db.get_session() as session:
            session.run("MERGE (p:PDF {doc_id: $doc_id}) SET p.embedding = $embedding", doc_id=doc_id, embedding=full_embedding)

        # Semantic chunking
        chunks = semantic_chunk_text(full_text, embedding_model)
        docs = [Document(page_content=chunk) for chunk in chunks if chunk.strip()]

        sections = []
        current_section = "Introduction"
        chunk_idx = 0
        for i, doc in enumerate(docs):
            if "[HEADING]" in doc.page_content:
                heading = doc.page_content.split("[HEADING] ")[1]
                sections.append((current_section, chunk_idx))
                current_section = heading
                doc.page_content = doc.page_content.replace("[HEADING] ", "")
            doc.metadata = {
                "doc_id": doc_id,
                "title": title,
                "chunk_index": i,
                "source": "pdf",
                "section": current_section
            }
            chunk_idx = i

        all_embeddings = []
        for i, doc in enumerate(docs):
            embedding = embedding_model.embed_documents([doc.page_content])[0]
            all_embeddings.append(embedding)
            chunk_id = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()[:8]

            try:
                if not doc.metadata:
                    logger.warning(f"Metadata is empty for chunk {chunk_id}, initializing with default values")
                    doc.metadata = {
                        "doc_id": doc_id,
                        "title": title,
                        "chunk_index": i,
                        "source": "pdf",
                        "section": current_section
                    }
                db.store_document(doc.page_content, embedding, doc.metadata)
                logger.debug(f"Stored document for chunk {chunk_id} in ChromaDB")
            except Exception as e:
                logger.error(f"Failed to store document for chunk {chunk_id}: {str(e)}")
                raise

            try:
                graph_db.add_node("Chunk", chunk_id, {
                    "content": doc.page_content[:1000],
                    "chunk_index": i,
                    "title": title[:200],
                    "doc_id": doc_id
                })
                with graph_db.get_session() as session:
                    session.run(
                        """
                        MATCH (p:PDF {doc_id: $doc_id})
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (c)-[:PART_OF]->(p)
                        SET c.embedding = $embedding
                        """,
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        embedding=embedding
                    )
                logger.debug(f"Stored chunk {chunk_id} in Neo4j")
            except Exception as e:
                logger.error(f"Failed to store chunk {chunk_id} in Neo4j: {str(e)}")
                raise

        prev_sec_id = None
        for sec_name, start_idx in sections:
            sec_id = hashlib.md5(sec_name.encode()).hexdigest()[:8]
            try:
                graph_db.add_node("Section", sec_id, {"name": sec_name, "doc_id": doc_id})
                with graph_db.get_session() as session:
                    session.run(
                        """
                        MATCH (p:PDF {doc_id: $doc_id})
                        MATCH (s:Section {id: $sec_id})
                        MERGE (p)-[:HAS_SECTION]->(s)
                        """,
                        doc_id=doc_id, sec_id=sec_id
                    )
                logger.debug(f"Created section {sec_id} in Neo4j")
            except Exception as e:
                logger.error(f"Failed to create section {sec_id}: {str(e)}")
                raise

            if prev_sec_id:
                try:
                    graph_db.add_relationship(prev_sec_id, sec_id, "NEXT")
                    logger.debug(f"Created section relationship {prev_sec_id} -> {sec_id}")
                except Exception as e:
                    logger.error(f"Failed to create section relationship {prev_sec_id} -> {sec_id}: {str(e)}")
                    raise
            prev_sec_id = sec_id

            for j in range(start_idx, len(docs)):
                if j + 1 < len(docs) and "[HEADING]" in docs[j+1].page_content:
                    break
                chunk_id = hashlib.md5(docs[j].page_content.encode('utf-8')).hexdigest()[:8]
                try:
                    graph_db.add_relationship(sec_id, chunk_id, "HAS_CHUNK")
                    logger.debug(f"Created chunk relationship {sec_id} -> {chunk_id}")
                except Exception as e:
                    logger.error(f"Failed to create chunk relationship {sec_id} -> {chunk_id}: {str(e)}")
                    raise

        for i in range(len(docs) - 1):
            chunk_id1 = hashlib.md5(docs[i].page_content.encode('utf-8')).hexdigest()[:8]
            chunk_id2 = hashlib.md5(docs[i+1].page_content.encode('utf-8')).hexdigest()[:8]
            try:
                graph_db.add_relationship(chunk_id1, chunk_id2, "NEXT")
                logger.debug(f"Created chunk relationship {chunk_id1} -> {chunk_id2}")
            except Exception as e:
                logger.error(f"Failed to create chunk relationship {chunk_id1} -> {chunk_id2}: {str(e)}")
                raise

        sample_texts = [doc.page_content for doc in docs if doc.page_content.strip() and len(doc.page_content.strip()) > 50]
        if not sample_texts:
            logger.warning(f"No valid sample texts (non-empty and >50 chars) available for graph extraction for doc_id {doc_id}")
            sample_texts = [full_text]  # Fallback to full text
        else:
            logger.debug(f"Attempting graph extraction with {len(sample_texts)} sample texts for doc_id {doc_id}")
            for idx, text in enumerate(sample_texts):
                logger.debug(f"Sample text {idx} for doc_id {doc_id}: {text[:200]}...")
        try:
            extract_graph_from_text(sample_texts, doc_id)
            logger.debug(f"Extracted graph for document {doc_id}")
        except Exception as e:
            logger.error(f"Failed to extract graph from text for doc_id {doc_id}: {str(e)}")
            logger.info(f"Continuing PDF processing despite graph extraction failure for doc_id {doc_id}")

        logger.info(f"Successfully processed PDF: {doc_id}, {len(docs)} chunks + 1 full document")
        return [full_doc] + docs, [full_embedding] + all_embeddings

    except Exception as e:
        logger.error(f"Error in PDF processing: {str(e)}")
        return [], []
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary file {temp_file_path}: {str(e)}")
