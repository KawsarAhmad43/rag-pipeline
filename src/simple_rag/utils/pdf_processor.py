from typing import List, Tuple
import io
import tempfile
import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

# Optimized chunk size and overlap matching Agentic RAG
CHUNK_SIZE = 1000
OVERLAP = 200  # Increased overlap for better context continuity

def process_pdf(file_stream, filename: str = "document.pdf") -> Tuple[List[Document], List[List[float]]]:
    """
    Process PDF with robust text extraction using PyMuPDF (fitz).
    Preserves document structure including tables, lists, and formatting.
    Matches Agentic RAG's approach for consistent, high-quality extraction.
    """
    try:
        from app.embedding.embedding import get_embedding_model
        import fitz  # PyMuPDF - superior PDF text extraction

        embedding_model = get_embedding_model()
        indexed_chunks = []
        all_embeddings = []

        # Save file stream to temporary file for PyMuPDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_path = tmp_file.name
            if hasattr(file_stream, 'read'):
                content = file_stream.read()
                tmp_file.write(content)
            else:
                # If already bytes
                tmp_file.write(file_stream)

        try:
            # Use PyMuPDF for robust extraction (same as Agentic RAG & Multimodal RAG)
            doc = fitz.open(tmp_path)
            full_text = ""
            page_map = []  # Track which page each chunk belongs to

            logger.info(f"📄 Processing PDF '{filename}' with {len(doc)} pages using PyMuPDF")

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()

                if page_text and page_text.strip():
                    # Track page boundaries for metadata
                    start_char = len(full_text)
                    full_text += page_text + "\n"  # Preserve structure with newlines
                    end_char = len(full_text)
                    page_map.append((page_num + 1, start_char, end_char))

            doc.close()

            if not full_text.strip():
                logger.warning(f"⚠️ No text extracted from PDF '{filename}'")
                return [], []

            logger.info(f"✅ Extracted {len(full_text)} characters from {len(page_map)} pages")

            # Use RecursiveCharacterTextSplitter (same as Agentic RAG & Multimodal RAG)
            # This intelligently preserves structure like tables, lists, paragraphs
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]  # Try to split on paragraphs first
            )

            chunks = text_splitter.split_text(full_text)
            logger.info(f"📦 Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={OVERLAP})")

            # Create documents with proper metadata
            for chunk_id, chunk in enumerate(chunks):
                cleaned_chunk = chunk.strip()
                if not cleaned_chunk or len(cleaned_chunk) < 10:
                    continue

                # Determine which page this chunk belongs to
                chunk_start = full_text.find(cleaned_chunk[:50])  # Find chunk position
                page_num = 1  # Default

                for page_n, start, end in page_map:
                    if start <= chunk_start < end:
                        page_num = page_n
                        break

                doc_obj = Document(
                    page_content=cleaned_chunk,
                    metadata={
                        "source": filename,
                        "page": page_num,
                        "chunk_id": chunk_id
                    }
                )
                indexed_chunks.append(doc_obj)

                # Generate embedding
                embedding = embedding_model.embed_query(cleaned_chunk)
                all_embeddings.append(embedding)

            logger.info(f"✅ Successfully processed '{filename}': {len(indexed_chunks)} chunks indexed")
            return indexed_chunks, all_embeddings

        except ImportError:
            logger.error("❌ PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF")
            raise

        except Exception as e:
            logger.error(f"❌ Error processing PDF with PyMuPDF: {str(e)}")
            # Fallback to text file processing
            try:
                with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()

                if not text.strip():
                    logger.warning("No text content in file")
                    return [], []

                logger.info(f"📝 Processing as text file: {len(text)} characters")

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=OVERLAP
                )
                chunks = text_splitter.split_text(text)

                for chunk_id, chunk in enumerate(chunks):
                    cleaned = chunk.strip()
                    if not cleaned:
                        continue

                    doc_obj = Document(
                        page_content=cleaned,
                        metadata={
                            "source": filename,
                            "page": "N/A",
                            "chunk_id": chunk_id
                        }
                    )
                    indexed_chunks.append(doc_obj)
                    embedding = embedding_model.embed_query(cleaned)
                    all_embeddings.append(embedding)

                logger.info(f"✅ Text file processed: {len(indexed_chunks)} chunks")
                return indexed_chunks, all_embeddings

            except Exception as fallback_error:
                logger.error(f"❌ Fallback text processing failed: {str(fallback_error)}")
                return [], []

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

    except Exception as e:
        logger.error(f"❌ Critical error in PDF processing: {str(e)}", exc_info=True)
        return [], []

pdf_processor = process_pdf
