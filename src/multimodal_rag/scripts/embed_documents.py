import sys
from pathlib import Path
import logging
import os

# ---------------- project root on sys.path ----------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ---------------- local imports ----------------
from config import config
from data_load.doc_load import PDFHandler
from vector_store.vector_db import ChromaDBHandler
from embedder.embedder import ColnomicEmbeddings   # 🔄 USE local-only embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _preflight_local_embedder() -> None:
    """Warm-up check: ensure the local embedding model folder is valid and loadable."""
    emb_path = Path(str(config.EMBEDDING_MODEL_PATH)).expanduser().resolve()
    logging.info(f"🔧 Preflighting local embedder at: {emb_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"EMBEDDING_MODEL_PATH does not exist: {emb_path}")

    # Load + quick encode + cleanup (CPU only, offline)
    emb = ColnomicEmbeddings(model_path=str(emb_path), device="cpu")
    _ = emb.embed_query("sanity check for local Nomic embeddings")
    emb.cleanup()
    logging.info("✅ Local embedder preflight OK.")


def process_all_pdfs_for_indexing():
    """
    Processes all PDF files found in the configured data directory.
    For each PDF, it extracts text chunks and indexes them into a session-specific
    ChromaDB instance (collection_name = PDF stem).
    """
    # Preflight the local embedder first so failures are clear
    _preflight_local_embedder()

    data_dir = Path(str(config.DATA_PATH)).expanduser().resolve()
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        logging.warning(f"No PDF files found in '{data_dir}'. Add PDFs there for indexing.")
        return

    logging.info(f"Found {len(pdf_files)} PDF file(s) to process for indexing.")

    # Ensure vector DB base path exists
    base_dir = Path(str(config.VECTOR_DB_BASE_PATH)).expanduser().resolve()
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created vector DB base directory at '{base_dir}'.")

    total_chunks_indexed = 0

    for pdf_path in pdf_files:
        session_id_for_db = pdf_path.stem  # one collection per PDF
        persist_dir = base_dir / session_id_for_db

        logging.info(f"--- Processing PDF: '{pdf_path.name}' → DB: '{session_id_for_db}' ---")
        pdf_handler = PDFHandler(str(pdf_path))

        try:
            chunks = pdf_handler.get_chunks()
            if not chunks:
                logging.warning(f"No text chunks extracted from '{pdf_path.name}'. Skipping indexing for this file.")
                continue

            # Prepare metadata and ids for each chunk
            metadatas = [{"source_pdf": pdf_path.name, "chunk_index": i} for i in range(len(chunks))]
            ids = [f"{session_id_for_db}_chunk_{i}" for i in range(len(chunks))]

            db_handler = ChromaDBHandler(
                session_specific_persist_directory=str(persist_dir),
                collection_name=session_id_for_db
            )

            # Build/rebuild index with metadata + ids
            db_handler.build_index(chunks, metadatas=metadatas, ids=ids)
            db_handler.cleanup()  # free embedder/model memory between PDFs

            logging.info(
                f"✅ Indexed {len(chunks)} text chunks from '{pdf_path.name}' "
                f"into database '{session_id_for_db}' at '{persist_dir}'."
            )
            total_chunks_indexed += len(chunks)
        except Exception as e:
            logging.error(f"❌ Failed to process and index '{pdf_path.name}': {e}", exc_info=True)

    logging.info(f"Finished processing PDFs. Total chunks indexed: {total_chunks_indexed}")


if __name__ == "__main__":
    process_all_pdfs_for_indexing()
