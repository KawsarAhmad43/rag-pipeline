import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import os
import textwrap

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PDFHandler:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        """Extracts text from PDF or TXT files."""
        if self.pdf_path.lower().endswith('.txt'):
            try:
                with open(self.pdf_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                logging.info(f"{len(text)} characters read from TXT file: {self.pdf_path}")
                return text
            except Exception as e:
                raise RuntimeError(f"Error reading TXT {self.pdf_path}: {str(e)}")

        try:
            doc = fitz.open(self.pdf_path)
            text = ""
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
            logging.info(f"{len(text)} characters extracted from PDF file: {self.pdf_path}")
        except FileNotFoundError:
            raise ValueError(f"File not found at path: {self.pdf_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading file {self.pdf_path}: {str(e)}")

        if not text.strip():
            logging.warning(f"No text extracted from PDF: {self.pdf_path}. Check if it's empty or scanned.")
            raise ValueError("No text extracted from the PDF.")

        return text

    def get_chunks(self, chunk_size=1000, chunk_overlap=200):
        """Extracts text and splits it into chunks."""
        text = self.extract_text()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        logging.info(f"Split text into {len(chunks)} chunks from {self.pdf_path}")
        return chunks
