import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import settings

class PDFHandler:
    """Handle PDF text extraction and chunking"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_text(self) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(self.pdf_path)
            text = ""
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
            doc.close()
            print(f"{len(text)} characters extracted from PDF file")
        except Exception as e:
            print(f"⚠️ PyMuPDF failed to open {self.pdf_path} (maybe not a valid PDF). Trying as plain text...")
            try:
                with open(self.pdf_path, "r", encoding="utf-8") as f:
                    text = f.read()
                print(f"✅ Successfully read as plain text: {len(text)} characters")
            except Exception as e2:
                raise RuntimeError(f"Error reading PDF or plain text: {str(e)} / {str(e2)}")

        if not text.strip():
            raise ValueError("No text extracted from the PDF. Check if it's empty or scanned.")

        return text

    def get_chunks(self, chunk_size: int = None, chunk_overlap: int = None) -> list:
        """Extract text and split into chunks"""
        text = self.extract_text()

        # Use config defaults if not provided
        chunk_size = chunk_size or settings.MAX_CHUNK_SIZE
        chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        print(f"PDF split into {len(chunks)} chunks")
        return chunks
