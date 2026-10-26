from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from pypdf import PdfReader

class PDFHandler:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        try:
            reader = PdfReader(self.pdf_path)
            text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            print(f"{len(text)} characters extracted from PDF file")
        except FileNotFoundError:
            raise ValueError(f"PDF file not found at path: {self.pdf_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading PDF: {str(e)}")

        if not text.strip():
            raise ValueError("No text extracted from the PDF. Check if it's empty or scanned.")

        return text

    def get_chunks(self, chunk_size=500, chunk_overlap=50):
        text = self.extract_text()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(text)
        return chunks
