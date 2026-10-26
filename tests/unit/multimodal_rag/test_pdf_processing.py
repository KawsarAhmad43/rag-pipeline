import pytest
from unittest.mock import MagicMock, patch
from src.multimodal_rag.data_load.doc_load import PDFHandler

class TestPDFHandler:
    @patch("src.multimodal_rag.data_load.doc_load.fitz.open")
    def test_extract_text_success(self, mock_fitz_open):
        """Test successful text extraction from a mocked PDF."""
        # Mock the document object and its pages
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is a test page."

        # Make the doc iterable (list of pages)
        mock_doc.__iter__.return_value = [mock_page]

        # Context manager for fitz.open
        mock_fitz_open.return_value = mock_doc

        handler = PDFHandler("dummy.pdf")
        text = handler.extract_text()

        assert "This is a test page." in text
        mock_fitz_open.assert_called_with("dummy.pdf")

    @patch("src.multimodal_rag.data_load.doc_load.fitz.open")
    def test_extract_text_empty(self, mock_fitz_open):
        """Test extraction from an empty PDF raises ValueError."""
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = [] # No pages
        mock_fitz_open.return_value = mock_doc

        handler = PDFHandler("empty.pdf")

        with pytest.raises(ValueError, match="No text extracted"):
            handler.extract_text()

    @patch("src.multimodal_rag.data_load.doc_load.RecursiveCharacterTextSplitter")
    @patch("src.multimodal_rag.data_load.doc_load.fitz.open")
    def test_get_chunks(self, mock_fitz_open, mock_splitter_cls):
        """Test text chunking logic."""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "text " * 100
        mock_doc.__iter__.return_value = [mock_page]
        mock_fitz_open.return_value = mock_doc

        # Configure splitter mock
        mock_splitter_instance = mock_splitter_cls.return_value
        mock_splitter_instance.split_text.return_value = ["chunk1", "chunk2"]

        handler = PDFHandler("dummy.pdf")
        chunks = handler.get_chunks(chunk_size=100, chunk_overlap=10)

        assert len(chunks) == 2
        assert chunks == ["chunk1", "chunk2"]
