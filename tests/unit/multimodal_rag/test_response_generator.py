import pytest
from unittest.mock import MagicMock, patch
import os

# Set environment before importing ResponseGenerator to avoid config validation exit
os.environ["GOOGLE_API_KEY"] = "dummy_key"
os.environ["MultiModal_RAG_GENERATOR_MODEL"] = "dummy_model"

from src.multimodal_rag.generation.response_generator import ResponseGenerator, LocalResponseGenerator

class TestResponseGenerator:
    @patch("src.multimodal_rag.generation.response_generator.ApiResponseGenerator")
    def test_api_generation_call(self, mock_api_cls):
        """Test that API mode calls the API generator correctly."""
        # Setup mock instance
        mock_instance = mock_api_cls.return_value
        mock_instance.generate.return_value = "API Response"

        # Initialize generator
        generator = ResponseGenerator()

        # Test generate
        response = generator.generate(
            query="Hello",
            context="context",
            session_id="test_session",
            mode="api"
        )

        assert response == "API Response"
        mock_instance.generate.assert_called_once()

    @patch("src.multimodal_rag.generation.response_generator.LocalResponseGenerator")
    def test_local_generation_call(self, mock_local_cls):
        """Test that Local mode calls the Local generator correctly."""
        # Ensure local is marked as available
        with patch("src.multimodal_rag.generation.response_generator.config") as mock_config:
            mock_config.GENERATOR_MODEL_PATH.is_dir.return_value = True

            mock_instance = mock_local_cls.return_value
            mock_instance.generate.return_value = "Local Response"

            generator = ResponseGenerator()
            # Force local available flag (since __init__ checked real path)
            generator.local_available = True

            response = generator.generate(
                query="Hello",
                context="context",
                session_id="test_session",
                mode="local"
            )

            assert response == "Local Response"
            mock_instance.generate.assert_called_once()

    @patch("src.multimodal_rag.generation.response_generator.LocalResponseGenerator")
    def test_direct_multimodal_local(self, mock_local_cls):
        """Test direct multimodal call routed to local generator."""
        from src.multimodal_rag.generation.prompt import multimodal_prompt
        mock_instance = mock_local_cls.return_value
        mock_instance._generate_response.return_value = "Image Description"

        generator = ResponseGenerator()
        generator.local_available = True

        response = generator.generate_direct_multimodal(
            prompt_text="Describe",
            image_path="/path/to/img.jpg",
            mode="local"
        )

        assert response == "Image Description"
        mock_instance._generate_response.assert_called_with(
            multimodal_prompt("Describe"),
            image_path="/path/to/img.jpg",
            pdf_text=None
        )

    def test_invalid_mode(self):
        """Test error handling for invalid generation mode."""
        generator = ResponseGenerator()
        response = generator.generate(
            query="Hi",
            context="",
            session_id="s1",
            mode="invalid_mode"
        )
        assert "Error: Unsupported generation mode" in response
