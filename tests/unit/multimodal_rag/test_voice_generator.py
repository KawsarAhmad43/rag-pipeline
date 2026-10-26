import pytest
from unittest.mock import MagicMock, patch
import os

# Ensure config environment matches logic
os.environ["GOOGLE_API_KEY"] = "dummy_key"
os.environ["MultiModal_RAG_GENERATOR_MODEL"] = "dummy_model"

from src.multimodal_rag.generation.response_generation_voice import ResponseGenerator, VOICE_NAMES

class TestVoiceGenerator:
    @patch("src.multimodal_rag.generation.response_generation_voice.load_engine")
    @patch("src.multimodal_rag.generation.response_generation_voice.ApiResponseGenerator")
    def test_generate_with_tts_api(self, mock_api_cls, mock_load_engine):
        """Test API generation with TTS enabled."""
        # Setup API mock
        mock_api_instance = mock_api_cls.return_value
        mock_api_instance.generate.return_value = "Hello Voice"

        # Setup TTS mock
        mock_engine = MagicMock()
        mock_engine.synthesize_to_wav.return_value = "/path/to/output.wav"
        mock_load_engine.return_value = mock_engine

        generator = ResponseGenerator()

        result = generator.generate(
            query="Hi",
            context="",
            session_id="s1",
            mode="api",
            speak=True,
            voice=VOICE_NAMES[0] # Use a valid voice
        )

        assert isinstance(result, dict)
        assert result["text"] == "Hello Voice"
        assert result["audio_path"] == "/path/to/output.wav"
        assert result["voice_used"] == VOICE_NAMES[0]

        # Verify calls
        mock_load_engine.assert_called_once()
        mock_engine.synthesize_to_wav.assert_called()

    @patch("src.multimodal_rag.generation.response_generation_voice.load_engine")
    @patch("src.multimodal_rag.generation.response_generation_voice.LocalResponseGenerator")
    def test_generate_with_tts_local(self, mock_local_cls, mock_load_engine):
        """Test Local generation with TTS enabled."""
        with patch("src.multimodal_rag.generation.response_generation_voice.config") as mock_config:
            mock_config.GENERATOR_MODEL_PATH.is_dir.return_value = True

            mock_instance = mock_local_cls.return_value
            mock_instance.generate.return_value = "Local Voice"

            mock_engine = MagicMock()
            mock_engine.synthesize_to_wav.return_value = "/local/out.wav"
            mock_load_engine.return_value = mock_engine

            generator = ResponseGenerator()
            generator.local_available = True

            result = generator.generate(
                query="Hi",
                context="",
                session_id="s1",
                mode="local",
                speak=True,
                voice=VOICE_NAMES[0]
            )

            assert result["text"] == "Local Voice"
            assert result["audio_path"] == "/local/out.wav"

    def test_invalid_voice_raises_error(self):
        """Test that providing an invalid voice raises ValueError."""
        generator = ResponseGenerator()

        # Depending on implementation, it might raise ValueError or return an error string.
        # Looking at code: `_maybe_tts` raises ValueError.
        # But `AgenticChatSystem.get_response` catches it?
        # Here we are testing `ResponseGenerator.generate`, which calls `_maybe_tts`.

        with pytest.raises(ValueError, match="Voice .* not in supported VOICE_NAMES"):
            generator.generate(
                query="Hi",
                context="",
                session_id="s1",
                mode="api",
                speak=True,
                voice="invalid_voice_name_XYZ"
            )

    @patch("src.multimodal_rag.generation.response_generation_voice.ApiResponseGenerator")
    def test_speak_false_returns_string(self, mock_api_cls):
        """Test that speak=False returns a plain string, not a dict."""
        mock_api_instance = mock_api_cls.return_value
        mock_api_instance.generate.return_value = "Text Only"

        generator = ResponseGenerator()

        result = generator.generate(
            query="Hi",
            context="",
            session_id="s1",
            mode="api",
            speak=False
        )

        assert isinstance(result, str)
        assert result == "Text Only"
