import sys
import os
from unittest.mock import MagicMock
import pytest

# Add project root to sys.path so we can import src modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Mock external heavy dependencies BEFORE they are imported by app logic
# This prevents model loading during test collection/execution

# Mock PyMuPDF (fitz)
sys.modules["fitz"] = MagicMock()

# Mock Transformers (AutoTokenizer, AutoModelForCausalLM)
mock_transformers = MagicMock()
sys.modules["transformers"] = mock_transformers

# Mock Torch
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False # Default to CPU for tests
sys.modules["torch"] = mock_torch

# Mock Google Generative AI
sys.modules["google.generativeai"] = MagicMock()

# Mock ChromaDB
sys.modules["chromadb"] = MagicMock()

# Mock Sentence Transformers
sys.modules["sentence_transformers"] = MagicMock()

# Mock LangChain (and submodules used)
mock_langchain = MagicMock()
sys.modules["langchain"] = mock_langchain
sys.modules["langchain.text_splitter"] = MagicMock()
sys.modules["langchain.memory"] = MagicMock()

# Mock the project-specific 'config' module
# This fixes the ambiguity between root 'config' dir and 'src/multimodal_rag/config.py'
mock_config_module = MagicMock()
inner_config = MagicMock()
inner_config.GOOGLE_API_KEY = "dummy_key"
inner_config.GENERATOR_MODEL_PATH = MagicMock()
inner_config.GENERATOR_MODEL_PATH.exists.return_value = True
inner_config.GENERATOR_MODEL_PATH.is_dir.return_value = True
inner_config.TRANSFORMERS_OFFLINE = False
inner_config.HF_HUB_OFFLINE = False
# Set attributes that might be accessed
mock_config_module.config = inner_config
sys.modules["config"] = mock_config_module

# Mock Audio Libraries for Voice Tests
sys.modules["soundfile"] = MagicMock()
sys.modules["phonemizer"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.io"] = MagicMock()
sys.modules["scipy.io.wavfile"] = MagicMock()
sys.modules["numpy"] = MagicMock() # Often used with audio
sys.modules["kokoro_onnx"] = MagicMock()

@pytest.fixture
def mock_config(monkeypatch):
    """Fixture to mock config values."""
    # We need to mock the Config object in src.multimodal_rag.config
    # But since it's already instantiated as 'config' object at module level,
    # we might need to patch attributes on that object.
    pass
