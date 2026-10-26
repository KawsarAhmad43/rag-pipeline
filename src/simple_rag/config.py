from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load .env file
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
env_path = os.path.abspath(env_path)
load_dotenv(dotenv_path=env_path)
logger.info(f"Loading .env from: {env_path}")

class Config:
    def __init__(self):
        # Validate environment variables
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
        # Use path relative to the config.py file
        # In container: base_dir = /app/src/simple_rag
        # On host: base_dir = /home/aiteam/Desktop/rag_pipelines/src/simple_rag
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.LOG_PATH = os.getenv("SIMPLE_LOG_PATH", os.path.join(base_dir, "data", "logs"))

        # Validate model paths
        self.LLM_MODEL = self._validate_path(
            os.getenv("SIMPLE_LLM_MODEL_NAME", "/home/aiteam/Desktop/all_downloaded_models/gemma-3-4b-it"),
            "LLM model",
            config_file="config.json"
        )
        # NEW CONFIG (Standalone Model)
        self.EMBEDDING_MODEL = self._validate_path(
            os.getenv("SIMPLE_EMBEDDING_MODEL_NAME", "/home/aiteam/Desktop/all_downloaded_models/gme-Qwen2-VL-2B-Instruct"),
            "Embedding model",
            config_file="config.json"
        )
        self.RERANKER_MODEL = self._validate_path(
            os.getenv("SIMPLE_RERANKER_MODEL_NAME", "/home/aiteam/Desktop/all_downloaded_models/bge-reranker-v2-m3"),
            "Reranker model",
            config_file="config.json"
        )
        # Handle both container and local environments
        # In container: base_dir = /app/src/simple_rag
        # On host: base_dir = /home/aiteam/Desktop/rag_pipelines/src/simple_rag
        default_vector_db = os.path.join(base_dir, "data", "vector_db")
        default_chat_memory = os.path.join(base_dir, "data", "chatmemory")

        self.VECTOR_DB_PATH = os.getenv("SIMPLE_VECTOR_DB_PATH", default_vector_db)
        self.CHAT_MEMORY_PATH = os.getenv("SIMPLE_CHAT_MEMORY_PATH", default_chat_memory)

    def _validate_path(self, path: str, model_type: str, config_file: str) -> str:
        """Validate that the specified path exists and is a directory with the specified config file."""
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            logger.error(f"{model_type} path does not exist or is not a directory: {abs_path}")
            raise ValueError(f"{model_type} path is invalid: {abs_path}")
        config_path = os.path.join(abs_path, config_file)
        if not os.path.exists(config_path):
            logger.error(f"{model_type} config file not found at: {config_path}")
            raise ValueError(f"{model_type} config file is missing: {config_path}")
        logger.info(f"{model_type} path validated successfully: {abs_path}")
        return abs_path

config = Config()
