import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging
load_dotenv()  # Load environment variables from .env file

# MONKEYPATCH: Bypass transformers version check for offline mode
import transformers.utils.versions
def dummy_require_version(*args, **kwargs): return None
transformers.utils.versions.require_version = dummy_require_version

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Determine Project Root ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print(PROJECT_ROOT)

# MODEL_Dir ="/home/aiteam/Desktop/rag_pipelines/models"
MODEL_Dir = "/home/aiteam/Desktop/all_downloaded_models"

# Data paths
PROJECT_ROOT2 = Path(__file__).resolve().parent
print(PROJECT_ROOT2 )

logging.info(f"Project root determined to be: {PROJECT_ROOT}")

# --- Add Project Root to sys.path ---
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    logging.info(f"Added '{PROJECT_ROOT}' to sys.path for module discovery.")

# Load environment variables from .env file
try:
    load_dotenv(PROJECT_ROOT2 / ".env")
    logging.info("Successfully loaded .env file.")
except Exception as e:
    logging.error(f"Could not load .env file from {PROJECT_ROOT2 / '.env'}: {e}")

# --- Central Configuration Class ---
class Config:
    """
    Central configuration class.
    Reads settings from environment variables with sensible defaults.
    """
    PROJECT_ROOT = PROJECT_ROOT

    # Model Names (from .env) - Stripped for safety
    EMBEDDER_MODEL_NAME = os.getenv("MultiModal_RAG_EMBEDDER_MODEL", "colnomic-embed-multimodal-3b").strip()
    RERANKER_MODEL_NAME = os.getenv("MultiModal_RAG_RERANKER_MODEL", "bge-reranker-v2-m3").strip()
    GENERATOR_MODEL_NAME = os.getenv("MultiModal_RAG_GENERATOR_MODEL", "gemma-3-4b-it").strip()


    # Model Cache Directory
    MODEL_CACHE_DIR = Path(os.getenv("MultiModal_RAG_MODEL_CACHE_DIR", "/home/aiteam/Desktop/all_downloaded_models").strip())

    # Full Model Paths (derived from MODEL_CACHE_DIR and model names)
    EMBEDDING_MODEL_PATH = MODEL_CACHE_DIR / EMBEDDER_MODEL_NAME
    RERANKER_MODEL_PATH = Path(os.getenv("MultiModal_RAG_RERANKER_PATH", str(MODEL_CACHE_DIR / RERANKER_MODEL_NAME)).strip())
    GENERATOR_MODEL_PATH = Path(os.getenv("MultiModal_RAG_GENERATOR_PATH", str(MODEL_CACHE_DIR / GENERATOR_MODEL_NAME)).strip())

    #   # Full Model Paths (derived from MODEL_CACHE_DIR and model names)
    # EMBEDDING_MODEL_PATH = MODEL_CACHE_DIR/ "colnomic-embed-multimodal-3b"
    # RERANKER_MODEL_PATH =  MODEL_CACHE_DIR/ "bge-reranker-v2-m3"
    # GENERATOR_MODEL_PATH = MODEL_CACHE_DIR / "gemma-3-4b-it"

    print(f"Using embedding model path: {EMBEDDING_MODEL_PATH}",'\n',
          f"Using reranker model path: {RERANKER_MODEL_PATH}",'\n',
          f"Using generator model path: {GENERATOR_MODEL_PATH}")

    # Data paths
    PROJECT_ROOT2 = Path(__file__).resolve().parent
    print(PROJECT_ROOT2 )
    DATA_PATH = PROJECT_ROOT2 / "data"

    ##(Changed) Vector DB settings
    VECTOR_DB_BASE_PATH = PROJECT_ROOT2 / os.getenv(
        "MultiModal_RAG_VECTOR_DB_PATH", "chroma_db"
    )
    CHROMA_COLLECTION_NAME = os.getenv(
        "MultiModal_RAG_CHROMA_COLLECTION_NAME", "rag_embeddings"
    )

    # API Settings (from .env)
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 5001))

    # RAG Specific Settings (from .env)
    TOP_K = int(os.getenv("MultiModal_RAG_TOP_K", 5))

    # Google API Configuration (from .env)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro")

    # Offline mode flags for Hugging Face (from .env)
    TRANSFORMERS_OFFLINE = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
    HF_HUB_OFFLINE = os.getenv("HF_HUB_OFFLINE", "0") == "1"

    # Generation Mode Options
    SUPPORTED_GENERATION_MODES = ["api", "local"]
    DEFAULT_GENERATION_MODE = "api" if GOOGLE_API_KEY else "local"




    @staticmethod
    def validate_paths():
        logging.info("--- Validating configuration paths ---")
        paths_to_check = {
            "Model Cache Directory": Config.MODEL_CACHE_DIR,
            "Embedder Model Path": Config.EMBEDDING_MODEL_PATH,
            "Reranker Model Path": Config.RERANKER_MODEL_PATH,
            "Generator Model Path": Config.GENERATOR_MODEL_PATH,
            "Data Directory": Config.DATA_PATH,
            "Vector DB Base Path": Config.VECTOR_DB_BASE_PATH,
        }
        all_good = True
        for name, path in paths_to_check.items():
            if "Model Path" in name and not path.is_dir():
                logging.warning(f"❌ Path for '{name}' does not exist or is not a directory: {path}")
                all_good = False
            elif not path.exists():
                logging.warning(f"❌ Path for '{name}' does not exist: {path}")
                all_good = False
            else:
                logging.info(f"✅ Path for '{name}' found: {path}")

        # Check API key separately
        if not Config.GOOGLE_API_KEY:
            logging.warning("⚠️ GOOGLE_API_KEY is missing in .env. Only local generation will be available.")

        if not all_good:
            logging.error(
                "One or more critical paths are missing. "
                "Please check your .env file and project structure."
            )
            raise FileNotFoundError("Critical configuration paths missing or invalid.")
        else:
            logging.info("--- All configuration paths validated successfully ---")
        return all_good

    @staticmethod
    def get_available_sessions():
        """Get list of available ChromaDB sessions"""
        sessions = []
        if Config.VECTOR_DB_BASE_PATH.exists():
            for item in Config.VECTOR_DB_BASE_PATH.iterdir():
                if item.is_dir():
                    sessions.append(item.name)
        return sorted(sessions)



config = Config()


# Create base directories if they don't exist
os.makedirs(config.MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(config.DATA_PATH, exist_ok=True)
os.makedirs(config.VECTOR_DB_BASE_PATH, exist_ok=True)

# Validate paths
try:
    config.validate_paths()
except FileNotFoundError:
    sys.exit("Exiting due to critical configuration errors.")

print(config.EMBEDDING_MODEL_PATH,'\n',config.RERANKER_MODEL_NAME,'\n',config.GENERATOR_MODEL_NAME)
print(config.DATA_PATH)
