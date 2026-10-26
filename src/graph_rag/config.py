# # ===========================Local envirnment config===========================
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
env_path = os.path.abspath(env_path)
load_dotenv(dotenv_path=env_path)
logger.info(f"Loading .env from: {env_path}")

class Config:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

        # Updated to use the standard port 7687
        self.NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        self.NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphragpassword")
        self.NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
        self.EMBEDDING_DIMENSION = 768

        self.LOG_PATH = os.path.abspath(os.getenv("GRAPH_RAG_LOG_PATH", "/home/aiteam/Desktop/rag_pipelines/src/graph_rag/data/logs"))
        self.VECTOR_DB_PATH = os.path.abspath(os.getenv("GRAPH_RAG_VECTOR_DB_PATH", "/home/aiteam/Desktop/rag_pipelines/src/graph_rag/data/vector_db"))
        self.CHAT_MEMORY_PATH = os.path.abspath(os.getenv("GRAPH_RAG_CHAT_MEMORY_PATH", "/home/aiteam/Desktop/rag_pipelines/src/graph_rag/data/chatmemory"))

        self.LLM_MODEL = self._validate_path(
            os.getenv("GRAPH_RAG_LLM_MODEL_NAME", "/home/aiteam/Desktop/rag_pipelines/models/gemma-3-4b-it"),
            "LLM model",
            config_file="config.json"
        )
        self.EMBEDDING_MODEL = self._validate_path(
            os.getenv("GRAPH_RAG_EMBEDDING_MODEL_NAME", "/home/aiteam/Desktop/rag_pipelines/models/colnomic-embed-multimodal-3b"),
            "Embedding model",
            config_file="preprocessor_config.json"
        )
        self.RERANKER_MODEL = self._validate_path(
            os.getenv("GRAPH_RERANKER_MODEL_NAME", "/home/aiteam/Desktop/rag_pipelines/models/bge-reranker-v2-m3"),
            "Reranker model",
            config_file="config.json"
        )
        self.SPACY_MODEL = os.getenv("GRAPH_RAG_SPACY_MODEL", "en_core_web_sm")
        self.MAX_DEPTH = int(os.getenv("GRAPH_RAG_MAX_DEPTH", "2"))
        self.SELECT_K = int(os.getenv("GRAPH_RAG_SELECT_K", "5"))
        self.START_K = int(os.getenv("GRAPH_RAG_START_K", "2"))

        # Dynamic search parameters
        self.DYNAMIC_SEARCH_MAX_DEPTH = int(os.getenv("DYNAMIC_SEARCH_MAX_DEPTH", "2"))
        self.DYNAMIC_SEARCH_K_VECTOR = int(os.getenv("DYNAMIC_SEARCH_K_VECTOR", "10"))
        self.DYNAMIC_SEARCH_K_FINAL = int(os.getenv("DYNAMIC_SEARCH_K_FINAL", "5"))

        # Graph traversal parameters
        self.GRAPH_TRAVERSAL_ENABLED = os.getenv("GRAPH_TRAVERSAL_ENABLED", "true").lower() == "true"
        self.SEMANTIC_SEARCH_ENABLED = os.getenv("SEMANTIC_SEARCH_ENABLED", "true").lower() == "true"
        self.NEIGHBORHOOD_ANALYSIS_ENABLED = os.getenv("NEIGHBORHOOD_ANALYSIS_ENABLED", "true").lower() == "true"
        self.HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
        self.HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))

        # RAGAs Configuration
        self.RAGAS_MODEL = self._validate_path(
            os.getenv("GRAPH_RAG_LLM_MODEL_NAME", "/home/aiteam/Desktop/rag_pipelines/models/gemma-3-4b-it"),
            "LLM model",
            config_file="config.json"
        )
        self.RAGAS_API_KEY = None
        self.RAGAS_EVAL_METRICS = ["context_relevance", "answer_relevance", "faithfulness", "answer_correctness"]
        self.RAGAS_DATA_PATH = "data/RAGAs_Template.csv"
        self.RAGAS_OUTPUT_PATH = "/home/aiteam/Desktop/rag_pipelines/src/graph_rag/data/evaluations/ragas_results.json"


    def _validate_path(self, path: str, model_type: str, config_file: str) -> str:
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
