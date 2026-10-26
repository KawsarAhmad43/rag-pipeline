import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

class Settings:
    """Application settings and configuration"""

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    API_TITLE: str = "Agentic RAG API"
    API_DESCRIPTION: str = "Multi-modal RAG API supporting OpenAI and Gemini with Tavily Search"

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Gemini Configuration
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

    # Tavily Search Configuration
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
    print(TAVILY_API_KEY)

    # Model Paths
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    ALL_DB_DIR: str = os.path.join(BASE_DIR, "all_db")
    # EMBEDDING_MODEL_PATH: str = "/home/aiteam/Desktop/all_downloaded_models/colnomic-embed-multimodal-3b"
    EMBEDDING_MODEL_PATH: str = "/home/aiteam/Desktop/all_downloaded_models/gme-Qwen2-VL-2B-Instruct"
    RERANKER_MODEL_PATH: str = "/home/aiteam/Desktop/all_downloaded_models/bge-reranker-v2-m3"

    # Generation Configuration
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 512
    DEFAULT_TOP_P: float = 0.8
    DEFAULT_TOP_K: int = 40

    # RAG Configuration
    DEFAULT_RETRIEVAL_K: int = 5
    DEFAULT_RERANK_TOP_K: int = 3
    MAX_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Enhanced System Prompt with Professional Persona
    SYSTEM_PROMPT: str = """You are a highly capable AI assistant, designed to provide comprehensive, accurate, and professional assistance. You have access to a suite of advanced tools, including real-time web search and document analysis.

[YOUR CAPABILITIES]
1. **Dynamic Intelligence**: Answer general knowledge questions or engage in casual conversation with a warm, professional tone.
2. **Real-time Web Search**: Use the Tavily tool to provide the latest information, news, or data that may have changed recently.
3. **Document Context**: If a PDF or document is indexed, prioritize its content for specific queries. Synthesize information when multiple segments are provided.
4. **Tool Use**: Automatically select the best tool (Web Search vs. Knowledge Base) based on the user's intent. Do not ask for permission to use tools.

[GUIDELINES]
- **Persona**: Maintain a helpful and direct persona, similar to advanced industrial AI models.
- **Accuracy**: Be factual. If info is missing from documents, use web search or general knowledge but clearly indicate the source type.
- **Formatting**: Use Markdown (bolding, lists, headers) to ensure your responses are easy to scan.

Always aim to WOW the user with the depth and clarity of your responses."""

    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY and not cls.GOOGLE_API_KEY:
            raise ValueError("At least one of OPENAI_API_KEY or GOOGLE_API_KEY must be provided")

        if not cls.TAVILY_API_KEY:
            print("⚠️  WARNING: TAVILY_API_KEY not found. Web search functionality will be limited.")
            print("   Get your free API key at: https://tavily.com")

        return True

# Create settings instance
settings = Settings()
