from langchain_google_genai import GoogleGenerativeAI
from config import config
import logging

logger = logging.getLogger(__name__)

def get_gemini_llm():
    logger.debug("Initializing Google Generative AI with model gemini-2.5-flash")
    try:
        llm = GoogleGenerativeAI(api_key=config.GEMINI_API_KEY, model="gemini-2.5-flash")
        logger.info("Google Generative AI initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Gemini LLM: {str(e)}")
        raise RuntimeError(f"Gemini LLM initialization failed: {str(e)}")
