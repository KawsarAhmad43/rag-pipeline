from langchain_openai import OpenAI
from config import config
import logging

logger = logging.getLogger(__name__)

def get_openai_llm():
    logger.debug("Initializing OpenAI LLM with model gpt-3.5-turbo")
    try:
        llm = OpenAI(api_key=config.OPENAI_API_KEY, model_name="gpt-3.5-turbo")
        logger.info("OpenAI LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI LLM: {str(e)}")
        raise RuntimeError(f"OpenAI LLM initialization failed: {str(e)}")
