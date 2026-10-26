from langchain_openai import ChatOpenAI
from config import config
from utils.logging import logger

def get_openai_llm():
    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=config.OPENAI_API_KEY,
            temperature=0.5
        )
        logger.info("OpenAI LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Error initializing OpenAI LLM: {str(e)}")
        raise
