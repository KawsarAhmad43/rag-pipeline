from config import config
import logging
import os
import torch
import psutil
import gc
from .llm_openai import get_openai_llm
from .llm_gemini import get_gemini_llm
from .llm_local_gemma import get_gemma_llm
from .llm_local_llama import get_llama_llm
from utils.system import get_memory_usage, clean_gpu_cache

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

_cached_llm_instance = None



def get_llm():
    global _cached_llm_instance

    if _cached_llm_instance is not None:
        logger.debug("Returning cached LLM instance.")
        return _cached_llm_instance

    try:
        cpu_mem, gpu_mem = get_memory_usage()
        logger.debug(f"Memory usage before loading LLM: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")

        model_path = os.path.abspath(config.LLM_MODEL)
        model_name = os.path.basename(model_path).lower()

        # Prioritize local Gemma model
        if "gemma" in model_name:
            logger.debug(f"Using local Gemma LLM from: {model_path}")
            _cached_llm_instance = get_gemma_llm(model_path)
        elif "llama" in model_name:
            logger.debug(f"Using local Llama LLM from: {model_path}")
            _cached_llm_instance = get_llama_llm(model_path)
        elif config.OPENAI_API_KEY:
            logger.debug("Using OpenAI LLM")
            _cached_llm_instance = get_openai_llm()
        elif config.GEMINI_API_KEY:
            logger.debug("Using Gemini LLM")
            _cached_llm_instance = get_gemini_llm()
        else:
            logger.error(f"Unsupported model name: {model_name}")
            raise ValueError(f"Unsupported model name: {model_name}")

        cpu_mem, gpu_mem = get_memory_usage()
        logger.info(f"LLM initialized successfully. Memory usage: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")
        clean_gpu_cache()
        return _cached_llm_instance

    except Exception as e:
        cpu_mem, gpu_mem = get_memory_usage()
        logger.error(f"Error initializing LLM: {str(e)}. Memory usage: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")
        raise RuntimeError(f"LLM initialization failed: {str(e)}")
