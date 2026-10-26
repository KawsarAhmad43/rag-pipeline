import logging
import os
import torch
import psutil
import gc
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variables to cache the reranker model and tokenizer
_cached_reranker_model = None
_cached_reranker_tokenizer = None

def get_memory_usage():
    """Log current CPU and GPU memory usage in MB."""
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / 1024 / 1024
    gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    return cpu_mem, gpu_mem

def get_reranker():
    global _cached_reranker_model, _cached_reranker_tokenizer

    if _cached_reranker_model is not None and _cached_reranker_tokenizer is not None:
        logger.debug("Returning cached reranker model and tokenizer.")
        return _cached_reranker_model, _cached_reranker_tokenizer

    try:
        model_path = os.path.abspath(config.RERANKER_MODEL)
        cpu_mem, gpu_mem = get_memory_usage()
        logger.debug(f"Loading reranker model from: {model_path}. Memory usage: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")
        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'config.json')):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            _cached_reranker_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=False
            )
            _cached_reranker_model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True,
                device_map="cpu", # Explicitly force CPU for reranker model
                torch_dtype=torch.float32, # Use float32 for CPU stability
                low_cpu_mem_usage=True,
                trust_remote_code=False,
                use_safetensors=True
            )

            cpu_mem, gpu_mem = get_memory_usage()
            logger.debug(f"Reranker model loaded successfully. Memory usage: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return _cached_reranker_model, _cached_reranker_tokenizer
        else:
            logger.error(f"Local reranker model not found or incompatible at {model_path}")
            raise ValueError(f"Local reranker model not found or incompatible at {model_path}")
    except Exception as e:
        cpu_mem, gpu_mem = get_memory_usage()
        logger.error(f"Error loading reranker model: {str(e)}. Memory usage: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")
        raise

def rank_documents(query, documents):
    """Custom ranking function using the transformers model."""
    try:
        model, tokenizer = get_reranker()

        model.eval()

        pairs = [[query, doc.page_content] for doc in documents]

        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to("cpu") # Ensure inputs are on CPU

        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()

        cpu_mem, gpu_mem = get_memory_usage()
        logger.debug(f"Ranked documents. Memory usage: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")

        ranked_indices = scores.argsort()[::-1]

        if ranked_indices.size > 0:
            return [documents[i] for i in ranked_indices[:1]]
        else:
            return []

    except Exception as e:
        cpu_mem, gpu_mem = get_memory_usage()
        logger.error(f"Error ranking documents: {str(e)}. Memory usage: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")
        raise
