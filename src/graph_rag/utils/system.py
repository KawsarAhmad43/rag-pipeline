import torch
import psutil
import gc
import logging

logger = logging.getLogger(__name__)

def get_memory_usage():
    """Log current CPU and GPU memory usage in MB."""
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / 1024 / 1024
    gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    return cpu_mem, gpu_mem

def clean_gpu_cache():
    """Clear GPU cache and perform garbage collection."""
    if torch.cuda.is_available():
        logger.info("Cleaning GPU cache...")
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU cache cleaned successfully")
    else:
        logger.warning("No CUDA device available, skipping GPU cache cleaning")
