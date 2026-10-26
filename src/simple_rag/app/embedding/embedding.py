import logging
import os
import torch
import numpy as np
import psutil
import gc
from transformers import AutoModel, AutoTokenizer
from config import config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variable to cache the embedding model instance
_cached_embedding_model_instance = None

def get_memory_usage():
    """Log current CPU and GPU memory usage in MB."""
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / 1024 / 1024
    gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    return cpu_mem, gpu_mem

# MONKEYPATCH: Bypass transformers strict version checks for offline loading
import transformers
if hasattr(transformers, "utils") and hasattr(transformers.utils, "versions"):
    original_require_version = transformers.utils.versions.require_version
    def monkey_require_version(requirement, hint=None):
        logger.debug(f"Monkeypatch bypassing version check: {requirement}")
        return
    transformers.utils.versions.require_version = monkey_require_version

class MultimodalEmbeddings:
    def __init__(self, model_path):
        cpu_mem, gpu_mem = get_memory_usage()
        logger.debug(f"Loading multimodal embedding model from: {model_path}. Memory usage: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")
        try:
            if os.path.isdir(model_path) and (os.path.exists(os.path.join(model_path, 'preprocessor_config.json')) or os.path.exists(os.path.join(model_path, 'config.json'))):
                logger.debug(f"Using local model from {model_path}")

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # # OLD LOGIC
                # self.tokenizer = AutoTokenizer.from_pretrained(
                #     model_path,
                #     local_files_only=True,
                #     trust_remote_code=False
                # )
                # self.model = AutoModel.from_pretrained(
                #     model_path,
                #     local_files_only=True,
                #     device_map="cpu", # Explicitly force CPU for embedding model
                #     torch_dtype=torch.float32, # Use float32 for CPU stability
                #     low_cpu_mem_usage=True,
                #     trust_remote_code=False,
                #     use_safetensors=True
                # )

                # NEW LOGIC
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True # Required for Nomic models
                )

                # Ensure pad_token is set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModel.from_pretrained(
                    model_path,
                    local_files_only=True,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    use_safetensors=True
                )

                cpu_mem, gpu_mem = get_memory_usage()
                logger.debug(f"Multimodal embedding model loaded successfully. Memory usage: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                logger.error(f"Local embedding model not found or incompatible at {model_path}")
                raise ValueError(f"Local embedding model not found or incompatible at {model_path}")
        except Exception as e:
            cpu_mem, gpu_mem = get_memory_usage()
            logger.error(f"Failed to load embedding model from {model_path}: {str(e)}. Memory usage: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")
            raise ValueError(f"Error loading embedding model: {str(e)}")

    def embed_query(self, text, image=None):
        try:
            if image is not None:
                raise NotImplementedError("Image processing not yet implemented")

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to("cpu") # Ensure inputs are on CPU

            with torch.no_grad():
                self.model.eval()
                outputs = self.model(**inputs)

                # Robust extraction of features
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                elif isinstance(outputs, torch.Tensor):
                    embeddings = outputs.mean(dim=1) if len(outputs.shape) == 3 else outputs
                else:
                    # Fallback for unexpected formats
                    try:
                        embeddings = outputs[0]
                        if len(embeddings.shape) == 3:
                            embeddings = embeddings.mean(dim=1)
                    except Exception:
                        embeddings = outputs

            embeddings_np = embeddings.squeeze().cpu().numpy()

            if not np.all(np.isfinite(embeddings_np)):
                logger.warning("Embedding contains NaN or inf values - replacing with zeros")
                embeddings_np = np.nan_to_num(embeddings_np)

            return embeddings_np.tolist()
        except Exception as e:
            cpu_mem, gpu_mem = get_memory_usage()
            logger.error(f"Error generating embedding for text: {str(e)}. Memory usage: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")
            raise

def get_embedding_model():
    global _cached_embedding_model_instance

    if _cached_embedding_model_instance is not None:
        logger.debug("Returning cached embedding model instance.")
        return _cached_embedding_model_instance

    try:
        _cached_embedding_model_instance = MultimodalEmbeddings(model_path=config.EMBEDDING_MODEL)
        return _cached_embedding_model_instance
    except Exception as e:
        cpu_mem, gpu_mem = get_memory_usage()
        logger.error(f"Error initializing embedding model: {str(e)}. Memory usage: CPU {cpu_mem:.2f} MB, GPU {gpu_mem:.2f} MB")
        raise
