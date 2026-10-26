

import logging
import os
from pathlib import Path
import sys
import gc
import numpy as np
import torch
from typing import List, Optional
from langchain.embeddings.base import Embeddings


FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Repo root (three levels up from this file)
REPO_ROOT = FILE_PATH.parents[3]
MODELS = REPO_ROOT / "models"
NOMIC_3B = MODELS / "colnomic-embed-multimodal-3b"  # your local folder


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# MONKEYPATCH: Bypass transformers version check for gme-Qwen2-VL-2B-Instruct
import transformers.utils.versions
def dummy_require_version(*args, **kwargs): return None
transformers.utils.versions.require_version = dummy_require_version

class ColnomicEmbeddings(Embeddings):
    """
    Local Nomic Embeddings with HF mean-pooling.
    Updated to respect config model paths.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu", torch_dtype=torch.float32):
        super().__init__()

        # Use model_path from config if not provided
        # self.model_path = Path(model_path or NOMIC_3B).resolve()
        self.model_path = Path(model_path or config.EMBEDDING_MODEL_PATH).resolve()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Local model not found at {self.model_path}")

        # force_cpu can be set via env var
        force_cpu = os.getenv("FORCE_CPU", "0") == "1"
        self.device = "cpu" if (force_cpu or not torch.cuda.is_available()) else device
        self.torch_dtype = torch.float32 if (force_cpu or not torch.cuda.is_available()) else torch_dtype
        self.model = None
        self.tokenizer = None

        logging.info(f"[Embeddings] Using model at: {self.model_path}")
        self._load_hf_mean_pool()

    def _load_hf_mean_pool(self):
        """Load HuggingFace encoder from local folder."""
        from transformers import AutoTokenizer, AutoModel

        # Ensure local_files_only=True and trust_remote_code=True for gme-Qwen2-VL
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            device_map="cpu", # Force CPU for embeddings stability
            torch_dtype=self.torch_dtype
        )
        self.model.eval()
        logging.info(f"HF encoder ready on {self.device}.")


    @torch.no_grad()
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        # Move to same device as model
        enc = {k: v.to(self.device) for k, v in enc.items()}
        outputs = self.model(**enc)

        # Handle different output formats
        if hasattr(outputs, "last_hidden_state"):
            last = outputs.last_hidden_state
        elif isinstance(outputs, torch.Tensor):
            last = outputs
        else:
            # Fallback for unexpected formats
            try:
                last = outputs[0]
            except Exception:
                last = outputs

        # Ensure last has 3 dimensions [batch, seq_len, hidden_dim]
        if len(last.shape) == 2:
            # If only 2D [batch, hidden_dim], already pooled - return directly
            return last.cpu().numpy().astype("float32").tolist()
        elif len(last.shape) != 3:
            # Unexpected shape, log and try to reshape
            logging.warning(f"Unexpected embedding shape: {last.shape}, attempting to fix")
            # Try to get the batch and hidden dimensions
            batch_size = enc["input_ids"].shape[0]
            if last.numel() % batch_size == 0:
                hidden_dim = last.numel() // batch_size
                last = last.view(batch_size, -1, hidden_dim)
            else:
                raise RuntimeError(f"Cannot reshape embeddings with shape {last.shape}")

        mask = enc["attention_mask"].unsqueeze(-1)
        # Verify dimensions match
        if last.shape[:2] != mask.shape[:2]:
            logging.error(f"Shape mismatch: last={last.shape}, mask={mask.shape}")
            # Try to fix by taking mean across sequence dimension
            if len(last.shape) == 3:
                mean = last.mean(dim=1)
                return mean.cpu().numpy().astype("float32").tolist()
            raise RuntimeError(f"Cannot handle embedding shape mismatch: last={last.shape}, mask={mask.shape}")

        mean = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return mean.cpu().numpy().astype("float32").tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


    @torch.no_grad()
    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        """Embed images using the vision encoder (if present in the Nomic model)."""
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=True)

        from PIL import Image
        images = [Image.open(p).convert("RGB") for p in image_paths]

        inputs = processor(images=images, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)
        return outputs.cpu().detach().numpy().astype("float32").tolist()


    def cleanup(self):
        logging.info("🧹 Unloading Nomic model from memory...")
        for attr in ["model", "tokenizer"]:
            if getattr(self, attr, None) is not None:
                delattr(self, attr)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
