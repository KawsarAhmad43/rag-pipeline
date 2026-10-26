import os
from pathlib import Path

class ModelPaths:
    """Manages model paths and availability."""

    def __init__(self):
        # Base path for models
        self.base_path = Path("/home/aiteam/Desktop/rag_pipelines/models")

        # Define model paths with priority order (smaller/more efficient first)
        self.model_paths = {
            # Smaller, more memory-efficient models first
            "Llama-3.2-1B": self.base_path / "Llama-3.2-1B",
            "Gemma LLM": self.base_path / "gemma-3-4b-it",  # 4B model
            "GME Qwen2 VL (Gen)": self.base_path / "gme-Qwen2-VL-2B-Instruct",
            "GME Qwen2 VL (Emb)": self.base_path / "gme-Qwen2-VL-2B-Instruct",

            # Embedding models (smaller first)
            "BGE Reranker": self.base_path / "bge-reranker-v2-m3",
            "Colnomic Embed 3B": self.base_path / "colnomic-embed-multimodal-3b",
            "Nomic Embed 7B": self.base_path / "nomic-embed-multimodal-7b",  # Largest, last priority
        }

        # Model type mapping for better selection
        self.model_types = {
            "llm": ["Llama-3.2-1B", "Gemma LLM", "GME Qwen2 VL (Gen)"],
            "embedding": ["BGE Reranker", "Colnomic Embed 3B", "Nomic Embed 7B"],
            "multimodal": ["GME Qwen2 VL (Gen)", "GME Qwen2 VL (Emb)"]
        }

    def check_availability(self):
        """Check which models are available and return them in priority order."""
        available_models = {}

        print("--- Model Availability Check ---")

        # Check each model in priority order
        for model_name, model_path in self.model_paths.items():
            if self._is_model_available(model_path):
                available_models[model_name] = str(model_path)
                print(f"✅ {model_name}: Available at {model_path}")
            else:
                print(f"❌ {model_name}: Not available at {model_path}")

        print(f"\nTotal available models: {len(available_models)}")

        return available_models

    def _is_model_available(self, model_path: Path) -> bool:
        """Check if a model is available at the given path."""
        if not model_path.exists():
            return False

        # Check for common model files
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors"
        ]

        # For some models, check for specific files
        if "gemma" in str(model_path).lower():
            required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        elif "sentence" in str(model_path).lower() or "embed" in str(model_path).lower():
            required_files = ["config.json", "modules.json"]

        # Check if at least some key files exist
        existing_files = [f for f in required_files if (model_path / f).exists()]
        return len(existing_files) >= 2  # At least 2 key files should exist

    def get_best_models(self, available_models: dict):
        """Get the best available models for each type, prioritizing smaller/more efficient ones."""
        best_models = {}

        # Get best LLM (prioritize smaller models)
        for llm_name in self.model_types["llm"]:
            if llm_name in available_models:
                best_models["llm"] = available_models[llm_name]
                break

        # Get best embedding model (prioritize smaller models)
        for emb_name in self.model_types["embedding"]:
            if emb_name in available_models:
                best_models["embedding"] = available_models[emb_name]
                break

        return best_models
