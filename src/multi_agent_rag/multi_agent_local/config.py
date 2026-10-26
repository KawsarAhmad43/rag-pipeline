import os

class ModelPaths:
    """Handles model path configurations and availability checks."""

    def __init__(self):
        # Adjust path to go three levels up from the current file to reach project root
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        self.paths = {
            "Nomic Embed 7B": {
                "path": os.path.join(self.project_root, "models", "nomic-embed-multimodal-7b"),
                "config_file": "preprocessor_config.json"
            },
            "Colnomic Embed 3B": {
                "path": os.path.join(self.project_root, "models", "colnomic-embed-multimodal-3b"),
                "config_file": "preprocessor_config.json"
            },
            "BGE Reranker": {
                "path": os.path.join(self.project_root, "models", "bge-reranker-v2-m3"),
                "config_file": "config.json"
            },
            "Gemma LLM": {
                "path": os.path.join(self.project_root, "models", "gemma-3-4b-it"),
                "config_file": "config.json"
            },
            "GME Qwen2 VL (Gen)": {
                "path": os.path.join(self.project_root, "models", "gme-Qwen2-VL-2B-Instruct"),
                "config_file": "config.json"
            },
            "GME Qwen2 VL (Emb)": {
                "path": os.path.join(self.project_root, "models", "gme-Qwen2-VL-2B-Instruct"),
                "config_file": "config.json"
            },
            "Llama-3.2-1B": {
                "path": os.path.join(self.project_root, "models", "Llama-3.2-1B"),
                "config_file": "config.json"
            }
        }

    def check_availability(self):
        """Check which models are available."""
        available_models = {}
        print("\n--- Model Availability Check ---")
        for model_name, model_info in self.paths.items():
            config_path = os.path.join(model_info["path"], model_info["config_file"])
            if os.path.exists(config_path):
                print(f"✅ {model_name}: Available at {config_path}")
                available_models[model_name] = model_info["path"]
            else:
                print(f"❌ {model_name}: Not found at {config_path}")
        print(f"\nTotal available models: {len(available_models)}")
        return available_models
