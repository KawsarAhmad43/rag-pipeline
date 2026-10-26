# Configuration settings
CHUNK_SIZE = 300
CHUNK_OVERLAP = 0
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "extending_context_window_llama_3"

# Model configurations
MODELS = {
    "openai": {
        "gpt-4o": {
            "name": "GPT-4o",
            "description": "OpenAI's most advanced model",
            "provider": "openai"
        },
        "gpt-4o-mini": {
            "name": "GPT-4o Mini",
            "description": "Fast and efficient GPT-4 model",
            "provider": "openai"
        },
        "gpt-3.5-turbo": {
            "name": "GPT-3.5 Turbo",
            "description": "Fast and cost-effective model",
            "provider": "openai"
        }
    },
    "gemini": {
        "gemini-1.5-pro": {
            "name": "Gemini 1.5 Pro",
            "description": "Google's most capable model",
            "provider": "gemini"
        },
        "gemini-1.5-flash": {
            "name": "Gemini 1.5 Flash",
            "description": "Fast and efficient Gemini model",
            "provider": "gemini"
        }
    }
}

# Default model
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_PROVIDER = "openai"

# Global variable to store current model selection
current_model = DEFAULT_MODEL
current_provider = DEFAULT_PROVIDER

def get_current_model():
    """Get the currently selected model"""
    global current_model
    return current_model

def get_current_provider():
    """Get the currently selected provider"""
    global current_provider
    return current_provider

def set_current_model(model_name: str):
    """Set the current model"""
    global current_model, current_provider

    # Find the provider for the given model
    for provider, models in MODELS.items():
        if model_name in models:
            current_model = model_name
            current_provider = provider
            return True

    # If model not found, use default
    current_model = DEFAULT_MODEL
    current_provider = DEFAULT_PROVIDER
    return False

def get_available_models():
    """Get all available models organized by provider"""
    return MODELS
