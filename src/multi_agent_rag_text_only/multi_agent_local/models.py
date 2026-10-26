import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.embeddings import Embeddings
from typing import List, Optional, Any
from pydantic import Field
import numpy as np
import traceback # Added for improved error logging
import asyncio # For async support

class LocalGemmaChatModel(BaseChatModel):
    """Custom chat model using local Gemma model."""

    model: Any = Field(...)  # Transformers AutoModelForCausalLM
    tokenizer: Any = Field(...)  # Transformers AutoTokenizer

    @property
    def _llm_type(self) -> str:
        return "local_gemma"

    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        text = self._messages_to_text(messages)
        # Ensure inputs are moved to the correct device where the model resides
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=generated_text))])

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        # Since the underlying model.generate is synchronous, run it in a thread pool
        # For LangChain BaseChatModel to be async, it needs an a_generate method.
        # This simple implementation uses a thread pool to avoid blocking the event loop.
        return await asyncio.to_thread(self._generate, messages, stop, **kwargs)

    def _messages_to_text(self, messages: List[BaseMessage]) -> str:
        text = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                text += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                text += f"Assistant: {message.content}\n"
        text += "Assistant: "
        return text

class LocalEmbeddings(Embeddings):
    """Custom embeddings using local embedding model."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Ensure embeddings are computed efficiently, optionally on GPU
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        # Ensure embeddings are computed efficiently, optionally on GPU
        embedding = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return embedding[0].tolist()

class ModelLoader:
    """Loads LLM and embedding models with proper memory management."""

    def __init__(self):
        # Determine device based on CUDA availability and memory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Clear GPU memory on initialization
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            # Check if we have enough memory for both models
            if not self._check_gpu_memory(required_memory_gb=12.0):  # Need ~12GB for both models
                print("⚠️  Limited GPU memory detected, forcing CPU usage for better stability...")
                self.device = "cpu"
                self.torch_dtype = torch.float32
                print("🧹 Cleared GPU memory and configured for CPU.")
            else:
                print("🧹 Cleared GPU memory and configured for CUDA.")
        else:
            print("Running on CPU.")

    def _check_gpu_memory(self, required_memory_gb: float = 8.0) -> bool:
        """Check if GPU has enough memory for the model."""
        if self.device != "cuda":
            return False

        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            free_memory = total_memory - allocated_memory

            print(f"GPU Memory: {total_memory:.2f}GB total, {allocated_memory:.2f}GB allocated, {free_memory:.2f}GB free")

            return free_memory >= required_memory_gb
        except Exception as e:
            print(f"Error checking GPU memory: {e}")
            return False

    def _load_model_on_cpu_if_needed(self, model_path: str, model_type: str = "llm"):
        """Load model on CPU if GPU memory is insufficient."""
        print(f"⚠️  Loading {model_type} on CPU due to GPU memory constraints...")

        try:
            if model_type == "llm":
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    device_map="cpu",
                    trust_remote_code=True
                )
                return LocalGemmaChatModel(model, tokenizer), tokenizer
            else:  # embedding model
                model = SentenceTransformer(model_path, device="cpu")
                return LocalEmbeddings(model)
        except Exception as e:
            print(f"❌ Error loading {model_type} on CPU: {e}")
            traceback.print_exc()
            return None, None if model_type == "llm" else None

    async def load_gemma_llm(self, model_path: str):
        """Load Gemma LLM with memory management."""
        return await asyncio.to_thread(self._load_gemma_llm_sync, model_path)

    def _load_gemma_llm_sync(self, model_path: str):
        try:
            print(f"🧠 Loading Gemma LLM...")

            # Check if we have enough GPU memory for the LLM
            if self.device == "cuda":
                if not self._check_gpu_memory(required_memory_gb=8.0):
                    print("⚠️  Insufficient GPU memory for LLM, falling back to CPU...")
                    return self._load_model_on_cpu_if_needed(model_path, "llm")

            # Try to load on GPU
            if self.device == "cuda":
                print(f"🧠 Loading Gemma LLM on {self.device}...")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=self.torch_dtype,
                    device_map={"": self.device},
                    trust_remote_code=True
                )
                print(f"✅ Gemma LLM loaded successfully on {next(model.parameters()).device}")
                return LocalGemmaChatModel(model, tokenizer), tokenizer
            else:
                # Load on CPU
                return self._load_model_on_cpu_if_needed(model_path, "llm")

        except Exception as e:
            print(f"❌ Error loading Gemma LLM on GPU: {e}")
            traceback.print_exc()

            # Fallback to CPU
            print("🔄 Falling back to CPU for LLM...")
            return self._load_model_on_cpu_if_needed(model_path, "llm")

    async def load_embedding_model(self, model_path: str):
        """Load embedding model with memory management."""
        return await asyncio.to_thread(self._load_embedding_model_sync, model_path)

    def _load_embedding_model_sync(self, model_path: str):
        try:
            print(f"\n🧠 Loading embedding model...")

            # Check if we have enough GPU memory for the embedding model
            if self.device == "cuda":
                if not self._check_gpu_memory(required_memory_gb=4.0):
                    print("⚠️  Insufficient GPU memory for embedding model, falling back to CPU...")
                    return self._load_model_on_cpu_if_needed(model_path, "embedding")

            # Try to load on GPU
            if self.device == "cuda":
                print(f"🧠 Loading embedding model on {self.device}...")
                model = SentenceTransformer(model_path, device=self.device)
                print(f"✅ Embedding model loaded successfully on {self.device}")
                return LocalEmbeddings(model)
            else:
                # Load on CPU
                return self._load_model_on_cpu_if_needed(model_path, "embedding")

        except Exception as e:
            print(f"❌ Error loading embedding model on GPU: {e}")
            traceback.print_exc()

            # Try fallback models
            try:
                print(f"🔄 Attempting to load fallback SentenceTransformer model (all-MiniLM-L6-v2)...")
                fallback_path = "all-MiniLM-L6-v2"

                if self.device == "cuda" and not self._check_gpu_memory(required_memory_gb=2.0):
                    print("⚠️  Insufficient GPU memory for fallback model, using CPU...")
                    model = SentenceTransformer(fallback_path, device="cpu")
                else:
                    model = SentenceTransformer(fallback_path, device=self.device)

                print(f"✅ Loaded fallback SentenceTransformer model on {self.device}")
                return LocalEmbeddings(model)
            except Exception as fallback_e:
                print(f"❌ Failed to load fallback model: {fallback_e}")
                traceback.print_exc()

                # Final fallback: use CPU with a very small model
                try:
                    print("🔄 Final fallback: Loading tiny model on CPU...")
                    model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")
                    print("✅ Loaded tiny fallback model on CPU")
                    return LocalEmbeddings(model)
                except Exception as final_e:
                    print(f"❌ All embedding model loading attempts failed: {final_e}")
                    return None
