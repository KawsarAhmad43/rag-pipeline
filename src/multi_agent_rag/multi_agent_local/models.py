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
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
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
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])
        return embedding[0].tolist()

class ModelLoader:
    """Loads LLM and embedding models."""

    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.device = "cpu"
        self.torch_dtype = torch.float32
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("🧹 Cleared GPU memory")

    def load_gemma_llm(self, model_path: str):
        try:
            print("🧠 Loading Gemma LLM...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                device_map={"": self.device},
                trust_remote_code=True
            )
            print(f"✅ Gemma LLM loaded successfully on {next(model.parameters()).device}")
            return LocalGemmaChatModel(model, tokenizer), tokenizer
        except Exception as e:
            print(f"❌ Error loading Gemma LLM: {e}")
            return None, None

    def load_embedding_model(self, model_path: str):
        try:
            print("\n🧠 Loading Colnomic Embedding Model...")
            model = SentenceTransformer(model_path, device=self.device)
            print(f"✅ Colnomic Embedding Model loaded successfully on {self.device}")
            return LocalEmbeddings(model)
        except Exception as e:
            print(f"❌ Error loading Colnomic: {e}")
            try:
                model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
                print("✅ Loaded fallback SentenceTransformer model (all-MiniLM-L6-v2)")
                return LocalEmbeddings(model)
            except Exception as fallback_e:
                print(f"❌ Failed to load fallback model: {fallback_e}")
                return None
