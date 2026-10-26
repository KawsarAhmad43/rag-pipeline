import os
import torch
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from typing import List
import re
import gc
from transformers import AutoTokenizer, AutoModel, pipeline
from transformers.utils.import_utils import is_flash_attn_2_available

# --- GLOBAL DEVICE AND DTYPE CONFIGURATION ---
CPU_DEVICE = "cpu"
CPU_DTYPE = torch.float32
GPU_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
GPU_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
GPU_ATTN_IMPL = "flash_attention_2" if torch.cuda.is_available() and is_flash_attn_2_available() else None

if torch.cuda.is_available():
    print(f"Models will load on {GPU_DEVICE} with {GPU_DTYPE}")
else:
    print("CUDA is NOT available. All models will run on CPU.")

# Local model paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EMBEDDING_MODEL_PATH = os.path.join(base_dir, "models", "nomic-embed-multimodal-7b")
GEMMA_MODEL_PATH = os.path.join(base_dir, "models", "gemma-3-4b-it")


class LocalEmbeddings:
    def __init__(self, model_path, device=CPU_DEVICE, torch_dtype=CPU_DTYPE):
        self.device = device
        self.torch_dtype = torch_dtype
        print(f"Loading embedding model from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling for sentence embeddings
                embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())

        return embeddings

    def embed_query(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

class LocalLLM:
    def __init__(self, model_path, device=CPU_DEVICE, torch_dtype=CPU_DTYPE):
        self.device = device
        self.torch_dtype = torch_dtype
        print(f"Loading LLM from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()

        # Create text generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            torch_dtype=torch_dtype,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )

    def generate(self, prompt):
        response = self.pipeline(prompt, max_new_tokens=512, temperature=0.7)
        return response[0]['generated_text'][len(prompt):].strip()

    def cleanup(self):
        del self.model
        del self.tokenizer
        del self.pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 1. Load your documents
def load_text_documents(file_path: str) -> List[Document]:
    # Simple text loader that splits file into paragraphs
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    paragraphs = content.split('\n\n')  # split by blank lines
    docs = [Document(page_content=p.strip()) for p in paragraphs if p.strip()]
    return docs

# 2. Build a vector store using ChromaDB with local embeddings
def build_vector_store(docs: List[Document], embeddings_model):
    # Convert each Document to LangChain's required format
    # We'll embed page_content for searching using local embeddings
    texts = [doc.page_content for doc in docs]
    embeddings = embeddings_model.embed_documents(texts)

    # Create ChromaDB with local embeddings
    vector_store = Chroma.from_embeddings(
        embeddings=embeddings,
        documents=docs,
        persist_directory="./chroma_db_local"
    )
    return vector_store

# 3. Keyword-based retrieval
def keyword_search(docs: List[Document], query: str, top_k: int = 2) -> List[Document]:
    """
    Very naive approach: returns docs containing the most occurrences of the query's keywords.
    """
    # Extract keywords (e.g., splitting by whitespace or a more advanced approach)
    keywords = query.lower().split()
    scored_docs = []
    for doc in docs:
        score = 0
        # Count how many keywords appear
        for kw in keywords:
            # naive count
            score += len(re.findall(rf'\b{re.escape(kw)}\b', doc.page_content.lower()))
        if score > 0:
            scored_docs.append((doc, score))
    # Sort by descending score
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]

# 4. Combine the results
def hybrid_retrieve(vector_store, all_docs: List[Document], query: str, top_k=2) -> List[Document]:
    # Vector similarity
    vector_hits = vector_store.similarity_search(query, k=top_k)
    # Keyword-based
    keyword_hits = keyword_search(all_docs, query, top_k=top_k)

    # Merge them
    combined = []
    # Use a set to avoid duplicates
    seen_contents = set()
    for doc in vector_hits + keyword_hits:
        if doc.page_content not in seen_contents:
            combined.append(doc)
            seen_contents.add(doc.page_content)
    return combined

def build_prompt_context(docs: List[Document], query: str) -> str:
    # Build a prompt with the top docs
    context = '\n\n'.join([doc.page_content for doc in docs])
    prompt = f"""You are a helpful AI assistant. Below is the user question and the relevant context.

Context:
{context}

Question: {query}

Provide an answer based on the context. If you are not sure, say you don't know."""
    return prompt

if __name__ == "__main__":
    print("�� Initializing Local Hybrid RAG System...")

    # Check if models exist
    if not os.path.exists(EMBEDDING_MODEL_PATH):
        print(f"❌ Embedding model not found at: {EMBEDDING_MODEL_PATH}")
        print("Please ensure the colnomic-embed-multimodal-3b model is downloaded.")
        exit(1)

    if not os.path.exists(GEMMA_MODEL_PATH):
        print(f"❌ LLM model not found at: {GEMMA_MODEL_PATH}")
        print("Please ensure the gemma-3-4b-it model is downloaded.")
        exit(1)

    try:
        # Initialize local models
        print("📥 Loading embedding model...")
        embeddings_model = LocalEmbeddings(EMBEDDING_MODEL_PATH, device=CPU_DEVICE, torch_dtype=CPU_DTYPE)

        print("📥 Loading LLM...")
        llm = LocalLLM(GEMMA_MODEL_PATH, device=CPU_DEVICE, torch_dtype=CPU_DTYPE)

        # 1. Load documents
        file_path = 'data/sample_docs.txt'
        documents = load_text_documents(file_path)
        print(f"✅ Loaded {len(documents)} documents")

        # 2. Build vector store
        print("🔍 Building vector store...")
        vector_store = build_vector_store(documents, embeddings_model)
        print("✅ Vector store built")

        # 3. Query
        user_query = "What is the recommended usage of your new product?"

        # 4. Perform hybrid retrieval
        print("🔍 Performing hybrid retrieval...")
        retrieved_docs = hybrid_retrieve(vector_store, documents, user_query, top_k=2)
        print(f"✅ Retrieved {len(retrieved_docs)} relevant documents")

        # 5. Build prompt
        final_prompt = build_prompt_context(retrieved_docs, user_query)

        # 6. Generate with local LLM
        print("🤖 Generating answer with local LLM...")
        answer = llm.generate(final_prompt)
        print("\n--- Hybrid RAG Answer (Local Models) ---\n", answer)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check your model paths and ensure all dependencies are installed.")

    finally:
        print("\n🧹 Cleaning up...")
        if 'llm' in locals():
            llm.cleanup()
        if 'embeddings_model' in locals():
            del embeddings_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ Cleanup complete.")
