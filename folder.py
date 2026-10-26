import os

# Existing implemented files under src/
existing_src_files = [
    "src/app/api.py",
    "src/app/ui.py",
    "src/ingestion/loader.py",
    "src/ingestion/splitter.py",
    "src/embeddings/qwen2_embedder.py",
    "src/embeddings/visual_embedder.py",
    "src/embeddings/embed_utils.py",
    "src/store/chroma_handler.py",
    "src/rerank/reranker.py",
    "src/llm/gemma_response.py"
]

# Top-level project files

def create_structure():
    base_dir = "rag_pipelines"



    # Create src structure and preserve existing files
    for file in existing_src_files:
        path = os.path.join(base_dir, file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(f"# {os.path.basename(file).replace('.py', '').capitalize()} module\n")

    print("✅ Full folder structure created and existing files preserved!")

if __name__ == "__main__":
    create_structure()
