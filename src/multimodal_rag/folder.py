# import os

# # Existing base project directory (must already exist)
# base_dir = "multimodal_rag"

# # Substructure inside base_dir
# structure = {
#     base_dir: ["main.py", "config.py", "requirements.txt", "main_embed_and_store.py"],
#     os.path.join(base_dir, "embeddings"): ["embedder.py"],
#     os.path.join(base_dir, "db"): ["chroma_handler.py"],
#     os.path.join(base_dir, "retrieval"): ["retriever.py", "reranker.py"],
#     os.path.join(base_dir, "generation"): ["response_generator.py"],
#     os.path.join(base_dir, "utils"): ["helpers.py"],
#     os.path.join(base_dir, "notebook"): ["demo_pipeline.ipynb"],
#     os.path.join(base_dir, "api"): ["app.py"],
#     os.path.join(base_dir, "web"): ["interface.py"],
# }

# # Check base_dir exists first

# for folder, files in structure.items():
#     os.makedirs(folder, exist_ok=True)
#     for file in files:
#         file_path = os.path.join(folder, file)
#         if not os.path.exists(file_path):
#             open(file_path, "a").close()  # create only if not exists

# print(f"[✓] Subfolders and files added inside ./{base_dir}")
