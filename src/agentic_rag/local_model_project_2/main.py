import os
import torch
from chatbot import Chatbot
from langchain.memory import ConversationBufferWindowMemory
from pdf_handler import PDFHandler
from reranker import Reranker
from vector_db import VectorDB
import gc
from transformers.utils.import_utils import is_flash_attn_2_available

def main():
    # --- GLOBAL DEVICE AND DTYPE CONFIGURATION ---
    CPU_DEVICE = "cpu"
    CPU_DTYPE = torch.float32
    GPU_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    GPU_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    GPU_ATTN_IMPL = "flash_attention_2" if torch.cuda.is_available() and is_flash_attn_2_available() else None
    if torch.cuda.is_available():
        print(f"Image-to-text model will load on {GPU_DEVICE} with {GPU_DTYPE} and attn_implementation: {GPU_ATTN_IMPL}")
        print(f"Text-to-text model (Gemma) will load on {CPU_DEVICE} with {CPU_DTYPE}")
    else:
        print("CUDA is NOT available. All models including Gemma and vision pipeline will run on CPU.")
    # ---------------------------------------------

    print("\n🤖 Initializing components...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    pdf_path = os.path.join(base_dir, "data", "company_profile.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    pdf_handler = PDFHandler(pdf_path)
    chunks = pdf_handler.get_chunks()
    print("✅ PDF Loading Done. Total chunks:", len(chunks))

    # embedding_model_path = os.path.join(base_dir, "models", "colnomic-embed-multimodal-3b")
    embedding_model_path = "/home/aiteam/Desktop/all_downloaded_models/colnomic-embed-multimodal-3b"
    vector_db = VectorDB(
        embedding_model_path,
        persist_directory=os.path.join(base_dir, "chroma_db"),
        device=CPU_DEVICE,
        torch_dtype=CPU_DTYPE
    )
    vector_db.build_index(chunks)
    print("✅ Vector Indexing Done.")

    # gemma_model_path = os.path.join(base_dir, "models", "gemma-3-4b-it")
    # reranker_model_path = os.path.join(base_dir, "models", "bge-reranker-v2-m3")
    gemma_model_path = "/home/aiteam/Desktop/all_downloaded_models/gemma-3-4b-it"
    reranker_model_path = "/home/aiteam/Desktop/all_downloaded_models/bge-reranker-v2-m3"
    reranker = Reranker(reranker_model_path, device=CPU_DEVICE)
    memory = ConversationBufferWindowMemory(k=5)  # Limit to last 5 exchanges

    chatbot = Chatbot(
        generation_model_path=gemma_model_path,
        vector_db=vector_db,
        reranker=reranker,
        memory=memory,
        gemma_device=CPU_DEVICE,  # Text-to-text on CPU
        gemma_torch_dtype=CPU_DTYPE,
        vision_device=GPU_DEVICE,  # Image-to-text on GPU
        vision_torch_dtype=GPU_DTYPE,
        vision_attn_implementation=GPU_ATTN_IMPL
    )

    print("\n✅ Chatbot initialized successfully!")
    print("🤖 Simple AI Chatbot Ready!")
    print("Type 'exit' to quit\n")

    try:
        while True:
            user_input = input("You (type 'exit', or use 'img:<path>' for image): ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("🤖 Goodbye!")
                break
            if user_input.lower().startswith("img:"):
                image_path = user_input[4:].strip()
                if not os.path.exists(image_path):
                    print(f"❌ Image file not found: {image_path}")
                else:
                    result = chatbot.describe_image(image_path)
                    print(f"🖼️ Description: {result}")
                continue

            try:
                response = chatbot.get_response(user_input)
                print(f"🤖 Bot: {response}")
            except Exception as e:
                print(f"🤖 Bot: Sorry, I had an error: {str(e)}. Can you try again?")

    finally:
        print("\nExiting application. Final cleanup.")
        if 'chatbot' in locals():
            chatbot.cleanup()
            del chatbot
        if 'vector_db' in locals():
            vector_db.cleanup()
            del vector_db
        if 'reranker' in locals():
            reranker.cleanup()
            del reranker
        if 'memory' in locals():
            del memory
        if 'pdf_handler' in locals():
            del pdf_handler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Application cleanup complete.")

if __name__ == "__main__":
    main()
