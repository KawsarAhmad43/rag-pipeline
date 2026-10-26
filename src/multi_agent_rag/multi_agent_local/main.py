from config import ModelPaths
from models import ModelLoader
from rag import RAGPipeline
from document_tools import DocumentTools
from agents import AgentSystem

def main():
    # Initialize model paths and check availability
    paths = ModelPaths()
    available_models = paths.check_availability()

    if not available_models:
        print("❌ No models available, exiting")
        return

    # Initialize model loader
    loader = ModelLoader()

    # Load Gemma LLM
    llm, tokenizer = loader.load_gemma_llm(available_models.get("Gemma LLM", ""))
    if llm is None:
        print("❌ Local LLM not available, exiting")
        return

    # Load embedding model
    embeddings = loader.load_embedding_model(available_models.get("Colnomic Embed 3B", ""))
    if embeddings is None:
        print("❌ Local embeddings not available, exiting")
        return

    # Initialize RAG pipeline
    rag = RAGPipeline(embeddings, llm)
    rag.load_documents("https://arxiv.org/pdf/2404.19553")
    rag.create_vectorstore()
    rag.create_rag_chain()

    # Initialize document tools
    document_tools = DocumentTools()

    # Initialize agent system
    agent_system = AgentSystem(llm, rag, document_tools)

    # Test the full system
    print("Testing Full Multi-Agent System...")
    agent_system.invoke(
        "Write a LinkedIn post on the paper 'Extending Llama-3's Context Ten-Fold Overnight'. "
        "First consult the research team. Then make sure you consult the LinkedIn team, "
        "and check for copy editing and dopeness, and write the file to disk."
    )

if __name__ == "__main__":
    main()
