import sys
import os
from app.embedding.embedding import get_embedding_model
from app.llm.llm import get_llm
from agents.agent import get_agent_response, search_documents, create_context_from_docs
from utils.database import db
from utils.pdf_processor import process_pdf

def chat_with_pdf(pdf_path, query):
    try:
        embedding_model = get_embedding_model()
        llm = get_llm()

        if not os.path.exists(pdf_path):
            print(f"Error: PDF file {pdf_path} does not exist")
            return

        with open(pdf_path, "rb") as file:
            docs, embeddings = process_pdf(file)
            if docs:
                for i, doc in enumerate(docs):
                    db.store_document(doc.page_content, embeddings[i])
                print(f"Processed PDF: {len(docs)} chunks stored")

        vector_results, graph_results = search_documents(query, context_type="pdf")
        context = create_context_from_docs(vector_results, graph_results)
        response = get_agent_response(query, context)
        db.store_chat_memory(query, response, embedding_model, context_type="pdf")
        print(f"Query: {query}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pdf_chat.py <pdf_path> <query>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    query = sys.argv[2]
    chat_with_pdf(pdf_path, query)
