from app.embedding.embedding import get_embedding_model
from app.llm.llm import get_llm
from agents.agent import get_agent_response  # Updated import
from langchain.memory import ConversationBufferMemory
from utils.database import db
from utils.pdf_processor import process_pdf
from utils.vector_search import vector_search

llm = get_llm()
embedding_model = get_embedding_model()
memory = ConversationBufferMemory()

def chat_with_pdf(query, pdf_content):
    memory_response = db.search_chat_memory(query, embedding_model)
    if memory_response:
        return memory_response

    docs, embeddings = process_pdf(pdf_content)
    if docs:
        db.store_document(docs[0].page_content, embedding_model)

    response = get_agent_response(query)  # Use agent for response
    db.store_chat_memory(query, response)
    return response
