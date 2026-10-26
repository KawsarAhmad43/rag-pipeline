import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from typing import List
import re

load_dotenv()  # Loads OPENAI_API_KEY from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Load your documents
def load_text_documents(file_path: str) -> List[Document]:
    # Simple text loader that splits file into paragraphs
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    paragraphs = content.split('\n\n')  # split by blank lines
    docs = [Document(page_content=p.strip()) for p in paragraphs if p.strip()]
    return docs

# 2. Build a vector store using ChromaDB
def build_vector_store(docs: List[Document]):
    # Convert each Document to LangChain's required format
    # We'll embed page_content for searching
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
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
    # 1. Load documents
    file_path = 'data/sample_docs.txt'
    documents = load_text_documents(file_path)

    # 2. Build vector store
    vector_store = build_vector_store(documents)

    # 3. Query
    user_query = "What is the recommended usage of your new product?"

    # 4. Perform hybrid retrieval
    retrieved_docs = hybrid_retrieve(vector_store, documents, user_query, top_k=2)

    # 5. Build prompt
    final_prompt = build_prompt_context(retrieved_docs, user_query)

    # 6. Generate with LLM (OpenAI GPT) using newer pattern
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    prompt_template = PromptTemplate(input_variables=["context"], template="{context}")

    # Use the newer RunnableSequence pattern
    chain = prompt_template | llm

    # Actually run the chain
    answer = chain.invoke({"context": final_prompt})
    print("\n--- Hybrid RAG Answer (OpenAI) ---\n", answer.content)
