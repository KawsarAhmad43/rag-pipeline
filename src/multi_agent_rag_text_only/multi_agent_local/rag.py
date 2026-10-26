import asyncio
import os
import tiktoken
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

class RAGPipeline:
    def __init__(self, embeddings, llm):
        self.embeddings = embeddings
        self.llm = llm
        self.split_chunks = []
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None

    async def load_documents(self, pdf_path: str):
        try:
            print(f"📁 Loading documents from: {pdf_path}")
            # PyMuPDFLoader().load() is synchronous, run in a thread pool
            if pdf_path.startswith(('http://', 'https://')):
                # URL loading can be network-bound, also use to_thread
                docs = await asyncio.to_thread(PyMuPDFLoader(pdf_path).load)
            else:
                if not os.path.exists(pdf_path):
                    raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                docs = await asyncio.to_thread(PyMuPDFLoader(pdf_path).load)

            print(f"📄 Loaded {len(docs)} documents")

            # Debug: Print actual document content
            for i, doc in enumerate(docs):
                print(f"📄 Document {i+1} content (first 300 chars): {doc.page_content[:300]}...")

            # Text splitting is CPU-bound, run in a thread pool if it's large
            def _split_documents_sync(documents):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=300,
                    chunk_overlap=0,
                    length_function=lambda text: len(tiktoken.encoding_for_model("gpt-4o-mini").encode(text))
                )
                return text_splitter.split_documents(documents)

            self.split_chunks = await asyncio.to_thread(_split_documents_sync, docs)
            print(f"✂️ Split into {len(self.split_chunks)} chunks")

            # Debug: Print chunk content
            for i, chunk in enumerate(self.split_chunks):
                print(f"📝 Chunk {i+1}: {chunk.page_content[:200]}...")

        except Exception as e:
            print(f"❌ Failed to load PDF: {e}")
            self.split_chunks = []
            import traceback
            traceback.print_exc()

    async def create_vectorstore(self):
        try:
            if not self.split_chunks:
                print("❌ No chunks available for vectorstore creation")
                return False

            # Test embedding creation first
            test_text = ["test embedding"]
            test_embeddings = await asyncio.to_thread(self.embeddings.embed_documents, test_text)
            print(f"✅ Embedding test successful: {len(test_embeddings)} embeddings")

            # Create vector store
            self.vectorstore = await asyncio.to_thread(
                Qdrant.from_documents,
                self.split_chunks,
                self.embeddings,
                location=":memory:",
                collection_name="local_rag"
            )

            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )

            print("✅ Vector store created with local embeddings")
            return True

        except Exception as e:
            print(f"❌ Failed to create vectorstore: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_rag_chain(self):
        try:
            if not self.retriever:
                print("❌ Retriever not available")
                return False

            # Create the RAG prompt template (matching working API version)
            rag_prompt = ChatPromptTemplate.from_template(
                """CONTEXT:
{context}

QUERY:
{question}

You are a helpful assistant. Use the available context to answer the question. If you can't answer the question based on the context, say you don't know.
"""
            )

            # Create the RAG chain (matching working API version structure)
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            self.rag_chain = (
                {
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | rag_prompt
                | self.llm
                | StrOutputParser()
            )

            print("✅ RAG chain created with local models")
            return True

        except Exception as e:
            print(f"❌ Failed to create RAG chain: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def invoke(self, question: str):
        if not self.rag_chain:
            print("❌ RAG chain not initialized")
            return "RAG chain not initialized. Please ensure models are loaded and document is processed."

        if not self.retriever:
            print("❌ Retriever not initialized")
            return "Document retriever not initialized. Please upload a document first."

        try:
            print(f"🔍 Invoking RAG chain with question: {question}")

            # Debug: Check what documents are retrieved
            print("🔍 Debugging retriever...")
            retrieved_docs = await asyncio.to_thread(self.retriever.invoke, question)
            print(f"📄 Retrieved {len(retrieved_docs)} documents:")
            for i, doc in enumerate(retrieved_docs):
                print(f"  Document {i+1}: {doc.page_content[:200]}...")

            # Now invoke the full RAG chain
            result = await self.rag_chain.ainvoke(question)
            print(f"🎯 RAG Result: {result}")

            if result is None:
                return "No response generated from the RAG system."

            return result
        except Exception as e:
            print(f"❌ Error in RAG chain: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing question: {str(e)}"
