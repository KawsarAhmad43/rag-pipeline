from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter

class RAGPipeline:
    """Handles the Retrieval-Augmented Generation pipeline."""

    def __init__(self, embeddings, llm):
        self.embeddings = embeddings
        self.llm = llm
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None

    def load_documents(self, pdf_url: str):
        try:
            docs = PyMuPDFLoader(pdf_url).load()
            print(f"Loaded {len(docs)} documents")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=0,
                length_function=lambda text: len(tiktoken.encoding_for_model("gpt-4o-mini").encode(text))
            )
            self.split_chunks = text_splitter.split_documents(docs)
            print(f"Split into {len(self.split_chunks)} chunks")
        except Exception as e:
            print(f"Failed to load PDF: {e}")
            self.split_chunks = []

    def create_vectorstore(self):
        if self.embeddings and self.split_chunks:
            try:
                test_embedding = self.embeddings.embed_documents(["Test sentence"])
                print(f"✅ Embedding test successful: {len(test_embedding)} embeddings")

                self.vectorstore = Qdrant.from_documents(
                    self.split_chunks,
                    self.embeddings,
                    location=":memory:",
                    collection_name="extending_context_window_llama_3"
                )
                self.retriever = self.vectorstore.as_retriever()
                print("✅ Vector store created with local embeddings")
            except Exception as e:
                print(f"❌ Failed to create vector store: {e}")
                self.retriever = None

    def create_rag_chain(self):
        if self.llm and self.retriever:
            rag_prompt = ChatPromptTemplate.from_template(
                """
                CONTEXT:
                {context}

                QUERY:
                {question}

                You are a helpful assistant. Use the available context to answer the question. If you can't answer the question, say you don't know.
                """
            )
            self.rag_chain = (
                {"context": itemgetter("question") | self.retriever, "question": itemgetter("question")}
                | rag_prompt | self.llm | StrOutputParser()
            )
            print("✅ RAG chain created with local models")
        else:
            print("❌ Local models not available, RAG chain not created")

    def invoke(self, question: str):
        if self.rag_chain:
            try:
                result = self.rag_chain.invoke({"question": question})
                print("RAG Result:")
                print(result)
                return result
            except Exception as e:
                print(f"Error testing RAG chain: {e}")
                return None
        return None
