# Graph RAG Chatbot

A multimodal Retrieval-Augmented Generation (RAG) application that integrates vector search and graph-based knowledge retrieval to answer queries, with a focus on processing PDF documents. Built with FastAPI, ChromaDB, Neo4j, and various language models, it supports natural language queries, PDF processing, and chat history management.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [Credits](#credits)
- [License](#license)

## Project Overview
The Graph RAG Chatbot enables users to query information from PDF documents using a hybrid approach combining vector-based search (ChromaDB) and graph-based knowledge representation (Neo4j). It supports multiple language models for natural language processing and entity extraction, making it suitable for applications like diabetes detection research. The system stores chat history for context-aware responses and provides a user-friendly web interface for interaction.

## Features
- **PDF Processing**: Extracts text, generates embeddings, and builds knowledge graphs from uploaded PDFs.
- **Hybrid Search**: Combines vector search (ChromaDB) and graph-based search (Neo4j) for precise answers.
- **Chat Memory**: Persists and retrieves chat history for contextual conversations.
- **Reranking**: Enhances result relevance using a reranker model.
- **Model Flexibility**: Supports local models (Gemma, Llama) and cloud-based models (OpenAI, Gemini).
- **Web Interface**: Responsive UI with Tailwind CSS for chat and PDF uploads.
- **Health Monitoring**: API health checks and detailed logging.
- **Chat Memory Management**: Clear all chat memory via API.

## Tech Stack
- **Backend**: FastAPI (Python)
- **Databases**:
  - Vector DB: ChromaDB
  - Graph DB: Neo4j
- **Language Models**:
  - Local: Gemma, Llama (Hugging Face Transformers)
  - Cloud: OpenAI GPT-3.5, Google Gemini
- **Embedding Model**: Hugging Face Colnomic Embed Multimodal
- **Reranker**: BGE Reranker V2
- **Entity Extraction**: spaCy
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Containerization**: Docker (Neo4j)

## Prerequisites
- Python 3.8+
- Docker (for Neo4j)
- CUDA-enabled GPU (optional, for local models)
- API keys for OpenAI, Google Gemini, or Hugging Face (if using cloud models)
- spaCy model: `en_core_web_sm`

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/graph-rag-chatbot.git
   cd graph-rag-chatbot
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   Running on GPU server PC(if virtual environment exists)
   ```bash
      conda activate graph
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install spaCy Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set Up Environment Variables**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   HUGGING_FACE_API_KEY=your_huggingface_api_key
   NEO4J_URI=bolt://127.0.0.1:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=graphragpassword
   NEO4J_DATABASE=neo4j
   GRAPH_RAG_LOG_PATH=/path/to/logs
   GRAPH_RAG_VECTOR_DB_PATH=/path/to/vector_db
   GRAPH_RAG_CHAT_MEMORY_PATH=/path/to/chatmemory
   GRAPH_RAG_LLM_MODEL_NAME=/path/to/models/gemma-3-4b-it
   GRAPH_RAG_EMBEDDING_MODEL_NAME=/path/to/models/colnomic-embed-multimodal-3b
   GRAPH_RERANKER_MODEL_NAME=/path/to/models/bge-reranker-v2-m3
   ```

6. **Start Neo4j**:
   ```bash
   docker-compose up -d
   ```

7. **Run the Application**:
   ```bash
   python run.py
   ```

   or

    ```bash
   uvicorn app.api.main:app --host 0.0.0.0 --port 2000 --reload --loop asyncio
   ```
   The FastAPI server will be available at `http://localhost:2000`.

## Configuration
- **Neo4j**: Configured via `docker-compose.yml` and `config.py`. Default port: 7687.
- **Logging**: Logs are stored at `GRAPH_RAG_LOG_PATH`.
- **Models**: Update `config.py` with paths for local models or API keys for cloud models.
- **Vector DB**: ChromaDB persists data at `GRAPH_RAG_VECTOR_DB_PATH`.

## Usage

0. **Swagger API**:
   - Visit `http://localhost:2000/graph/docs` for the swagger api documentation.


1. **Web Interface**:
   - Visit `http://localhost:2000/graph/` for the chat interface.
   - Upload PDFs at `http://localhost:2000/graph/upload/`.

2. **Command-Line PDF Chat**:
   ```bash
   python tools/pdf_chat.py /path/to/pdf "Your query here"
   ```

3. **API Interaction**:
   - **Query**: `POST /graph/chat/`
     ```json
     {"text": "What is diabetes?", "pdf_context": true}
     ```
   - **Upload PDF**: `POST /graph/upload-pdf/` (multipart/form-data with PDF file)
   - **Clear Chat Memory**: `DELETE /graph/chat-memory/`
   - **View Chat History**: `GET /graph/chat-memory/`
   - **Health Check**: `GET /health`

## API Endpoints
- **GET /health**: Check server and model status.
- **GET /graph/**: Serve the chat interface.
- **GET /graph/upload/**: Serve the PDF upload page.
- **POST /graph/upload-pdf/**: Upload and process PDFs.
- **POST /graph/chat/**: Query with text and optional PDF context.
- **GET /graph/chat-memory/**: Retrieve all chat history.
- **DELETE /graph/chat-memory/**: Clear all chat memory.
- **GET /docs**: Swagger UI for API documentation.

## Directory Structure
```
graph-rag-chatbot/
├── app/
│   ├── api/
│   │   ├── main.py         # FastAPI application
│   │   └── __init__.py
│   ├── embedding/
│   │   ├── embedding.py    # Embedding model logic
│   │   └── __init__.py
│   ├── llm/
│   │   ├── llm.py         # LLM initialization
│   │   ├── llm_openai.py  # OpenAI LLM
│   │   ├── llm_gemini.py  # Gemini LLM
│   │   ├── llm_local_gemma.py  # Local Gemma LLM
│   │   ├── llm_local_llama.py  # Local Llama LLM
│   │   └── __init__.py
│   ├── reranker/
│   │   ├── reranker.py    # Reranker model logic
│   │   └── __init__.py
│   ├── graph/
│   │   ├── graph.py       # Neo4j graph operations
│   │   └── __init__.py
│   └── frontend/
│       └── static/
│           ├── index.html      # Chat interface
│           └── pdf_upload.html # PDF upload page
├── utils/
│   ├── database.py        # ChromaDB operations
│   ├── pdf_processor.py   # PDF text extraction
│   ├── vector_search.py   # Vector search utilities
│   ├── graph_processor.py # Graph extraction logic
│   ├── logging.py         # Logging configuration
│   └── __init__.py
├── tools/
│   ├── pdf_chat.py        # Command-line PDF query tool
│   └── __init__.py
├── config.py              # Configuration management
├── run.py                 # Application entry point
├── docker-compose.yml     # Neo4j service configuration
├── .env                   # Environment variables
└── README.md              # Project documentation
```

## Contributing
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m "Add feature"`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.

## Credits
Special thanks to the **MW AI Teams** for their invaluable contributions to the development and testing of this project. Their expertise in AI, data processing, and system architecture was instrumental in bringing this chatbot to life.

#### Corresponding Engineer:
 - Mushfiqur Rahman
 - Kawsar Ahmad


## License
MIT License. See `LICENSE` file for details.
