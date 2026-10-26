# Simple RAG System

The Simple RAG (Retrieval-Augmented Generation) System is a web-based application designed to enable users to interact with a conversational AI that leverages both general knowledge and context from uploaded PDF documents. Built with a modern tech stack, it combines a FastAPI backend for efficient API handling and a sleek, responsive HTML frontend powered by Tailwind CSS. The system supports document retrieval, chat functionality, and memory of previous interactions, making it ideal for knowledge extraction and conversational tasks.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Models Used](#models-used)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [Entrypoints](#entrypoints)
- [API Endpoints](#api-endpoints)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

## Overview
The Simple RAG System integrates retrieval-augmented generation to provide context-aware responses. Users can upload PDF documents, which are processed into chunks, embedded using an embedding model, and stored in a vector database (ChromaDB). Queries can be answered with or without PDF context, and the system maintains a chat history for quick reference. The frontend, served as static HTML files, offers a pixel-perfect UI with glassmorphism effects, responsive design, and real-time system status updates.

## Features
- **Conversational AI**:
  - Users can submit queries via a chat interface.
  - Supports general knowledge queries and context-aware responses using uploaded PDFs.
  - Option to toggle PDF context for queries via a checkbox.
- **PDF Document Processing**:
  - Upload PDF files through a dedicated interface.
  - PDFs are split into chunks, embedded, and stored in a vector database for retrieval.
  - Supports efficient document retrieval for context-aware responses.
- **Chat Memory**:
  - Stores previous queries and responses in a vector database.
  - Displays recent chats in a sidebar for quick access.
  - Allows users to revisit and view past conversations.
- **System Status Monitoring**:
  - Real-time status updates in the UI:
    - **System Standby** (green): System is healthy and ready.
    - **System Initializing** (orange): System is loading models or initializing.
    - **System Down** (red): System encountered an error.
- **Responsive UI**:
  - Pixel-perfect design with Tailwind CSS and glassmorphism effects.
  - Sidebar navigation for New Chat, History, and PDF Upload.
  - Mobile-friendly with toggleable sidebar and adaptive layouts.
- **Detailed Logging**:
  - Logs server startup, API calls, model initialization, and errors to both terminal and files (`logs/run.log`, `logs/simple_rag.log`).
  - Facilitates debugging and monitoring.
- **Scalable Backend**:
  - Built with FastAPI for high-performance API serving.
  - Supports CORS for flexible client-side interactions.
  - Handles model initialization and GPU memory management.

## Models Used
The Simple RAG System leverages the following models for its functionality:
- **Embedding Model** (`app/embedding/embedding.py`):
  - **Model**: Any model can be used here we have available Hugging Face's `colnomic-embed-multimodal-3b` model.
  - **Purpose**: Converts text (PDF chunks and queries) into vector embeddings for storage and retrieval in ChromaDB.
  - **Details**: Lightweight, fast, and optimized for semantic similarity tasks.
- **Reranker Model** (`app/reranker/reranker.py`):
  - **Model**: Hugging Face's `bge-reranker-v2-m3`.
  - **Purpose**: Ranks retrieved documents based on relevance to the query for improved context accuracy.
  - **Details**: Enhances retrieval precision by scoring document-query pairs.
- **Language Model (LLM)** (`app/llm/*.py`):
  - **Options**:
    - **OpenAI GPT** (`llm_openai.py`): Uses OpenAI's GPT models (e.g., `gpt-3.5-turbo`) for cloud-based inference.
    - **Google Gemini** (`llm_gemini.py`): Uses Google's Gemini models (e.g., `gemini-2.5-flash`) for alternative cloud-based inference.
    - **Local Gemma** (`llm_local_gemma.py`): Uses Hugging Face's `gemma-3-4b-it` for on-premises inference.
    - **Local LLaMA** (`llm_local_llama.py`): Uses Meta AI's LLaMA models (e.g., `Llama-3.2-1B`) for on-premises inference.
  - **Purpose**: Generates human-like responses based on query and retrieved context.
  - **Details**: Configurable via `config.py` to select the desired LLM provider (default: OpenAI if API keys are set).
- **Vector Database**:
  - **Tool**: ChromaDB (`utils/database.py`).
  - **Purpose**: Stores document embeddings and chat memory for efficient retrieval.
  - **Details**: In-memory or persistent storage, configurable via `config.py`.

## Project Structure
```
simple_rag/
├── agents/
│   ├── __init__.py
│   └── agent.py
│
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── embedding/
│   │   └── embedding.py
│   ├── frontend/
│   │   └── static/
│   └── llm/
│       ├── llm_gemini.py
│       ├── llm_local_gemma.py
│       ├── llm_local_llama.py
│       ├── llm_openai.py
│       └── llm.py
│
├── reranker/
│   ├── __init__.py
│   └── reranker.py
│
├── data/
│   ├── chatmemory/
│   ├── logs/
│   ├── uploads/
│   └── vector_db/
│
├── tools/
│   ├── __init__.py
│   └── pdf_chat.py
│
├── utils/
│   ├── __init__.py
│   ├── database.py
│   ├── logging.py
│   ├── pdf_processor.py
│   ├── vector_search.py
│   └── codetotext.py
│
├── config.py
├── run.py
└── README.md

```

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd simple_rag
   ```

2. **Install Dependencies**:
   - Install using requirements.txt
   ```bash
   pip install -r requirements.txt
   ```
   - Install using Astral UV
   ```bash
   uv sync
   ```


3. **Configure Environment**:
   Create a `.env` file in the root directory with API keys (if using cloud-based LLMs):
   ```
   OPENAI_API_KEY=your-openai-api-key
   GOOGLE_API_KEY=your-google-api-key
   ```


4. **Prepare Storage**:
   Ensure the `data/uploads` directory exists for PDF storage:
   ```bash
   mkdir -p data/uploads
   ```

## Running the Project
There are two ways to start the Simple RAG System:

### Option 1: Using Uvicorn Directly
Run the FastAPI server directly with Uvicorn:
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 9000
```
- **Pros**: Simple, lightweight, ideal for development.
- **Cons**: No automatic port conflict resolution or GPU cleanup.

### Option 2: Using `run.py` Script
Run the provided `run.py` script, which includes port checking, GPU cleanup, and detailed logging:
```bash
python run.py
```
- **Pros**: Handles port conflicts, cleans GPU memory, logs to `logs/run.log`.
- **Cons**: Slightly more overhead due to additional checks.

**Expected Output**:
- The server starts on `http://localhost:9000`.
- Logs are written to `logs/run.log` and printed to the terminal, e.g.:
  ```
  2025-07-31 17:29:00,123 - INFO - Starting application...
  2025-07-31 17:29:00,125 - INFO - Checking if port 9000 is available for FastAPI...
  2025-07-31 17:29:00,126 - INFO - Starting FastAPI server...
  2025-07-31 17:29:03,456 - INFO - FastAPI server started successfully on http://localhost:9000
  ```

## Entrypoints
- **Frontend**:
  - **URL**: `http://localhost:9000/`
  - **Description**: Loads the chat interface (`index.html`) with a sidebar for navigation, recent chats, and a chat input form. Supports real-time status updates and PDF context toggling.
  - **Navigation**:
    - **Chat**: `http://localhost:9000/` (main chat interface).
    - **PDF Upload**: `http://localhost:9000/static/pdf_upload.html` (PDF upload interface).
- **Backend**:
  - **URL**: `http://localhost:9000/docs#/`
  - **Description**: Swagger UI for API documentation, listing all endpoints and allowing interactive testing.

## API Endpoints
The backend exposes the following endpoints under the `/simple/` prefix:
- **GET /health**:
  - Checks system health, including model initialization.
  - Response: `{"status": "healthy"}` or `{"status": "unhealthy", "error": "..."}`.
- **POST /simple/upload-pdf/**:
  - Uploads a PDF file, processes it into chunks, and stores embeddings in ChromaDB.
  - Request: Form-data with `file` (PDF file).
  - Response: `{"message": "PDF processed and stored successfully - X chunks extracted"}`.
- **POST /simple/chat/**:
  - Processes a user query, optionally using PDF context.
  - Request: JSON body with `text` (query string) and `pdf_context` (boolean).
  - Response: `{"response": "...", "context": "pdf|not_pdf|memory|error"}`.
- **GET /simple/chat-memory/**:
  - Retrieves all chat history entries.
  - Response: `{"chat_memory": [{"id": "...", "query": "...", "response": "...", "context_type": "..."}], "message": "..."}`.

Test endpoints using `curl` or Postman:
```bash
curl http://localhost:9000/health
curl -X POST http://localhost:9000/simple/chat/ -H "Content-Type: application/json" -d '{"text": "What is Teflon?", "pdf_context": false}'
curl -X POST http://localhost:9000/simple/upload-pdf/ -F "file=@/path/to/sample.pdf"
curl http://localhost:9000/simple/chat-memory/
```

## Logging
- **Log Files**:
  - `logs/run.log`: Server startup, port checks, and runtime errors.
  - `logs/simple_rag.log`: Application-specific logs (API calls, model initialization, PDF processing).
- **Log Format**:
  ```
  2025-07-31 17:29:05,789 - INFO - Serving index.html at root
  2025-07-31 17:29:06,123 - INFO - Health check requested
  2025-07-31 17:29:10,123 - INFO - Processing query: 'What is Teflon?'...
  2025-07-31 17:29:12,123 - INFO - Uploaded and stored PDF: sample.pdf, 5 chunks
  ```
- **Purpose**: Facilitates debugging, performance monitoring, and error tracking.

## Troubleshooting
- **Server Fails to Start**:
  - **Issue**: Port 9000 in use.
  - **Solution**: Check `logs/run.log`. If using `run.py`, it attempts to free the port. Alternatively, change the port in `run.py` or `uvicorn` command (e.g., `--port 9001`) and update `API_URL` in `index.html` and `pdf_upload.html`.
- **UI Not Loading**:
  - **Issue**: `index.html` or `pdf_upload.html` not found.
  - **Solution**: Verify files exist in `app/frontend/static/`. Check browser console (F12) for 404 errors. Ensure `app.mount("/static", ...)` is in `main.py`.
- **API Errors**:
  - **Issue**: `/health` returns `unhealthy` or endpoints fail.
  - **Solution**: Check `logs/simple_rag.log` for model initialization errors. Ensure API keys in `.env` are valid and models are downloaded (for local LLMs).
- **Chat Memory Empty**:
  - **Issue**: Recent chats not populating.
  - **Solution**: Verify `/simple/chat-memory/` returns data. Check ChromaDB configuration in `config.py`.
- **PDF Upload Fails**:
  - **Issue**: Uploads fail or no chunks extracted.
  - **Solution**: Ensure `data/uploads` directory exists. Check `logs/simple_rag.log` for PDF processing errors.
## Author: MD. Kawsar Ahnad
## Credit:  AI Team of Mediusware
 - Mushfiqur Rahman
 - Zihad Bin Zahangir
 - Monisha
 - Kawsar Ahmad
