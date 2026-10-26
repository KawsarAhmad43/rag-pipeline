#!/usr/bin/env python3
"""
Agentic RAG API - Refactored Version

A clean, modular RAG API supporting both OpenAI and Gemini providers.
"""

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from api.routes import router, session_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    print("🚀 Starting Agentic RAG API...")

    # Validate configuration
    try:
        settings.validate_config()
        print("✅ Configuration validated")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        raise

    # Pre-load models to prevent 504 timeouts
    try:
        from spech_to_text import get_stt_model
        from text_to_spech import get_tts_model

        print("🧠 Pre-loading models...")
        # Pre-load STT (Whisper)
        get_stt_model()
        print("✅ STT model loaded")

        # Pre-load TTS (Kokoro)
        get_tts_model()
        print("✅ TTS model loaded")

    except Exception as e:
        print(f"⚠️ Error pre-loading models: {e}")
        # We don't raise here to allow the API to start even if local models have issues
        # (e.g. if the user only wants to use OpenAI/Gemini)

    yield

    # Shutdown - cleanup
    print("🛑 Shutting down Agentic RAG API...")
    session_manager.cleanup_all()
    print("✅ Cleanup completed")

# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "api_title": settings.API_TITLE
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Agentic RAG API v2.0",
        "docs": "/docs",
        "health": "/health",
        "api_prefix": "/api/v1"
    }

if __name__ == "__main__":
    print(f"Starting {settings.API_TITLE} on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False  # Set to True for development
    )
