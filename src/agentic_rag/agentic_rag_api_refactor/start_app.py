#!/usr/bin/env python3
"""
Startup script for Agentic RAG API with Speech Features

This script provides an easy way to start both the API server and the frontend.
"""

import os
import sys
import time
import subprocess
import threading
import signal
import requests
from pathlib import Path

# Configuration
API_HOST = "localhost"
API_PORT = 8001
FRONTEND_PORT = 8501

def print_status(message, status="INFO"):
    """Print a formatted status message"""
    colors = {
        "INFO": "\033[94m",    # Blue
        "SUCCESS": "\033[92m", # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
        "RESET": "\033[0m"     # Reset
    }
    print(f"{colors.get(status, colors['INFO'])}[{status}]{colors['RESET']} {message}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "streamlit",
        "requests",
        "torch",
        "transformers",
        "soundfile"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print_status(f"Missing packages: {', '.join(missing_packages)}", "ERROR")
        print_status("Please install missing packages with: pip install -r requirements.txt", "INFO")
        return False

    print_status("✅ All required packages are installed", "SUCCESS")
    return True

def check_speech_models():
    """Check if speech models are available"""
    try:
        # Check if Whisper model can be loaded
        from transformers import pipeline
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=-1)
        print_status("✅ Whisper model available", "SUCCESS")
    except Exception as e:
        print_status(f"⚠️ Whisper model not available: {e}", "WARNING")

    try:
        # Check if Kokoro is available
        import kokoro_onnx
        print_status("✅ Kokoro model available", "SUCCESS")
    except ImportError:
        print_status("⚠️ Kokoro model not available - text-to-speech may not work", "WARNING")

def wait_for_api(host, port, timeout=30):
    """Wait for the API to be ready"""
    print_status(f"Waiting for API to be ready at {host}:{port}...", "INFO")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=2)
            if response.status_code == 200:
                print_status("✅ API is ready!", "SUCCESS")
                return True
        except:
            pass
        time.sleep(1)

    print_status("❌ API failed to start within timeout", "ERROR")
    return False

def start_api_server():
    """Start the FastAPI server"""
    print_status("Starting API server...", "INFO")

    try:
        # Start the API server
        process = subprocess.Popen([
            sys.executable, "main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for API to be ready
        if wait_for_api(API_HOST, API_PORT):
            return process
        else:
            process.terminate()
            return None

    except Exception as e:
        print_status(f"❌ Failed to start API server: {e}", "ERROR")
        return None

def start_frontend():
    """Start the Streamlit frontend"""
    print_status("Starting frontend...", "INFO")

    try:
        # Start Streamlit frontend
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "frontend.py",
            "--server.port", str(FRONTEND_PORT),
            "--server.address", "localhost"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait a bit for frontend to start
        time.sleep(3)

        print_status(f"✅ Frontend started at http://localhost:{FRONTEND_PORT}", "SUCCESS")
        return process

    except Exception as e:
        print_status(f"❌ Failed to start frontend: {e}", "ERROR")
        return None

def cleanup(api_process, frontend_process):
    """Clean up processes on exit"""
    print_status("\n🛑 Shutting down...", "INFO")

    if frontend_process:
        frontend_process.terminate()
        print_status("Frontend stopped", "INFO")

    if api_process:
        api_process.terminate()
        print_status("API server stopped", "INFO")

def main():
    """Main startup function"""
    print_status("🚀 Starting Agentic RAG API with Speech Features", "INFO")
    print_status("=" * 60, "INFO")

    # Check dependencies
    if not check_dependencies():
        return False

    # Check speech models
    check_speech_models()

    print_status("=" * 60, "INFO")

    # Start API server
    api_process = start_api_server()
    if not api_process:
        return False

    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        api_process.terminate()
        return False

    print_status("=" * 60, "INFO")
    print_status("🎉 Application started successfully!", "SUCCESS")
    print_status(f"📡 API: http://localhost:{API_PORT}", "INFO")
    print_status(f"🌐 Frontend: http://localhost:{FRONTEND_PORT}", "INFO")
    print_status(f"📚 API Docs: http://localhost:{API_PORT}/docs", "INFO")
    print_status("=" * 60, "INFO")
    print_status("Press Ctrl+C to stop the application", "INFO")

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        cleanup(api_process, frontend_process)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)

            # Check if processes are still running
            if api_process.poll() is not None:
                print_status("❌ API server stopped unexpectedly", "ERROR")
                break

            if frontend_process.poll() is not None:
                print_status("❌ Frontend stopped unexpectedly", "ERROR")
                break

    except KeyboardInterrupt:
        print_status("\n🛑 Received interrupt signal", "INFO")
    finally:
        cleanup(api_process, frontend_process)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
