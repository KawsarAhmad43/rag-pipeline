#!/usr/bin/env python3
"""
Startup script for the Local Multi-Agent RAG System
Runs both the FastAPI backend and Streamlit frontend
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_api():
    """Run the FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "api:app",
            "--host", "0.0.0.0",
            "--port", "8004",
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except Exception as e:
        print(f"❌ Error starting API: {e}")

def run_frontend():
    """Run the Streamlit frontend"""
    print("🎨 Starting Streamlit frontend...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit",
            "run", "frontend.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")

def run_both():
    """Run both API and frontend in separate processes"""
    print("🤖 Starting Local Multi-Agent RAG System...")
    print("📋 System will be available at:")
    print("   • Frontend: http://localhost:8501")
    print("   • API: http://localhost:8004")
    print("   • API Docs: http://localhost:8004/docs")
    print()

    try:
        # Start API in background
        api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "api:app",
            "--host", "0.0.0.0",
            "--port", "8004"
        ])

        # Wait a moment for API to start
        time.sleep(3)

        # Start frontend
        frontend_process = subprocess.Popen([
            sys.executable, "-m", "streamlit",
            "run", "frontend.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])

        print("✅ Both servers started successfully!")
        print("Press Ctrl+C to stop both servers")

        # Wait for either process to finish
        try:
            api_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping servers...")
            api_process.terminate()
            frontend_process.terminate()
            api_process.wait()
            frontend_process.wait()
            print("✅ Servers stopped")

    except Exception as e:
        print(f"❌ Error: {e}")
        if 'api_process' in locals():
            api_process.terminate()
        if 'frontend_process' in locals():
            frontend_process.terminate()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "api":
            run_api()
        elif command == "frontend":
            run_frontend()
        elif command == "both":
            run_both()
        else:
            print("Usage: python run_system.py [api|frontend|both]")
            print("  api      - Run only the FastAPI backend")
            print("  frontend - Run only the Streamlit frontend")
            print("  both     - Run both servers (default)")
    else:
        run_both()
