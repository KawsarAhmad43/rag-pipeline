#!/usr/bin/env python3
"""
Startup script for the Multi-Agent RAG System
Runs both the FastAPI backend and Streamlit frontend
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import streamlit
        import langchain
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  Warning: .env file not found")
        print("Please create a .env file with your API keys:")
        print("OPENAI_API_KEY=your_openai_api_key_here")
        print("TAVILY_API_KEY=your_tavily_api_key_here")
        return False
    else:
        print("✅ .env file found")
        return True

def run_backend():
    """Run the FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend failed to start: {e}")

def run_frontend():
    """Run the Streamlit frontend"""
    print("🚀 Starting Streamlit frontend...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend failed to start: {e}")

def main():
    """Main function"""
    print("🤖 Multi-Agent RAG System Startup")
    print("=" * 40)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check environment
    check_env_file()

    print("\nChoose an option:")
    print("1. Run Backend only (FastAPI)")
    print("2. Run Frontend only (Streamlit)")
    print("3. Run Both (Backend + Frontend)")
    print("4. Exit")

    while True:
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            run_backend()
            break
        elif choice == "2":
            run_frontend()
            break
        elif choice == "3":
            print("\n🔄 Starting both services...")
            print("Backend will be available at: http://localhost:")
            print("Frontend will be available at: http://localhost:8501")
            print("Press Ctrl+C to stop both services")

            try:
                # Start backend in background
                backend_process = subprocess.Popen([sys.executable, "main.py"])

                # Wait a moment for backend to start
                time.sleep(3)

                # Start frontend
                frontend_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend.py"])

                # Wait for either process to finish
                while True:
                    if backend_process.poll() is not None:
                        print("🛑 Backend stopped")
                        frontend_process.terminate()
                        break
                    if frontend_process.poll() is not None:
                        print("🛑 Frontend stopped")
                        backend_process.terminate()
                        break
                    time.sleep(1)

            except KeyboardInterrupt:
                print("\n🛑 Stopping services...")
                backend_process.terminate()
                frontend_process.terminate()
                print("✅ Services stopped")
            break
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()
