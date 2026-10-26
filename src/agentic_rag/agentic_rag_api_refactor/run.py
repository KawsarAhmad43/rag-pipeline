#!/usr/bin/env python3
"""
Run script for Agentic RAG API v2.0

This script helps start both the API server and frontend UI.
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

def run_api():
    """Run the FastAPI server"""
    print("🚀 Starting Agentic RAG API...")
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped by user")
    except Exception as e:
        print(f"❌ Error starting API: {e}")

def run_frontend():
    """Run the Streamlit frontend"""
    print("🎨 Starting Streamlit frontend...")
    try:
        subprocess.run(["streamlit", "run", "frontend.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped by user")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")

def run_both():
    """Run both API and frontend in parallel"""
    import threading

    print("🚀 Starting both API and Frontend...")

    # Start API in a separate thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    # Wait a moment for API to start
    time.sleep(3)

    # Start frontend
    run_frontend()

def check_dependencies():
    """Check if required dependencies are available"""
    required = ['fastapi', 'uvicorn', 'streamlit']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("📦 Install with: pip install -r requirements.txt")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description="Run Agentic RAG API v2.0")
    parser.add_argument(
        "--mode",
        choices=["api", "frontend", "both"],
        default="both",
        help="What to run (default: both)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check dependencies only"
    )

    args = parser.parse_args()

    # Change to script directory
    os.chdir(Path(__file__).parent)

    if args.check:
        if check_dependencies():
            print("✅ All dependencies are available")
        return

    if not check_dependencies():
        return

    print(f"🤖 Agentic RAG API v2.0")
    print("=" * 40)

    if args.mode == "api":
        run_api()
    elif args.mode == "frontend":
        run_frontend()
    else:
        run_both()

if __name__ == "__main__":
    main()
