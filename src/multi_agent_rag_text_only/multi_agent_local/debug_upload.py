#!/usr/bin/env python3
"""
Debug script for testing document upload functionality
"""

import requests
import tempfile
import os
from pathlib import Path

def test_api_health():
    """Test API health"""
    try:
        response = requests.get("http://localhost:8004/health")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_models():
    """Test model availability"""
    try:
        response = requests.get("http://localhost:8004/test-models")
        print(f"Model test: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Model test failed: {e}")
        return False

def test_session_creation():
    """Test session creation"""
    try:
        response = requests.post("http://localhost:8004/create-session")
        print(f"Session creation: {response.status_code}")
        if response.status_code == 200:
            session_data = response.json()
            print(f"Session ID: {session_data['session_id']}")
            return session_data['session_id']
        else:
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Session creation failed: {e}")
        return None

def create_test_pdf():
    """Create a simple test PDF"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_path = temp_file.name
        temp_file.close()

        c = canvas.Canvas(temp_path, pagesize=letter)
        c.drawString(100, 750, "Test Document")
        c.drawString(100, 700, "This is a test PDF for debugging.")
        c.save()

        return temp_path
    except ImportError:
        print("reportlab not available, creating a dummy file")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_path = temp_file.name
        temp_file.write(b"%PDF-1.4\nTest PDF content\n%%EOF")
        temp_file.close()
        return temp_path

def test_upload(session_id):
    """Test document upload"""
    try:
        # Create test PDF
        pdf_path = create_test_pdf()
        print(f"Created test PDF: {pdf_path}")

        # Upload file
        with open(pdf_path, 'rb') as f:
            files = {'file': ('test.pdf', f, 'application/pdf')}
            data = {'session_id': session_id}

            print(f"Uploading with session_id: {session_id}")
            response = requests.post("http://localhost:8004/upload-document", files=files, data=data)

        # Clean up test file
        os.unlink(pdf_path)

        print(f"Upload response: {response.status_code}")
        print(f"Response text: {response.text}")

        return response.status_code == 200
    except Exception as e:
        print(f"Upload test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔍 Debugging Upload Issue")
    print("=" * 40)

    # Test 1: API Health
    print("\n1. Testing API Health...")
    if not test_api_health():
        print("❌ API is not running")
        return

    # Test 2: Model Availability
    print("\n2. Testing Model Availability...")
    test_models()

    # Test 3: Session Creation
    print("\n3. Testing Session Creation...")
    session_id = test_session_creation()
    if not session_id:
        print("❌ Session creation failed")
        return

    # Test 4: Document Upload
    print("\n4. Testing Document Upload...")
    if test_upload(session_id):
        print("✅ Upload successful!")
    else:
        print("❌ Upload failed")

    print("\n" + "=" * 40)
    print("Debug complete!")

if __name__ == "__main__":
    main()
