import os
import shutil
from typing import Optional
from fastapi import HTTPException, UploadFile

from config.settings import settings
from models.api_models import UploadResponse, AIProvider
from core.session_manager import SessionManager
from utils.validation import validate_uuid

class FileService:
    """Service for handling file operations"""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    async def upload_pdf(
        self,
        file: UploadFile,
        session_id: str,
        ai_provider: AIProvider = AIProvider.GEMINI
    ) -> UploadResponse:
        """Upload and process PDF file"""
        print(f"🔵 Agentic RAG: upload_pdf called with session_id={session_id}, filename={file.filename}")

        # Validate session ID
        if not validate_uuid(session_id):
            raise HTTPException(status_code=400, detail="Invalid session ID")

        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        try:
            # Ensure data directory exists
            os.makedirs(settings.DATA_DIR, exist_ok=True)
            print(f"🔵 Agentic RAG: DATA_DIR = {settings.DATA_DIR}")

            # Save uploaded file using await file.read() for better reliability
            pdf_path = os.path.join(settings.DATA_DIR, f"{session_id}_{file.filename}")
            print(f"🔵 Agentic RAG: Will save file to: {pdf_path}")
            content = await file.read()
            print(f"🔵 Agentic RAG: Read {len(content)} bytes from upload")
            with open(pdf_path, "wb") as f:
                f.write(content)
            print(f"✅ Agentic RAG: Saved uploaded file to {pdf_path} ({len(content)} bytes)")
            print(f"✅ Agentic RAG: File exists check: {os.path.exists(pdf_path)}")

            # Clear existing session if it exists (to replace simple with document-based)
            if self.session_manager.has_session(session_id):
                self.session_manager.clear_session(session_id)
                print(f"🔄 Replacing existing session with document-based session for {session_id}")

            # Create new document-based session
            result = self.session_manager.create_session(
                session_id=session_id,
                ai_provider=ai_provider,
                pdf_file_path=pdf_path
            )

            if not result["success"]:
                # Clean up file if session creation failed
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                raise HTTPException(
                    status_code=500,
                    detail="Failed to initialize chatbot with uploaded PDF"
                )

            chunk_count = result.get("chunk_count", 0)

            return UploadResponse(
                success=True,
                message=f"PDF uploaded and processed successfully. Indexed {chunk_count} chunks. You can now ask questions about the document!",
                pdf_path=pdf_path,
                session_id=session_id,
                file_name=file.filename,
                chunk_count=chunk_count
            )

        except HTTPException:
            raise
        except Exception as e:
            # Clean up file on error
            try:
                if 'pdf_path' in locals() and os.path.exists(pdf_path):
                    os.remove(pdf_path)
            except:
                pass

            print(f"Error uploading PDF: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error uploading PDF: {str(e)}"
            )
        finally:
            # Ensure file handle is closed
            if hasattr(file, 'file'):
                try:
                    file.file.close()
                except:
                    pass

    async def save_temporary_file(
        self,
        file: UploadFile,
        session_id: str
    ) -> str:
        """Save temporary file for processing"""
        # Validate session ID
        if not validate_uuid(session_id):
            raise HTTPException(status_code=400, detail="Invalid session ID")

        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.pdf'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail="Only JPG, JPEG, PNG, or PDF files are allowed"
            )

        try:
            # Ensure data directory exists
            os.makedirs(settings.DATA_DIR, exist_ok=True)

            # Save temporary file
            file_path = os.path.join(settings.DATA_DIR, f"{session_id}_{file.filename}")
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            return file_path

        except Exception as e:
            print(f"Error saving temporary file: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error saving file: {str(e)}"
            )
        finally:
            # Ensure file handle is closed
            if hasattr(file, 'file'):
                try:
                    file.file.close()
                except:
                    pass

    def cleanup_temporary_file(self, file_path: str):
        """Clean up temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            print(f"Error cleaning up temporary file {file_path}: {str(e)}")

    def get_file_info(self, file_path: str) -> dict:
        """Get information about a file"""
        try:
            if not os.path.exists(file_path):
                return {"exists": False}

            stat = os.stat(file_path)
            return {
                "exists": True,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "name": os.path.basename(file_path)
            }
        except Exception as e:
            return {"exists": False, "error": str(e)}
