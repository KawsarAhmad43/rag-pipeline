# src/orchestrator/adapters/http_adapter.py
import httpx
import logging
from typing import Any, Dict, Optional
from .base import BaseAdapter

logger = logging.getLogger(__name__)

class HttpAdapter(BaseAdapter):
    def __init__(self, pipeline_id: str, base_url: str, display_name: str = None, capabilities: dict = None):
        super().__init__(pipeline_id, display_name, capabilities)
        self.base_url = base_url.rstrip("/")

    async def ingest(self, session_id: str, file_paths: list[str], options: dict) -> Dict[str, Any]:
        # Different pipelines use different endpoints and parameter names
        if self.pipeline_id == "agentic_rag":
            url = f"{self.base_url}/api/v1/upload_pdf"
            file_param_name = "file"
        elif self.pipeline_id == "simple_rag":
            url = f"{self.base_url}/simple/upload-pdf/"
            file_param_name = "file"
        elif self.pipeline_id == "graph_rag":
            url = f"{self.base_url}/graph/upload-pdf/"
            file_param_name = "file"
        else:
            # Default for multimodal_rag and others
            url = f"{self.base_url}/initialize_session"
            file_param_name = "pdf_file"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                import os
                data = {"session_id": session_id}
                files = []

                # Properly read file content and send with filename
                for p in file_paths:
                    filename = os.path.basename(p)
                    # Determine content type based on extension
                    if filename.lower().endswith('.pdf'):
                        content_type = 'application/pdf'
                    elif filename.lower().endswith('.txt'):
                        content_type = 'text/plain'
                    else:
                        content_type = 'application/octet-stream'

                    # Read file content
                    with open(p, "rb") as f:
                        file_content = f.read()

                    # Append with correct parameter name per pipeline
                    files.append((file_param_name, (filename, file_content, content_type)))

                logger.info(f"HttpAdapter({self.pipeline_id}): Ingesting {len(file_paths)} files to {url}")
                res = await client.post(url, data=data, files=files)
                res.raise_for_status()
                return res.json()
        except Exception as e:
            logger.exception(f"HttpAdapter({self.pipeline_id}): ingest failed")
            return {"status": "error", "error": str(e)}

    async def chat(self, session_id: str, message: str, options: dict) -> Dict[str, Any]:
        # Determine URL based on pipeline
        if self.pipeline_id == "agentic_rag":
            url = f"{self.base_url}/api/v1/chat"
        elif self.pipeline_id == "simple_rag":
            url = f"{self.base_url}/simple/chat/"
        elif self.pipeline_id == "graph_rag":
            url = f"{self.base_url}/graph/chat/"
        else:
            url = f"{self.base_url}/chat"
        # # OLD LOGIC (Form Data Only)
        # try:
        #     async with httpx.AsyncClient(timeout=120.0) as client:
        #         # MultiModal RAG /chat expects Form data: user_input, session_id, generation_mode
        #         data = {
        #             "user_input": message,
        #             "session_id": session_id,
        #             "generation_mode": options.get("generation_mode", "api")
        #         }
        #
        #         logger.debug(f"HttpAdapter({self.pipeline_id}): Sending chat to {url}")
        #         res = await client.post(url, data=data)
        #         res.raise_for_status()
        #         result = res.json()
        #
        #         # Normalize response for Orchestrator
        #         return {
        #             "answer": result.get("response", ""),
        #             "sources": result.get("sources", []),
        #             "debug": {"backend_url": url, "session_id": session_id}
        #         }
        # except Exception as e:
        #     logger.exception(f"HttpAdapter({self.pipeline_id}): chat failed")
        #     return {
        #         "answer": f"(HTTP error) {str(e)}",
        #         "sources": [],
        #         "debug": {"error": str(e)}
        #     }

        # NEW LOGIC (Generalized JSON with Form Fallback)
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                if self.pipeline_id == "multimodal_rag":
                    # Multimodal still uses Form Data for unified support of attachments
                    data = {
                        "user_input": message,
                        "session_id": session_id,
                        "generation_mode": options.get("generation_mode", "api")
                    }
                    if options.get("attachment"):
                        # Handling attachments would go here if passed as byte stream
                        pass
                    res = await client.post(url, data=data)
                else:
                    if self.pipeline_id == "agentic_rag":
                        json_payload = {
                            "session_id": session_id,
                            "user_input": message,
                            "options": options
                        }
                    elif self.pipeline_id == "simple_rag":
                        json_payload = {
                            "text": message,
                            "pdf_context": options.get("pdf_context", False),
                            "pdf_analyzer": options.get("pdf_analyzer", False),
                            "web_search": options.get("web_search", False)
                        }
                        logger.info(f"🔍 Sending to Simple RAG - message: '{message[:50]}...', pdf_context: {options.get('pdf_context', False)}, pdf_analyzer: {options.get('pdf_analyzer', False)}")
                    elif self.pipeline_id == "graph_rag":
                        json_payload = {
                            "prompt": message
                        }
                    else:
                        json_payload = {
                            "session_id": session_id,
                            "message": message,
                            "options": options
                        }
                    res = await client.post(url, json=json_payload)

                res.raise_for_status()
                result = res.json()

                # Normalize response keys
                answer = result.get("answer") or result.get("response") or result.get("text", "")
                sources = result.get("sources") or result.get("context", [])

                return {
                    "answer": answer,
                    "sources": sources,
                    "debug": {"backend_url": url, "session_id": session_id, "raw_response": result}
                }
        except Exception as e:
            logger.exception(f"HttpAdapter({self.pipeline_id}): chat failed")
            return {
                "answer": f"(HTTP error) {str(e)}",
                "sources": [],
                "debug": {"error": str(e)}
            }

    async def reset_session(self, session_id: str) -> Dict[str, Any]:
        """Reset/clear session data for a specific RAG pipeline"""
        # Determine reset URL based on pipeline
        if self.pipeline_id == "agentic_rag":
            url = f"{self.base_url}/api/v1/clear_session/{session_id}"
        elif self.pipeline_id == "simple_rag":
            # Simple RAG doesn't have a dedicated reset endpoint, so we'll return success
            # as it doesn't maintain persistent sessions
            return {"status": "success", "message": "Simple RAG uses stateless processing"}
        elif self.pipeline_id == "multimodal_rag":
            url = f"{self.base_url}/reset_session/{session_id}"
        elif self.pipeline_id == "graph_rag":
            url = f"{self.base_url}/graph/reset_session/{session_id}"
        else:
            return {"status": "skipped", "message": f"No reset endpoint defined for {self.pipeline_id}"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.info(f"HttpAdapter({self.pipeline_id}): Resetting session {session_id} at {url}")
                res = await client.delete(url)
                res.raise_for_status()
                return {"status": "success", "response": res.json() if res.text else {}}
        except Exception as e:
            logger.exception(f"HttpAdapter({self.pipeline_id}): reset_session failed")
            return {"status": "error", "error": str(e)}
