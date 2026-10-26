from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict
import openai
import google.generativeai as genai
from config.settings import settings
from models.api_models import AIProvider

class BaseAIProvider(ABC):
    """Base abstract class for AI providers"""

    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.initialize()

    @abstractmethod
    def initialize(self):
        """Initialize the AI provider"""
        pass

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text response"""
        pass

    @abstractmethod
    def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> str:
        """Generate response with image input"""
        pass

    @abstractmethod
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for texts"""
        pass

    def get_llm_wrapper(self):
        """Get LangChain-compatible wrapper for this provider"""
        from langchain.llms.base import LLM
        from pydantic import Field
        from typing import Any, List, Optional

        class LangChainAIWrapper(LLM):
            """Wrapper to make any AI provider compatible with LangChain"""

            ai_provider: BaseAIProvider = Field(...)

            class Config:
                arbitrary_types_allowed = True

            @property
            def _llm_type(self) -> str:
                return "custom_ai_provider"

            def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
                try:
                    response = self.ai_provider.generate_text(prompt, **kwargs)
                    return response
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    print(f"ERROR: {error_msg}")
                    return error_msg

        return LangChainAIWrapper(ai_provider=self)

class OpenAIProvider(BaseAIProvider):
    """OpenAI provider implementation"""

    def initialize(self):
        """Initialize OpenAI client"""
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI"""
        try:
            temperature = kwargs.get('temperature', settings.DEFAULT_TEMPERATURE)
            max_tokens = kwargs.get('max_tokens', settings.DEFAULT_MAX_TOKENS)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Simple token tracking log (can be extended to a database if needed)
            usage = response.usage
            print(f"📊 [OpenAI] Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")

            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI generation error: {str(e)}")

    def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> str:
        """Generate response with image using OpenAI Vision"""
        try:
            import base64
            from pathlib import Path

            # Read and encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Determine image format
            image_format = Path(image_path).suffix.lower()
            if image_format == '.png':
                mime_type = "image/png"
            elif image_format in ['.jpg', '.jpeg']:
                mime_type = "image/jpeg"
            elif image_format == '.webp':
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"  # Default fallback

            # Use the current GPT-4 vision model
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Updated to current vision model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": "high"  # Use high detail for better analysis
                                }
                            }
                        ]
                    }
                ],
                max_tokens=kwargs.get('max_tokens', settings.DEFAULT_MAX_TOKENS),
                temperature=kwargs.get('temperature', 0.1)  # Lower temperature for more accurate image analysis
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI vision generation error: {str(e)}")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            raise Exception(f"OpenAI embedding error: {str(e)}")

class GeminiProvider(BaseAIProvider):
    """Gemini provider implementation"""

    def initialize(self):
        """Initialize Gemini client"""
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": settings.DEFAULT_TEMPERATURE,
                "max_output_tokens": settings.DEFAULT_MAX_TOKENS,
                "top_p": settings.DEFAULT_TOP_P,
                "top_k": settings.DEFAULT_TOP_K
            }
        )

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using Gemini"""
        try:
            response = self.model.generate_content(prompt)

            # Token usage for Gemini
            try:
                # Gemini usage metadata might vary by SDK version
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    print(f"📊 [Gemini] Prompt: {usage.prompt_token_count}, Candidates: {usage.candidates_token_count}, Total: {usage.total_token_count}")
            except Exception as usage_err:
                 print(f"⚠️ Could not retrieve token usage: {usage_err}")

            return response.text.strip()
        except Exception as e:
            raise Exception(f"Gemini generation error: {str(e)}")

    def generate_with_image(self, prompt: str, image_path: str, **kwargs) -> str:
        """Generate response with image using Gemini"""
        try:
            from pathlib import Path
            import time

            # Determine MIME type based on file extension
            file_ext = Path(image_path).suffix.lower()
            if file_ext == '.pdf':
                mime_type = 'application/pdf'
            elif file_ext == '.png':
                mime_type = 'image/png'
            elif file_ext in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif file_ext == '.webp':
                mime_type = 'image/webp'
            else:
                mime_type = 'image/jpeg'  # Default fallback

            print(f"📸 Uploading image to Gemini with MIME type: {mime_type}")

            # Upload file to Gemini
            uploaded_file = genai.upload_file(path=image_path, mime_type=mime_type)

            # Wait for file to be ready
            print("⏳ Waiting for file to be processed...")
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(1)
                uploaded_file = genai.get_file(uploaded_file.name)

            if uploaded_file.state.name == "FAILED":
                raise Exception("File upload failed")

            print("✅ File uploaded successfully")

            # Create a vision-specific model with adjusted safety settings
            vision_model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",  # Use the more efficient model for free tier
                generation_config={
                    "temperature": 0.1,  # Lower temperature for image analysis
                    "max_output_tokens": settings.DEFAULT_MAX_TOKENS,
                    "top_p": 0.8,
                    "top_k": 5
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
            )

            # Generate content with uploaded file
            content = [uploaded_file, prompt]
            response = vision_model.generate_content(content)

            # Enhanced error handling for Gemini responses
            if not response.parts:
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    raise Exception(f"Content was blocked: {response.prompt_feedback.block_reason}")
                else:
                    raise Exception("No response generated - content may have been filtered")

            # Check if response was blocked
            if hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    if candidate.finish_reason == 2:  # SAFETY
                        raise Exception("Response blocked due to safety concerns. Try rephrasing your question or use a different image.")
                    elif candidate.finish_reason == 3:  # RECITATION
                        raise Exception("Response blocked due to recitation concerns.")
                    elif candidate.finish_reason == 4:  # OTHER
                        raise Exception("Response generation failed due to other reasons.")

            return response.text.strip()

        except Exception as e:
            error_msg = str(e)
            if "finish_reason" in error_msg and "2" in error_msg:
                raise Exception("Image analysis blocked by safety filters. The image may contain sensitive content. Please try a different image or rephrase your question.")
            elif "block" in error_msg.lower():
                raise Exception(f"Content was blocked by Gemini's safety filters: {error_msg}")
            elif "429" in error_msg or "quota" in error_msg.lower():
                raise Exception("🚫 Gemini API quota exceeded. Please try again later or switch to OpenAI for image analysis. The free tier has daily/hourly limits.")
            elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
                raise Exception("🚫 Gemini API rate limit reached. Please wait a moment and try again, or switch to OpenAI.")
            else:
                raise Exception(f"Gemini vision generation error: {error_msg}")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using Gemini (fallback to local model)"""
        # Note: Gemini doesn't have a direct embedding API, so we'd use local models
        # This is a placeholder - in practice, you'd use the local embedding model
        raise NotImplementedError("Gemini embeddings should use local embedding model")

class AIProviderFactory:
    """Factory for creating AI providers"""

    @staticmethod
    def create_provider(provider_type: AIProvider) -> BaseAIProvider:
        """Create AI provider based on type"""
        if provider_type == AIProvider.OPENAI:
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not configured")
            return OpenAIProvider(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.OPENAI_MODEL_NAME
            )
        elif provider_type == AIProvider.GEMINI:
            if not settings.GOOGLE_API_KEY:
                raise ValueError("Google API key not configured")
            return GeminiProvider(
                api_key=settings.GOOGLE_API_KEY,
                model_name=settings.GEMINI_MODEL_NAME
            )
        else:
            raise ValueError(f"Unsupported AI provider: {provider_type}")

    @staticmethod
    def get_available_providers() -> List[AIProvider]:
        """Get list of available providers based on configured API keys"""
        available = []
        if settings.OPENAI_API_KEY:
            available.append(AIProvider.OPENAI)
        if settings.GOOGLE_API_KEY:
            available.append(AIProvider.GEMINI)
        return available
