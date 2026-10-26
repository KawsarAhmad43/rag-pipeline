import google.generativeai as genai
import logging
from langchain.memory import ConversationBufferWindowMemory
from typing import Optional, Union, Literal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import gc
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Add project root to path for module imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
print(project_root)

from generation.prompt import text_generation_prompt, multimodal_prompt, error_prompt
from PIL import Image
from config import config


os.environ["TRANSFORMERS_VERBOSITY"] = "info"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_gpu_memory():
    """Centralized GPU memory cleanup function."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

class LocalResponseGenerator:
    """Local model response generator using the same model for both text and vision."""

    def __init__(self):

        model_path = str(config.GENERATOR_MODEL_PATH)
        logging.info(f"Loading local generator from {model_path}")
        if not config.GENERATOR_MODEL_PATH.exists():
            raise FileNotFoundError(f"Local generator model not found at: {model_path}")

        if config.TRANSFORMERS_OFFLINE:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if config.HF_HUB_OFFLINE:
            os.environ["HF_HUB_OFFLINE"] = "1"

        clean_gpu_memory()

        try:
            self.device = torch.device("cpu")
            logging.warning("⚠️ GPU unstable (device-side asserts). Falling back to CPU.")

            self.dtype = torch.float32

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                local_files_only=True,
                trust_remote_code=True,
                device_map=self.device
            ).eval()


            logging.info(f"Successfully loaded model. Model class: {type(self.model)}")
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'model_type'):
                logging.info(f"Model config type: {self.model.config.model_type}")


            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
            logging.info(f"Local unified model and processor loaded successfully on {self.device}")
        except Exception as e:
            logging.critical(f"Failed to load local generator: {e}", exc_info=True)
            raise RuntimeError(f"Could not load local generator model: {e}")

    def generate(self, query: str, context: str, history: str = "") -> str:
        if not query:
            return "Please provide a valid query."
        prompt = text_generation_prompt(query, context, history)
        return self._generate_response(prompt)

    def generate_multimodal(self, prompt_text: str, image_path: str) -> str:
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"

        try:
            multimodal_prompt_text = multimodal_prompt(prompt_text)
            return self._generate_response(multimodal_prompt_text, image_path=image_path)
        except Exception as e:
            logging.error(f"Error in local multimodal generation: {e}", exc_info=True)
            return error_prompt(str(e))

    def _generate_response(self, prompt: str, image_path: Optional[str] = None, pdf_text: Optional[str] = None) -> str:
        try:
            if pdf_text:
                prompt = f"Additional Document Context:\n{pdf_text}\n\n{prompt}"

            if image_path:
                prompt = f"<image>\n{prompt}"

            # Handle multimodal inputs
            if image_path and os.path.exists(image_path):
                raw_image = Image.open(image_path)
                # Note: Assuming self.dtype is defined or using default float32 for CPU
                inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt").to(self.device)
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(self.device)

            input_token_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = self.model.generate(

                     **inputs,
                    max_new_tokens=256,
                    do_sample= False,
                    pad_token_id=self.tokenizer.eos_token_id
                )


            response_tokens = outputs[0][input_token_length:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            logging.info(f"Local model generated response: '{response_text[:200]}...'")
            return response_text if response_text else "I'm having trouble generating a response. Could you please rephrase your question?"
        except Exception as e:
            logging.error(f"Error during local model generation: {e}", exc_info=True)
            return f"I encountered an error while processing your request: {str(e)}"

    def cleanup(self):
        logging.info("Cleaning up local generator models...")
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        clean_gpu_memory()


class ApiResponseGenerator:
    def __init__(self):
        from config import config
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is missing.")
        genai.configure(api_key=config.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL_NAME,
            generation_config={"temperature": 0.7, "max_output_tokens": 300}
        )
        logging.info(f"API generator initialized with Gemini model: {config.GEMINI_MODEL_NAME}")

    def generate(self, query: str, context: str, history: str = "") -> str:
        prompt = text_generation_prompt(query, context, history)
        try:
            response = self.model.generate_content(prompt)
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            elif hasattr(response, 'parts') and response.parts:
                return response.parts[0].text.strip()
            else:
                return "I could not find relevant information in the context. Could you rephrase your question?"
        except Exception as e:
            logging.error(f"Error during API model generation: {e}", exc_info=True)
            return error_prompt(str(e))

    def generate_direct_multimodal(self, prompt_text: str, uploaded_genai_file) -> str:
        if not uploaded_genai_file:
            return "No file attachment found."


        multimodal_request = multimodal_prompt(prompt_text)
        content = [uploaded_genai_file, multimodal_request]

        try:
            response = self.model.generate_content(content)

            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content.parts:
                    text_output = candidate.content.parts[0].text.strip()
                    if text_output:
                        return text_output

                # Retry once with enhanced prompt
                logging.warning("Gemini multimodal returned empty response. Retrying with enhanced prompt...")
                retry_prompt = multimodal_prompt(f"{prompt_text} Please give your best possible interpretation.")
                retry_content = [uploaded_genai_file, retry_prompt]
                retry_response = self.model.generate_content(retry_content)
                if hasattr(retry_response, "candidates") and retry_response.candidates:
                    retry_candidate = retry_response.candidates[0]
                    if hasattr(retry_candidate, "content") and retry_candidate.content.parts:
                        retry_text = retry_candidate.content.parts[0].text.strip()
                        if retry_text:
                            return retry_text

            return "⚠️ I reviewed the uploaded file/image but could not extract meaningful details. Could you ask a more specific question about it?"

        except Exception as e:
            logging.error(f"Error during Gemini multimodal generation: {e}", exc_info=True)
            return error_prompt(str(e))


class ResponseGenerator:
    def __init__(self):
        from config import config
        self.api_generator: Optional[ApiResponseGenerator] = None
        self.local_generator: Optional[LocalResponseGenerator] = None
        self.api_available = bool(config.GOOGLE_API_KEY)
        self.local_available = config.GENERATOR_MODEL_PATH.is_dir()
        self.memory_store: dict[str, ConversationBufferWindowMemory] = {}
        logging.info(f"ResponseGenerator initialized. API available: {self.api_available}, Local available: {self.local_available}")

    def get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        if session_id not in self.memory_store:
            self.memory_store[session_id] = ConversationBufferWindowMemory(k=4, return_messages=False)
        return self.memory_store[session_id]

    def generate(self, query: str, context: str, session_id: str, mode: Literal["api", "local"], uploaded_genai_file=None, pdf_text: Optional[str] = None) -> str:
        """Handles both text and image queries in a unified way."""
        memory = self.get_memory(session_id)
        current_history = memory.load_memory_variables({})[memory.memory_key]


        # --- DEBUG: log raw context before it's processed ---
        logging.info("\n===== RAW EXTRACTED CONTEXT START =====")
        if context and context.strip():
            logging.info(context)
        else:
            logging.info("[EMPTY TEXT]")
        logging.info("===== RAW EXTRACTED CONTEXT END =====\n")
        # --- END DEBUG ---

        # If user uploaded an image and query is generic, directly go multimodal
        if uploaded_genai_file and (query.strip().lower() in ["", "hi", "describe the image", "what's in the image"]):
            logging.info("Image detected with generic query — forcing multimodal analysis.")
            return self.generate_direct_multimodal(query or "Describe this image", uploaded_genai_file=uploaded_genai_file, mode=mode, pdf_text=pdf_text)

        if pdf_text:
            context = f"{context}\n\n[ATTACHED PDF CONTENT]:\n{pdf_text}"

        if mode == "api":
            if not self.api_available:
                raise ValueError("API mode requested, but GOOGLE_API_KEY is not configured.")
            if self.api_generator is None:
                self.api_generator = ApiResponseGenerator()
            response_text = self.api_generator.generate(query, context, current_history)
        elif mode == "local":
            if not self.local_available:
                raise ValueError("Local mode requested, but local model path is not configured or found.")
            if self.local_generator is None:
                self.local_generator = LocalResponseGenerator()
                print(query,context, current_history)
            response_text = self.local_generator.generate(query, context, current_history)
        else:
            return "Error: Unsupported generation mode."

        if response_text and response_text.strip():
            memory.save_context({"input": query}, {"output": response_text})

        return response_text

    def generate_direct_multimodal(self, prompt_text: str, uploaded_genai_file=None, image_path=None, mode: Literal["api", "local"] = "api", context: str = "", pdf_text: Optional[str] = None) -> str:
        if mode == "api":
            if not self.api_available:
                return "Error: Multimodal generation requires the API model, which is not available."
            if self.api_generator is None:
                self.api_generator = ApiResponseGenerator()
            return self.api_generator.generate_direct_multimodal(prompt_text, uploaded_genai_file)

        elif mode == "local":
            if not self.local_available:
                return "Error: Local multimodal generation requires the local model, which is not available."
            if self.local_generator is None:
                self.local_generator = LocalResponseGenerator()
            if not image_path:
                return "Error: Local multimodal generation requires an image path."

            prompt = multimodal_prompt(prompt_text)
            return self.local_generator._generate_response(prompt, image_path=str(image_path), pdf_text=pdf_text)

        else:
            return "Error: Unsupported generation mode for multimodal."

    def cleanup(self):
        if self.local_generator:
            self.local_generator.cleanup()
