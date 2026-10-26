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



# folder containing the current file
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(PROJECT_ROOT)
print(PROJECT_ROOT)



from generation.prompt import text_generation_prompt, multimodal_prompt, error_prompt
from speech.TTS import tts_say, load_engine, VOICE_NAMES  # uses your cached local Kokoro



# from prompt import text_generation_prompt, multimodal_prompt, error_prompt
from PIL import Image
from config import config

# ---- TTS: local Kokoro engine ----
# PATHS (reference only, as requested)
PATHS = {
    # "kokoro_onnx": "/home/aiteam/Desktop/rag_pipelines/models/kokoro/kokoro-v1.0.int8.onnx",
    # "kokoro_voices": "/home/aiteam/Desktop/rag_pipelines/models/kokoro/voices-v1.0.bin",
    # "wav_out": "/home/aiteam/Desktop/rag_pipelines/src/multimodal_rag/voice_notebook/output.wav"
    "kokoro_onnx": "/home/aiteam/Desktop/all_downloaded_models/kokoro/kokoro-v1.0.int8.onnx",
    "kokoro_voices": "/home/aiteam/Desktop/all_downloaded_models/kokoro/voices-v1.0.bin",
    "wav_out": "/home/aiteam/Desktop/rag_pipelines/src/multimodal_rag/voice_notebook/output.wav"
}


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
            # force_cpu can be set via env var
            # FORCE CPU to fix "device-side assert" error
            force_cpu = True # os.getenv("FORCE_CPU", "0") == "1"

            # GPU Auto-Detection
            if torch.cuda.is_available() and not force_cpu:
                self.device = torch.device("cuda")
                self.dtype = torch.float16  # Faster on GPU
                logging.info("🚀 CUDA detected. Switching to GPU mode with float16 precision.")
            else:
                self.device = torch.device("cpu")
                self.dtype = torch.float32  # Standard for CPU
                if force_cpu:
                    logging.info("💾 FORCE_CPU=1 detected. Using CPU mode.")
                else:
                    logging.info("💻 No CUDA detected. Using CPU mode with float32 precision.")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                local_files_only=True,
                trust_remote_code=True,
                device_map="auto" if (torch.cuda.is_available() and not force_cpu) else None
            )

            if not torch.cuda.is_available() or force_cpu:
                self.model = self.model.to(self.device)

            self.model.eval()

            logging.info(f"Successfully loaded model. Model class: {type(self.model)}")
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'model_type'):
                logging.info(f"Model config type: {self.model.config.model_type}")

            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
            logging.info(f"Local unified model and processor loaded successfully on {self.device}")
        except Exception as e:
            logging.critical(f"Failed to load local generator: {e}", exc_info=True)
            raise RuntimeError(f"Could not load local generator model: {e}")

    def generate(self, query: str, context: str, history: str = "", for_speech: bool = False) -> str:
        if not query:
            return "Please provide a valid query."
        prompt = text_generation_prompt(query, context, history, for_speech=for_speech)
        return self._generate_response(prompt)

    def generate_multimodal(self, prompt_text: str, image_path: str, context: str = "", for_speech: bool = False) -> str:
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"
        try:
             # Combine context with prompt if available
            full_prompt = prompt_text
            if context:
                full_prompt = f"Context from Knowledge Base:\n{context}\n\nUser Question: {prompt_text}"

            multimodal_prompt_text = multimodal_prompt(full_prompt, for_speech=for_speech)
            # Re-implementing multimodal support via _generate_response
            return self._generate_response(multimodal_prompt_text, image_path=image_path)
        except Exception as e:
            logging.error(f"Error in local multimodal generation: {e}", exc_info=True)
            return error_prompt(str(e))

    def _generate_response(self, prompt: str, image_path: Optional[str] = None, pdf_text: Optional[str] = None) -> str:
        try:
            # Combine PDF text with prompt if available (for local context)
            if pdf_text:
                prompt = f"Additional Document Context:\n{pdf_text}\n\n{prompt}"

            # Prepend <image> tag if image is provided (Gemma 3 vision requirement)
            if image_path:
                prompt = f"<image>\n{prompt}"

            # Gemma 3 Instruct Template: <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
            # We must ensure <image> is present for the processor to replace it with image features.
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # If apply_chat_template didn't include our <image> tag correctly (though it should be in content),
                # we double check. For Gemma-3, it usually works if <image> is in the string.
            except Exception:
                formatted_prompt = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

            logging.info(f"Final formatted prompt (length {len(formatted_prompt)}): {formatted_prompt[:200]}...")

            # Handle multimodal inputs if image exists
            if image_path and os.path.exists(image_path):
                raw_image = Image.open(image_path)
                inputs = self.processor(text=formatted_prompt, images=raw_image, return_tensors="pt").to(self.device, self.dtype)
            else:
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt", max_length=2048, truncation=True).to(self.device)

            input_token_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512, # Increased for better quality
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
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

    def generate(self, query: str, context: str, history: str = "", for_speech: bool = False) -> str:
        prompt = text_generation_prompt(query, context, history, for_speech=for_speech)
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

    def generate_direct_multimodal(
        self,
        prompt_text: str,
        uploaded_genai_file,
        context: str = "",
        for_speech: bool = False
    ) -> str:
        if not uploaded_genai_file:
            return "No file attachment found."

        # Combine context with prompt if available
        full_prompt = prompt_text
        if context:
            full_prompt = f"Context from Knowledge Base:\n{context}\n\nUser Question: {prompt_text}"

        multimodal_request = multimodal_prompt(full_prompt, for_speech=for_speech)
        content = [uploaded_genai_file, multimodal_request]

        try:
            response = self.model.generate_content(content)
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content.parts:
                    text_output = candidate.content.parts[0].text.strip()
                    if text_output:
                        return text_output

                logging.warning("Gemini multimodal returned empty response. Retrying with enhanced prompt...")
                retry_prompt = multimodal_prompt(f"{full_prompt} Please give your best possible interpretation.", for_speech=for_speech)
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

    def _maybe_tts(self, text: str, voice: Optional[str], audio_out_path: Optional[str]) -> Optional[Path]:
        """
        If voice is provided, synthesize with Kokoro and return the WAV path.
        If audio_out_path is given, write there; otherwise use engine cache path.
        """
        if not text or voice is None:
            return None
        if voice not in VOICE_NAMES:
            raise ValueError(f"Voice '{voice}' not in supported VOICE_NAMES: {VOICE_NAMES}")
        eng = load_engine()
        if audio_out_path:
            out = Path(audio_out_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            return eng.synthesize_to_wav(text, out_path=out, voice=voice, use_cache=True)
        return eng.synthesize_to_wav(text, voice=voice, use_cache=True)

    def generate(
        self,
        query: str,
        context: str,
        session_id: str,
        mode: Literal["api", "local"],
        uploaded_genai_file=None,
        *,
        speak: bool = False,                 # NEW: enable Kokoro TTS
        voice: Optional[str] = None,         # NEW: pick from VOICE_NAMES; None = no audio
        audio_out_path: Optional[str] = None, # NEW: optional fixed output path
        pdf_text: Optional[str] = None       # NEW: optional extracted text from attachment
    ) -> Union[str, dict]:
        """Handles both text and image queries in a unified way. Returns string by default; dict if speak=True."""
        memory = self.get_memory(session_id)
        current_history = memory.load_memory_variables({})[memory.memory_key]

        logging.info("\n===== RAW EXTRACTED CONTEXT START =====")
        logging.info(context if (context and context.strip()) else "[EMPTY TEXT]")
        logging.info("===== RAW EXTRACTED CONTEXT END =====\n")

        # If user uploaded a file/image, use multimodal flow to handle it correctly
        if uploaded_genai_file:
            logging.info("File/Image attachment detected — using multimodal analysis flow.")
            return self.generate_direct_multimodal(
                query,
                uploaded_genai_file=uploaded_genai_file,
                mode=mode,
                context=context,
                speak=speak,
                voice=voice,
                audio_out_path=audio_out_path,
                pdf_text=pdf_text
            )

        # Merge pdf_text into context for text RAG flow
        if pdf_text:
            context = f"{context}\n\n[ATTACHED PDF CONTENT]:\n{pdf_text}"

        # Generate text
        if mode == "api":
            if not self.api_available:
                raise ValueError("API mode requested, but GOOGLE_API_KEY is not configured.")
            if self.api_generator is None:
                self.api_generator = ApiResponseGenerator()
            response_text = self.api_generator.generate(query, context, current_history, for_speech=speak)
        elif mode == "local":
            if not self.local_available:
                raise ValueError("Local mode requested, but local model path is not configured or found.")
            if self.local_generator is None:
                self.local_generator = LocalResponseGenerator()
            response_text = self.local_generator.generate(query, context, current_history, for_speech=speak)
        else:
            return "Error: Unsupported generation mode."

        if response_text and response_text.strip():
            memory.save_context({"input": query}, {"output": response_text})

        if not speak:
            return response_text

        # TTS
        wav_path = self._maybe_tts(response_text, voice=voice, audio_out_path=audio_out_path)
        return {
            "text": response_text,
            "audio_path": str(wav_path) if wav_path else None,
            "voice_used": voice if voice else None
        }

    def generate_direct_multimodal(
        self,
        prompt_text: str,
        uploaded_genai_file=None,
        image_path=None,
        mode: Literal["api", "local"] = "api",
        context: str = "",
        *,
        speak: bool = False,
        voice: Optional[str] = None,
        audio_out_path: Optional[str] = None,
        pdf_text: Optional[str] = None
    ) -> Union[str, dict]:
        # Produce text
        if mode == "api":
            if not self.api_available:
                return "Error: Multimodal generation requires the API model, which is not available."
            if self.api_generator is None:
                self.api_generator = ApiResponseGenerator()
            # Pass pdf_text context to API multimodal call
            text = self.api_generator.generate_direct_multimodal(prompt_text, uploaded_genai_file, context=f"{context}\n{pdf_text}" if pdf_text else context, for_speech=speak)
        elif mode == "local":
            if not self.local_available:
                return "Error: Local multimodal generation requires the local model, which is not available."
            if self.local_generator is None:
                self.local_generator = LocalResponseGenerator()

            # If image_path is None, try to use uploaded_genai_file (which might be the local path)
            actual_image_path = image_path or (uploaded_genai_file if isinstance(uploaded_genai_file, (str, Path)) else None)

            if not actual_image_path:
                return "Error: Local multimodal generation requires an image path."

            # Combine context with prompt for local multimodal
            full_multimodal_prompt = prompt_text
            if context:
                full_multimodal_prompt = f"Context from Knowledge Base:\n{context}\n\nUser Question: {prompt_text}"

            # Wrap with multimodal prompt template
            prompt_with_template = multimodal_prompt(full_multimodal_prompt, for_speech=speak)
            text = self.local_generator._generate_response(prompt_with_template, image_path=str(actual_image_path), pdf_text=pdf_text)
        else:
            return "Error: Unsupported generation mode for multimodal."

        if not speak:
            return text

        # TTS
        wav_path = self._maybe_tts(text, voice=voice, audio_out_path=audio_out_path)
        return {
            "text": text,
            "audio_path": str(wav_path) if wav_path else None,
            "voice_used": voice if voice else None
        }

    def cleanup(self):
        if self.local_generator:
            self.local_generator.cleanup()
