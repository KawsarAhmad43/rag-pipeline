import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np
import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LocalSTT:
    """Local Speech-to-Text using Whisper-tiny."""
    def __init__(self, model_path: str):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Whisper model not found at {self.model_path}")

        logger.info(f"Loading local Whisper model from {self.model_path}")

        # force_cpu can be set via env var
        force_cpu = os.getenv("FORCE_CPU", "0") == "1"

        # GPU Auto-Detection
        if torch.cuda.is_available() and not force_cpu:
            self.device = torch.device("cuda")
            self.torch_dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            self.torch_dtype = torch.float32

        self.processor = WhisperProcessor.from_pretrained(
            str(self.model_path),
            local_files_only=True
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(self.model_path),
            torch_dtype=self.torch_dtype,
            local_files_only=True
        ).to(self.device).eval()

        logger.info(f"Whisper model loaded on {self.device}")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text."""
        try:
            # Load and resample audio to 16kHz (Whisper requirement)
            audio, sr = librosa.load(audio_path, sr=16000)

            input_features = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device).to(self.torch_dtype)

            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)

            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            logger.info(f"Transcription complete: '{transcription.strip()[:100]}...'")
            return transcription.strip()
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            return f"Error: Could not transcribe audio. {str(e)}"

# Singleton helper
_stt_instance = None

def get_stt_engine() -> LocalSTT:
    global _stt_instance
    if _stt_instance is None:
        model_path = os.getenv("WHISPER_MODEL_PATH", "/home/aiteam/Desktop/all_downloaded_models/whisper-tiny/")
        _stt_instance = LocalSTT(model_path)
    return _stt_instance
