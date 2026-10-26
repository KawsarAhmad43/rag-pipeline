import os
import tempfile
import torch
from transformers import pipeline
import soundfile as sf
import numpy as np
from typing import Optional, Tuple

class SpeechToText:
    def __init__(self, model_path: str = None):
        """
        Initialize the Speech-to-Text model using Whisper.

        Args:
            model_path: Path to the local Whisper model. If None, uses default path.
        """
        if model_path is None:
            # # Use the default path relative to the project root
            # base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            # model_path = os.path.join(base_dir, "models", "whisper-tiny")
            model_path = "/home/aiteam/Desktop/all_downloaded_models/whisper-tiny"


        self.model_path = model_path
        self.pipe = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        try:
            print(f"Loading Whisper model from: {self.model_path}")

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path not found: {self.model_path}")

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_path,
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Whisper model loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            self.pipe = None

    def transcribe_audio_file(self, audio_file_path: str) -> Optional[str]:
        """
        Transcribe an audio file to text.

        Args:
            audio_file_path: Path to the audio file

        Returns:
            Transcribed text or None if failed
        """
        if not self.pipe:
            print("❌ Whisper model not loaded")
            return None

        try:
            if not os.path.exists(audio_file_path):
                print(f"❌ Audio file not found: {audio_file_path}")
                return None

            print(f"Transcribing audio: {audio_file_path}")
            result = self.pipe(audio_file_path, return_timestamps=False)
            transcribed_text = result["text"].strip()

            print(f"✅ Transcription successful: {transcribed_text}")
            return transcribed_text

        except Exception as e:
            print(f"❌ Error transcribing audio: {e}")
            return None

    def transcribe_audio_data(self, audio_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        """
        Transcribe audio data (bytes) to text.

        Args:
            audio_data: Audio data as bytes
            sample_rate: Sample rate of the audio (default: 16000)

        Returns:
            Transcribed text or None if failed
        """
        if not self.pipe:
            print("❌ Whisper model not loaded")
            return None

        try:
            # Create a temporary file to save the audio data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                # Convert bytes to numpy array and save as WAV
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                sf.write(temp_file.name, audio_array, sample_rate)
                temp_path = temp_file.name

            # Transcribe the temporary file
            result = self.transcribe_audio_file(temp_path)

            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass

            return result

        except Exception as e:
            print(f"❌ Error transcribing audio data: {e}")
            return None

    def is_model_loaded(self) -> bool:
        """Check if the model is loaded successfully."""
        return self.pipe is not None

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "is_loaded": self.is_model_loaded(),
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

# Global instance for easy access
stt_model = None

def get_stt_model() -> SpeechToText:
    """Get or create the global Speech-to-Text model instance."""
    global stt_model
    if stt_model is None:
        stt_model = SpeechToText()
    return stt_model

def transcribe_audio(audio_file_path: str) -> Optional[str]:
    """Convenience function to transcribe audio file."""
    model = get_stt_model()
    return model.transcribe_audio_file(audio_file_path)

def transcribe_audio_bytes(audio_data: bytes, sample_rate: int = 16000) -> Optional[str]:
    """Convenience function to transcribe audio data."""
    model = get_stt_model()
    return model.transcribe_audio_data(audio_data, sample_rate)
