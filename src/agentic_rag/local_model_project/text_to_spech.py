# import os
# import tempfile
# import soundfile as sf
# from kokoro_onnx import Kokoro
# from typing import Optional, Tuple, Union
# import numpy as np

# class TextToSpeech:
#     def __init__(self, model_path: str = None, voices_path: str = None):
#         """
#         Initialize the Text-to-Speech model using Kokoro.

#         Args:
#             model_path: Path to the local Kokoro model. If None, uses default path.
#             voices_path: Path to the voices file. If None, uses default path.
#         """
#         if model_path is None or voices_path is None:
#             # Use the default paths relative to the project root
#             base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
#             if model_path is None:
#                 model_path = os.path.join(base_dir, "models", "kokoro", "kokoro-v1.0.int8.onnx")
#             if voices_path is None:
#                 voices_path = os.path.join(base_dir, "models", "kokoro", "voices-v1.0.bin")

#         self.model_path = model_path
#         self.voices_path = voices_path
#         self.kokoro_model = None
#         self._load_model()

#     def _load_model(self):
#         """Load the Kokoro model."""
#         try:
#             print(f"Loading Kokoro model from: {self.model_path}")
#             print(f"Loading voices from: {self.voices_path}")

#             if not os.path.exists(self.model_path):
#                 raise FileNotFoundError(f"Model path not found: {self.model_path}")

#             if not os.path.exists(self.voices_path):
#                 raise FileNotFoundError(f"Voices path not found: {self.voices_path}")

#             self.kokoro_model = Kokoro(self.model_path, self.voices_path)
#             print("✅ Kokoro model loaded successfully!")

#         except Exception as e:
#             print(f"❌ Error loading Kokoro model: {e}")
#             self.kokoro_model = None

#     def synthesize_speech(self,
#                          text: str,
#                          voice: str = "af_heart",
#                          speed: float = 1.0,
#                          lang: str = "en-us",
#                          output_path: Optional[str] = None) -> Optional[Tuple[np.ndarray, int]]:
#         """
#         Synthesize speech from text.

#         Args:
#             text: Text to convert to speech
#             voice: Voice to use (default: "af_heart")
#             speed: Speech speed (default: 1.0)
#             lang: Language code (default: "en-us")
#             output_path: Optional path to save the audio file

#         Returns:
#             Tuple of (audio_samples, sample_rate) or None if failed
#         """
#         if not self.kokoro_model:
#             print("❌ Kokoro model not loaded")
#             return None

#         try:
#             print(f"Synthesizing speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")

#             # Generate audio
#             samples, sample_rate = self.kokoro_model.create(
#                 text,
#                 voice=voice,
#                 speed=speed,
#                 lang=lang
#             )

#             print(f"✅ Speech synthesis successful! Sample rate: {sample_rate}Hz")

#             # Save to file if output_path is provided
#             if output_path:
#                 try:
#                     sf.write(output_path, samples, sample_rate)
#                     print(f"✅ Audio saved to: {output_path}")
#                 except Exception as e:
#                     print(f"⚠️ Warning: Could not save audio file: {e}")

#             return samples, sample_rate

#         except Exception as e:
#             print(f"❌ Error synthesizing speech: {e}")
#             return None

#     def synthesize_to_file(self,
#                           text: str,
#                           output_path: str,
#                           voice: str = "af_heart",
#                           speed: float = 1.0,
#                           lang: str = "en-us") -> bool:
#         """
#         Synthesize speech and save directly to file.

#         Args:
#             text: Text to convert to speech
#             output_path: Path to save the audio file
#             voice: Voice to use (default: "af_heart")
#             speed: Speech speed (default: 1.0)
#             lang: Language code (default: "en-us")

#         Returns:
#             True if successful, False otherwise
#         """
#         result = self.synthesize_speech(text, voice, speed, lang, output_path)
#         return result is not None

#     def synthesize_to_temp_file(self,
#                                text: str,
#                                voice: str = "af_heart",
#                                speed: float = 1.0,
#                                lang: str = "en-us") -> Optional[str]:
#         """
#         Synthesize speech and save to a temporary file.

#         Args:
#             text: Text to convert to speech
#             voice: Voice to use (default: "af_heart")
#             speed: Speech speed (default: 1.0)
#             lang: Language code (default: "en-us")

#         Returns:
#             Path to the temporary audio file or None if failed
#         """
#         try:
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
#                 temp_path = temp_file.name

#             if self.synthesize_to_file(text, temp_path, voice, speed, lang):
#                 return temp_path
#             else:
#                 # Clean up if synthesis failed
#                 try:
#                     os.unlink(temp_path)
#                 except:
#                     pass
#                 return None

#         except Exception as e:
#             print(f"❌ Error creating temporary audio file: {e}")
#             return None

#     def get_available_voices(self) -> list:
#         """Get list of available voices."""
#         # Common Kokoro voices - you might want to make this dynamic
#         return [
#             "af_heart", "af_soft", "af_strong", "af_weak",
#             "am_heart", "am_soft", "am_strong", "am_weak",
#             "bm_heart", "bm_soft", "bm_strong", "bm_weak",
#             "cm_heart", "cm_soft", "cm_strong", "cm_weak"
#         ]

#     def is_model_loaded(self) -> bool:
#         """Check if the model is loaded successfully."""
#         return self.kokoro_model is not None

#     def get_model_info(self) -> dict:
#         """Get information about the loaded model."""
#         return {
#             "model_path": self.model_path,
#             "voices_path": self.voices_path,
#             "is_loaded": self.is_model_loaded(),
#             "available_voices": self.get_available_voices()
#         }

# # Global instance for easy access
# tts_model = None

# def get_tts_model() -> TextToSpeech:
#     """Get or create the global Text-to-Speech model instance."""
#     global tts_model
#     if tts_model is None:
#         tts_model = TextToSpeech()
#     return tts_model

# def text_to_speech(text: str,
#                   voice: str = "af_heart",
#                   speed: float = 1.0,
#                   lang: str = "en-us",
#                   output_path: Optional[str] = None) -> Optional[Tuple[np.ndarray, int]]:
#     """Convenience function to convert text to speech."""
#     model = get_tts_model()
#     return model.synthesize_speech(text, voice, speed, lang, output_path)

# def text_to_speech_file(text: str,
#                        output_path: str,
#                        voice: str = "af_heart",
#                        speed: float = 1.0,
#                        lang: str = "en-us") -> bool:
#     """Convenience function to convert text to speech and save to file."""
#     model = get_tts_model()
#     return model.synthesize_to_file(text, output_path, voice, speed, lang)

# def text_to_speech_temp(text: str,
#                        voice: str = "af_heart",
#                        speed: float = 1.0,
#                        lang: str = "en-us") -> Optional[str]:
#     """Convenience function to convert text to speech and save to temporary file."""
#     model = get_tts_model()
#     return model.synthesize_to_temp_file(text, voice, speed, lang)



import os
import tempfile
import soundfile as sf
from kokoro_onnx import Kokoro
from typing import Optional, Tuple, Union
import numpy as np

class TextToSpeech:
    def __init__(self, model_path: str = None, voices_path: str = None):
        """
        Initialize the Text-to-Speech model using Kokoro.

        Args:
            model_path: Path to the local Kokoro model. If None, uses default path.
            voices_path: Path to the voices file. If None, uses default path.
        """
        if model_path is None or voices_path is None:
            # # Use the default paths relative to the project root
            # base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            if model_path is None:
                # model_path = os.path.join(base_dir, "models", "kokoro", "kokoro-v1.0.int8.onnx")
                model_path = "/home/aiteam/Desktop/all_downloaded_models/kokoro/kokoro-v1.0.int8.onnx"
            if voices_path is None:
                # voices_path = os.path.join(base_dir, "models", "kokoro", "voices-v1.0.bin")
                voices_path = "/home/aiteam/Desktop/all_downloaded_models/kokoro/voices-v1.0.bin"

        self.model_path = model_path
        self.voices_path = voices_path
        self.kokoro_model = None
        self._load_model()

    def _load_model(self):
        """Load the Kokoro model."""
        try:
            print(f"Loading Kokoro model from: {self.model_path}")
            print(f"Loading voices from: {self.voices_path}")

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path not found: {self.model_path}")

            if not os.path.exists(self.voices_path):
                raise FileNotFoundError(f"Voices path not found: {self.voices_path}")

            self.kokoro_model = Kokoro(self.model_path, self.voices_path)
            print("✅ Kokoro model loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading Kokoro model: {e}")
            self.kokoro_model = None

    def synthesize_speech(self,
                         text: str,
                         voice: str = "af_heart",
                         speed: float = 1.0,
                         lang: str = "en-us",
                         output_path: Optional[str] = None) -> Optional[Tuple[np.ndarray, int]]:
        """
        Synthesize speech from text.

        Args:
            text: Text to convert to speech
            voice: Voice to use (default: "af_heart")
            speed: Speech speed (default: 1.0)
            lang: Language code (default: "en-us")
            output_path: Optional path to save the audio file

        Returns:
            Tuple of (audio_samples, sample_rate) or None if failed
        """
        if not self.kokoro_model:
            print("❌ Kokoro model not loaded")
            return None

        try:
            print(f"Synthesizing speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")

            # Generate audio
            samples, sample_rate = self.kokoro_model.create(
                text,
                voice=voice,
                speed=speed,
                lang=lang
            )

            print(f"✅ Speech synthesis successful! Sample rate: {sample_rate}Hz")

            # Save to file if output_path is provided
            if output_path:
                try:
                    sf.write(output_path, samples, sample_rate)
                    print(f"✅ Audio saved to: {output_path}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not save audio file: {e}")

            return samples, sample_rate

        except Exception as e:
            print(f"❌ Error synthesizing speech: {e}")
            return None

    def synthesize_to_file(self,
                          text: str,
                          output_path: str,
                          voice: str = "af_heart",
                          speed: float = 1.0,
                          lang: str = "en-us") -> bool:
        """
        Synthesize speech and save directly to file.

        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file
            voice: Voice to use (default: "af_heart")
            speed: Speech speed (default: 1.0)
            lang: Language code (default: "en-us")

        Returns:
            True if successful, False otherwise
        """
        result = self.synthesize_speech(text, voice, speed, lang, output_path)
        return result is not None

    def synthesize_to_temp_file(self,
                               text: str,
                               voice: str = "af_heart",
                               speed: float = 1.0,
                               lang: str = "en-us") -> Optional[str]:
        """
        Synthesize speech and save to a temporary file.

        Args:
            text: Text to convert to speech
            voice: Voice to use (default: "af_heart")
            speed: Speech speed (default: 1.0)
            lang: Language code (default: "en-us")

        Returns:
            Path to the temporary audio file or None if failed
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

            if self.synthesize_to_file(text, temp_path, voice, speed, lang):
                return temp_path
            else:
                # Clean up if synthesis failed
                try:
                    os.unlink(temp_path)
                except:
                    pass
                return None

        except Exception as e:
            print(f"❌ Error creating temporary audio file: {e}")
            return None

    def get_available_voices(self) -> list:
        """Get list of available voices."""
        # Common Kokoro voices - you might want to make this dynamic
        return [
            "af_heart", "af_soft", "af_strong", "af_weak",
            "am_heart", "am_soft", "am_strong", "am_weak",
            "bm_heart", "bm_soft", "bm_strong", "bm_weak",
            "cm_heart", "cm_soft", "cm_strong", "cm_weak"
        ]

    def is_model_loaded(self) -> bool:
        """Check if the model is loaded successfully."""
        return self.kokoro_model is not None

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "voices_path": self.voices_path,
            "is_loaded": self.is_model_loaded(),
            "available_voices": self.get_available_voices()
        }

# Global instance for easy access
tts_model = None

def get_tts_model() -> TextToSpeech:
    """Get or create the global Text-to-Speech model instance."""
    global tts_model
    if tts_model is None:
        tts_model = TextToSpeech()
    return tts_model

def text_to_speech(text: str,
                  voice: str = "af_heart",
                  speed: float = 1.0,
                  lang: str = "en-us",
                  output_path: Optional[str] = None) -> Optional[Tuple[np.ndarray, int]]:
    """Convenience function to convert text to speech."""
    model = get_tts_model()
    return model.synthesize_speech(text, voice, speed, lang, output_path)

def text_to_speech_file(text: str,
                       output_path: str,
                       voice: str = "af_heart",
                       speed: float = 1.0,
                       lang: str = "en-us") -> bool:
    """Convenience function to convert text to speech and save to file."""
    model = get_tts_model()
    return model.synthesize_to_file(text, output_path, voice, speed, lang)

def text_to_speech_temp(text: str,
                       voice: str = "af_heart",
                       speed: float = 1.0,
                       lang: str = "en-us") -> Optional[str]:
    """Convenience function to convert text to speech and save to temporary file."""
    model = get_tts_model()
    return model.synthesize_to_temp_file(text, voice, speed, lang)
