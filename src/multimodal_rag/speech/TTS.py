# multimodal_rag/speech/TTS.py
from __future__ import annotations
import os, hashlib, time, argparse
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import soundfile as sf
import wave

# ======= Your fixed paths =======
THIS_FILE = Path(__file__).resolve()
# multimodal_rag root
ROOT = THIS_FILE.parents[1]
CACHE_DEFAULT_DIR = ROOT / "voice_notebook"         # default cache dir for synthesized WAVs
print(f"[TTS] ROOT: {ROOT} | cache: {CACHE_DEFAULT_DIR}")


from kokoro_onnx import Kokoro  # throws if package missing

# NOTE (as requested): predefined model paths you can copy/paste anywhere
PATHS = {
    "kokoro_onnx": "/home/aiteam/Desktop/rag_pipelines/models/kokoro/kokoro-v1.0.int8.onnx",
    "kokoro_voices": "/home/aiteam/Desktop/rag_pipelines/models/kokoro/voices-v1.0.bin",
    "wav_out": "/home/aiteam/Desktop/rag_pipelines/src/multimodal_rag/voice_notebook/output.wav"
}

# Prefer soundfile; fall back to wave if not available
try:
    _HAS_SF = True
except Exception:

    _HAS_SF = False

# ===== Voice pack =====
VOICE_NAMES: List[str] = [
    "af", "af_bella", "af_sarah",
    "am_adam", "am_michael",
    "bf_emma", "bf_isabella",
    "bm_george", "bm_lewis",
    "af_nicole", "af_sky",
]

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _resolve_model_paths() -> tuple[Path, Path]:
    """
    Resolve Kokoro ONNX + voices .bin.
    1) Use env overrides if provided.
    2) Otherwise use your project layout under <repo_root>/models/kokoro/.
    """
    env_model  = os.getenv("KOKORO_MODEL_PATH")
    env_voices = os.getenv("KOKORO_VOICES_PATH")

    repo_root = ROOT.parents[1]  # /home/.../rag_pipelines
    candidates_model = [
        env_model,
        repo_root / "models" / "kokoro" / "kokoro-v1.0.int8.onnx",
        ROOT      / "models" / "kokoro" / "kokoro-v1.0.int8.onnx",
    ]
    candidates_voices = [
        env_voices,
        repo_root / "models" / "kokoro" / "voices-v1.0.bin",
        ROOT      / "models" / "kokoro" / "voices-v1.0.bin",
    ]
    def _first_existing(paths):
        for p in paths:
            if not p: continue
            p = Path(p).expanduser()
            if p.exists(): return p.resolve()
        return None

    m = _first_existing(candidates_model)
    v = _first_existing(candidates_voices)

    # Fallback to absolute paths if everything else fails
    fallback_m = Path("/home/aiteam/Desktop/all_downloaded_models/kokoro/kokoro-v1.0.int8.onnx")
    fallback_v = Path("/home/aiteam/Desktop/all_downloaded_models/kokoro/voices-v1.0.bin")

    if not m and fallback_m.exists():
        m = fallback_m.resolve()
        logging.info(f"[TTS] Using fallback model path: {m}")
    if not v and fallback_v.exists():
        v = fallback_v.resolve()
        logging.info(f"[TTS] Using fallback voices path: {v}")

    if not m:
        raise FileNotFoundError(f"Kokoro model not found. Checked: {candidates_model} and fallback {fallback_m}")
    if not v:
        raise FileNotFoundError(f"Kokoro voices not found. Checked: {candidates_voices} and fallback {fallback_v}")

    return m, v


class KokoroTTS:
    """
    Local Kokoro-ONNX wrapper.
    - Loads your local .onnx and voices .bin
    - Validates voice names
    - Caches WAVs in ROOT/speech (or env override)
    """
    def __init__(
        self,
        model_path: Path | str | None = None,
        voices_path: Path | str | None = None,
        default_voice: str = os.getenv("TTS_DEFAULT_VOICE", "af_sarah"),
        default_speed: float = float(os.getenv("TTS_SPEED", "1.0")),
        default_lang: str = os.getenv("TTS_LANG", "en-us"),
        audio_cache_dir: Path | str = os.getenv("TTS_AUDIO_DIR", str(CACHE_DEFAULT_DIR)),
    ):


        if model_path is None or voices_path is None:
            model_path, voices_path = _resolve_model_paths()

        self.model_path  = Path(model_path).expanduser().resolve()
        self.voices_path = Path(voices_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Kokoro .onnx not found: {self.model_path}")
        if not self.voices_path.exists():
            raise FileNotFoundError(f"Kokoro voices .bin not found: {self.voices_path}")

        self.default_voice = default_voice
        self.default_speed = float(default_speed)
        self.default_lang  = default_lang

        self.cache_dir = Path(audio_cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._engine = Kokoro(str(self.model_path), str(self.voices_path))

    @staticmethod
    def voices() -> list[str]:
        return VOICE_NAMES.copy()

    def _validate_voice(self, voice: str):
        if voice not in VOICE_NAMES:
            raise ValueError(f"Unknown voice '{voice}'. Choose one of: {VOICE_NAMES}")

    def _cache_key(self, text: str, voice: str, speed: float, lang: str) -> str:
        sig = f"{text}|{voice}|{speed}|{lang}|{self.model_path.stat().st_mtime}|{self.voices_path.stat().st_mtime}"
        return _sha1(sig)

    def synth(self, text: str, voice: str | None = None, speed: float | None = None, lang: str | None = None) -> tuple[np.ndarray, int]:
        v = voice or self.default_voice
        s = float(speed if speed is not None else self.default_speed)
        l = lang  or self.default_lang
        self._validate_voice(v)
        samples, sr = self._engine.create(text, voice=v, speed=s, lang=l)  # ← Kokoro API
        return np.asarray(samples, dtype=np.float32).reshape(-1), int(sr)

    def synthesize_to_wav(
        self,
        text: str,
        out_path: Path | str | None = None,
        voice: str | None = None,
        speed: float | None = None,
        lang: str | None = None,
        use_cache: bool = True,
    ) -> Path:
        v = voice or self.default_voice
        s = float(speed if speed is not None else self.default_speed)
        l = lang  or self.default_lang
        key = self._cache_key(text, v, s, l)
        out_path = Path(out_path) if out_path else self.cache_dir / f"{key}.wav"

        if use_cache and out_path.exists():
            return out_path

        samples, sr = self.synth(text, voice=v, speed=s, lang=l)

        if _HAS_SF:
            sf.write(out_path, samples, sr)
        else:
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                wf.writeframes(pcm.tobytes())
        return out_path

    def synthesize_to_bytes(self, text: str, voice: str | None = None, speed: float | None = None, lang: str | None = None) -> bytes:
        tmp = self.cache_dir / f"tmp_{int(time.time()*1000)}.wav"
        path = self.synthesize_to_wav(text, out_path=tmp, voice=voice, speed=speed, lang=lang, use_cache=False)
        data = path.read_bytes()
        try: tmp.unlink()
        except Exception: pass
        return data


# ===== Singleton helpers =====
_engine_singleton: Optional[KokoroTTS] = None

def load_engine() -> KokoroTTS:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = KokoroTTS()
    return _engine_singleton

def tts_say(text: str, voice: str | None = None, speed: float | None = None, lang: str = "en-us") -> Path:
    return load_engine().synthesize_to_wav(text, voice=voice, speed=speed, lang=lang)
