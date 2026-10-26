
import os
import sys
import logging
import time
from pathlib import Path
from src.multimodal_rag.main_voice import AgenticChatSystem
# ---------------- Project root ----------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


os.environ["TRANSFORMERS_VERBOSITY"] = "info"
# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def test_text_interactive(agent, generation_mode):
    """Interactive text-based testing with user input."""
    logging.info("\nEnter 'exit' to quit interactive text mode.")
    while True:
        user_input = input("Enter your text query: ").strip()
        if user_input.lower() == 'exit':
            break
        if not user_input:
            logging.warning("Empty input. Try again.")
            continue

        logging.info(f"\nTesting text query: {user_input}")
        start_time = time.time()

        # normal text response
        response = agent.get_response(user_input, generation_mode=generation_mode)
        end_time = time.time()
        logging.info(f"Text Response: {response}")
        logging.info(f"Time taken: {end_time - start_time:.2f} seconds")
        if isinstance(response, dict):
            logging.info(f"Audio saved at: {response.get('audio_path')}")


def test_tts_interactive(agent, generation_mode):
    """Interactive text-to-speech testing."""
    logging.info("\nEnter 'exit' to quit TTS mode.")
    available_voices = agent.available_voices()
    logging.info(f"Available voices: {available_voices}")

    while True:
        user_input = input("Enter your text for TTS: ").strip()
        if user_input.lower() == 'exit':
            break
        if not user_input:
            logging.warning("Empty input. Try again.")
            continue

        voice_choice = input(f"Choose voice from list {available_voices} (or press Enter for default): ").strip()
        if not voice_choice or voice_choice not in available_voices:
            voice_choice = "af_sarah"  # default

        audio_out = Path(project_root / "voice_notebook" / f"tts_output_{int(time.time())}.wav")
        audio_out.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Generating TTS with voice '{voice_choice}'...")
        start_time = time.time()
        response = agent.get_response(
            user_input,
            generation_mode=generation_mode,
            speak=True,
            voice=voice_choice,
            audio_out_path=audio_out
        )
        end_time = time.time()

        logging.info(f"TTS Response Text: {response.get('text') if isinstance(response, dict) else response}")
        logging.info(f"Audio saved at: {response.get('audio_path') if isinstance(response, dict) else audio_out}")
        logging.info(f"Time taken: {end_time - start_time:.2f} seconds")


def main_test():
    session_id = "tts_test_session"
    pdf_file_path = None
    generation_mode = "local"

    logging.info("Initializing AgenticChatSystem...")
    agent = AgenticChatSystem(session_id=session_id, pdf_file_path=pdf_file_path)

    # Interactive text testing
    test_text_interactive(agent, generation_mode)

    # TTS testing
    test_tts_interactive(agent, generation_mode)

    # Cleanup
    agent.cleanup()


if __name__ == "__main__":
    main_test()
