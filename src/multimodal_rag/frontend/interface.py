import streamlit as st
import requests
import uuid
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from config import config

st.set_page_config(page_title="Multimodal Agentic RAG Chatbot", layout="wide")
st.title("🤖 Multimodal Agentic RAG Chatbot")

# API Settings
API_BASE = f"http://{os.getenv('MultiModal_RAG_HOST', '0.0.0.0')}:{os.getenv('MultiModal_RAG_PORT', '5001')}"

# ---------------- Session State ----------------
def initialize_session_state():
    defaults = {
        "messages": [],
        "session_id": str(uuid.uuid4()),
        "uploaded_file": None,
        "rag_session_initialized": False,
        "selected_generation_mode": config.DEFAULT_GENERATION_MODE,
        "tts_enabled": False,
        "selected_voice": "af_sarah",
        "available_rag_sessions": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

initialize_session_state()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("⚙️ Settings")

    # Model Selection
    mode_options = []
    if config.GOOGLE_API_KEY:
        mode_options.append("api")
    if Path(config.GENERATOR_MODEL_PATH).is_dir():
        mode_options.append("local")
    if mode_options:
        st.session_state.selected_generation_mode = st.radio(
            "Choose Model:", mode_options, key="mode_radio"
        )
    else:
        st.error("No generation model available.")

    # RAG Sessions
    def fetch_sessions():
        try:
            res = requests.get(f"{API_BASE}/get_available_sessions")
            return res.json().get("sessions", [])
        except:
            return []

    st.session_state.available_rag_sessions = fetch_sessions()
    options = ["Create New Session"] + st.session_state.available_rag_sessions
    selected = st.selectbox("Session:", options)

    if selected == "Create New Session":
        st.session_state.session_id = st.text_input("New Session ID:", st.session_state.session_id)
    else:
        st.session_state.session_id = selected

    uploaded_file = st.file_uploader("📄 Upload PDF or Image (optional)", type=["pdf", "jpg", "jpeg", "png"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

    # TTS Settings
    st.write("🔊 Text-to-Speech (TTS)")
    st.session_state.tts_enabled = st.checkbox("Enable TTS", value=st.session_state.tts_enabled)
    if st.session_state.tts_enabled:
        voices =  [
            "af", "af_bella", "af_sarah",
            "am_adam", "am_michael",
            "bf_emma", "bf_isabella",
            "bm_george", "bm_lewis",
            "af_nicole", "af_sky",
        ]
        st.session_state.selected_voice = st.selectbox("Select Voice:", voices, index=0)

    if st.button("🔁 Initialize Session"):
        with st.spinner("Initializing..."):
            try:
                files = None
                if st.session_state.uploaded_file:
                    file = st.session_state.uploaded_file
                    files = {"file": (file.name, file.getvalue(), file.type)}
                res = requests.post(
                    f"{API_BASE}/initialize_session",
                    data={"session_id": st.session_state.session_id},
                    files=files
                )
                res.raise_for_status()
                st.session_state.rag_session_initialized = True
                st.session_state.messages = []
                st.success("Session initialized.")
            except Exception as e:
                st.session_state.rag_session_initialized = False
                st.error(f"Initialization failed: {e}")
        st.rerun()

# ---------------- Main Chat ----------------
st.subheader("💬 Chat Interface")

if st.session_state.rag_session_initialized:
    if st.button("🩹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # # Display chat history
    # for msg in st.session_state.messages:
    #     with st.chat_message(msg["role"]):
    #         st.markdown(msg["content"])
    #         if "audio_path" in msg:
    #             st.audio(msg["audio_path"])

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # Show the main content if available
            if "content" in msg:
                st.markdown(msg["content"])

            # If there's audio, play it
            if "audio_path" in msg:
                st.audio(msg["audio_path"])

            # NEW: Show transcript under audio
            if "text" in msg:
                st.markdown(f"**Transcript:** {msg['text']}")


    # User input
    user_input = st.chat_input("Ask a question...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Determine endpoint and payload
        if st.session_state.tts_enabled:
            endpoint = f"{API_BASE}/tts"
            payload = {
                "user_input": user_input,
                "session_id": st.session_state.session_id,
                "generation_mode": st.session_state.selected_generation_mode,
                "voice": st.session_state.selected_voice
            }
            files = None
            res = requests.post(endpoint, data=payload)
        else:
            endpoint = f"{API_BASE}/chat"
            payload = {
                "user_input": user_input,
                "session_id": st.session_state.session_id,
                "generation_mode": st.session_state.selected_generation_mode
            }
            file = st.session_state.uploaded_file
            files = {"attachment": (file.name, file.getvalue(), file.type)} if file else None
            res = requests.post(endpoint, data=payload, files=files)

        # Process response
        try:
            res.raise_for_status()
            result = res.json()
            if st.session_state.tts_enabled:
                answer = result.get("text", "")
                audio_path = result.get("audio_path")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "audio_path": audio_path
                })
            else:
                answer = result.get("response", "")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
        except Exception as e:
            st.error(f"❌ Error: {e}")

        st.rerun()

else:
    st.info("Initialize a session from the sidebar to start chatting.")
