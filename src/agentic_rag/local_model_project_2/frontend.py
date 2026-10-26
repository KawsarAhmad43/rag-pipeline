import streamlit as st
import requests
import json
import time
from PIL import Image
import io
import base64
import tempfile
import os
from st_audiorec import st_audiorec
import numpy as np
import soundfile as sf
import scipy.signal as sps

# Configuration
API_BASE_URL = "http://localhost:6001"

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Chatbot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with improved text visibility
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid;
        color: #000000 !important;
        font-weight: 500;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
        color: #000000 !important;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
        color: #000000 !important;
    }
    .image-analysis {
        background-color: #fff3e0;
        border-left-color: #ff9800;
        color: #000000 !important;
    }
    .web-search {
        background-color: #e8f5e9;
        border-left-color: #4caf50;
        color: #000000 !important;
    }
    .chat-message strong {
        color: #000000 !important;
        font-weight: 700;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        height: 3rem;
        font-size: 1.1rem;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    /* Ensure all text in chat messages is black and visible */
    .chat-message * {
        color: #000000 !important;
    }
    /* Override any Streamlit default text colors */
    .stMarkdown .chat-message {
        color: #000000 !important;
    }
    .stMarkdown .chat-message p {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"
    if 'api_status' not in st.session_state:
        st.session_state.api_status = "unknown"
    if 'chat_input_key' not in st.session_state:
        st.session_state.chat_input_key = 0
    if 'transcript' not in st.session_state:
        st.session_state.transcript = None

def check_api_status():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            st.session_state.api_status = "connected"
            return True
        else:
            st.session_state.api_status = "error"
            return False
    except requests.exceptions.RequestException:
        st.session_state.api_status = "disconnected"
        return False

def send_chat_message(message):
    """Send a chat message to the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "message": message,
                "session_id": st.session_state.session_id
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.json().get('detail', 'Unknown error')}"
    except requests.exceptions.RequestException as e:
        return f"Connection error: {str(e)}"

def send_web_search(query):
    """Send a web search request to the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={
                "query": query,
                "session_id": st.session_state.session_id,
                "max_results": 5
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["response"], response.json()["search_results"]
        else:
            return f"Error: {response.json().get('detail', 'Unknown error')}", []
    except requests.exceptions.RequestException as e:
        return f"Connection error: {str(e)}", []

def analyze_image(image_file, prompt="Describe this image"):
    """Send an image for analysis"""
    try:
        # Debug information
        st.write(f"Debug: File name: {image_file.name}")
        st.write(f"Debug: File type: {image_file.type}")
        st.write(f"Debug: File size: {len(image_file.getvalue())} bytes")

        # Reset file pointer
        image_file.seek(0)

        # Prepare the multipart form data
        files = {"file": (image_file.name, image_file, image_file.type)}
        data = {
            "prompt": prompt,
            "session_id": st.session_state.session_id
        }

        response = requests.post(
            f"{API_BASE_URL}/analyze-image",
            files=files,
            data=data,
            timeout=60
        )

        if response.status_code == 200:
            return response.json()["description"]
        else:
            error_detail = response.json().get('detail', 'Unknown error') if response.content else 'Empty response'
            return f"Error: {error_detail}"
    except requests.exceptions.RequestException as e:
        return f"Connection error: {str(e)}"
    except json.JSONDecodeError as e:
        return f"JSON parsing error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def clear_chat_history():
    """Clear the chat history"""
    try:
        response = requests.delete(f"{API_BASE_URL}/session/{st.session_state.session_id}")
        if response.status_code == 200:
            st.session_state.messages = []
            st.success("Chat history cleared!")
        else:
            st.error("Failed to clear chat history")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")

def send_speech_to_text(audio_file):
    """Send audio file for speech-to-text conversion"""
    try:
        # Reset file pointer to beginning
        audio_file.seek(0)

        # Debug information - use stored audio data if available
        if "current_audio_data" in st.session_state:
            audio_size = len(st.session_state.current_audio_data)
        else:
            audio_size = len(audio_file.getvalue())
            audio_file.seek(0)  # Reset again after getting size

        print(f"Debug: Audio file size: {audio_size} bytes")
        print(f"Debug: Audio file type: {type(audio_file)}")
        print(f"Debug: Audio file name: {getattr(audio_file, 'name', 'No name')}")
        print(f"Debug: Audio file type: {getattr(audio_file, 'type', 'No type')}")

        # Prepare the file for upload with proper filename and content type
        # For UploadedFile objects, we can use them directly
        files = {
            "file": (
                getattr(audio_file, 'name', 'recording.wav'),  # Use the original name if available
                audio_file,  # The UploadedFile object
                getattr(audio_file, 'type', 'audio/wav')  # Use the original type if available
            )
        }

        print(f"Debug: Sending request to {API_BASE_URL}/speech-to-text")
        response = requests.post(f"{API_BASE_URL}/speech-to-text", files=files, timeout=60)

        print(f"Debug: Response status: {response.status_code}")
        print(f"Debug: Response content: {response.text[:500]}...")

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Debug: Request error: {str(e)}")
        st.error(f"Request error in speech-to-text: {str(e)}")
        return None
    except Exception as e:
        print(f"Debug: General error: {str(e)}")
        st.error(f"Error in speech-to-text: {str(e)}")
        return None

def send_text_to_speech(text, voice="af_heart", speed=1.0, lang="en-us"):
    """Send text for text-to-speech conversion"""
    try:
        data = {
            "text": text,
            "voice": voice,
            "speed": speed,
            "lang": lang
        }
        response = requests.post(f"{API_BASE_URL}/text-to-speech", json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error in text-to-speech: {str(e)}")
        return None

def get_speech_status():
    """Get status of speech models"""
    try:
        response = requests.get(f"{API_BASE_URL}/speech-status")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting speech status: {str(e)}")
        return None

def send_document_upload(document_file):
    """Send document for processing"""
    try:
        files = {"file": document_file}
        data = {"session_id": st.session_state.session_id}
        response = requests.post(f"{API_BASE_URL}/upload-document", files=files, data=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error uploading document: {str(e)}")
        return None

def get_session_status():
    """Get status of current session"""
    try:
        response = requests.get(f"{API_BASE_URL}/session-status/{st.session_state.session_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting session status: {str(e)}")
        return None

def create_proper_wav_file(audio_bytes, output_path):
    """Create a proper WAV file with correct headers from raw audio data"""
    try:
        # Assume the audio data is 16-bit PCM at 16kHz sample rate
        sample_rate = 16000
        channels = 1
        sample_width = 2  # 16-bit = 2 bytes

        # Calculate WAV file size
        data_size = len(audio_bytes)
        file_size = 36 + data_size

        # Create WAV header
        wav_header = bytearray()

        # RIFF header
        wav_header.extend(b'RIFF')
        wav_header.extend(file_size.to_bytes(4, 'little'))
        wav_header.extend(b'WAVE')

        # fmt chunk
        wav_header.extend(b'fmt ')
        wav_header.extend((16).to_bytes(4, 'little'))  # fmt chunk size
        wav_header.extend((1).to_bytes(2, 'little'))   # PCM format
        wav_header.extend(channels.to_bytes(2, 'little'))
        wav_header.extend(sample_rate.to_bytes(4, 'little'))
        wav_header.extend((sample_rate * channels * sample_width).to_bytes(4, 'little'))  # byte rate
        wav_header.extend((channels * sample_width).to_bytes(2, 'little'))  # block align
        wav_header.extend(sample_width.to_bytes(2, 'little'))  # bits per sample

        # data chunk
        wav_header.extend(b'data')
        wav_header.extend(data_size.to_bytes(4, 'little'))

        # Write the complete WAV file
        with open(output_path, 'wb') as f:
            f.write(wav_header)
            f.write(audio_bytes)

        return True
    except Exception as e:
        print(f"Error creating WAV file: {e}")
        return False

def list_saved_audio_files():
    """List all saved audio files in the audio_files directory"""
    audio_folder = "audio_files"
    if not os.path.exists(audio_folder):
        return []

    audio_files = []
    for filename in os.listdir(audio_folder):
        if filename.endswith('.wav'):
            filepath = os.path.join(audio_folder, filename)
            file_size = os.path.getsize(filepath)
            audio_files.append({
                'filename': filename,
                'filepath': filepath,
                'size': file_size,
                'session_id': filename.replace('_recording.wav', '').replace('_test.wav', '')
            })

    return sorted(audio_files, key=lambda x: x['filename'])

def main():
    # Initialize session state
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header"> Multimodal AI Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("**Tip**: Prefix your message with 'web:' to perform a web search (e.g., 'web: latest AI news').", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # API Status
        api_status = check_api_status()
        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.info("Make sure the FastAPI backend is running on localhost:6001")

        # Speech Models Status
        speech_status = get_speech_status()
        if speech_status:
            st.subheader("🎤 Speech Models")
            if speech_status.get("speech_to_text", {}).get("loaded", False):
                st.success("✅ Speech-to-Text Ready")
            else:
                st.error("❌ Speech-to-Text Not Available")

            if speech_status.get("text_to_speech", {}).get("loaded", False):
                st.success("✅ Text-to-Speech Ready")

                # Voice selection
                available_voices = speech_status.get("text_to_speech", {}).get("available_voices", [])
                if available_voices:
                    selected_voice = st.selectbox(
                        "Select Voice:",
                        available_voices,
                        index=0,
                        key="voice_selector"
                    )
                    st.session_state.selected_voice = selected_voice

                # Speed control
                speech_speed = st.slider(
                    "Speech Speed:",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    key="speed_selector"
                )
                st.session_state.speech_speed = speech_speed
            else:
                st.error("❌ Text-to-Speech Not Available")

        st.divider()

        # Session Info
        st.subheader("📋 Session Info")
        st.text(f"Session ID: {st.session_state.session_id}")
        st.text(f"Messages: {len(st.session_state.messages)}")

        # Session Status
        session_status = get_session_status()
        if session_status:
            st.subheader("🔧 Session Status")
            if session_status.get("chatbot_available"):
                st.success("✅ Chatbot Ready")
            else:
                st.error("❌ Chatbot Not Available")

            if session_status.get("vector_db_available"):
                st.success("✅ Vector DB Ready")
            else:
                st.error("❌ Vector DB Not Available")

            if session_status.get("document_uploaded"):
                st.success("✅ Document Loaded")
            else:
                st.info("📄 No documents uploaded yet")

        st.divider()

        # Controls
        st.subheader("🛠️ Controls")
        if st.button("🗑️ Clear Chat History", type="secondary"):
            clear_chat_history()

        if st.button(" New Session", type="secondary"):
            st.session_state.session_id = f"session_{int(time.time())}"
            st.session_state.messages = []
            st.rerun()

        # Test speech-to-text functionality
        if st.button("🧪 Test Speech-to-Text", type="secondary"):
            st.info("Testing speech-to-text API connection...")
            try:
                # Test with a simple request to check if the endpoint is accessible
                response = requests.get(f"{API_BASE_URL}/speech-status", timeout=10)
                if response.status_code == 200:
                    status_data = response.json()
                    st.success("✅ Speech-to-Text API is accessible!")
                    st.json(status_data)
                else:
                    st.error(f"❌ Speech-to-Text API returned status {response.status_code}")
            except Exception as e:
                st.error(f"❌ Error testing speech-to-text API: {str(e)}")

        # Test audio data storage
        if st.button("🔊 Test Audio Data", type="secondary"):
            if "current_audio_data" in st.session_state:
                st.success(f"✅ Audio data stored: {len(st.session_state.current_audio_data)} bytes")
                # Try to create a simple audio player
                try:
                    # Create audio folder if it doesn't exist
                    audio_folder = "audio_files"
                    if not os.path.exists(audio_folder):
                        os.makedirs(audio_folder)

                    # Save test audio file
                    test_filename = f"{st.session_state.session_id}_test.wav"
                    test_filepath = os.path.join(audio_folder, test_filename)

                    with open(test_filepath, "wb") as f:
                        f.write(st.session_state.current_audio_data)

                    st.audio(test_filepath, format="audio/wav")
                    st.info(f"Test audio file created at: {test_filepath}")
                    st.write(f"File size: {os.path.getsize(test_filepath)} bytes")

                    # Clean up test file
                    os.unlink(test_filepath)

                except Exception as e:
                    st.error(f"❌ Error testing audio playback: {str(e)}")
            else:
                st.info("ℹ️ No audio data currently stored")

        # Saved Audio Files
        st.divider()
        st.subheader("🎵 Saved Audio Files")

        # Refresh button
        col_refresh, col_clear = st.columns([1, 1])
        with col_refresh:
            if st.button("🔄 Refresh", type="secondary", key="refresh_audio_files"):
                st.rerun()
        with col_clear:
            if st.button("🗑️ Clear All", type="secondary", key="clear_all_audio"):
                saved_files = list_saved_audio_files()
                deleted_count = 0
                for audio_file in saved_files:
                    try:
                        os.unlink(audio_file['filepath'])
                        deleted_count += 1
                    except:
                        pass
                st.success(f"✅ Deleted {deleted_count} audio file(s)")
                st.rerun()

        saved_files = list_saved_audio_files()
        if saved_files:
            st.success(f"Found {len(saved_files)} saved audio file(s)")

            for audio_file in saved_files:
                with st.expander(f"📁 {audio_file['filename']} ({audio_file['size']} bytes)"):
                    # Display audio player
                    st.audio(audio_file['filepath'], format="audio/wav")

                    # Action buttons
                    col_load, col_download, col_delete = st.columns([1, 1, 1])

                    with col_load:
                        # Load and use button
                        if st.button(f"🎤 Load", key=f"load_{audio_file['filename']}"):
                            # Load the audio file into session state
                            with open(audio_file['filepath'], "rb") as f:
                                st.session_state.current_audio_data = f.read()
                            st.session_state.current_audio_filepath = audio_file['filepath']
                            st.success(f"✅ Loaded audio file: {audio_file['filename']}")
                            st.rerun()

                    with col_download:
                        # Download button
                        if st.button(f"📥 Download", key=f"download_{audio_file['filename']}"):
                            try:
                                with open(audio_file['filepath'], "rb") as f:
                                    audio_data = f.read()

                                st.download_button(
                                    label="💾 Download",
                                    data=audio_data,
                                    file_name=audio_file['filename'],
                                    mime="audio/wav",
                                    key=f"download_btn_{audio_file['filename']}"
                                )
                            except Exception as e:
                                st.error(f"❌ Error downloading: {str(e)}")

                    with col_delete:
                        # Delete button
                        if st.button(f"🗑️ Delete", key=f"delete_{audio_file['filename']}"):
                            try:
                                os.unlink(audio_file['filepath'])
                                st.success(f"✅ Deleted: {audio_file['filename']}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error deleting file: {str(e)}")
        else:
            st.info("ℹ️ No saved audio files found")

        # Document upload option
        st.divider()
        st.subheader("📄 Upload Documents")
        uploaded_document = st.file_uploader(
            "Upload documents for RAG",
            type=['pdf', 'txt', 'docx', 'doc'],
            key="sidebar_document_uploader",
            help="Upload documents to build your knowledge base"
        )

        if uploaded_document:
            st.info(f"📁 File: {uploaded_document.name} ({len(uploaded_document.getvalue())} bytes)")
            if st.button("📚 Process Document", type="secondary", key="sidebar_process_doc_btn"):
                with st.spinner("📚 Processing document..."):
                    result = send_document_upload(uploaded_document)
                    if result and result.get("success"):
                        chunks_count = result.get("chunks_count", 0)
                        st.success(f"✅ Document processed! {chunks_count} chunks created.")
                        st.rerun()
                    else:
                        st.error("❌ Failed to process document")

        # Audio upload option
        st.divider()
        st.subheader("📁 Upload Audio")
        uploaded_audio = st.file_uploader(
            "Upload pre-recorded audio",
            type=['wav', 'mp3', 'm4a', 'ogg'],
            key="sidebar_audio_uploader",
            help="Upload audio files for transcription"
        )

        if uploaded_audio:
            st.audio(uploaded_audio, format="audio/wav")
            if st.button("🎤 Transcribe & Send", type="secondary", key="sidebar_transcribe_btn"):
                with st.spinner("🎤 Converting speech to text..."):
                    result = send_speech_to_text(uploaded_audio)
                    if result and result.get("success"):
                        transcribed_text = result.get("text", "")
                        st.session_state.transcribed_text = transcribed_text
                        st.success(f"✅ Transcribed: {transcribed_text}")
                        st.rerun()
                    else:
                        st.error("❌ Failed to transcribe audio")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Chat Interface")

        # Chat messages display with improved visibility
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message" style="color: #000000 !important;">
                        <strong style="color: #000000 !important;">You:</strong>
                        <span style="color: #000000 !important;">{message["content"]}</span>
                    </div>
                    """, unsafe_allow_html=True)
                elif message["role"] == "assistant":
                    st.markdown(f"""
                    <div class="chat-message bot-message" style="color: #000000 !important;">
                        <strong style="color: #000000 !important;">🤖 Bot:</strong>
                        <span style="color: #000000 !important;">{message["content"]}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Add audio playback button for bot responses
                    if st.button(f"🔊 Listen to Response", key=f"listen_{i}"):
                        with st.spinner("🔊 Generating speech..."):
                            # Use selected voice and speed or defaults
                            selected_voice = st.session_state.get("selected_voice", "af_heart")
                            speech_speed = st.session_state.get("speech_speed", 1.0)
                            result = send_text_to_speech(message["content"], voice=selected_voice, speed=speech_speed)
                            if result and result.get("success"):
                                audio_file = result.get("audio_file")
                                if audio_file:
                                    # Get the filename from the path
                                    filename = os.path.basename(audio_file)
                                    audio_url = f"{API_BASE_URL}/audio/{filename}"
                                    st.audio(audio_url, format="audio/wav")
                                else:
                                    st.error("Failed to generate audio file")
                            else:
                                st.error("Failed to generate speech")
                elif message["role"] == "image":
                    st.markdown(f"""
                    <div class="chat-message image-analysis" style="color: #000000 !important;">
                        <strong style="color: #000000 !important;">️ Image Analysis:</strong>
                        <span style="color: #000000 !important;">{message["content"]}</span>
                    </div>
                    """, unsafe_allow_html=True)
                elif message["role"] == "web":
                    st.markdown(f"""
                    <div class="chat-message web-search" style="color: #000000 !important;">
                        <strong style="color: #000000 !important;">🌐 Web Search:</strong>
                        <span style="color: #000000 !important;">{message["content"]}</span>
                    </div>
                    """, unsafe_allow_html=True)

        # Chat input section
        st.subheader("💬 Chat Input")

        # Record button above text input
        col_record_above, col_space = st.columns([1, 4])

        with col_record_above:
            if st.button("🎤 Record Audio", type="primary", help="Record audio for speech-to-text", key="record_audio_btn"):
                # Set a flag to show recording interface
                st.session_state.show_recording = True
                st.session_state.transcript = None
                # Clear any previous audio data
                if "current_audio_data" in st.session_state:
                    del st.session_state.current_audio_data
                # Clean up any existing saved audio file
                if "current_audio_filepath" in st.session_state:
                    try:
                        os.unlink(st.session_state.current_audio_filepath)
                        del st.session_state.current_audio_filepath
                    except:
                        pass
                st.rerun()

        # Text input
        user_input = st.text_input(
            "Type your message here (use 'web:' for web search)...",
            key=f"user_input_{st.session_state.chat_input_key}",
            placeholder="Ask me anything, upload an image, or use 'web:' for web search!",
            value=st.session_state.get("transcribed_text", "")
        )

        # Clear transcribed text after use
        if "transcribed_text" in st.session_state:
            del st.session_state.transcribed_text

        # Recording interface
        if st.session_state.get("show_recording", False):
            st.markdown("---")
            st.subheader("🎤 Voice Recording")

            # Audio recorder with better parameters
            st.info("🎤 Click the microphone button below to record your message")

            # Try the streamlit-audiorec component with error handling
            try:
                # Audio recording widget using streamlit-audiorec
                wav_audio_data = st_audiorec()

                # Show recorded audio and processing options
                if wav_audio_data is not None:
                    st.success("✅ Audio recorded successfully!")

                    # Store raw audio data in session state
                    st.session_state.current_audio_data = wav_audio_data

                    # --- MODIFICATION START: Save audio directly without resampling ---
                    audio_folder = "audio_files"
                    if not os.path.exists(audio_folder):
                        os.makedirs(audio_folder)

                    audio_filename = f"{st.session_state.session_id}_recording.wav"
                    audio_filepath = os.path.join(audio_folder, audio_filename)

                    try:
                        # streamlit-audiorec returns WAV data directly, so we can save it as-is
                        with open(audio_filepath, "wb") as f:
                            f.write(wav_audio_data)

                        file_size = os.path.getsize(audio_filepath)
                        if file_size > 44:  # Check if file is not just an empty header
                            st.session_state.current_audio_filepath = audio_filepath

                            # Show the processed audio (this should match the original)
                            st.audio(audio_filepath, format="audio/wav")

                            # Show file info
                            st.info(f"Recording saved: {audio_filename} ({file_size} bytes)")
                        else:
                            st.error("❌ Error: Saved audio file is empty!")

                    except Exception as e:
                        st.error(f"❌ Error saving audio file: {e}")
                        st.info("Attempting to play raw bytes as fallback...")
                        st.audio(wav_audio_data, format="audio/wav")
                    # --- MODIFICATION END ---

                    # --- MODIFICATION START: Display transcript and improve UI flow ---
                    if st.session_state.transcript:
                        st.subheader("📝 Transcript")
                        st.text_area("Result:", value=st.session_state.transcript, height=100, key="transcript_display")
                        if st.button("✏️ Use Transcript", key="use_transcript_btn"):
                            st.session_state.transcribed_text = st.session_state.transcript
                            st.session_state.transcript = None
                            st.session_state.show_recording = False
                            st.rerun()
                    # --- MODIFICATION END ---

                    # Action buttons
                    col_convert, col_send, col_cancel = st.columns([1, 1, 1])

                    with col_convert:
                        if st.button("🎤 Convert to Text", type="secondary", key="convert_only_btn"):
                            with st.spinner("🎤 Converting speech to text..."):
                                if "current_audio_filepath" in st.session_state and os.path.exists(st.session_state.current_audio_filepath):
                                    with open(st.session_state.current_audio_filepath, "rb") as f:
                                        result = send_speech_to_text(f)
                                    if result and result.get("success"):
                                        st.session_state.transcript = result.get("text", "")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to transcribe audio. Please try again.")
                                else:
                                    st.error("❌ Could not find processed audio file. Please record again.")

                    with col_send:
                        if st.button("🚀 Convert & Send", type="primary", key="convert_send_btn"):
                            with st.spinner("🎤 Converting and sending..."):
                                if "current_audio_filepath" in st.session_state and os.path.exists(st.session_state.current_audio_filepath):
                                    with open(st.session_state.current_audio_filepath, "rb") as f:
                                        result = send_speech_to_text(f)

                                    if result and result.get("success"):
                                        transcribed_text = result.get("text", "")
                                        st.success(f"✅ Transcribed: {transcribed_text}")

                                        st.session_state.messages.append({"role": "user", "content": transcribed_text})

                                        with st.spinner("🤖 Thinking..."):
                                            bot_response = send_chat_message(transcribed_text)
                                            st.session_state.messages.append({"role": "assistant", "content": bot_response})

                                        st.session_state.show_recording = False
                                        st.session_state.transcript = None
                                        st.session_state.chat_input_key += 1
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to transcribe audio. Please try again.")
                                else:
                                    st.error("❌ Could not find processed audio file. Please record again.")

                    with col_cancel:
                        if st.button("❌ Cancel", type="secondary", key="cancel_recording_btn"):
                            st.session_state.show_recording = False
                            st.session_state.transcript = None
                            st.rerun()

                # Cancel button (when no audio recorded yet)
                if wav_audio_data is None:
                    if st.button("❌ Cancel Recording", type="secondary", key="cancel_no_audio_btn"):
                        st.session_state.show_recording = False
                        st.session_state.transcript = None
                        st.rerun()

            except Exception as e:
                st.error(f"❌ Audio recorder error: {str(e)}")
                st.warning("Audio recorder component failed. You can try uploading an audio file from the sidebar.")
                if st.button("Go Back", key="fallback_cancel_btn"):
                    st.session_state.show_recording = False
                    st.rerun()

        # Send message button
        if st.button("🚀 Send Message", type="primary", key="send_text_btn"):
            if user_input.strip():
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": user_input})

                # Check if it's a web search
                if user_input.lower().startswith("web:"):
                    query = user_input[4:].strip()
                    if query:
                        with st.spinner("🌐 Searching the web..."):
                            bot_response, search_results = send_web_search(query)
                            content = f"Response: {bot_response}\n\nSearch Results:\n" + \
                                     "\n".join(f"- [{r['title']}]({r['href']}): {r['body']}" for r in search_results) if search_results else "No search results found."
                            st.session_state.messages.append({"role": "web", "content": content})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "Error: Web search query is empty."})
                else:
                    # Normal chat
                    with st.spinner("🤖 Thinking..."):
                        bot_response = send_chat_message(user_input)
                        st.session_state.messages.append({"role": "assistant", "content": bot_response})

                # Increment the key to clear the input field
                st.session_state.chat_input_key += 1
                st.rerun()

        with col2:
            st.subheader("️ Image Analysis")

            # Image upload section
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload an image to analyze",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                help="Upload an image and I'll describe it for you!"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                # Image analysis prompt
                analysis_prompt = st.text_input(
                    "Analysis prompt (optional):",
                    value="Describe this image in detail",
                    key="analysis_prompt"
                )

                if st.button("🔍 Analyze Image", type="primary", key="analyze_image_btn"):
                    with st.spinner("🔍 Analyzing image..."):
                        analysis_result = analyze_image(uploaded_file, analysis_prompt)

                    # Add image analysis to chat
                    st.session_state.messages.append({
                        "role": "image",
                        "content": f"Prompt: {analysis_prompt}\n\nAnalysis: {analysis_result}"
                    })
                    st.rerun()

            # Quick actions
            st.subheader("⚡ Quick Actions")
            if st.button("👋 Say Hello", key="say_hello_btn"):
                st.session_state.messages.append({"role": "user", "content": "Hello!"})
                with st.spinner("🤖 Thinking..."):
                    bot_response = send_chat_message("Hello!")
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                st.rerun()

            if st.button("❓ Ask for Help", key="ask_help_btn"):
                st.session_state.messages.append({"role": "user", "content": "What can you help me with?"})
                with st.spinner("🤖 Thinking..."):
                    bot_response = send_chat_message("What can you help me with?")
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                st.rerun()

            if st.button("🌐 Web Search Example", key="web_search_example_btn"):
                query = "web: latest AI news"
                st.session_state.messages.append({"role": "user", "content": query})
                with st.spinner("🌐 Searching the web..."):
                    bot_response, search_results = send_web_search("latest AI news")
                    content = f"Response: {bot_response}\n\nSearch Results:\n" + \
                             "\n".join(f"- [{r['title']}]({r['href']}): {r['body']}" for r in search_results) if search_results else "No search results found."
                    st.session_state.messages.append({"role": "web", "content": content})
                st.rerun()

    # Footer
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            Multimodal AI Chatbot | Powered by FastAPI & Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
