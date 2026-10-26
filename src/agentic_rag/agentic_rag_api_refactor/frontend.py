import streamlit as st
import requests
import uuid
import base64
from PIL import Image
import io
import os
import time
from typing import Optional

# Try to import optional dependencies
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

# Try to import st_audiorec
try:
    from st_audiorec import st_audiorec
    AUDIOREC_AVAILABLE = True
except ImportError:
    AUDIOREC_AVAILABLE = False
    st_audiorec = None

# Configuration
API_BASE_URL = "http://localhost:8001/api/v1"

st.set_page_config(
    page_title="Agentic RAG Chatbot v2.0",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG Chatbot v2.0")
st.caption("Enhanced with OpenAI, Gemini support, Web Search, and Speech Features")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "upload_status" not in st.session_state:
    st.session_state.upload_status = ""
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "attachment_key" not in st.session_state:
    st.session_state.attachment_key = 0
if "ai_provider" not in st.session_state:
    st.session_state.ai_provider = "gemini"
if "available_providers" not in st.session_state:
    st.session_state.available_providers = []
if "selected_voice" not in st.session_state:
    st.session_state.selected_voice = "af_heart"
if "speech_speed" not in st.session_state:
    st.session_state.speech_speed = 1.0
if "speech_status" not in st.session_state:
    st.session_state.speech_status = None

# Speech functions
def get_speech_status():
    """Get status of speech models"""
    try:
        response = requests.get(f"{API_BASE_URL}/speech-status", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def send_speech_to_text(audio_file):
    """Send audio file for speech-to-text conversion"""
    try:
        # Reset file pointer to beginning
        audio_file.seek(0)

        files = {
            "file": (
                getattr(audio_file, 'name', 'recording.wav'),
                audio_file,
                getattr(audio_file, 'type', 'audio/wav')
            )
        }

        response = requests.post(f"{API_BASE_URL}/speech-to-text", files=files, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
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
        response = requests.post(f"{API_BASE_URL}/text-to-speech", data=data, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error in text-to-speech: {str(e)}")
        return None

def create_proper_wav_file(audio_bytes: bytes, output_path: str) -> bool:
    try:
        if PYDUB_AVAILABLE:
            # Use pydub for advanced audio processing
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 16kHz, mono, 16-bit PCM
            audio.export(output_path, format="wav")
            return True
        else:
            # Simple fallback: save audio data directly
            # Check if it's already a WAV file
            if len(audio_bytes) > 44 and audio_bytes[:4] == b'RIFF':
                # It's already a WAV file, save it directly
                with open(output_path, 'wb') as f:
                    f.write(audio_bytes)
                return True
            else:
                # Try to create a basic WAV file
                try:
                    # Assume 16kHz, mono, 16-bit PCM
                    sample_rate = 16000
                    channels = 1
                    sample_width = 2

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
                    st.error(f"Error creating WAV file: {str(e)}")
                    return False
    except Exception as e:
        st.error(f"Error converting audio: {str(e)}")
        return False

# Fetch available AI providers
@st.cache_data(ttl=300)
def get_available_providers():
    try:
        response = requests.get(f"{API_BASE_URL}/available_providers", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("providers", ["gemini"])
        else:
            return ["gemini"]
    except:
        return ["gemini"]

# Load available providers
st.session_state.available_providers = get_available_providers()

# Sidebar for settings
with st.sidebar:
    st.subheader("⚙️ Settings")

    # AI Provider Selection
    st.subheader("🧠 AI Provider")
    provider_options = {
        "gemini": "🔮 Google Gemini",
        "openai": "🤖 OpenAI GPT"
    }

    available_options = {
        k: v for k, v in provider_options.items()
        if k in st.session_state.available_providers
    }

    if not available_options:
        st.error("No AI providers available. Please check your API keys.")
        st.stop()

    selected_provider = st.radio(
        "Choose AI Provider:",
        options=list(available_options.keys()),
        format_func=lambda x: available_options[x],
        index=0 if st.session_state.ai_provider not in available_options else list(available_options.keys()).index(st.session_state.ai_provider)
    )

    if selected_provider != st.session_state.ai_provider:
        st.session_state.ai_provider = selected_provider
        st.rerun()

    st.divider()

    # Speech Models Status
    st.subheader("🎤 Speech Models")
    if st.button("🔄 Refresh Speech Status", key="refresh_speech_status"):
        st.session_state.speech_status = get_speech_status()
        st.rerun()

    if st.button("🧪 Test Speech Models", key="test_speech_models"):
        try:
            response = requests.get(f"{API_BASE_URL}/test-speech-models", timeout=30)
            if response.status_code == 200:
                test_results = response.json()
                with st.expander("🧪 Speech Models Test Results", expanded=True):
                    st.json(test_results)
            else:
                st.error("Failed to test speech models")
        except Exception as e:
            st.error(f"Error testing speech models: {str(e)}")

    if st.session_state.speech_status is None:
        st.session_state.speech_status = get_speech_status()

    if st.session_state.speech_status:
        # Speech-to-Text Status
        stt_status = st.session_state.speech_status.get("speech_to_text", {})
        if stt_status.get("loaded", False):
            st.success("✅ Speech-to-Text Ready")
        else:
            st.error("❌ Speech-to-Text Not Available")

        # Text-to-Speech Status
        tts_status = st.session_state.speech_status.get("text_to_speech", {})
        if tts_status.get("loaded", False):
            st.success("✅ Text-to-Speech Ready")

            # Voice selection
            available_voices = tts_status.get("available_voices", [])
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
    else:
        st.warning("⚠️ Speech status unavailable")

    st.divider()

    # Web Search Section
    st.subheader("🌐 Web Search")
    search_query = st.text_input("Search the web:", placeholder="Enter search query...")
    if st.button("🔍 Search", use_container_width=True):
        if search_query.strip():
            try:
                with st.spinner("Searching..."):
                    response = requests.post(
                        f"{API_BASE_URL}/web_search",
                        params={"query": search_query, "session_id": st.session_state.session_id}
                    )
                    if response.status_code == 200:
                        results = response.json()["results"]
                        with st.expander("Search Results", expanded=True):
                            st.markdown(results)
                    else:
                        st.error("Search failed")
            except Exception as e:
                st.error(f"Search error: {str(e)}")

    st.divider()

    # Session Info
    st.subheader("ℹ️ Session Info")
    st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
    st.write(f"**AI Provider:** {available_options[st.session_state.ai_provider]}")
    st.write(f"**PDF Uploaded:** {'✅' if st.session_state.pdf_uploaded else '❌'}")

    if st.button("🔄 New Session", use_container_width=True):
        try:
            requests.delete(f"{API_BASE_URL}/clear_session/{st.session_state.session_id}")
        except:
            pass

        st.session_state.messages = []
        st.session_state.pdf_uploaded = False
        st.session_state.upload_status = ""
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.attachment_key = 0
        st.success("New session started!")
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📄 Upload Document (Optional)")
    st.info("Upload a PDF to enable document-based Q&A, or chat without documents for general conversation.")

with col2:
    st.subheader("📊 Status")
    try:
        response = requests.get(f"{API_BASE_URL}/session_status/{st.session_state.session_id}", timeout=5)
        if response.status_code == 200:
            status = response.json()
            st.success(f"Agent: {status['agent_type'].title()}")
            if status.get('ai_provider'):
                st.info(f"Provider: {status['ai_provider'].title()}")
        else:
            st.warning("No active session")
    except:
        st.error("Unable to get status")

# PDF Upload
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf",
    key="pdf_uploader",
    help="Upload a PDF document to enable document-specific Q&A"
)

if uploaded_file is not None:
    try:
        with st.spinner("Processing PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            data = {
                "session_id": st.session_state.session_id,
                "ai_provider": st.session_state.ai_provider
            }
            response = requests.post(
                f"{API_BASE_URL}/upload_pdf",
                files=files,
                data=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            st.session_state.pdf_uploaded = True
            st.session_state.upload_status = f"✅ {result['message']}"
            st.success(st.session_state.upload_status)
    except requests.exceptions.RequestException as e:
        st.session_state.upload_status = f"❌ Error uploading PDF: {str(e)}"
        st.error(st.session_state.upload_status)

# Image Upload with Drag and Drop
st.subheader("🖼️ Upload Image for Analysis")
st.info("Drag and drop an image or click to upload for AI analysis with your selected provider.")

# Create columns for drag and drop area
img_col1, img_col2, img_col3 = st.columns([1, 2, 1])

with img_col2:
    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
        help="Upload an image for AI analysis"
    )

# Display uploaded image with analysis option
if uploaded_image is not None:
    try:
        # Display the image
        image = Image.open(uploaded_image)
        st.image(image, caption=f"Uploaded: {uploaded_image.name}", use_column_width=True)

        # Image analysis section
        col1, col2 = st.columns([2, 1])

        with col1:
            analysis_prompt = st.text_area(
                "What would you like to know about this image?",
                placeholder="Describe what you see, analyze the content, extract text, etc.",
                height=100
            )

        with col2:
            st.write("**Image Details:**")
            st.write(f"📁 Name: {uploaded_image.name}")
            st.write(f"📏 Size: {image.size}")
            st.write(f"🎨 Mode: {image.mode}")

        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            if analysis_prompt.strip():
                try:
                    with st.spinner("Analyzing image..."):
                        # Reset file pointer
                        uploaded_image.seek(0)

                        files = {
                            "file": (uploaded_image.name, uploaded_image, f"image/{uploaded_image.type.split('/')[-1]}")
                        }
                        data = {
                            "prompt": analysis_prompt,
                            "session_id": st.session_state.session_id,
                            "ai_provider": st.session_state.ai_provider
                        }

                        response = requests.post(
                            f"{API_BASE_URL}/ai_direct",
                            files=files,
                            data=data,
                            timeout=60
                        )
                        response.raise_for_status()
                        result = response.json()

                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "user",
                            "content": f"🖼️ Image Analysis: {analysis_prompt}"
                        })
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["response"]
                        })

                        st.success("Image analyzed! Check the chat below.")
                        st.rerun()

                except requests.exceptions.RequestException as e:
                    st.error(f"Error analyzing image: {str(e)}")
            else:
                st.warning("Please enter a question about the image.")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Chat Interface
st.subheader("💬 Chat")

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Add audio playback button for bot responses
        if message["role"] == "assistant":
            if st.button(f"🔊 Listen to Response", key=f"listen_{i}"):
                with st.spinner("🔊 Generating speech..."):
                    selected_voice = st.session_state.get("selected_voice", "af_heart")
                    speech_speed = st.session_state.get("speech_speed", 1.0)
                    result = send_text_to_speech(message["content"], voice=selected_voice, speed=speech_speed)
                    if result and result.get("success"):
                        audio_file = result.get("audio_file")
                        if audio_file:
                            filename = os.path.basename(audio_file)
                            audio_url = f"{API_BASE_URL}/audio/{filename}"
                            st.audio(audio_url, format="audio/wav")
                        else:
                            st.error("Failed to generate audio file")
                    else:
                        st.error("Failed to generate speech")

# Chat input section
st.subheader("💬 Chat Input")

# Record button above text input
col_record_above, col_space = st.columns([1, 4])

with col_record_above:
    if st.button("🎤 Record Audio", type="primary", help="Record audio for speech-to-text", key="record_audio_btn"):
        st.session_state.show_recording = True
        st.rerun()

# Text input
user_input = st.text_input(
    "Type your message here (use 'web:' for web search)...",
    key="user_input",
    placeholder="Ask me anything, upload an image, or use 'web:' for web search!"
)

# Audio recording interface
if st.session_state.get("show_recording", False):
    st.markdown("---")
    st.subheader("🎤 Voice Recording")

    if not AUDIOREC_AVAILABLE:
        st.error("❌ Audio recording not available. Please install: `pip install streamlit-audiorec`")
        if st.button("❌ Cancel", key="cancel_no_audiorec"):
            st.session_state.show_recording = False
            st.rerun()
    else:
        st.info("🎤 Click the microphone button below to record your message")
        audio_data = st_audiorec()

        if audio_data:
            st.success("✅ Audio recorded successfully!")
            st.info(f"📊 Audio data size: {len(audio_data)} bytes")

            with st.spinner("🎤 Processing audio..."):
                temp_wav = f"temp_audio_{int(time.time())}.wav"
                try:
                    # Save and standardize audio
                    if create_proper_wav_file(audio_data, temp_wav):
                        # Show the processed audio
                        st.audio(temp_wav, format="audio/wav")

                        # Send to speech-to-text endpoint
                        with open(temp_wav, "rb") as f:
                            files = {
                                "file": (
                                    "recording.wav",
                                    f,
                                    "audio/wav"
                                )
                            }
                            response = requests.post(f"{API_BASE_URL}/speech-to-text", files=files)

                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                transcribed_text = result.get("text", "")
                                st.success(f"✅ Transcribed: {transcribed_text}")

                                # Add to chat history
                                st.session_state.messages.append({"role": "user", "content": transcribed_text})
                                with st.chat_message("user"):
                                    st.markdown(transcribed_text)
                                    st.audio(temp_wav, format="audio/wav")

                                # Process transcribed text as a chat message
                                with st.spinner("🤖 Getting response..."):
                                    data = {
                                        "user_input": transcribed_text,
                                        "session_id": st.session_state.session_id,
                                        "ai_provider": st.session_state.ai_provider
                                    }
                                    response = requests.post(
                                        f"{API_BASE_URL}/chat",
                                        json=data,
                                        timeout=60
                                    )
                                    response.raise_for_status()
                                    result = response.json()
                                    bot_response = result["response"]
                                    st.session_state.messages.append({"role": "assistant", "content": bot_response})

                                st.session_state.show_recording = False
                                st.rerun()
                            else:
                                st.error(f"❌ Speech-to-text failed: {result.get('error', 'Unknown error')}")
                        else:
                            st.error(f"❌ Speech-to-text request failed: {response.text}")
                            # Add debugging information
                            st.info(f"📊 Response status: {response.status_code}")
                            try:
                                error_detail = response.json()
                                st.info(f"📋 Error details: {error_detail}")
                            except:
                                st.info(f"📋 Raw response: {response.text}")
                    else:
                        st.error("❌ Failed to process audio file")
                except Exception as e:
                    st.error(f"❌ Error processing audio: {str(e)}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_wav):
                        try:
                            os.remove(temp_wav)
                        except:
                            pass

        # Cancel button (when no audio recorded yet)
        if audio_data is None:
            if st.button("❌ Cancel Recording", type="secondary", key="cancel_no_audio_btn"):
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
                    response = requests.post(
                        f"{API_BASE_URL}/web_search",
                        params={"query": query, "session_id": st.session_state.session_id}
                    )
                    if response.status_code == 200:
                        results = response.json()["results"]
                        content = f"Response: {results}"
                        st.session_state.messages.append({"role": "assistant", "content": content})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "Error: Web search failed."})
            else:
                st.session_state.messages.append({"role": "assistant", "content": "Error: Web search query is empty."})
        else:
            # Normal chat
            with st.spinner("🤖 Thinking..."):
                data = {
                    "user_input": user_input,
                    "session_id": st.session_state.session_id,
                    "ai_provider": st.session_state.ai_provider
                }
                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json=data,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                bot_response = result["response"]
                st.session_state.messages.append({"role": "assistant", "content": bot_response})

        st.rerun()

# Footer
st.divider()
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔍 Session Details", use_container_width=True):
        try:
            response = requests.get(f"{API_BASE_URL}/conversation_history/{st.session_state.session_id}")
            if response.status_code == 200:
                data = response.json()
                with st.expander("Session Details", expanded=True):
                    st.json(data)
        except:
            st.error("Could not retrieve session details")

with col2:
    if st.button("📋 Copy Session ID", use_container_width=True):
        st.code(st.session_state.session_id)

with col3:
    if st.button("🌐 Test Web Search", use_container_width=True):
        try:
            response = requests.post(
                f"{API_BASE_URL}/web_search",
                params={"query": "latest AI news", "session_id": st.session_state.session_id}
            )
            if response.status_code == 200:
                st.success("Web search working ✅")
            else:
                st.error("Web search failed ❌")
        except:
            st.error("Cannot reach web search ❌")

with col4:
    if st.button("🔊 Test Speech", use_container_width=True):
        try:
            response = requests.get(f"{API_BASE_URL}/speech-status")
            if response.status_code == 200:
                status_data = response.json()
                st.success("Speech API working ✅")
                with st.expander("Speech Status", expanded=True):
                    st.json(status_data)
            else:
                st.error("Speech API failed ❌")
        except:
            st.error("Cannot reach speech API ❌")
