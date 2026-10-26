import streamlit as st
import requests
import uuid
import base64
from PIL import Image
import io

# Configuration
API_BASE_URL = "http://localhost:8001/api/v1"

st.set_page_config(
    page_title="Agentic RAG Chatbot v2.0",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG Chatbot v2.0")
st.caption("Enhanced with OpenAI, Gemini support, and Web Search")

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
        "openai": "�� OpenAI GPT"
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
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input section
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.chat_input(
        f"Chat with {available_options[st.session_state.ai_provider]}...",
        key="chat_input"
    )

with col2:
    attachment_file = st.file_uploader(
        "📎",
        type=["jpg", "jpeg", "png", "pdf"],
        key=f"attachment_{st.session_state.attachment_key}",
        label_visibility="collapsed",
        help="Attach file for analysis"
    )

# Process chat input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        with st.spinner("Thinking..."):
            if attachment_file:
                files = {
                    "file": (
                        attachment_file.name,
                        attachment_file,
                        "application/pdf" if attachment_file.name.lower().endswith('.pdf') else "image/jpeg"
                    )
                }
                data = {
                    "prompt": user_input,
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
                bot_response = result["response"]

                st.session_state.messages.append({
                    "role": "user",
                    "content": f"📎 Attachment: {attachment_file.name}"
                })
                with st.chat_message("user"):
                    st.markdown(f"📎 Attachment: {attachment_file.name}")

                st.session_state.attachment_key += 1

            else:
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
        with st.chat_message("assistant"):
            st.markdown(bot_response)

    except requests.exceptions.RequestException as e:
        error_msg = f"❌ Error: Could not connect to the backend. {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant"):
            st.markdown(error_msg)

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
    if st.button("�� API Status", use_container_width=True):
        try:
            response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/health")
            if response.status_code == 200:
                st.success("API is healthy ✅")
            else:
                st.error("API is unhealthy ❌")
        except:
            st.error("Cannot reach API ❌")
