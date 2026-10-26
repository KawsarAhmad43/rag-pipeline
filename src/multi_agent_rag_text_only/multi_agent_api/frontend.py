import streamlit as st
import requests
import json
import time
from pathlib import Path
import tempfile
import os
from io import BytesIO

# Configure the page
st.set_page_config(
    page_title="Multi-Agent RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8003"

def parse_agent_response(response_data):
    """Parse multi-agent response and extract clean, readable content"""
    if not response_data or not isinstance(response_data, list):
        return "No response received."

    clean_responses = []

    for step in response_data:
        if isinstance(step, dict):
            # Check if this is a simple RAG response
            if "messages" in step and len(step["messages"]) > 0:
                for message in step["messages"]:
                    if isinstance(message, dict) and "content" in message:
                        content = message["content"]
                        # Check if it's a simple RAG response
                        if content.startswith("Simple RAG Response: "):
                            clean_content = content.replace("Simple RAG Response: ", "")
                            clean_responses.append(clean_content)
                        # Check if it's a regular agent response
                        elif "name" in message and message["name"]:
                            # Skip technical step messages
                            if not content.startswith("Step:") and not content.startswith("Selected:"):
                                clean_responses.append(content)
                        # Regular content without agent name
                        elif not content.startswith("Step:") and not content.startswith("Selected:"):
                            clean_responses.append(content)

            # Handle other response formats
            elif "content" in step:
                content = step["content"]
                if not content.startswith("Step:") and not content.startswith("Selected:"):
                    clean_responses.append(content)

    # If no clean responses found, try to extract any readable content
    if not clean_responses:
        for step in response_data:
            if isinstance(step, str):
                if not step.startswith("Step:") and not step.startswith("Selected:"):
                    clean_responses.append(step)
            elif isinstance(step, dict):
                # Try to find any string content in the dict
                for key, value in step.items():
                    if isinstance(value, str) and len(value) > 20:  # Only meaningful content
                        if not value.startswith("Step:") and not value.startswith("Selected:"):
                            clean_responses.append(value)

    # Combine all responses
    if clean_responses:
        return "\n\n".join(clean_responses)
    else:
        return "Processing completed. Please try a different query."

def get_available_models():
    """Get available models from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        response.raise_for_status()
        return response.json()["models"]
    except:
        return {}

def get_current_model():
    """Get current model from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/current-model")
        response.raise_for_status()
        return response.json()
    except:
        return {"model": "gpt-4o-mini", "provider": "openai"}

def set_model(model_name):
    """Set the current model"""
    try:
        data = {"model": model_name}
        response = requests.post(f"{API_BASE_URL}/set-model", json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error setting model: {str(e)}")
        return None

def upload_document(file):
    """Upload a document to the API"""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/upload-document", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading document: {str(e)}")
        return None

def process_message(message):
    """Process a message through the API"""
    try:
        data = {"message": message}
        response = requests.post(f"{API_BASE_URL}/process-message", json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            error_detail = e.response.json().get('detail', 'Unknown error')
            st.error(f"⚠️ {error_detail}")
        else:
            st.error(f"Error processing message: {str(e)}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error processing message: {str(e)}")
        return None

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except:
        return False

# Main UI
# Add some styling
st.markdown("""
<style>
    /* Buttons */
    .stButton > button {
        width: 100%;
        margin-bottom: 10px;
        background: linear-gradient(45deg, #ff6b6b, #ff8e53);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 12px;
        font-weight: bold;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        background: linear-gradient(45deg, #ff8e53, #ff6b6b);
    }

    /* File Uploader */
    .stFileUploader {
        border: 2px dashed #ff6b6b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background-color: #fff5f5;
        transition: border-color 0.3s;
    }
    .stFileUploader:hover {
        border-color: #ff8e53;
        background-color: #fef0e7;
    }

    /* Chat Messages */
    .stChatMessage {
        background: linear-gradient(135deg, #e0eafc, #cfdef3);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        color: #1a1a1a !important;
    }

/* Alerts */
.stSuccess {
    background: linear-gradient(90deg, #1c7a33, #249645);
    border-color: #1c7a33;
    color: rgb(0, 0, 0) !important;
    border-radius: 8px;
    padding: 12px;
}
.stWarning {
    background: linear-gradient(90deg, #cc9a22, #c49227);
    border-color: #cc9a22;
    color: #000000 !important;
    border-radius: 8px;
    padding: 12px;
}
.stInfo {
    background: linear-gradient(90deg, #107a8a, #1296a3);
    border-color: #107a8a;
    color: rgb(0, 0, 0) !important;
    border-radius: 8px;
    padding: 12px;
}

    /* Headers */
    .main-header {
        background: linear-gradient(90deg, #ff4b91, #6b48ff);
        padding: 20px;
        border-radius: 12px;
        color: white !important;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .sidebar-header {
        background: linear-gradient(135deg, #6b48ff, #ff4b91);
        padding: 15px;
        border-radius: 8px;
        color: white !important;
        text-align: center;
        margin-bottom: 20px;
    }

    /* General Text Styling */
    .stApp, .stMarkdown, .stChatMessage, .stChatMessageContent, .stTextInput > div > div > input {
        color: #1a1a1a !important;
    }
    .stMarkdown p, .stChatMessage p, .stChatMessage div, .element-container .stMarkdown {
        color: #1a1a1a !important;
    }

    /* Chat Input */
    .stTextInput > div > div > input {
        border: 2px solid #ff6b6b;
        border-radius: 8px;
        padding: 10px;
        transition: border-color 0.3s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #ff8e53;
        box-shadow: 0 0 5px rgba(255, 107, 107, 0.5);
    }

    /* Global Background and Animations */
    .stApp {
        background: linear-gradient(180deg, #f8f9ff, #e0eafc);
        transition: background 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 Multi-Agent RAG System</h1>
    <p>Upload a document and chat with our AI agents to analyze and understand your content</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # API Status
    if check_api_health():
        st.success("✅ API Connected")
    else:
        st.error("❌ API Not Connected")
        st.info("Please start the FastAPI server first:")
        st.code("python main.py")

    # Model Selection
    st.subheader("�� Model Selection")

    # Get available models
    models = get_available_models()
    current_model_info = get_current_model()

    if models:
        # Create model selection interface
        selected_provider = st.selectbox(
            "Choose Provider:",
            list(models.keys()),
            index=0 if current_model_info["provider"] == "openai" else 1
        )

        # Get models for selected provider
        provider_models = models[selected_provider]
        model_names = list(provider_models.keys())
        model_descriptions = [provider_models[model]["description"] for model in model_names]

        # Create model selection with descriptions
        selected_model_index = st.selectbox(
            "Choose Model:",
            range(len(model_names)),
            format_func=lambda x: f"{model_names[x]} - {model_descriptions[x]}",
            index=model_names.index(current_model_info["model"]) if current_model_info["model"] in model_names else 0
        )

        selected_model = model_names[selected_model_index]

        # Show current model info
        st.info(f"**Current Model:** {current_model_info['model']}")
        st.info(f"**Provider:** {current_model_info['provider']}")

        # Change model button
        if st.button("🔄 Change Model"):
            if selected_model != current_model_info["model"]:
                result = set_model(selected_model)
                if result:
                    st.success(f"✅ Model changed to {selected_model}")
                    st.rerun()
                else:
                    st.error("❌ Failed to change model")
            else:
                st.info("Model is already selected")
    else:
        st.warning("⚠️ Could not load available models")

    # Example Messages
    st.subheader("💡 Example Messages")
    example_messages = [
        "Write a LinkedIn post about this document",
        "Summarize the key findings",
        "Create an executive summary",
        "What are the main conclusions?",
        "Generate a blog post about this research"
    ]

    for msg in example_messages:
        if st.button(msg, key=f"example_{msg}"):
            st.session_state.user_message = msg

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📄 Document Upload")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to analyze"
    )

    if uploaded_file is not None:
        # Display file info
        st.info(f"📎 File: {uploaded_file.name}")
        st.info(f"📏 Size: {uploaded_file.size} bytes")

        # Upload button
        if st.button("🚀 Upload Document", type="primary"):
            with st.spinner("Uploading document..."):
                result = upload_document(uploaded_file)
                if result:
                    st.success("✅ Document uploaded successfully!")
                    st.session_state.document_uploaded = True
                    st.session_state.uploaded_filename = uploaded_file.name

with col2:
    st.header("💬 Chat Interface")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "document_uploaded" not in st.session_state:
        st.session_state.document_uploaded = False

    # Check if document is uploaded
    if not st.session_state.document_uploaded:
        st.warning("⚠️ Please upload a document first before starting a conversation.")
    else:
        st.success(f"✅ Ready to chat about: {st.session_state.uploaded_filename}")

        # Show current model info in chat area
        current_model_info = get_current_model()
        st.info(f"🤖 Using: {current_model_info['model']} ({current_model_info['provider']})")

        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about your document..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process with API
            with st.chat_message("assistant"):
                with st.spinner("🤖 Processing with multi-agent system..."):
                    response = process_message(prompt)

                    if response and "response" in response:
                        # Display the response
                        parsed_response = parse_agent_response(response["response"])
                        st.markdown(parsed_response)

                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": parsed_response})
                    elif response is None:
                        # Error already displayed by process_message function
                        pass
                    else:
                        st.error("Failed to get response from the API")

        # Quick action button
        if st.button("🔍 Summarize this research", type="secondary"):
            # Add user message to chat history
            summary_prompt = "Summarize this research"
            st.session_state.messages.append({"role": "user", "content": summary_prompt})
            with st.chat_message("user"):
                st.markdown(summary_prompt)

            # Process with API
            with st.chat_message("assistant"):
                with st.spinner("🤖 Generating summary..."):
                    response = process_message(summary_prompt)

                    if response and "response" in response:
                        # Display the response
                        parsed_response = parse_agent_response(response["response"])
                        st.markdown(parsed_response)

                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": parsed_response})
                    elif response is None:
                        # Error already displayed by process_message function
                        pass
                    else:
                        st.error("Failed to get response from the API")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Multi-Agent RAG System | Powered by LangChain and Streamlit</p>
</div>
""", unsafe_allow_html=True)
