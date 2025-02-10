import base64
import json
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage, SystemMessage
from graph import invoke_our_graph as invoke_gpt_graph
from graph_gemini import invoke_our_graph as invoke_gemini_graph
from graph_anthropic import invoke_our_graph as invoke_anthropic_graph
from util import display_message as display_message_gpt, render_conversation_history as render_conversation_history_gpt, get_conversation_summary as get_conversation_summary_gpt
from util_gemini import display_message as display_message_gemini, render_conversation_history as render_conversation_history_gemini, get_conversation_summary as get_conversation_summary_gemini
from util_anthropic import display_message as display_message_anthropic, render_conversation_history as render_conversation_history_anthropic, get_conversation_summary as get_conversation_summary_anthropic
from speech_to_text import input_from_mic, convert_text_to_speech
from datetime import datetime
from prompt import system_prompt

# Load environment variables
load_dotenv()

# Initialize session state if not present
if "page" not in st.session_state:
    st.session_state["page"] = "OpenAI"

if "final_state" not in st.session_state:
    st.session_state["final_state"] = {
        "messages": [SystemMessage(content=system_prompt)]
    }
if "audio_transcription" not in st.session_state:
    st.session_state["audio_transcription"] = None

# Add custom CSS with theme-aware styling
st.markdown("""
<style>
    /* Custom styling for the main title */
    .main-title {
        text-align: center;
        color: #1E88E5;
        padding: 1rem 0;
        border-bottom: 2px solid #1E88E5;
        margin-bottom: 2rem;
    }
    
    /* Provider selection styling */
    .provider-section {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        background-color: #1E88E5;
        color: white;
    }
    
    .new-chat-button > button {
        background-color: #4CAF50 !important;
        margin: 1rem 0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #E3F2FD;
    }
    
    .ai-message {
        background-color: #F5F5F5;
    }
    
    /* Form styling */
    .stForm {
        background-color: var(--background-color);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Image upload area styling */
    [data-testid="stFileUploader"] {
        background-color: var(--background-color);
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed var(--primary-color);
    }
    
    /* Dark mode styles */
    @media (prefers-color-scheme: dark) {
        .main-title {
            color: #90CAF9;
            border-bottom-color: #90CAF9;
        }
        
        .provider-section {
            background-color: #1a1a1a;
        }
        
        .user-message {
            background-color: #263238;
        }
        
        .ai-message {
            background-color: #1a1a1a;
        }
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
        padding: 10px 20px;
        border: 2px solid #1E88E5;
    }
    
    /* Submit button hover effect */
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    /* Tab hover effect */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f0f8ff;
        transition: background-color 0.3s ease;
    }

    /* API key setup styling */
    .api-key-setup {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid var(--primary-color);
    }

    /* Audio instructions styling */
    .audio-instructions {
        background-color: var(--secondary-background-color);
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        border: 1px solid var(--primary-color);
    }
    
    /* Main chat interface title styling */
    .chat-title {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 15px;
        background: linear-gradient(90deg, var(--secondary-background-color), transparent);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .robot-icon {
        font-size: 24px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    .provider-name {
        color: var(--primary-color);
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Set up Streamlit layout
st.markdown('<h1 class="main-title">🤖 Spike Sorting Agent</h1>', unsafe_allow_html=True)

# Navigation in sidebar with improved styling
st.sidebar.markdown('<div class="provider-section">', unsafe_allow_html=True)
st.sidebar.title("🎯 Navigation")

PROVIDER_CONFIGS = {
    "OpenAI": {
        "icon": "🟢",
        "color": "#10A37F",
        "hover_color": "#1A7F64"
    },
    "Anthropic": {
        "icon": "🟣",
        "color": "#6B48FF",
        "hover_color": "#5438CC"
    },
    "Gemini": {
        "icon": "🔵",
        "color": "#1E88E5",
        "hover_color": "#1976D2"
    }
}

# Then update the provider selection
provider_options = [f"{PROVIDER_CONFIGS[p]['icon']} {p}" for p in ["OpenAI", "Anthropic", "Gemini"]]
selected = st.sidebar.radio("Select LLM Provider Family", provider_options)
page = selected.split(" ")[1]  # Extract provider name without emoji
st.session_state["page"] = page

# Set provider-specific functions and variables
if page == "OpenAI":
    HISTORY_DIR = "conversation_histories_gpt"
    invoke_graph = invoke_gpt_graph
    display_message = display_message_gpt
    render_conversation_history = render_conversation_history_gpt
    get_conversation_summary = get_conversation_summary_gpt
    available_models = ["gpt-4o", "gpt-4o-mini", "o1", "gpt-4-turbo", "gpt-3.5-turbo"]
elif page == "Gemini":
    HISTORY_DIR = "conversation_histories_gemini"
    invoke_graph = invoke_gemini_graph
    display_message = display_message_gemini
    render_conversation_history = render_conversation_history_gemini
    get_conversation_summary = get_conversation_summary_gemini
    available_models = [
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro"
    ]
else:  # Anthropic
    HISTORY_DIR = "conversation_histories_anthropic"
    invoke_graph = invoke_anthropic_graph
    display_message = display_message_anthropic
    render_conversation_history = render_conversation_history_anthropic
    get_conversation_summary = get_conversation_summary_anthropic
    available_models = [
        "claude_3_5_sonnet_20241022",
        "claude_3_5_sonnet",
        "claude_3_5_haiku",
        "claude_3_opus",
        "claude_3_haiku",
        "claude_3_sonnet"
    ]

# Add model selection with improved styling
selected_model = st.sidebar.selectbox(f"🔧 Select {page} Model:", available_models, index=0)

# Add New Chat button with custom styling
st.sidebar.markdown('<div class="new-chat-button">', unsafe_allow_html=True)
if st.sidebar.button("🔄 Start New Chat"):
    st.session_state["final_state"] = {
        "messages": [SystemMessage(content=system_prompt)]
    }
    st.session_state["last_summary_point"] = 0
    st.session_state["last_summary_title"] = "Default Title"
    st.session_state["last_summary_summary"] = "This is the default summary for short conversations."
    st.rerun()
st.sidebar.markdown('</div>', unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Set up environment for API keys
if page == "OpenAI" and not os.getenv('OPENAI_API_KEY'):
    st.sidebar.markdown("""
        <div class="api-key-setup">
            <h3>🔑 OpenAI API Key Setup</h3>
        </div>
    """, unsafe_allow_html=True)
    api_key = st.sidebar.text_input(label="OpenAI API Key", type="password", label_visibility="collapsed")
    os.environ["OPENAI_API_KEY"] = api_key
    if not api_key:
        st.info("Please enter your OpenAI API Key in the sidebar.")
        st.stop()
elif page == "Gemini" and not os.getenv('GOOGLE_API_KEY'):
    st.sidebar.header("Google API Key Setup")
    api_key = st.sidebar.text_input(label="Google API Key", type="password", label_visibility="collapsed")
    os.environ["GOOGLE_API_KEY"] = api_key
    if not api_key:
        st.info("Please enter your Google API Key in the sidebar.")
        st.stop()
elif page == "Anthropic" and not os.getenv('ANTHROPIC_API_KEY'):
    st.sidebar.header("Anthropic API Key Setup")
    api_key = st.sidebar.text_input(label="Anthropic API Key", type="password", label_visibility="collapsed")
    os.environ["ANTHROPIC_API_KEY"] = api_key
    if not api_key:
        st.info("Please enter your Anthropic API Key in the sidebar.")
        st.stop()

os.makedirs(HISTORY_DIR, exist_ok=True)

# Helper Functions for Conversation Management
def save_history(title: str, summary: str):
    """Save the current conversation history to a file with title and summary."""
    history_data = {
        "title": title,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
        "messages": messages_to_dicts(st.session_state["final_state"]["messages"])
    }
    filename = f"{HISTORY_DIR}/{title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(history_data, f)
    st.rerun()

def load_all_histories():
    """Load all saved conversation histories as a list of metadata for display."""
    histories = []
    for file in os.listdir(HISTORY_DIR):
        if file.endswith(".json"):
            with open(os.path.join(HISTORY_DIR, file), "r") as f:
                history = json.load(f)
                histories.append({
                    "title": history["title"],
                    "summary": history["summary"],
                    "timestamp": history["timestamp"],
                    "filename": file
                })
    return sorted(histories, key=lambda x: x["timestamp"], reverse=True)

def load_history(filename: str):
    """Load a specific conversation history file into session state."""
    try:
        with open(os.path.join(HISTORY_DIR, filename), "r") as f:
            history_data = json.load(f)
            st.session_state["final_state"]["messages"] = dicts_to_messages(history_data["messages"])
        st.sidebar.success(f"Conversation '{history_data['title']}' loaded successfully")
    except FileNotFoundError:
        st.sidebar.error("Conversation history not found.")

def delete_history(filename: str):
    """Delete a specific conversation history file."""
    os.remove(os.path.join(HISTORY_DIR, filename))
    st.sidebar.success("Conversation history deleted.")
    st.rerun()

# Convert messages to serializable dictionaries and vice versa
def messages_to_dicts(messages):
    return [msg.dict() for msg in messages]

def dicts_to_messages(dicts):
    reconstructed_messages = []
    for d in dicts:
        if d["type"] == "ai":
            reconstructed_messages.append(AIMessage(**d))
        elif d["type"] == "human":
            reconstructed_messages.append(HumanMessage(**d))
        elif d["type"] == "tool":
            reconstructed_messages.append(ToolMessage(**d))
    return reconstructed_messages

# Organize Sidebar with Tabs and improved styling
st.sidebar.title("⚙️ Settings")
tab1, tab2, tab3 = st.sidebar.tabs(["💬 Conversation", "🎤 Voice", "🖼️ Image"])

# Initialize session state variables
if "last_summary_point" not in st.session_state:
    st.session_state["last_summary_point"] = 0
if "last_summary_title" not in st.session_state:
    st.session_state["last_summary_title"] = "Default Title"
if "last_summary_summary" not in st.session_state:
    st.session_state["last_summary_summary"] = "This is the default summary for short conversations."

# Tab 1: Conversation Management
with tab1:
    st.subheader("History")
    histories = load_all_histories()
    if histories:
        st.markdown("### Saved Histories")
        for history in histories:
            with st.expander(f"{history['title']} ({history['timestamp'][:10]})"):
                st.write(history["summary"])
                if st.button("Load", key=f"load_{history['filename']}"):
                    load_history(history["filename"])
                if st.button("Delete", key=f"delete_{history['filename']}"):
                    delete_history(history["filename"])

    # Determine title and summary based on message count and last summary point
    message_count = len(st.session_state["final_state"]["messages"])
    if message_count > 5 and (message_count - 5) % 10 == 0 and message_count != st.session_state["last_summary_point"]:
        generated_title, generated_summary = get_conversation_summary(st.session_state["final_state"]["messages"])
        st.session_state["last_summary_title"] = generated_title
        st.session_state["last_summary_summary"] = generated_summary
        st.session_state["last_summary_point"] = message_count
    elif message_count <= 5:
        st.session_state["last_summary_title"] = "Default Title"
        st.session_state["last_summary_summary"] = "This is the default summary for short conversations."

    title = st.text_input("Conversation Title", value=st.session_state["last_summary_title"])
    summary = st.text_area("Conversation Summary", value=st.session_state["last_summary_summary"])

    if st.button("Save Conversation"):
        save_history(title, summary)
        st.sidebar.success(f"Conversation saved as '{title}'")

# Tab 2: Voice Options
with tab2:
    st.subheader("Audio Options")
    use_audio_input = st.checkbox("Enable Voice Input", value=False)
    if use_audio_input:
        with st.form("audio_input_form", clear_on_submit=True):
            st.markdown("""
                <div class="audio-instructions">
                    <strong>Instructions for Recording Audio:</strong>
                    <ol style="padding-left: 20px; line-height: 1.5;">
                        <li>Click <strong>Submit Audio</strong> below to activate the audio recorder.</li>
                        <li>Once activated, click <strong>Start Recording</strong> to begin capturing audio.</li>
                        <li>When finished, click <strong>Stop</strong> to end the recording.</li>
                        <li>Finally, click <strong>Submit Audio</strong> again to use the recorded audio.</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)
            submitted_audio = st.form_submit_button("Submit Audio")
            if submitted_audio:
                audio_transcript = input_from_mic()
                if audio_transcript:
                    st.session_state["audio_transcription"] = audio_transcript
                    prompt = st.session_state["audio_transcription"]
                else:
                    st.session_state["audio_transcription"] = None

    use_voice_response = st.checkbox("Enable Voice Response", value=False)
    if use_voice_response:
        st.write("If the voice response is too long, a summarized version will generate.")

# Tab 3: Image Upload
with tab3:
    st.subheader("Image")
    with st.form("image_upload_form", clear_on_submit=True):
        uploaded_images = st.file_uploader("Upload one or more images (optional)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        submitted = st.form_submit_button("Submit Images")
        if submitted:
            if uploaded_images:
                st.session_state["uploaded_images_data"] = [
                    base64.b64encode(image.read()).decode("utf-8") for image in uploaded_images
                ]
            else:
                st.session_state["uploaded_images_data"] = []

# Initialize prompt variable
prompt = st.session_state.get("audio_transcription")

# Main chat interface
st.markdown(f"""
    <div class="chat-title">
        <span class="robot-icon">🤖</span>
        <span>Chat with Spike Sorting Agent</span>
    </div>
""", unsafe_allow_html=True)

render_conversation_history(st.session_state["final_state"]["messages"][0:])

# Capture text input if no audio input
if prompt is None:
    prompt = st.chat_input()

# Process new user input if available
if prompt:
    content_list = [{"type": "text", "text": prompt}]
    if "uploaded_images_data" in st.session_state and st.session_state["uploaded_images_data"]:
        content_list.extend([
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
            for img_data in st.session_state["uploaded_images_data"]
        ])
        st.session_state["uploaded_images_data"] = []
    
    user_message = HumanMessage(content=content_list)
    st.session_state["final_state"]["messages"].append(user_message)
    render_conversation_history([user_message])

    with st.spinner(f"Agent is thinking..."):
        previous_message_count = len(st.session_state["final_state"]["messages"])
        updated_state = invoke_graph(st.session_state["final_state"]["messages"], selected_model)
    
    st.session_state["final_state"] = updated_state
    new_messages = st.session_state["final_state"]["messages"][previous_message_count:]
    
    if st.session_state.get("render_last_message", True):
        render_conversation_history([st.session_state["final_state"]["messages"][-1]])
    
    if use_voice_response:
        audio_file = convert_text_to_speech(new_messages[-1].content)
        if audio_file:
            st.audio(audio_file)
    
    st.session_state["audio_transcription"] = None 



