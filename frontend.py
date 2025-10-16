
"""
Streamlit app. This file depends on graph_rag.py in the same folder.
Run with: streamlit run app.py
"""
import streamlit as st
import base64
import re
import logging
import os
import asyncio

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from graph_rag import GraphRAGChatbot, DocumentProcessor

# Configure logging for the app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom CSS styling for the application
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    .think-output {
        color: #ffcc00;
        font-style: italic;
    }
    .actual-output {
        color: #00ff00;
        font-weight: bold;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    .graph-info {
        background-color: #2d2d2d;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #ffcc00;
    }
    .user-image {
        max-width: 300px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# Helper to stream and format AI output
def stream_formatted_output(raw_stream, placeholder):
    """Stream AI response token by token and format output."""
    full_response = ""
    for chunk in raw_stream:
        if chunk:
            full_response += chunk
            think_part = ""
            actual_part = ""

            if "<think>" in full_response and "</think>" in full_response:
                think_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
                if think_match:
                    think_part = think_match.group(1)
                    actual_part = full_response.split("</think>")[1].strip()
            elif "<think>" in full_response:
                think_part = full_response.split("<think>")[1]
            else:
                actual_part = full_response

            html_output = ""
            if think_part:
                html_output += f'<div class="think-container"><span class="think-output"><think>{think_part}</think></span></div>'
            if actual_part:
                html_output += f'<div class="actual-container"><span class="actual-output">{actual_part}</span></div>'

            placeholder.markdown(html_output, unsafe_allow_html=True)

    return full_response


# Initialize session state
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm your AI Code Assistant(Debugging, Knowledge Graph) 🖥️👁️"}]

if "graph_rag" not in st.session_state:
    st.session_state.graph_rag = GraphRAGChatbot()

if "graph_stats" not in st.session_state:
    st.session_state.graph_stats = {"nodes": 0, "relationships": 0}

# Application title and caption
st.title("🧠 AI Code Companion with GraphRAG")
# st.caption("🚀 Your Multimodal AI Pair Programmer with Vision, Debugging & Knowledge Graph Integration")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    selected_model = st.selectbox(
        "Choose Ollama Model",
        ["llama3.2:latest", "deepscaler:latest"],
        index=0
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1
    )

    st.divider()

    st.header("📊 GraphRAG Configuration")

    use_graph_rag = st.checkbox("Enable GraphRAG", value=True)

    if use_graph_rag:
        st.write(f"**Graph Status:** {'✅ Connected' if st.session_state.graph_rag.graph_db.is_connected() else '❌ Not Connected'}")

        stats = st.session_state.graph_rag.graph_db.get_graph_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nodes", stats.get("nodes", 0))
        with col2:
            st.metric("Relationships", stats.get("relationships", 0))

        if st.button("🗑️ Clear Knowledge Graph"):
            st.session_state.graph_rag.graph_db.clear_database()
            st.success("Knowledge graph cleared!")
            st.rerun()

    st.divider()

    st.header("📄 Upload Documents for Graph")
    uploaded_docs = st.file_uploader(
        "Upload documents (PDF, DOCX, CSV, TXT)",
        type=["pdf", "docx", "csv", "txt"],
        accept_multiple_files=True,
        key="doc_uploader"
    )

    if uploaded_docs and use_graph_rag:
        if st.button("📈 Process Documents to Graph"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, doc_file in enumerate(uploaded_docs):
                try:
                    status_text.text(f"Processing {doc_file.name}...")

                    temp_path = f"temp_{doc_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(doc_file.getbuffer())

                    text = DocumentProcessor.process_document(temp_path)

                    result = asyncio.run(st.session_state.graph_rag.extract_and_upload_graph(text))

                    if result:
                        st.success(f"✅ {doc_file.name}: {result['nodes']} nodes, {result['relationships']} relationships")

                    os.remove(temp_path)
                    progress_bar.progress((idx + 1) / len(uploaded_docs))

                except Exception as e:
                    st.error(f"Error processing {doc_file.name}: {str(e)}")

    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐛 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    - 👁️ Image Analysis
    - 🧠 GraphRAG Knowledge
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/) | [Neo4j](https://neo4j.com/)")


# Initialize LLM engine with Ollama (used for prompt pipeline)
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=temperature,
    stream=True
)

# Enhanced system prompt with GraphRAG context
system_prompt = """ 
Before you answer, produce clear plan and outline. Then give the final answer.
You are an expert specialist(IQ 150)(advanced reasoning) in designing and building knowledge graphs.
Use knowledge from the graph database to enhance your responses with relevant context.
don't halucinate any information.
"""

# Create chat interface
chat_container = st.container()

# Display chat messages
with chat_container:
    for msg in st.session_state.message_log:
        with st.chat_message(msg["role"]):
            if msg["role"] == "ai":
                st.markdown(msg["content"], unsafe_allow_html=True)
            else:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item["type"] == "text":
                            st.markdown(item["text"])
                        elif item["type"] == "image_url":
                            st.markdown(f'<img src="{item["image_url"]}" class="user-image" alt="Uploaded Image">', unsafe_allow_html=True)
                else:
                    st.markdown(msg["content"])


# User input section
# col1, col2 = st.columns([4, 3])
# with col1:
#     user_query = st.chat_input("Type your coding question here...")
user_query = st.chat_input("Type your coding question here...")
# with col2:
    # uploaded_image = st.file_uploader("📸", type=["jpg", "png", "jpeg"], key="image_uploader", label_visibility="collapsed")


# Function to build prompt chain
def build_prompt_chain():
    prompt_sequence = [SystemMessage(content=system_prompt)]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessage(content=msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)


# Process user input and generate response
if user_query:
    with st.spinner("🧠 Processing with GraphRAG..."):
        # if uploaded_image is not None:
        #     file_type = uploaded_image.type
        #     image_data = uploaded_image.read()
        #     image_base64 = base64.b64encode(image_data).decode("utf-8")
        #     content = [
        #         {"type": "text", "text": user_query},
        #         {"type": "image_url", "image_url": f"data:{file_type};base64,{image_base64}"}
        #     ]
        # else:
        #     content = user_query

        graph_context = ""
        if use_graph_rag:
            graph_context = st.session_state.graph_rag.retrieve_context_from_graph(user_query)
            if graph_context:
                full_query = f"{user_query}\n\n{graph_context}"
            else:
                full_query = user_query
        else:
            full_query = user_query

        # Initialize content as full_query (since image upload is commented out)
        content = full_query

        if isinstance(content, list):
            content[0]["text"] = full_query
        else:
            content = full_query

        st.session_state.message_log.append({"role": "user", "content": content})

        prompt_chain = build_prompt_chain()
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()

        with st.chat_message("ai"):
            response_placeholder = st.empty()
            raw_stream = processing_pipeline.stream({})
            ai_response = stream_formatted_output(raw_stream, response_placeholder)

        st.session_state.message_log.append({"role": "ai", "content": ai_response})

        # if use_graph_rag and st.session_state.graph_rag.graph_transformer:
        #     try:
        #         combined_text = f"User Query: {user_query}\n\nAI Response: {ai_response}"
        #         result = asyncio.run(st.session_state.graph_rag.extract_and_upload_graph(combined_text))
        #         if result:
        #             with st.chat_message("assistant"):
        #                 st.markdown(f"""
        #                 <div class="graph-info">
        #                 📊 <strong>Graph Updated:</strong> +{result['nodes']} nodes, +{result['relationships']} relationships<br>
        #                 <strong>Total Graph Size:</strong> {result['stats']['nodes']} nodes, {result['stats']['relationships']} relationships
        #                 </div>
        #                 """, unsafe_allow_html=True)
        #     except Exception as e:
        #         logger.warning(f"Could not auto-extract graph: {e}")

        st.rerun()
