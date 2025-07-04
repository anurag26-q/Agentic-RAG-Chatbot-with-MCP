import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP crash

import sys
import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent
from agents.coordinator_agent import CoordinatorAgent

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.title("🧠 Agentic RAG Chatbot with MCP")

# Session state for file + response memory
if "file_path" not in st.session_state:
    st.session_state.file_path = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Agents
ingestion_agent = IngestionAgent()
retrieval_agent = RetrievalAgent(api_key)
llm_agent = LLMResponseAgent(api_key)
coordinator = CoordinatorAgent(ingestion_agent, retrieval_agent, llm_agent)

# Upload section
uploaded_file = st.file_uploader("📄 Upload a document", type=["pdf", "docx", "csv", "txt", "pptx"])
if uploaded_file:
    os.makedirs("documents", exist_ok=True)
    file_path = f"documents/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.file_path = file_path
    st.success("✅ Document uploaded and ready!")

# Query section
query = st.text_input("💬 Enter your question")
ask_button = st.button("🧠 Ask")
reset_button = st.button("🔁 Reset Chat")

# Reset handler
if reset_button:
    st.session_state.clear()
    st.rerun()

# Ask handler
if ask_button:
    if st.session_state.file_path and query.strip():
        try:
            with st.spinner("🤖 Thinking..."):
                answer, sources = coordinator.handle_file_and_query(st.session_state.file_path, query)
                st.session_state.chat_history.append((query, answer, sources))

        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Please upload a document and enter a question.")

# Show chat history
if st.session_state.chat_history:
    st.markdown("### 💬 Chat History")
    for idx, (q, a, srcs) in enumerate(st.session_state.chat_history[::-1], 1):
        st.markdown(f"**Q{idx}: {q}**")
        st.markdown(f"**A{idx}:** {a}")
        st.markdown("<hr>", unsafe_allow_html=True)
