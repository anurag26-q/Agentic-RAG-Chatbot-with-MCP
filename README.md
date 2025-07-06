# Agentic RAG Chatbot with MCP

## 📌 Overview

This project implements an **Agentic Retrieval-Augmented Generation (RAG)** chatbot using:

* 🧠 **Agents** for modular tasks
* 📚 **Multi-format document ingestion** (PDF, DOCX, PPTX, CSV, TXT)
* 🧾 **Model Context Protocol (MCP)** for message-based communication
* 🔍 **FAISS** as the vector store
* 🌐 **Gemini (Google Generative AI)** for embeddings & LLM
* 📱 **Streamlit UI** for interaction

---

## 💡 Problem Statement

> Build a document-based chatbot that:
>
> * Accepts multiple document formats
> * Uses an agent-based architecture with clearly defined roles
> * Implements MCP for message passing
> * Responds to user queries with relevant document context

---

## 🧠 Architecture

### 👷 Agents

| Agent              | Responsibility                           |
| ------------------ | ---------------------------------------- |
| `IngestionAgent`   | Load & split documents                   |
| `RetrievalAgent`   | Embed & retrieve relevant chunks         |
| `LLMResponseAgent` | Use Gemini LLM to answer based on chunks |
| `CoordinatorAgent` | Orchestrates the flow via MCP messages   |

### 🔁 MCP Message Format

```json
{
  "sender": "IngestionAgent",
  "receiver": "RetrievalAgent",
  "type": "PARSED_DOCUMENTS",
  "trace_id": "uuid-123",
  "payload": { "documents": [...] }
}
```

---

## 📦 Tech Stack

* LangChain + FAISS
* Gemini Pro (LLM + embeddings)
* Python 3.10+
* Streamlit
* dotenv

---

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/anurag26-q/Agentic-RAG-Chatbot-with-MCP
cd agentic-rag-chatbot
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Add `.env`

```env
GOOGLE_API_KEY=your_google_genai_key
```

### 4. Run the App

```bash
streamlit run ui/app.py
```

---

## 🖥️ Features

* 📄 Upload any document: PDF, DOCX, CSV, TXT, PPTX
* 💬 Ask questions from the uploaded document
* 📂 View response and source chunks
* 🔁 Reset chat with one click

---

## 📊 Folder Structure

```
.
├── agents/
│   ├── ingestion_agent.py
│   ├── retrieval_agent.py
│   ├── llm_response_agent.py
│   └── coordinator_agent.py
├── mcp/
│   └── message.py
├── ui/
│   └── app.py
├── requirements.txt
├── README.md
```

---

---

## 📧 Contact

Created by Anurag Sharma. Reach out at: `anuragparashar111@gmail.com`

---


