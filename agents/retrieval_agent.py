# agents/retrieval_agent.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from mcp.message_protocol import MCPMessage
import os
import shutil

class RetrievalAgent:
    def __init__(self, api_key, persist_dir="./faiss_index"):
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        self.persist_dir = persist_dir

        # Clean old index for debug (you can remove this in prod)
        if os.path.exists(persist_dir):
            print("🧹 Removing old FAISS index")
            shutil.rmtree(persist_dir)

        self.vector_store = None  # Initialize after first add

    def add_documents(self, mcp_msg):
        docs = mcp_msg.payload["documents"]
        print(f"🧪 Embedding test on {len(docs)} documents...")

        filtered_docs = []
        for i, doc in enumerate(docs):
            try:
                _ = self.embedding_model.embed_query(doc.page_content)
                print(f"✅ Chunk {i} embedded. Length: {len(doc.page_content)}")
                filtered_docs.append(doc)
            except Exception as e:
                print(f"❌ Chunk {i} failed to embed: {e}")

        print(f"✅ {len(filtered_docs)} documents passed embedding. Adding to FAISS...")

        try:
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(filtered_docs, self.embedding_model)
            else:
                self.vector_store.add_documents(filtered_docs)

            self.vector_store.save_local(self.persist_dir)
            print("💾 FAISS index saved.")

        except Exception as e:
            print(f"❌ Failed to add to FAISS: {e}")
            raise e

    def search(self, query, trace_id):
        try:
            if self.vector_store is None:
                self.vector_store = FAISS.load_local(self.persist_dir, self.embedding_model)

            print(f"🔎 Searching FAISS for query: {query}")
            results = self.vector_store.similarity_search(query, k=3)
            print(f"✅ Retrieved {len(results)} results")

            return MCPMessage(
                sender="RetrievalAgent",
                receiver="LLMResponseAgent",
                type_="RETRIEVAL_RESULT",
                payload={"top_chunks": results, "query": query},
                trace_id=trace_id
            )
        except Exception as e:
            print(f"❌ FAISS search failed: {e}")
            raise e
