# agents/coordinator_agent.py
from mcp.message_protocol import MCPMessage

class CoordinatorAgent:
    def __init__(self, ingestion_agent, retrieval_agent, llm_agent):
        self.ingestion_agent = ingestion_agent
        self.retrieval_agent = retrieval_agent
        self.llm_agent = llm_agent

    def handle_file_and_query(self, file_path, query):
        try:
            print("🔍 Step 1: Parsing and splitting document...")
            mcp_doc_msg = self.ingestion_agent.parse_and_split(file_path)
            print(f"✅ Parsed {len(mcp_doc_msg.payload['documents'])} chunks")

            print("📦 Step 2: Adding documents to vector store...")
            self.retrieval_agent.add_documents(mcp_doc_msg)
            print("✅ Documents added to vector store.")

            print("🔎 Step 3: Performing similarity search...")
            mcp_search_msg = self.retrieval_agent.search(query, trace_id=mcp_doc_msg.trace_id)
            print(f"✅ Retrieved {len(mcp_search_msg.payload['top_chunks'])} relevant chunks")

            print("🤖 Step 4: Generating answer from LLM...")
            try:
                answer = self.llm_agent.generate_answer(mcp_search_msg)
                print("✅ LLM generated the answer.")
            except Exception as e:
                print(f"❌ Error generating LLM answer: {e}")
                raise e

            # EXTRACT SOURCES FROM THE MESSAGE
            sources = mcp_search_msg.payload["top_chunks"]
            print(f"🔍 DEBUG: Found {len(sources)} sources")
            
            # DEBUG: Print first source to verify
            if sources:
                print(f"🔍 DEBUG: First source preview: {sources[0].page_content[:100]}...")
            
            return answer, sources

        except Exception as e:
            print(f"❌ Error in CoordinatorAgent: {e}")
            raise e