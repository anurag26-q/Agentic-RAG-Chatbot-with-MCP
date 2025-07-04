# agents/llm_response_agent.py
from mcp.message_protocol import MCPMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class LLMResponseAgent:
    def __init__(self, api_key):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=api_key
        )

    def generate_answer(self, mcp_msg):
        chunks = mcp_msg.payload["top_chunks"]
        query = mcp_msg.payload["query"]
        context = "\n".join([doc.page_content for doc in chunks])

        template = """You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question:
{question}
"""
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        answer = chain.invoke({"context": context, "question": query})

        return answer
