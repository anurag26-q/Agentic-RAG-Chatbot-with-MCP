# agents/ingestion_agent.py
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_docling import DoclingLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mcp.message_protocol import MCPMessage

class IngestionAgent:
    def parse_and_split(self, file_path, trace_id=""):
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = DoclingLoader(file_path)
        elif file_path.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(file_path)
        else:
            raise ValueError("Unsupported file type")

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        return MCPMessage(
            sender="IngestionAgent",
            receiver="RetrievalAgent",
            type_="PARSED_DOCUMENTS",
            payload={"documents": chunks},
            trace_id=trace_id
        )
