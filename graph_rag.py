# ===== file: graph_rag.py =====
"""
Graph backend code: DocumentProcessor, Neo4jGraphDB, GraphRAGChatbot
Place this file next to app.py and import GraphRAGChatbot from it.
"""
import os
import logging
import asyncio
from pathlib import Path
from typing import List, Optional

import nest_asyncio
from dotenv import load_dotenv
import pypdf
import docx
import csv
from neo4j import GraphDatabase

# LLM / LangChain imports (used only if available)
from langchain_core.documents import Document
try:
    from langchain_ollama import ChatOllama
    from langchain_experimental.graph_transformers import LLMGraphTransformer
except Exception:
    # If these packages are not available at import time, delayed import is handled in code that needs them.
    ChatOllama = None
    LLMGraphTransformer = None

# allow nested event loops when using asyncio.run inside Streamlit
nest_asyncio.apply()

# load .env if present
load_dotenv()

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process multiple document formats into plain text."""

    @staticmethod
    def extract_pdf(file_path: str) -> str:
        try:
            text = ""
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = pypdf.PdfReader(pdf_file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page.extract_text() or ""
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
            return text
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

    @staticmethod
    def extract_docx(file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                    text += "\n"
            return text
        except Exception as e:
            logger.error(f"Error processing DOCX: {e}")
            raise

    @staticmethod
    def extract_csv(file_path: str) -> str:
        try:
            text = ""
            with open(file_path, 'r', encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    text += str(row) + "\n"
            return text
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise

    @staticmethod
    def extract_txt(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as txt_file:
                text = txt_file.read()
            return text
        except Exception as e:
            logger.error(f"Error processing TXT: {e}")
            raise

    @staticmethod
    def process_document(file_path: str) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()
        processors = {
            '.pdf': DocumentProcessor.extract_pdf,
            '.docx': DocumentProcessor.extract_docx,
            '.csv': DocumentProcessor.extract_csv,
            '.txt': DocumentProcessor.extract_txt
        }

        if extension not in processors:
            raise ValueError(f"Unsupported file format: {extension}")

        return processors[extension](str(file_path))


class Neo4jGraphDB:
    """Manage Neo4j graph database operations."""

    def __init__(self, uri: str, user: str, password: str):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def is_connected(self) -> bool:
        return self.driver is not None

    def close(self):
        if self.driver:
            self.driver.close()

    def clear_database(self):
        if not self.is_connected():
            return
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")

    def upload_graph_data(self, graph_documents):
        if not self.is_connected():
            logger.warning("Not connected to Neo4j")
            return

        try:
            with self.driver.session() as session:
                # Upload nodes
                for node in graph_documents[0].nodes:
                    query = f"""
                    MERGE (n:`{node.type}` {{id: $id, name: $name}})
                    """
                    session.run(query, id=node.id, name=node.id)

                # Upload relationships
                for rel in graph_documents[0].relationships:
                    query = f"""
                    MATCH (source:`{rel.source.type}` {{id: $source_id}})
                    MATCH (target:`{rel.target.type}` {{id: $target_id}})
                    MERGE (source)-[r:`{rel.type}`]->(target)
                    """
                    try:
                        session.run(query, source_id=rel.source.id, target_id=rel.target.id)
                    except Exception as e:
                        logger.warning(f"Error creating relationship: {e}")

                logger.info(f"Uploaded {len(graph_documents[0].nodes)} nodes and {len(graph_documents[0].relationships)} relationships")
        except Exception as e:
            logger.error(f"Error uploading graph data: {e}")

    def query_graph(self, cypher_query: str) -> List[dict]:
        if not self.is_connected():
            return []

        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []

    def get_graph_stats(self) -> dict:
        if not self.is_connected():
            return {}

        try:
            with self.driver.session() as session:
                node_count = session.run("MATCH (n) RETURN COUNT(n) as count").single()["count"]
                rel_count = session.run("MATCH ()-[r]-() RETURN COUNT(r) as count").single()["count"]
                return {"nodes": node_count, "relationships": rel_count}
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {"nodes": 0, "relationships": 0}


class GraphRAGChatbot:
    """GraphRAG backend class. Keeps LLM + Neo4j logic separate from Streamlit UI."""

    def __init__(self, model: str = "llama3.2:latest", base_url: str = "http://localhost:11434", temperature: float = 0.6):
        self.document_processor = DocumentProcessor()

        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        self.graph_db = Neo4jGraphDB(neo4j_uri, neo4j_user, neo4j_password)

        self.llm_gpt = None
        self.graph_transformer = None

        # try to initialize LLM; failure is non-fatal for the backend object
        try:
            self.init_llm(model=model, base_url=base_url, temperature=temperature)
        except Exception as e:
            logger.warning(f"Ollama initialization failed in GraphRAGChatbot.__init__: {e}")
            self.llm_gpt = None
            self.graph_transformer = None

    def init_llm(self, model: str = "llama3.2:latest", base_url: str = "http://localhost:11434", temperature: float = 0.6, stream: bool = True):
        """
        Initialize or update the local Ollama model and attach an LLMGraphTransformer.
        Call this at runtime to change model or temperature.
        """
        # clean previous
        self.llm_gpt = None
        self.graph_transformer = None

        if not model:
            logger.info("No Ollama model specified; skipping LLM init.")
            return

        if ChatOllama is None or LLMGraphTransformer is None:
            raise RuntimeError("langchain_ollama or LLMGraphTransformer not available. Please install required packages.")

        try:
            logger.info(f"Initializing Ollama model '{model}' at {base_url} (temp={temperature})")
            self.llm_gpt = ChatOllama(
                model=model,
                base_url=base_url,
                temperature=temperature,
                stream=stream
            )
            self.graph_transformer = LLMGraphTransformer(llm=self.llm_gpt)
            logger.info("Ollama LLM and graph transformer initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            self.llm_gpt = None
            self.graph_transformer = None
            raise

    async def extract_and_upload_graph(self, text: str) -> Optional[dict]:
        """Extract graph from text and upload to Neo4j. Returns stats or None."""
        if not self.graph_transformer:
            logger.warning("Graph transformer not available")
            return None

        try:
            documents = [Document(page_content=text)]
            graph_documents = await self.graph_transformer.aconvert_to_graph_documents(documents)

            if graph_documents and len(graph_documents) > 0:
                self.graph_db.upload_graph_data(graph_documents)
                stats = self.graph_db.get_graph_stats()
                return {
                    "nodes": len(graph_documents[0].nodes),
                    "relationships": len(graph_documents[0].relationships),
                    "stats": stats
                }
        except Exception as e:
            logger.error(f"Error extracting graph: {e}")

        return None

    def retrieve_context_from_graph(self, query: str) -> str:
        """Return a short text context pulled from Neo4j for the UI to include in prompts."""
        if not self.graph_db.is_connected():
            return ""

        try:
            cypher_query = f"""
            MATCH (n)-[r]-(m)
            RETURN DISTINCT n.name as source_name, TYPE(r) as rel_type, m.name as target_name
            LIMIT 10
            """
            results = self.graph_db.query_graph(cypher_query)

            if results:
                context = "Knowledge Graph Context:\n"
                for result in results:
                    context += f"- {result.get('source_name', 'Unknown')} {result.get('rel_type', 'RELATED_TO')} {result.get('target_name', 'Unknown')}\n"
                return context
            return ""
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""
