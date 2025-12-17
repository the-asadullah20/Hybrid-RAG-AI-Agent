"""Configuration settings for the Hybrid RAG AI Agent"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Gemini API
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    
    # Neo4j Configuration
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://127.0.0.1:7687')
    NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'asadullah12345')
    
    # ChromaDB Configuration
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './database/chroma_db')
    CHROMA_COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'hybrid_rag_collection')
    
    # SQLite Database
    SQLITE_DB_PATH = os.getenv('SQLITE_DB_PATH', './database/hybrid_rag.db')
    
    # Application Settings
    MAX_CONTEXT_LENGTH = 4000
    VECTOR_SEARCH_TOP_K = 5
    GRAPH_SEARCH_TOP_K = 5
    TEMPERATURE = 0.7
    
    # Web Scraping (Disabled - using only database and PDF)
    ENABLE_WEB_SCRAPING = False
    MAX_WEB_RESULTS = 3
    
    # PDF Processing
    PDF_UPLOAD_FOLDER = './data/uploads'
    MAX_PDF_SIZE = 10 * 1024 * 1024  # 10MB

