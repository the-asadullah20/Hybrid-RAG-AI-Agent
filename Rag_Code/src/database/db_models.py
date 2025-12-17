"""SQLAlchemy models for SQLite database"""
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from src.config import Config

Base = declarative_base()

class Chat(Base):
    __tablename__ = 'chats'
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    messages = relationship('Message', back_populates='chat', cascade='all, delete-orphan')

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(String, primary_key=True)
    chat_id = Column(String, ForeignKey('chats.id'), nullable=False)
    content = Column(Text, nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    chat = relationship('Chat', back_populates='messages')
    query_log = relationship('QueryLog', back_populates='message', uselist=False)

class QueryLog(Base):
    __tablename__ = 'query_logs'
    
    id = Column(String, primary_key=True)
    message_id = Column(String, ForeignKey('messages.id'), nullable=False)
    query_text = Column(Text, nullable=False)
    query_type = Column(String)  # 'vector', 'graph', 'hybrid', 'web'
    response_time = Column(Float)
    vector_results_count = Column(Integer, default=0)
    graph_results_count = Column(Integer, default=0)
    web_results_count = Column(Integer, default=0)
    confidence_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    message = relationship('Message', back_populates='query_log')

class AgentMemory(Base):
    __tablename__ = 'agent_memory'
    
    id = Column(String, primary_key=True)
    chat_id = Column(String, ForeignKey('chats.id'))
    memory_type = Column(String)  # 'short_term', 'long_term', 'context'
    content = Column(Text, nullable=False)
    importance_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)

class PDFDocument(Base):
    __tablename__ = 'pdf_documents'
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    page_count = Column(Integer)
    extracted_text = Column(Text)
    processed = Column(Integer, default=0)  # 0 or 1

# Database setup
engine = create_engine(f'sqlite:///{Config.SQLITE_DB_PATH}', echo=False)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

