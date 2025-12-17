"""Pydantic models for validation"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class QueryType(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"
    WEB = "web"

class MessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=5000, description="Message content")
    role: MessageRole = Field(default=MessageRole.USER, description="Message role")
    chat_id: Optional[str] = Field(None, description="Chat ID")
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()

class MessageResponse(BaseModel):
    id: str
    content: str
    role: MessageRole
    timestamp: datetime
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    sources: Optional[List[str]] = Field(default_factory=list, description="Source references")
    query_type: Optional[QueryType] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ChatResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class QueryLogResponse(BaseModel):
    id: str
    query_text: str
    query_type: QueryType
    response_time: float
    vector_results_count: int = Field(ge=0)
    graph_results_count: int = Field(ge=0)
    web_results_count: int = Field(ge=0)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AgentResponse(BaseModel):
    """Validated agent response with completeness scoring"""
    content: str = Field(..., min_length=1, description="Response content")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    sources: List[str] = Field(default_factory=list, description="Source references")
    query_type: QueryType
    vector_results_count: int = Field(ge=0)
    graph_results_count: int = Field(ge=0)
    web_results_count: int = Field(ge=0)
    reasoning: Optional[str] = Field(None, description="Agent reasoning process")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Answer completeness")
    
    @validator('completeness_score')
    def calculate_completeness(cls, v, values):
        """Calculate completeness based on content length and sources"""
        content = values.get('content', '')
        sources_count = len(values.get('sources', []))
        
        # Base score from content length
        length_score = min(len(content) / 500, 1.0)
        
        # Bonus for sources
        source_score = min(sources_count * 0.2, 0.4)
        
        return min(length_score + source_score, 1.0)

class PDFUploadResponse(BaseModel):
    id: str
    filename: str
    page_count: int
    processed: bool
    uploaded_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class RecommendationResponse(BaseModel):
    similar_queries: List[str] = Field(default_factory=list)
    suggested_topics: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

