"""Agent Memory Management Service"""
from typing import List, Dict, Optional
from datetime import datetime
import uuid
from sqlalchemy.orm import Session
from src.database.db_models import AgentMemory, Chat, Message
import logging

logger = logging.getLogger(__name__)

class AgentMemoryService:
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def add_memory(
        self, 
        chat_id: str, 
        content: str, 
        memory_type: str = 'short_term',
        importance_score: float = 0.5
    ) -> str:
        """Add memory to agent memory store"""
        memory_id = str(uuid.uuid4())
        memory = AgentMemory(
            id=memory_id,
            chat_id=chat_id,
            memory_type=memory_type,
            content=content,
            importance_score=importance_score
        )
        self.db.add(memory)
        self.db.commit()
        return memory_id
    
    def get_chat_memory(self, chat_id: str, memory_type: Optional[str] = None) -> List[Dict]:
        """Retrieve memories for a chat"""
        query = self.db.query(AgentMemory).filter(AgentMemory.chat_id == chat_id)
        
        if memory_type:
            query = query.filter(AgentMemory.memory_type == memory_type)
        
        memories = query.order_by(AgentMemory.importance_score.desc()).all()
        
        return [{
            'id': m.id,
            'content': m.content,
            'type': m.memory_type,
            'importance': m.importance_score,
            'created_at': m.created_at.isoformat()
        } for m in memories]
    
    def get_context_memory(self, chat_id: str, limit: int = 5) -> str:
        """Get context from memory for LLM"""
        memories = self.get_chat_memory(chat_id)
        context_parts = []
        
        for memory in memories[:limit]:
            context_parts.append(f"- {memory['content']}")
        
        return '\n'.join(context_parts) if context_parts else ""
    
    def store_chat_summary(self, chat_id: str, summary: str, topics: List[str] = None):
        """Store a summary of the chat conversation and topics discussed"""
        topics_str = ", ".join(topics) if topics else "general discussion"
        content = f"Chat Summary: {summary}\nTopics Discussed: {topics_str}"
        return self.add_memory(
            chat_id=chat_id,
            content=content,
            memory_type='chat_summary',
            importance_score=0.9  # High importance for summaries
        )
    
    def get_chat_summary(self, chat_id: str) -> Optional[Dict]:
        """Get the chat summary if available"""
        memories = self.get_chat_memory(chat_id, memory_type='chat_summary')
        return memories[0] if memories else None
    
    def update_importance(self, memory_id: str, new_score: float):
        """Update importance score of memory"""
        memory = self.db.query(AgentMemory).filter(AgentMemory.id == memory_id).first()
        if memory:
            memory.importance_score = new_score
            memory.last_accessed = datetime.utcnow()
            self.db.commit()
    
    def cleanup_old_memories(self, chat_id: str, keep_top_n: int = 10):
        """Remove less important old memories"""
        memories = self.get_chat_memory(chat_id)
        if len(memories) > keep_top_n:
            to_delete = memories[keep_top_n:]
            for memory in to_delete:
                self.db.query(AgentMemory).filter(AgentMemory.id == memory['id']).delete()
            self.db.commit()

