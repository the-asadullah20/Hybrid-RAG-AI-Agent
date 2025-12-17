"""Recommendation Service for similar queries"""
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from src.database.db_models import Message, QueryLog
from src.services.vector_store import VectorStore
import logging
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(self, db_session: Session, vector_store: VectorStore, graph_store=None):
        self.db = db_session
        self.vector_store = vector_store
        self.graph_store = graph_store
    
    def get_similar_queries(self, current_query: str, limit: int = 5) -> List[str]:
        """Find similar past queries using vector search with embeddings"""
        # Get all past queries from database
        past_queries = self.db.query(QueryLog).order_by(QueryLog.timestamp.desc()).limit(100).all()
        
        if not past_queries:
            return []
        
        # Use embedding-based similarity
        try:
            # Get embedding for current query
            current_embedding = self.vector_store.embedder.encode([current_query])[0]
            
            # Calculate similarities with past queries
            similarities = []
            for query_log in past_queries:
                try:
                    query_embedding = self.vector_store.embedder.encode([query_log.query_text])[0]
                    # Cosine similarity
                    similarity = np.dot(current_embedding, query_embedding) / (
                        np.linalg.norm(current_embedding) * np.linalg.norm(query_embedding)
                    )
                    similarities.append((similarity, query_log.query_text))
                except Exception as e:
                    logger.warning(f"Error encoding query: {e}")
                    continue
            
            # Sort by similarity and return top matches
            similarities.sort(reverse=True, key=lambda x: x[0])
            similar = [q[1] for q in similarities[:limit] if q[0] > 0.3]  # Threshold 0.3
            
            return similar
        except Exception as e:
            logger.error(f"Error in vector similarity: {e}")
            # Fallback to keyword matching
            similar = []
            current_lower = current_query.lower()
            for query_log in past_queries:
                query_text = query_log.query_text.lower()
                if any(word in query_text for word in current_lower.split() if len(word) > 3):
                    if query_text not in [s.lower() for s in similar]:
                        similar.append(query_log.query_text)
                        if len(similar) >= limit:
                            break
            return similar
    
    def suggest_similar_documents(self, current_query: str, limit: int = 5) -> List[Dict]:
        """Suggest semantically similar documents using vector search"""
        try:
            # Search for similar documents
            results = self.vector_store.search(current_query, top_k=limit * 2)
            
            # Filter and format results
            suggestions = []
            seen_ids = set()
            for result in results:
                doc_id = result.get('id')
                if doc_id and doc_id not in seen_ids:
                    metadata = result.get('metadata', {})
                    suggestions.append({
                        'id': doc_id,
                        'content': result.get('content', '')[:200] + '...',
                        'source': metadata.get('source', 'unknown'),
                        'filename': metadata.get('filename', ''),
                        'similarity': 1 - result.get('distance', 1.0) if result.get('distance') else 0.8
                    })
                    seen_ids.add(doc_id)
                    if len(suggestions) >= limit:
                        break
            
            return suggestions
        except Exception as e:
            logger.error(f"Error suggesting documents: {e}")
            return []
    
    def suggest_connected_nodes(self, current_query: str, limit: int = 5) -> List[Dict]:
        """Suggest connected nodes from graph database"""
        if not self.graph_store:
            return []
        
        try:
            # Search for nodes related to query
            keyword_results = self.graph_store.search_by_keyword(current_query, limit=limit)
            
            suggestions = []
            for result in keyword_results:
                node_data = result.get('n', {}) or {}
                if isinstance(node_data, dict):
                    # Get related nodes
                    node_id = node_data.get('id') or result.get('id')
                    if node_id:
                        try:
                            related = self.graph_store.get_related_nodes(int(node_id), depth=1)
                            for rel_node in related[:3]:  # Top 3 related
                                rel_data = rel_node.get('related', {})
                                if isinstance(rel_data, dict):
                                    suggestions.append({
                                        'name': rel_data.get('name') or rel_data.get('title', 'Unknown'),
                                        'type': ', '.join(rel_node.get('labels', [])),
                                        'source_node': node_data.get('name') or node_data.get('title', 'Unknown')
                                    })
                                    if len(suggestions) >= limit:
                                        break
                        except:
                            pass
                    
                    if len(suggestions) >= limit:
                        break
            
            return suggestions[:limit]
        except Exception as e:
            logger.error(f"Error suggesting connected nodes: {e}")
            return []
    
    def suggest_related_concepts(self, current_query: str, chat_id: Optional[str] = None, limit: int = 5) -> List[str]:
        """Suggest related concepts based on query and graph relationships"""
        concepts = []
        
        # Extract concepts from graph if available
        if self.graph_store:
            try:
                keyword_results = self.graph_store.search_by_keyword(current_query, limit=10)
                for result in keyword_results:
                    node_data = result.get('n', {}) or {}
                    if isinstance(node_data, dict):
                        # Check if it's a Concept node
                        labels = result.get('labels', [])
                        if 'Concept' in labels:
                            concept_name = node_data.get('name')
                            if concept_name and concept_name not in concepts:
                                concepts.append(concept_name)
                                if len(concepts) >= limit:
                                    break
            except Exception as e:
                logger.warning(f"Error extracting concepts from graph: {e}")
        
        # Also extract from query logs
        if len(concepts) < limit:
            query_logs = self.db.query(QueryLog).order_by(QueryLog.timestamp.desc()).limit(50).all()
            query_texts = [q.query_text.lower() for q in query_logs]
            
            # Find common terms
            all_words = []
            for text in query_texts:
                words = [w for w in text.split() if len(w) > 4]  # Words longer than 4 chars
                all_words.extend(words)
            
            # Get most common terms
            word_freq = Counter(all_words)
            common_terms = [word for word, count in word_freq.most_common(limit * 2) if count > 1]
            
            for term in common_terms:
                if term not in [c.lower() for c in concepts]:
                    concepts.append(term.title())
                    if len(concepts) >= limit:
                        break
        
        return concepts[:limit]
    
    def get_suggested_topics(self, chat_id: str) -> List[str]:
        """Get suggested topics based on chat history"""
        messages = self.db.query(Message).filter(
            Message.chat_id == chat_id,
            Message.role == 'user'
        ).order_by(Message.timestamp.desc()).limit(10).all()
        
        # Extract keywords/topics from messages
        topics = set()
        common_topics = ['data structures', 'algorithms', 'machine learning', 'neural networks', 
                        'rag', 'langchain', 'vector database', 'graph database', 'embeddings']
        
        for message in messages:
            content_lower = message.content.lower()
            for topic in common_topics:
                if topic in content_lower:
                    topics.add(topic.title())
        
        return list(topics)[:5]
    
    def get_recommendations(self, current_query: str, chat_id: str) -> Dict:
        """Get comprehensive recommendations with enhanced features"""
        similar_queries = self.get_similar_queries(current_query)
        suggested_topics = self.get_suggested_topics(chat_id)
        similar_docs = self.suggest_similar_documents(current_query)
        connected_nodes = self.suggest_connected_nodes(current_query)
        related_concepts = self.suggest_related_concepts(current_query, chat_id)
        
        return {
            'similar_queries': similar_queries,
            'suggested_topics': suggested_topics,
            'similar_documents': similar_docs,
            'connected_nodes': connected_nodes,
            'related_concepts': related_concepts,
            'confidence': min(len(similar_queries) / 5.0, 1.0)
        }

