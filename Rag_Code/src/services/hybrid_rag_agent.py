"""Hybrid RAG Agent - Main Agent Service"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
import uuid
import numpy as np
from collections import Counter, defaultdict
from sqlalchemy.orm import Session

from src.services.vector_store import VectorStore
from src.services.graph_store import GraphStore
from src.services.gemini_service import GeminiService
from src.services.web_scraper import WebScraper
from src.services.agent_memory import AgentMemoryService
from src.services.recommendation_service import RecommendationService
from src.models.pydantic_models import AgentResponse, QueryType
from src.database.db_models import Message, QueryLog, Chat
from src.config import Config
import logging

logger = logging.getLogger(__name__)

class HybridRAGAgent:
    def __init__(self, db_session: Session):
        self.db = db_session
        try:
            self.vector_store = VectorStore()
        except Exception as e:
            logger.error(f"Error initializing VectorStore: {e}")
            raise
        
        try:
            self.graph_store = GraphStore()
            logger.info("GraphStore initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing GraphStore: {e}", exc_info=True)
            logger.warning("Will continue without graph features - only vector search will be available")
            logger.warning(f"Neo4j URI: {Config.NEO4J_URI}, User: {Config.NEO4J_USER}")
            self.graph_store = None
        
        try:
            self.gemini = GeminiService()
        except Exception as e:
            logger.error(f"Error initializing GeminiService: {e}")
            raise
        
        self.web_scraper = WebScraper() if Config.ENABLE_WEB_SCRAPING else None
        self.memory_service = AgentMemoryService(db_session)
        self.recommendation_service = RecommendationService(db_session, self.vector_store, self.graph_store)
    
    def process_query(
        self, 
        user_query: str, 
        chat_id: str,
        chat_history: Optional[List[Dict]] = None
    ) -> Dict:
        """Process user query through Hybrid RAG pipeline"""
        start_time = time.time()
        
        # Step 1: Enhanced Vector Search - comprehensive search across ALL PDFs and documents
        # Search with increased top_k to get more comprehensive results
        search_top_k = max(Config.VECTOR_SEARCH_TOP_K * 3, 15)  # Get more results to ensure we find PDF content
        vector_results = self.vector_store.search(user_query, top_k=search_top_k, chat_id=chat_id)
        
        logger.info(f"Vector search returned {len(vector_results)} total results for query: '{user_query[:50]}...'")
        
        # Separate PDF results from general vector results
        pdf_results = [r for r in vector_results if r.get('metadata', {}).get('source') == 'pdf']
        general_results = [r for r in vector_results if r.get('metadata', {}).get('source') != 'pdf']
        
        logger.info(f"Found {len(pdf_results)} PDF results and {len(general_results)} general results")
        
        # Log PDF details for debugging
        if pdf_results:
            pdf_files = {}
            for pdf_result in pdf_results:
                filename = pdf_result.get('metadata', {}).get('filename', 'unknown')
                if filename not in pdf_files:
                    pdf_files[filename] = 0
                pdf_files[filename] += 1
            logger.info(f"PDF results breakdown: {pdf_files}")
            # Log sample PDF content
            for i, pdf_result in enumerate(pdf_results[:2]):
                filename = pdf_result.get('metadata', {}).get('filename', 'unknown')
                content_preview = pdf_result.get('content', '')[:150]
                logger.info(f"PDF result {i+1} from {filename}: {content_preview}...")
        else:
            logger.warning("⚠️ NO PDF RESULTS FOUND! This might indicate PDFs are not indexed or search is not finding them.")
        
        # PRIORITIZE PDF RESULTS: If PDFs exist, use them FIRST, then database
        # Search flow: PDF → Database → Sorry message
        if pdf_results:
            # Check if query is asking for summary/overview - if yes, include ALL PDF chunks
            is_summary_query = any(word in user_query.lower() for word in ['summary', 'summarize', 'overview', 'detailed', 'comprehensive', 'describe', 'explain', 'tell me about'])
            
            combined_results = []
            seen_ids = set()
            
            # FIRST: Add ALL PDF results (they're most important - user uploaded them!)
            # For summary queries, include ALL PDF chunks. For specific queries, include up to 20 chunks
            max_pdf_chunks = len(pdf_results) if is_summary_query else min(len(pdf_results), 20)
            logger.info(f"Query type: {'SUMMARY' if is_summary_query else 'SPECIFIC'} - Including {max_pdf_chunks} PDF chunks")
            
            for pdf_result in pdf_results[:max_pdf_chunks]:
                pdf_id = pdf_result.get('id')
                if pdf_id and pdf_id not in seen_ids:
                    combined_results.append(pdf_result)
                    seen_ids.add(pdf_id)
            
            # THEN: Add general database results (only if we have space)
            for gen_result in general_results[:5]:  # Add up to 5 general results
                gen_id = gen_result.get('id')
                if gen_id and gen_id not in seen_ids and len(combined_results) < search_top_k * 2:
                    combined_results.append(gen_result)
                    seen_ids.add(gen_id)
            
            vector_results = combined_results
            logger.info(f"PRIORITIZED PDF: Using {len([r for r in vector_results if r.get('metadata', {}).get('source') == 'pdf'])} PDF chunks + {len([r for r in vector_results if r.get('metadata', {}).get('source') != 'pdf'])} general results")
        else:
            # No PDFs, use general database results
            vector_results = general_results[:search_top_k]
            logger.info(f"No PDFs found - using {len(vector_results)} general database results")
        
        logger.info(f"Final vector results: {len(vector_results)} (PDFs: {len([r for r in vector_results if r.get('metadata', {}).get('source') == 'pdf'])})")
        
        # Step 2: Graph Search (Generate Cypher and execute)
        graph_results = []
        if self.graph_store:
            try:
                logger.info(f"Starting graph search for query: {user_query[:50]}...")
                
                # First try keyword search - this should always work if Neo4j has data
                keyword_results = self.graph_store.search_by_keyword(user_query, limit=Config.GRAPH_SEARCH_TOP_K)
                logger.info(f"Keyword search returned {len(keyword_results)} results")
                if keyword_results:
                    graph_results.extend(keyword_results)
                
                # Then try Cypher query generation for more specific queries
                try:
                    cypher_query = self.gemini.generate_cypher_query(user_query)
                    logger.info(f"Generated Cypher query: {cypher_query}")
                    cypher_results = self.graph_store.execute_cypher(cypher_query)
                    logger.info(f"Cypher query returned {len(cypher_results)} results")
                    if cypher_results:
                        graph_results.extend(cypher_results)
                except Exception as cypher_error:
                    logger.warning(f"Cypher generation error: {cypher_error}, using keyword search results")
                
                # Remove duplicates more intelligently
                seen = set()
                unique_results = []
                for result in graph_results:
                    # Use node ID for better duplicate detection
                    node_data = result.get('n', {}) or result.get('related', {}) or {}
                    if isinstance(node_data, dict):
                        node_id = node_data.get('id') or result.get('node_id') or str(id(node_data))
                    else:
                        node_id = str(id(result))
                    
                    if node_id not in seen:
                        seen.add(node_id)
                        unique_results.append(result)
                
                # Keep more results for comprehensive search
                graph_results = unique_results[:Config.GRAPH_SEARCH_TOP_K * 2]
                
                logger.info(f"Graph search completed: {len(graph_results)} unique results")
            except Exception as e:
                logger.error(f"Graph search error: {e}", exc_info=True)
                graph_results = []
        else:
            logger.warning("Graph store not initialized - skipping graph search")
        
        # Web scraping disabled - using only database and PDF
        web_results = []
        
        # Step 3: Get memory context for this chat
        memory_context = self.memory_service.get_context_memory(chat_id, limit=3)
        chat_summary = self.memory_service.get_chat_summary(chat_id)
        
        # Step 4: Generate response
        response_data = self.gemini.generate_response(
            user_query,
            vector_results,
            graph_results,
            web_results,
            chat_history,
            memory_context=memory_context,
            chat_summary=chat_summary
        )
        
        # Step 5: Determine query type (prioritize hybrid if both available)
        # IMPORTANT: Count graph search if graph_results exist, even if vector_results also exist
        logger.info(f"Determining query type - vector: {len(vector_results)}, graph: {len(graph_results)}, web: {len(web_results)}")
        
        if graph_results:
            if vector_results:
                query_type = QueryType.HYBRID
                logger.info(f"✓ Using HYBRID search: {len(vector_results)} vector + {len(graph_results)} graph results")
            else:
                query_type = QueryType.GRAPH
                logger.info(f"✓ Using GRAPH search: {len(graph_results)} results")
        elif web_results:
            query_type = QueryType.WEB
            logger.info(f"Using WEB search: {len(web_results)} results")
        elif vector_results:
            query_type = QueryType.VECTOR
            logger.info(f"Using VECTOR search: {len(vector_results)} results")
        else:
            query_type = QueryType.VECTOR  # Default
            logger.warning("No results found from any source")
        
        logger.info(f"Final query_type: {query_type.value}")
        
        # Step 6: Extract sources with better formatting - show all sources found
        sources = []
        seen_pdfs = set()  # Track unique PDFs to avoid duplicates
        
        if vector_results:
            # Include more sources (up to 5) to show all PDFs searched
            for r in vector_results[:5]:
                metadata = r.get('metadata', {})
                if metadata.get('source') == 'pdf':
                    filename = metadata.get('filename', 'uploaded document')
                    # Only add each PDF once (even if multiple chunks from same PDF)
                    if filename not in seen_pdfs:
                        page = metadata.get('page', '')
                        page_str = f" (Page {page})" if page else ""
                        sources.append(f"PDF: {filename}{page_str}")
                        seen_pdfs.add(filename)
                else:
                    # Add vector DB sources
                    doc_id = r.get('id', 'unknown')
                    if doc_id not in [s for s in sources if 'Vector DB' in s]:
                        sources.append(f"Vector DB: {doc_id[:20]}")
        
        if graph_results:
            # Include more graph sources (up to 5)
            seen_graph_nodes = set()
            for i, result in enumerate(graph_results[:5], 1):
                node_data = result.get('n', {}) or result.get('related', {}) or {}
                if isinstance(node_data, dict):
                    name = node_data.get('name', node_data.get('title', f'Node {i}'))
                    # Avoid duplicate graph node names
                    if name not in seen_graph_nodes:
                        sources.append(f"Graph DB: {name}")
                        seen_graph_nodes.add(name)
                else:
                    sources.append(f"Graph DB: Result {i}")
        if web_results:
            sources.extend([f"Web: {r.get('source', 'unknown')}" for r in web_results[:2]])
        
        # Step 7: Calculate confidence score based on database results
        confidence_score = self._calculate_confidence_score(
            len(vector_results),
            len(graph_results),
            len(web_results)
        )
        
        # Additional check: If response indicates no information found, set confidence to 10%
        response_content_lower = response_data['content'].lower()
        no_info_indicators = [
            "cannot answer",
            "unable to provide",
            "do not contain",
            "no information",
            "not found",
            "don't know",
            "i'm sorry, i'm unable",
            "does not contain",
            "doesn't contain",
            "apologize, but",
            "cannot provide",
            "couldn't find",
            "could not find",
            "i cannot find",
            "i'm sorry, i couldn't find"
        ]
        
        # Check if response says it can't find info
        says_cannot_find = any(indicator in response_content_lower for indicator in no_info_indicators)
        
        # If no results OR response says can't find → 10% confidence
        if len(vector_results) == 0 and len(graph_results) == 0:
            confidence_score = 0.1
            logger.info("No results from any source - forcing confidence to 10%")
        elif says_cannot_find:
            # Response says can't find - check if we actually have PDF results
            pdf_results_count = len([r for r in vector_results if r.get('metadata', {}).get('source') == 'pdf'])
            if pdf_results_count > 0:
                # We have PDF results but agent says can't find - this is wrong, but still set to 10%
                logger.warning(f"Agent says can't find but we have {pdf_results_count} PDF results - setting confidence to 10%")
            confidence_score = 0.1
            logger.info("Response indicates no information found - forcing confidence to 10%")
        
        # Override confidence from Gemini service with our calculated confidence
        response_data['confidence'] = confidence_score
        
        # Step 8: Calculate metrics
        response_time = time.time() - start_time
        
        # Step 9: Store memory with better context
        if response_data['has_context']:
            # Store full Q&A with context
            memory_content = f"User Question: {user_query}\nAssistant Response: {response_data['content'][:300]}"
            
            # Add context about what was discussed
            if chat_history and len(chat_history) > 2:
                # Extract topics from conversation
                all_user_messages = [msg.get('content', '') for msg in chat_history if msg.get('role') == 'user']
                if all_user_messages:
                    memory_content += f"\nConversation Context: This is message {len(chat_history)} in the conversation."
            
            self.memory_service.add_memory(
                chat_id=chat_id,
                content=memory_content,
                memory_type='short_term',
                importance_score=response_data['confidence']
            )
            
            # Periodically extract and store chat summary (every 5 messages)
            if chat_history and len(chat_history) % 5 == 0:
                self._extract_and_store_chat_summary(chat_id, chat_history)
        
        # Step 10: Validate with Pydantic
        try:
            agent_response = AgentResponse(
                content=response_data['content'],
                confidence_score=response_data['confidence'],
                sources=sources,
                query_type=query_type,
                vector_results_count=len(vector_results),
                graph_results_count=len(graph_results),
                web_results_count=len(web_results),
                reasoning=f"Retrieved {len(vector_results)} vector results, {len(graph_results)} graph results",
                completeness_score=0.0  # Will be calculated by validator
            )
            # Recalculate completeness
            agent_response.completeness_score = self._calculate_completeness(agent_response)
        except Exception as e:
            logger.error(f"Pydantic validation error: {e}")
            # Return basic response if validation fails
            # Calculate completeness manually
            content_len = len(response_data['content'])
            sources_count = len(sources)
            length_score = min(content_len / 500, 1.0)
            source_score = min(sources_count * 0.2, 0.4)
            completeness = min(length_score + source_score, 1.0)
            
            agent_response = {
                'content': response_data['content'],
                'confidence_score': response_data['confidence'],
                'completeness_score': completeness,
                'sources': sources,
                'query_type': query_type.value,
                'vector_results_count': len(vector_results),
                'graph_results_count': len(graph_results),
                'web_results_count': len(web_results)
            }
        
        # Step 11: Get recommendations
        recommendations = self.recommendation_service.get_recommendations(user_query, chat_id)
        
        return {
            'response': agent_response.dict() if hasattr(agent_response, 'dict') else agent_response,
            'query_type': query_type.value,
            'response_time': response_time,
            'vector_count': len(vector_results),
            'graph_count': len(graph_results),
            'web_count': len(web_results),
            'recommendations': recommendations
        }
    
    def _calculate_confidence_score(
        self, 
        vector_count: int, 
        graph_count: int, 
        web_count: int
    ) -> float:
        """
        Calculate confidence score based on total data found:
        - No data found → 10% (0.1)
        - Little data (1-2 results) → 30% (0.3)
        - A bit more (3-4 results) → 40% (0.4)
        - More (5-6 results) → 50-55% (0.5-0.55)
        - Even more (7-10 results) → 60% (0.6)
        - Maximum (11+ results) → 90% (0.9)
        """
        # Calculate total results from all sources
        total_results = vector_count + graph_count + web_count
        
        # Case 1: No results from any database → 10%
        if total_results == 0:
            logger.info("No results from any database - confidence: 0.1 (10%)")
            return 0.1
        
        # Case 2: Little data (1-2 results) → 30%
        if total_results <= 2:
            confidence = 0.30
            logger.info(f"Little data found ({total_results} results: vector={vector_count}, graph={graph_count}, web={web_count}) - confidence: {confidence} (30%)")
            return confidence
        
        # Case 3: A bit more (3-4 results) → 40%
        if total_results <= 4:
            confidence = 0.40
            logger.info(f"A bit more data found ({total_results} results: vector={vector_count}, graph={graph_count}, web={web_count}) - confidence: {confidence} (40%)")
            return confidence
        
        # Case 4: More (5-6 results) → 50-55%
        if total_results <= 6:
            # Slightly higher if we have both vector and graph
            if vector_count > 0 and graph_count > 0:
                confidence = 0.55
            else:
                confidence = 0.50
            logger.info(f"More data found ({total_results} results: vector={vector_count}, graph={graph_count}, web={web_count}) - confidence: {confidence} ({confidence*100:.0f}%)")
            return confidence
        
        # Case 5: Even more (7-10 results) → 60%
        if total_results <= 10:
            confidence = 0.60
            logger.info(f"Even more data found ({total_results} results: vector={vector_count}, graph={graph_count}, web={web_count}) - confidence: {confidence} (60%)")
            return confidence
        
        # Case 6: Maximum (11+ results) → 90%
        confidence = 0.90
        logger.info(f"Maximum data found ({total_results} results: vector={vector_count}, graph={graph_count}, web={web_count}) - confidence: {confidence} (90%)")
        return confidence
    
    def teach_knowledge(self, knowledge_text: str, chat_id: str = None) -> Dict:
        """
        Teach the agent new knowledge - stores in both vector and graph databases
        Example: "Karachi is the capital of Pakistan"
        """
        import uuid
        from datetime import datetime
        
        logger.info(f"Teaching new knowledge: {knowledge_text[:50]}...")
        
        # Step 1: Add to Vector Store (ChromaDB)
        doc_id = f"taught_{uuid.uuid4().hex[:12]}"
        try:
            self.vector_store.add_documents(
                documents=[knowledge_text],
                metadatas=[{
                    'source': 'user_taught',
                    'chat_id': chat_id or 'global',
                    'taught_at': datetime.utcnow().isoformat(),
                    'type': 'knowledge'
                }],
                ids=[doc_id]
            )
            logger.info(f"Added knowledge to ChromaDB: {doc_id}")
        except Exception as e:
            logger.error(f"Error adding to vector store: {e}", exc_info=True)
            return {'success': False, 'error': f'Vector store error: {str(e)}'}
        
        # Step 2: Add to Graph Store (Neo4j)
        graph_nodes_created = 0
        graph_relationships_created = 0
        
        if self.graph_store:
            try:
                # Create Document node for the knowledge
                doc_title = knowledge_text.split('.')[0][:50] if '.' in knowledge_text else knowledge_text[:50]
                doc_node_props = {
                    'title': doc_title,
                    'content': knowledge_text,
                    'source': 'user_taught',
                    'chat_id': chat_id or 'global',
                    'taught_at': datetime.utcnow().isoformat(),
                    'type': 'knowledge'
                }
                
                doc_node_id = self.graph_store.create_node("Document", doc_node_props)
                if doc_node_id:
                    graph_nodes_created += 1
                    logger.info(f"Created Document node in Neo4j: {doc_title}")
                    
                    # Extract concepts from knowledge using Gemini
                    try:
                        extract_prompt = f"""Extract main entities, concepts, or key terms from this knowledge statement. 
Return only a comma-separated list of 3-5 important entities/concepts.

Knowledge: {knowledge_text}

Entities/Concepts (comma-separated):"""
                        
                        extraction_response = self.gemini.model.generate_content(extract_prompt)
                        concepts_text = extraction_response.text.strip()
                        
                        # Parse concepts
                        concepts = [c.strip() for c in concepts_text.split(',') if c.strip()][:5]
                        
                        # Create Concept nodes and link to Document
                        concept_ids = {}
                        for concept_name in concepts:
                            if len(concept_name) > 2:
                                concept_props = {
                                    'name': concept_name,
                                    'source': 'user_taught',
                                    'knowledge': knowledge_text[:200]
                                }
                                concept_id = self.graph_store.create_node("Concept", concept_props)
                                if concept_id:
                                    concept_ids[concept_name] = int(concept_id)
                                    graph_nodes_created += 1
                        
                        # Create relationships: Document → DESCRIBES → Concepts
                        for concept_name, concept_id in concept_ids.items():
                            try:
                                self.graph_store.create_relationship(
                                    int(doc_node_id),
                                    concept_id,
                                    "DESCRIBES"
                                )
                                graph_relationships_created += 1
                            except Exception as rel_error:
                                logger.warning(f"Could not create relationship: {rel_error}")
                        
                        logger.info(f"Extracted and linked {len(concept_ids)} concepts from knowledge")
                        
                    except Exception as extract_error:
                        logger.warning(f"Could not extract concepts from knowledge: {extract_error}")
                    
                    # Link to "User Taught Knowledge" topic
                    topic_name = "User Taught Knowledge"
                    topic_props = {
                        'name': topic_name,
                        'topic_id': 'user_taught',
                        'description': 'Knowledge taught by users'
                    }
                    topic_id = self.graph_store.create_node("Topic", topic_props)
                    if topic_id:
                        try:
                            self.graph_store.create_relationship(
                                int(topic_id),
                                int(doc_node_id),
                                "CONTAINS"
                            )
                            graph_relationships_created += 1
                        except:
                            pass
                    
                    logger.info(f"Stored knowledge in Neo4j: {graph_nodes_created} nodes, {graph_relationships_created} relationships")
                    
            except Exception as graph_error:
                logger.error(f"Error storing knowledge in Neo4j: {graph_error}", exc_info=True)
                # Continue even if graph storage fails
        
        return {
            'success': True,
            'vector_id': doc_id,
            'graph_nodes': graph_nodes_created,
            'graph_relationships': graph_relationships_created,
            'message': 'Knowledge successfully stored in databases'
        }
    
    def _calculate_completeness(self, response: AgentResponse) -> float:
        """Calculate completeness score"""
        content_len = len(response.content)
        sources_count = len(response.sources)
        
        length_score = min(content_len / 500.0, 0.6)
        source_score = min(sources_count * 0.1, 0.4)
        
        return min(length_score + source_score, 1.0)
    
    def process_pdf_and_index(self, pdf_path: str, chat_id: str, filename: str = None) -> Dict:
        """Process PDF and add to both vector store (ChromaDB) and graph store (Neo4j)"""
        from src.services.pdf_processor import PDFProcessor
        import uuid
        
        processor = PDFProcessor()
        result = processor.process_pdf(pdf_path)
        
        if result['success']:
            # Step 1: Chunk and add to vector store (ChromaDB)
            chunks = processor.chunk_text(result['text'])
            chunk_ids = [f"pdf_{chat_id}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
            
            ids = self.vector_store.add_documents(
                documents=chunks,
                metadatas=[{
                    'source': 'pdf', 
                    'chat_id': chat_id,
                    'filename': filename or 'uploaded_document.pdf',
                    'page': (i // 3) + 1,
                    'chunk_index': i
                } for i in range(len(chunks))],
                ids=chunk_ids
            )
            
            logger.info(f"Indexed {len(ids)} chunks to ChromaDB: {filename}")
            
            # Step 2: Store PDF in Neo4j graph database with nodes and relationships
            graph_nodes_created = 0
            graph_relationships_created = 0
            
            if self.graph_store:
                try:
                    # Create PDF Document node
                    pdf_title = filename.replace('.pdf', '')[:50] if filename else 'Uploaded Document'
                    pdf_node_props = {
                        'title': pdf_title,
                        'content': result['text'][:5000],  # First 5k chars for search
                        'filename': filename or 'uploaded_document.pdf',
                        'source': 'pdf',
                        'chat_id': chat_id,
                        'page_count': result['page_count'],
                        'uploaded_at': datetime.utcnow().isoformat()
                    }
                    
                    pdf_node_id = self.graph_store.create_node("Document", pdf_node_props)
                    if pdf_node_id:
                        graph_nodes_created += 1
                        logger.info(f"Created PDF Document node in Neo4j: {pdf_title}")
                        
                        # Extract key concepts/topics from PDF using Gemini
                        try:
                            # Get main topics/concepts from PDF
                            extract_prompt = f"""Extract main topics, concepts, and key terms from this PDF content. 
Return only a comma-separated list of 5-10 important topics/concepts.

PDF Content (first 2000 chars):
{result['text'][:2000]}

Topics/Concepts (comma-separated):"""
                            
                            extraction_response = self.gemini.model.generate_content(extract_prompt)
                            concepts_text = extraction_response.text.strip()
                            
                            # Parse concepts
                            concepts = [c.strip() for c in concepts_text.split(',') if c.strip()][:10]
                            
                            # Create Concept nodes and link to PDF
                            concept_ids = {}
                            for concept_name in concepts:
                                if len(concept_name) > 2:  # Valid concept
                                    concept_props = {
                                        'name': concept_name,
                                        'source': 'pdf_extracted',
                                        'pdf_filename': filename
                                    }
                                    concept_id = self.graph_store.create_node("Concept", concept_props)
                                    if concept_id:
                                        concept_ids[concept_name] = int(concept_id)
                                        graph_nodes_created += 1
                            
                            # Create relationships: PDF -> DESCRIBES -> Concepts
                            for concept_name, concept_id in concept_ids.items():
                                try:
                                    self.graph_store.create_relationship(
                                        int(pdf_node_id), 
                                        concept_id, 
                                        "DESCRIBES"
                                    )
                                    graph_relationships_created += 1
                                except Exception as rel_error:
                                    logger.warning(f"Could not create relationship: {rel_error}")
                            
                            logger.info(f"Extracted and linked {len(concept_ids)} concepts from PDF")
                            
                        except Exception as extract_error:
                            logger.warning(f"Could not extract concepts from PDF: {extract_error}")
                        
                        # Create Topic node for PDF category
                        topic_name = "User Uploaded Documents"
                        topic_props = {
                            'name': topic_name,
                            'topic_id': 'user_uploads',
                            'description': 'Documents uploaded by users'
                        }
                        topic_id = self.graph_store.create_node("Topic", topic_props)
                        if topic_id:
                            try:
                                self.graph_store.create_relationship(
                                    int(topic_id),
                                    int(pdf_node_id),
                                    "CONTAINS"
                                )
                                graph_relationships_created += 1
                            except:
                                pass
                    
                    logger.info(f"Stored PDF in Neo4j: {graph_nodes_created} nodes, {graph_relationships_created} relationships")
                    
                except Exception as graph_error:
                    logger.error(f"Error storing PDF in Neo4j: {graph_error}", exc_info=True)
                    # Continue even if graph storage fails - vector store is primary
            
            return {
                'success': True,
                'chunks_added': len(ids),
                'page_count': result['page_count'],
                'graph_nodes': graph_nodes_created,
                'graph_relationships': graph_relationships_created
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'Unknown error')
            }
    
    def detect_patterns(self, chat_id: str) -> List[Dict]:
        """Detect patterns in chat history: cluster similar queries, detect recurring entity paths, frequent vector vs graph queries"""
        messages = self.db.query(Message).filter(
            Message.chat_id == chat_id,
            Message.role == 'user'
        ).all()
        
        if not messages:
            return []
        
        patterns = []
        query_types = {}
        query_clusters = {}
        vector_vs_graph_queries = []
        
        # Get query logs for this chat
        query_logs = self.db.query(QueryLog).filter(
            QueryLog.message_id.in_(
                self.db.query(Message.id).filter(Message.chat_id == chat_id)
            )
        ).all()
        
        # Cluster similar queries using embeddings
        if len(messages) > 1:
            try:
                # Group queries by similarity
                query_texts = [msg.content for msg in messages]
                embeddings = self.vector_store.embedder.encode(query_texts)
                
                # Simple clustering: group queries with similarity > 0.7
                clusters = []
                used = set()
                for i, query1 in enumerate(query_texts):
                    if i in used:
                        continue
                    cluster = [query1]
                    used.add(i)
                    for j, query2 in enumerate(query_texts[i+1:], i+1):
                        if j in used:
                            continue
                        # Calculate cosine similarity
                        similarity = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        )
                        if similarity > 0.7:
                            cluster.append(query2)
                            used.add(j)
                    
                    if len(cluster) > 1:
                        clusters.append(cluster)
                
                # Add cluster patterns
                for cluster in clusters:
                    patterns.append({
                        'type': 'similar_queries',
                        'frequency': len(cluster),
                        'description': f"Cluster of {len(cluster)} similar queries",
                        'examples': cluster[:3]  # Show first 3 examples
                    })
            except Exception as e:
                logger.warning(f"Error clustering queries: {e}")
        
        # Detect recurring entity paths from graph
        if self.graph_store and query_logs:
            try:
                # Get entities mentioned in queries
                entity_paths = {}
                for query_log in query_logs:
                    # Search graph for entities in query
                    graph_results = self.graph_store.search_by_keyword(query_log.query_text, limit=5)
                    for result in graph_results:
                        node_data = result.get('n', {}) or {}
                        if isinstance(node_data, dict):
                            node_name = node_data.get('name') or node_data.get('title', '')
                            if node_name:
                                # Get relationships
                                node_id = node_data.get('id')
                                if node_id:
                                    try:
                                        related = self.graph_store.get_related_nodes(int(node_id), depth=1)
                                        path_key = f"{node_name} -> {len(related)} related"
                                        entity_paths[path_key] = entity_paths.get(path_key, 0) + 1
                                    except:
                                        pass
                
                # Add recurring paths
                for path, count in entity_paths.items():
                    if count > 1:
                        patterns.append({
                            'type': 'recurring_entity_path',
                            'frequency': count,
                            'description': f"Recurring entity path: {path}",
                            'path': path
                        })
            except Exception as e:
                logger.warning(f"Error detecting entity paths: {e}")
        
        # Detect frequent "vector vs graph" queries
        for query_log in query_logs:
            query_lower = query_log.query_text.lower()
            # Check for comparison queries
            if any(term in query_lower for term in ['vector', 'graph', 'vs', 'versus', 'compare', 'difference']):
                if 'vector' in query_lower or 'graph' in query_lower:
                    vector_vs_graph_queries.append({
                        'query': query_log.query_text,
                        'query_type': query_log.query_type,
                        'timestamp': query_log.timestamp.isoformat() if query_log.timestamp else None
                    })
        
        if len(vector_vs_graph_queries) > 0:
            patterns.append({
                'type': 'vector_vs_graph_queries',
                'frequency': len(vector_vs_graph_queries),
                'description': f"Frequent vector vs graph comparison queries ({len(vector_vs_graph_queries)} found)",
                'examples': [q['query'] for q in vector_vs_graph_queries[:3]]
            })
        
        # Basic query type patterns
        for msg in messages:
            content_lower = msg.content.lower()
            if 'what' in content_lower or 'explain' in content_lower or 'tell me about' in content_lower:
                query_types['explanatory'] = query_types.get('explanatory', 0) + 1
            if 'how' in content_lower or 'how to' in content_lower or 'how do' in content_lower:
                query_types['how-to'] = query_types.get('how-to', 0) + 1
            if 'compare' in content_lower or 'difference' in content_lower or 'vs' in content_lower or 'versus' in content_lower:
                query_types['comparison'] = query_types.get('comparison', 0) + 1
            if 'why' in content_lower:
                query_types['reasoning'] = query_types.get('reasoning', 0) + 1
            if 'when' in content_lower or 'where' in content_lower:
                query_types['factual'] = query_types.get('factual', 0) + 1
        
        for pattern_type, count in query_types.items():
            if count > 0:
                patterns.append({
                    'type': pattern_type,
                    'frequency': count,
                    'description': f"User asks {pattern_type} questions ({count} time{'s' if count > 1 else ''})"
                })
        
        return patterns
    
    def detect_anomalies(self, chat_id: str) -> List[Dict]:
        """Detect anomalies: low similarity/missing chunks, missing relationships, low-confidence flags"""
        query_logs = self.db.query(QueryLog).filter(
            QueryLog.message_id.in_(
                self.db.query(Message.id).filter(Message.chat_id == chat_id)
            )
        ).all()
        
        anomalies = []
        
        if not query_logs:
            return anomalies
        
        # Filter out None values for calculation
        confidence_scores = [q.confidence_score for q in query_logs if q.confidence_score is not None]
        response_times = [q.response_time for q in query_logs if q.response_time is not None]
        
        # Calculate averages if we have multiple queries
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else None
        avg_response_time = sum(response_times) / len(response_times) if response_times else None
        
        for log in query_logs:
            # 1. Detect low similarity / missing chunks
            if log.vector_results_count == 0 and log.query_text:
                # Check if query should have matched something
                try:
                    # Search vector store to see if there are any similar chunks
                    test_results = self.vector_store.search(log.query_text, top_k=5)
                    if test_results:
                        # There are results but they weren't used - low similarity
                        max_distance = max([r.get('distance', 1.0) for r in test_results if r.get('distance')])
                        if max_distance > 0.8:  # High distance = low similarity
                            anomalies.append({
                                'type': 'low_similarity_missing_chunks',
                                'query': log.query_text[:50],
                                'max_distance': round(max_distance, 3),
                                'description': f'Low similarity found (distance: {max_distance:.3f}) - chunks may be missing or poorly indexed'
                            })
                    else:
                        # No results at all - missing chunks
                        anomalies.append({
                            'type': 'missing_chunks',
                            'query': log.query_text[:50],
                            'description': 'No matching chunks found in vector store - content may be missing'
                        })
                except Exception as e:
                    logger.warning(f"Error checking similarity: {e}")
            
            # 2. Detect missing relationships in graph
            if self.graph_store and log.graph_results_count == 0 and log.query_text:
                try:
                    # Check if graph has related nodes that should have been found
                    keyword_results = self.graph_store.search_by_keyword(log.query_text, limit=5)
                    if keyword_results:
                        # Graph has nodes but relationships might be missing
                        has_relationships = False
                        for result in keyword_results:
                            node_data = result.get('n', {}) or {}
                            if isinstance(node_data, dict):
                                node_id = node_data.get('id')
                                if node_id:
                                    try:
                                        related = self.graph_store.get_related_nodes(int(node_id), depth=1)
                                        if related:
                                            has_relationships = True
                                            break
                                    except:
                                        pass
                        
                        if not has_relationships:
                            anomalies.append({
                                'type': 'missing_relationships',
                                'query': log.query_text[:50],
                                'description': 'Graph nodes found but no relationships detected - graph structure may be incomplete'
                            })
                except Exception as e:
                    logger.warning(f"Error checking relationships: {e}")
            
            # 3. Detect "can't answer" - very low confidence (< 0.5) regardless of average
            if log.confidence_score is not None and log.confidence_score < 0.5:
                anomalies.append({
                    'type': 'low_confidence_flagged',
                    'query': log.query_text[:50],
                    'confidence': log.confidence_score,
                    'description': f"Low-confidence answer flagged (confidence: {log.confidence_score:.2f})"
                })
            
            # 4. Detect no results from any source (agent has no data)
            if (log.vector_results_count == 0 and 
                log.graph_results_count == 0 and 
                log.web_results_count == 0):
                anomalies.append({
                    'type': 'no_results',
                    'query': log.query_text[:50],
                    'description': 'No results found from vector, graph, or web search'
                })
            
            # 5. Detect low confidence compared to average (if we have average)
            if avg_confidence and log.confidence_score is not None:
                if log.confidence_score < avg_confidence * 0.6:
                    anomalies.append({
                        'type': 'low_confidence',
                        'query': log.query_text[:50],
                        'confidence': log.confidence_score,
                        'average_confidence': round(avg_confidence, 3),
                        'description': f'Low confidence ({log.confidence_score:.2f}) compared to average ({avg_confidence:.2f})'
                    })
            
            # 6. Detect slow response compared to average (if we have average)
            if avg_response_time and log.response_time is not None:
                if log.response_time > avg_response_time * 1.5:
                    anomalies.append({
                        'type': 'slow_response',
                        'query': log.query_text[:50],
                        'response_time': round(log.response_time, 2),
                        'average_response_time': round(avg_response_time, 2),
                        'description': f'Slow response ({log.response_time:.2f}s) compared to average ({avg_response_time:.2f}s)'
                    })
            
            # 7. Detect very low result counts (might indicate poor retrieval)
            total_results = (log.vector_results_count or 0) + (log.graph_results_count or 0) + (log.web_results_count or 0)
            if total_results > 0 and total_results < 2:  # Very few results
                anomalies.append({
                    'type': 'few_results',
                    'query': log.query_text[:50],
                    'total_results': total_results,
                    'description': f'Very few results found ({total_results} total)'
                })
        
        # Remove duplicates (same query might trigger multiple anomaly types)
        seen_queries = set()
        unique_anomalies = []
        for anomaly in anomalies:
            query_key = anomaly['query']
            if query_key not in seen_queries:
                seen_queries.add(query_key)
                unique_anomalies.append(anomaly)
        
        return unique_anomalies
    
    def analyze_trends(self, chat_id: Optional[str] = None, days: int = 30) -> Dict:
        """Trend Analysis: aggregate embeddings over time, graph query frequency, rising topic detection"""
        trends = {
            'embedding_trends': [],
            'graph_query_frequency': {},
            'rising_topics': [],
            'query_type_distribution': {}
        }
        
        # Get query logs (for specific chat or all)
        query_filter = {}
        if chat_id:
            query_filter['message_id'] = self.db.query(Message.id).filter(Message.chat_id == chat_id)
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query_logs = self.db.query(QueryLog).filter(
            QueryLog.timestamp >= cutoff_date
        ).order_by(QueryLog.timestamp.asc()).all()
        
        if not query_logs:
            return trends
        
        # 1. Aggregate embeddings over time
        try:
            # Group queries by time periods (daily)
            daily_embeddings = defaultdict(list)
            for log in query_logs:
                date_key = log.timestamp.date() if log.timestamp else datetime.utcnow().date()
                daily_embeddings[date_key].append(log.query_text)
            
            # Calculate average embedding per day
            for date_key, queries in sorted(daily_embeddings.items()):
                if queries:
                    try:
                        embeddings = self.vector_store.embedder.encode(queries)
                        avg_embedding = np.mean(embeddings, axis=0)
                        # Calculate embedding magnitude as a trend indicator
                        magnitude = np.linalg.norm(avg_embedding)
                        trends['embedding_trends'].append({
                            'date': date_key.isoformat(),
                            'query_count': len(queries),
                            'embedding_magnitude': float(magnitude),
                            'sample_queries': queries[:3]
                        })
                    except Exception as e:
                        logger.warning(f"Error calculating embeddings for {date_key}: {e}")
        except Exception as e:
            logger.error(f"Error in embedding trend analysis: {e}")
        
        # 2. Graph query frequency
        graph_query_counts = Counter()
        vector_query_counts = Counter()
        hybrid_query_counts = Counter()
        
        for log in query_logs:
            if log.query_type == 'graph':
                graph_query_counts[log.timestamp.date() if log.timestamp else datetime.utcnow().date()] += 1
            elif log.query_type == 'vector':
                vector_query_counts[log.timestamp.date() if log.timestamp else datetime.utcnow().date()] += 1
            elif log.query_type == 'hybrid':
                hybrid_query_counts[log.timestamp.date() if log.timestamp else datetime.utcnow().date()] += 1
        
        trends['graph_query_frequency'] = {
            'graph': {str(k): v for k, v in sorted(graph_query_counts.items())},
            'vector': {str(k): v for k, v in sorted(vector_query_counts.items())},
            'hybrid': {str(k): v for k, v in sorted(hybrid_query_counts.items())},
            'total_graph': sum(graph_query_counts.values()),
            'total_vector': sum(vector_query_counts.values()),
            'total_hybrid': sum(hybrid_query_counts.values())
        }
        
        # 3. Rising topic detection
        try:
            # Split time into two periods
            mid_date = datetime.utcnow() - timedelta(days=days // 2)
            early_queries = [log.query_text for log in query_logs if log.timestamp and log.timestamp < mid_date]
            recent_queries = [log.query_text for log in query_logs if log.timestamp and log.timestamp >= mid_date]
            
            # Extract topics/keywords from queries
            def extract_topics(queries):
                all_words = []
                for query in queries:
                    words = [w.lower() for w in query.split() if len(w) > 4]  # Words > 4 chars
                    all_words.extend(words)
                return Counter(all_words)
            
            early_topics = extract_topics(early_queries)
            recent_topics = extract_topics(recent_queries)
            
            # Find topics that increased significantly
            rising_topics = []
            for topic, recent_count in recent_topics.items():
                early_count = early_topics.get(topic, 0)
                if recent_count > early_count and recent_count >= 2:  # At least 2 mentions recently
                    growth_rate = (recent_count - early_count) / max(early_count, 1)
                    if growth_rate > 0.5:  # 50%+ growth
                        rising_topics.append({
                            'topic': topic,
                            'recent_count': recent_count,
                            'early_count': early_count,
                            'growth_rate': round(growth_rate * 100, 1)
                        })
            
            # Sort by growth rate
            rising_topics.sort(reverse=True, key=lambda x: x['growth_rate'])
            trends['rising_topics'] = rising_topics[:10]  # Top 10 rising topics
            
        except Exception as e:
            logger.error(f"Error in rising topic detection: {e}")
        
        # 4. Query type distribution
        query_type_counts = Counter([log.query_type for log in query_logs if log.query_type])
        trends['query_type_distribution'] = dict(query_type_counts)
        
        return trends
    
    def _extract_and_store_chat_summary(self, chat_id: str, chat_history: List[Dict]):
        """Extract chat summary and topics using LLM"""
        try:
            # Get all user messages to extract topics
            user_messages = [msg.get('content', '') for msg in chat_history if msg.get('role') == 'user']
            if not user_messages or len(user_messages) < 2:
                return
            
            # Create a prompt to extract summary and topics
            conversation_text = "\n".join([f"User: {msg}" for msg in user_messages[-5:]])  # Last 5 user messages
            
            summary_prompt = f"""Analyze this conversation and provide:
1. Main topic/subject being discussed (1-2 sentences)
2. Key topics covered (comma-separated list)

Conversation:
{conversation_text}

Format your response as:
TOPIC: [main topic]
KEY_TOPICS: [comma-separated topics]"""
            
            try:
                summary_response = self.gemini.model.generate_content(summary_prompt)
                summary_text = summary_response.text.strip()
                
                # Parse the response
                topic = "General discussion"
                topics = []
                
                lines = summary_text.split('\n')
                for line in lines:
                    if line.startswith('TOPIC:'):
                        topic = line.replace('TOPIC:', '').strip()
                    elif line.startswith('KEY_TOPICS:'):
                        topics_str = line.replace('KEY_TOPICS:', '').strip()
                        topics = [t.strip() for t in topics_str.split(',') if t.strip()]
                
                # Store the summary
                self.memory_service.store_chat_summary(chat_id, topic, topics)
                logger.info(f"Stored chat summary for chat {chat_id}: {topic}")
                
            except Exception as extract_error:
                logger.warning(f"Could not extract chat summary: {extract_error}")
                
        except Exception as e:
            logger.warning(f"Error in chat summary extraction: {e}")

