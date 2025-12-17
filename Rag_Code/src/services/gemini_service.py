"""Google Gemini AI Service"""
import google.generativeai as genai
from typing import List, Dict, Optional
from src.config import Config
import logging

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in environment variables")
        
        genai.configure(api_key=Config.GEMINI_API_KEY)
        # Use available model - try gemini-2.5-flash (fastest), then gemini-2.5-pro
        try:
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Using gemini-2.5-flash model")
        except Exception as e:
            try:
                self.model = genai.GenerativeModel('gemini-2.5-pro')
                logger.info("Using gemini-2.5-pro model")
            except Exception as e2:
                try:
                    self.model = genai.GenerativeModel('gemini-1.5-flash')
                    logger.info("Using gemini-1.5-flash model")
                except Exception as e3:
                    # Last resort
                    self.model = genai.GenerativeModel('gemini-1.5-pro')
                    logger.info("Using gemini-1.5-pro model")
    
    def generate_cypher_query(self, user_query: str, schema_info: Optional[str] = None) -> str:
        """Generate Cypher query from natural language"""
        schema_prompt = schema_info or """
        Node labels: Document (has: title, content, topic, category), Topic (has: name, topic_id), Concept (has: name, keywords)
        Relationships: CONTAINS, DESCRIBES, USES, INCLUDES, STORES, SUPPORTS
        """
        
        # Extract main keywords from query
        keywords = [w.lower() for w in user_query.split() if len(w) > 3][:3]
        main_keyword = keywords[0] if keywords else user_query.split()[0].lower()
        
        prompt = f"""Generate a valid Cypher query for Neo4j.

Schema: Document nodes have (title, content, topic, category), Topic nodes have (name), Concept nodes have (name)
Relationships: CONTAINS, DESCRIBES, USES

User query: {user_query}

Return ONLY a complete, valid Cypher query. Examples:
- For "what is RAG": MATCH (n:Document) WHERE toLower(n.content) CONTAINS 'rag' OR toLower(n.title) CONTAINS 'rag' RETURN n LIMIT 5
- For relationships: MATCH (a)-[r]->(b) WHERE toLower(a.name) CONTAINS 'rag' OR toLower(b.name) CONTAINS 'rag' RETURN a, r, b LIMIT 5
- For general search: MATCH (n:Document) WHERE toLower(n.content) CONTAINS toLower('{main_keyword}') RETURN n LIMIT 5

Cypher query:"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.2,
                    'max_output_tokens': 300,
                    'top_p': 0.8
                }
            )
            cypher = response.text.strip()
            
            # Clean up markdown
            if '```' in cypher:
                lines = cypher.split('```')
                for line in lines:
                    if 'MATCH' in line.upper():
                        cypher = line.replace('cypher', '').replace('sql', '').strip()
                        break
            
            # Remove any trailing incomplete parts
            if cypher.count('(') > cypher.count(')'):
                # Incomplete query, use fallback
                raise ValueError("Incomplete Cypher query generated")
            
            # Validate it starts with MATCH
            if not cypher.upper().startswith('MATCH'):
                raise ValueError("Invalid Cypher query - must start with MATCH")
            
            # Ensure it has RETURN and LIMIT
            if 'RETURN' not in cypher.upper():
                cypher += " RETURN n LIMIT 5"
            elif 'LIMIT' not in cypher.upper():
                cypher += " LIMIT 5"
            
            logger.info(f"Generated valid Cypher: {cypher[:100]}...")
            return cypher
        except Exception as e:
            logger.warning(f"Cypher generation failed: {e}, using fallback")
            # Better fallback - search Document nodes by content
            return f"MATCH (n:Document) WHERE toLower(n.content) CONTAINS toLower('{main_keyword}') OR toLower(n.title) CONTAINS toLower('{main_keyword}') RETURN n LIMIT 5"
    
    def generate_response(
        self, 
        user_query: str, 
        vector_results: List[Dict], 
        graph_results: List[Dict],
        web_results: Optional[List[Dict]] = None,
        chat_history: Optional[List[Dict]] = None,
        memory_context: Optional[str] = None,
        chat_summary: Optional[Dict] = None
    ) -> Dict:
        """Generate response from RAG results"""
        
        # Format context
        context_parts = []
        
        if vector_results:
            # Separate PDF and general results for better formatting
            pdf_results = [r for r in vector_results if r.get('metadata', {}).get('source') == 'pdf']
            general_results = [r for r in vector_results if r.get('metadata', {}).get('source') != 'pdf']
            
            if pdf_results:
                context_parts.append("=== PDF Document Content (User Uploaded) - CRITICAL: Answer questions from this content ===")
                # Include ALL PDF results - no limit for comprehensive summaries
                # For summary requests, we need ALL chunks to generate comprehensive answers
                pdf_chunks_to_include = pdf_results  # Include ALL PDF chunks, no limit
                logger.info(f"Including {len(pdf_chunks_to_include)} PDF chunks in context for comprehensive answer generation")
                
                for i, result in enumerate(pdf_chunks_to_include, 1):
                    content = result.get('content', '')
                    metadata = result.get('metadata', {})
                    filename = metadata.get('filename', 'uploaded document')
                    page = metadata.get('page', '')
                    page_info = f" (Page {page})" if page else ""
                    # Include FULL content (up to 2000 chars) for comprehensive summaries
                    # This ensures LLM gets all information to generate detailed summaries
                    full_content = content[:2000]  # Increased to 2000 chars per chunk
                    context_parts.append(f"{i}. [From PDF: {filename}{page_info}]\n{full_content}")
                
                context_parts.append("=== END PDF CONTENT ===")
                context_parts.append("CRITICAL: The PDF content above contains ALL chunks from user's uploaded documents. You MUST read through ALL chunks and generate comprehensive, detailed answers using information from ALL of them.")
            
            if general_results:
                context_parts.append("=== General Knowledge Base Results ===")
                # Include more general results (up to 5)
                for i, result in enumerate(general_results[:5], 1):
                    content = result.get('content', '')
                    # Include more content (500 chars) for better context
                    context_parts.append(f"{i}. {content[:500]}")
        
        if graph_results:
            context_parts.append("\n=== Graph Database Results ===")
            # Include more graph results (up to 5)
            for i, result in enumerate(graph_results[:5], 1):
                # Format Neo4j result
                node_data = result.get('n', {}) or result.get('related', {})
                if isinstance(node_data, dict):
                    # Include more properties for better context
                    props = ', '.join([f"{k}: {v}" for k, v in node_data.items() if k not in ['id', 'node_id']][:5])
                    context_parts.append(f"{i}. {props}")
        
        # Web scraping disabled - no web results
        context = "\n".join(context_parts) if context_parts else "No relevant data found in database."
        
        # Log context size for debugging
        context_size = len(context)
        logger.info(f"Context size being sent to LLM: {context_size} characters")
        if pdf_results:
            logger.info(f"PDF content in context: {len([p for p in context_parts if '[From PDF' in p or 'From PDF' in p])} chunks")
        
        # Build prompt with FULL chat history for complete context
        history_text = ""
        if chat_history:
            history_text = "\n\n=== FULL CONVERSATION HISTORY (This Chat) ===\n"
            history_text += "IMPORTANT: You have access to the ENTIRE conversation history. Use this context to:\n"
            history_text += "- Understand what topics have been discussed\n"
            history_text += "- Answer follow-up questions based on previous messages\n"
            history_text += "- Maintain conversation continuity\n"
            history_text += "- Understand references to earlier parts of the conversation\n\n"
            
            # Include ALL messages from the chat (not just last 3)
            for i, msg in enumerate(chat_history, 1):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                # Truncate very long messages to prevent context overflow, but keep most of it
                if len(content) > 500:
                    content = content[:500] + "... [truncated]"
                history_text += f"[Message {i}] {role.capitalize()}: {content}\n"
            
            history_text += "\n=== END OF CONVERSATION HISTORY ===\n"
        
        # Add memory context and chat summary if available
        memory_section = ""
        if chat_summary:
            summary_content = chat_summary.get('content', '')
            memory_section += f"\n=== CHAT SUMMARY & TOPICS ===\n{summary_content}\n=== END CHAT SUMMARY ===\n"
        
        if memory_context:
            memory_section += f"\n=== IMPORTANT PREVIOUS MEMORIES ===\n{memory_context}\n=== END MEMORIES ===\n"
        
        if memory_section:
            history_text += memory_section
        
        # Check if PDF content is in context
        has_pdf_content = any('From PDF' in part or '[From PDF' in part or 'PDF Document Content' in part for part in context_parts)
        has_graph_content = any('Graph Database' in part or 'Neo4j' in part for part in context_parts)
        
        pdf_instruction = ""
        if has_pdf_content:
            # Check if query is asking for summary
            is_summary_query = any(word in user_query.lower() for word in ['summary', 'summarize', 'overview', 'detailed', 'comprehensive', 'describe', 'explain'])
            
            summary_instruction = ""
            if is_summary_query:
                summary_instruction = "\n   - FOR SUMMARY/OVERVIEW REQUESTS: You MUST combine information from ALL PDF chunks provided above to create a comprehensive, detailed summary.\n   - Read through ALL PDF chunks and synthesize the information into a well-structured, complete summary.\n   - Do NOT say 'didn't find summary' or 'no summary available' - instead, create a summary from all the PDF content provided.\n   - Include all relevant details, facts, dates, locations, and information from the PDF chunks.\n   - Structure your summary with clear sections if the content covers multiple topics.\n"
            
            pdf_instruction = f"\n6. CRITICAL PDF PRIORITY RULE:\n   - The '=== PDF Document Content (User Uploaded) ===' section contains content from user's uploaded PDF documents.\n   - You MUST answer questions using PDF content FIRST if it contains ANY relevant information.\n   - Search through ALL PDF chunks provided - extract dates, names, facts, deadlines, instructions, etc.\n   - NEVER say 'cannot find', 'not in PDF', 'does not contain', or 'unable to provide' if the PDF content above has ANY relevant information.\n   - If the query asks about something that might be in the PDF, carefully read through ALL PDF chunks and provide the answer.\n{summary_instruction}   - Only if PDF content truly doesn't have the answer, then use database results.\n   - Always reference the PDF filename when citing: 'According to [PDF filename]...'"
        
        graph_instruction = ""
        if has_graph_content:
            graph_instruction = "\n7. If you see '=== Graph Database Results (Neo4j) ===', use this structured relationship data to provide comprehensive answers about connections and relationships between concepts. This is from the graph database."
        
        no_data_instruction = ""
        if not context_parts or context == "No relevant data found in database.":
            no_data_instruction = "\n8. If there is NO context available (no results from database AND no PDF content), you MUST say EXACTLY: 'I'm sorry, I couldn't find anything like this in my database.' Keep it simple and clear."
        
        # Add footer for summary requests
        summary_footer = ""
        is_summary_query_check = any(word in user_query.lower() for word in ['summary', 'summarize', 'overview', 'detailed', 'comprehensive', 'describe', 'explain', 'tell me about'])
        if is_summary_query_check and has_pdf_content:
            summary_footer = "\n\n⚠️ REMINDER FOR SUMMARY: You MUST create a DETAILED, MULTI-PARAGRAPH summary using ALL PDF chunks above. Do NOT give short 1-2 sentence answers. Combine ALL information from ALL chunks into a comprehensive, well-structured summary."
        
        prompt = f"""You are an AI assistant with access to a knowledge base and user-uploaded PDF documents.

{history_text}

Context from knowledge base:
{context}

User question: {user_query}

CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. CONVERSATION CONTEXT: The "=== FULL CONVERSATION HISTORY (This Chat) ===" section above contains the ENTIRE conversation history from this chat. You MUST:
   - Remember what was discussed earlier in this conversation
   - Understand the context and topic of the ongoing conversation
   - Answer follow-up questions that refer to previous messages
   - Maintain continuity with earlier parts of the conversation
   - If the user asks "what did I ask before?" or "what were we discussing?", refer to the conversation history above
   - Understand references like "that thing", "the previous question", "as I mentioned" based on conversation history

2. SEARCH ORDER: PDF content FIRST → Database results SECOND → If nothing found, say sorry
3. If you see "=== PDF Document Content (User Uploaded) ===" section above, it contains content from user's uploaded PDFs
3. You MUST search through ALL PDF chunks provided above and extract the answer if it exists
4. NEVER say "cannot find", "not in PDF", "does not contain", or "unable to provide" if PDF content has ANY relevant information
5. Read PDF chunks carefully - look for dates, names, deadlines, instructions, facts, etc.
6. For SUMMARY/OVERVIEW requests: 
   - Read through EVERY SINGLE PDF chunk provided above
   - Extract ALL relevant information from ALL chunks
   - Combine and synthesize information from ALL chunks to create a COMPREHENSIVE, DETAILED summary
   - Do NOT give short 1-2 sentence answers - provide a DETAILED summary with multiple paragraphs
   - Include ALL available details: background, location, history, key facts, dates, descriptions, etc.
   - Your summary should be comprehensive (200-300+ words) using information from ALL PDF chunks
   - Structure it properly with introduction, main content, and conclusion if appropriate
7. If PDF has the answer, provide it with source citation: "According to [PDF filename]..."
8. Only if PDF truly doesn't have the answer, then check database results
9. If database has relevant info, use it
10. Only if BOTH PDF and database have nothing, then say: "I'm sorry, I couldn't find anything like this in my database."
{pdf_instruction}
{graph_instruction}
{no_data_instruction}
{summary_footer}

Response:"""
        
        # Check if query is asking for summary/overview - increase tokens for comprehensive answers
        is_summary_query = any(word in user_query.lower() for word in ['summary', 'summarize', 'overview', 'detailed', 'comprehensive', 'describe', 'explain', 'tell me about'])
        max_tokens = 3000 if is_summary_query else 1000  # More tokens for summaries (increased to 3000)
        
        logger.info(f"Query type: {'SUMMARY' if is_summary_query else 'SPECIFIC'}, Max tokens: {max_tokens}, PDF chunks: {len(pdf_results) if pdf_results else 0}")
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.7 if is_summary_query else Config.TEMPERATURE,  # Slightly higher temp for creative summaries
                    'max_output_tokens': max_tokens
                }
            )
            
            # Handle response - it might be a string or have .text attribute
            if hasattr(response, 'text'):
                response_text = response.text.strip()
            elif isinstance(response, str):
                response_text = response.strip()
            else:
                response_text = str(response).strip()
            
            # Log response length for debugging
            logger.info(f"LLM response length: {len(response_text)} characters")
            
            # Determine confidence
            # Only database (vector/graph) and PDF results are used
            has_database_context = bool(vector_results or graph_results)
            
            if has_database_context:
                confidence = 0.8  # Database has relevant data
            else:
                confidence = 0.3  # No results - unable to provide answer
            
            return {
                'content': response_text,
                'confidence': confidence,
                'has_context': has_database_context
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            # Check if it's an API key error
            error_str = str(e).lower()
            api_key_errors = ['api key', 'apikey', 'authentication', 'unauthorized', 'invalid key', 'quota', 'billing', 'permission denied']
            is_api_key_error = any(keyword in error_str for keyword in api_key_errors)
            
            if is_api_key_error:
                error_msg = "I'm sorry, there's an issue with the API service. Please check your API key configuration."
                confidence = 0.1  # 10% for API key errors
                logger.warning("API key error detected - setting confidence to 10%")
            else:
                error_msg = f"I'm sorry, I encountered an error while processing your request. Error: {str(e)}"
                confidence = 0.1  # 10% for any error
            
            return {
                'content': error_msg,
                'confidence': confidence,
                'has_context': False
            }
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """Summarize text"""
        prompt = f"Summarize the following text in {max_length} words or less:\n\n{text}"
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error summarizing: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def classify(self, text: str, categories: List[str]) -> str:
        """Classify text into categories"""
        categories_str = ', '.join(categories)
        prompt = f"Classify the following text into one of these categories: {categories_str}\n\nText: {text}\n\nCategory:"
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error classifying: {e}")
            return categories[0] if categories else "Unknown"

