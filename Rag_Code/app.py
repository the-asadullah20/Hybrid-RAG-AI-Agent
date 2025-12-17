"""Flask Application - Hybrid RAG AI Agent"""
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime
import uuid
import os
import logging
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from src.config import Config
from src.database.db_models import init_db, get_db, Chat, Message, QueryLog, PDFDocument
from src.services.hybrid_rag_agent import HybridRAGAgent
from src.services.pdf_processor import PDFProcessor
from src.models.pydantic_models import MessageRequest, MessageResponse, ChatResponse, PDFUploadResponse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_PDF_SIZE
app.config['UPLOAD_FOLDER'] = Config.PDF_UPLOAD_FOLDER

# Initialize database
init_db()

# Statistics tracking
statistics = {
    'total_chats': 0,
    'total_messages': 0,
    'average_response_time': 0.0,
    'vector_searches': 0,
    'graph_searches': 0
}

def get_db_session():
    """Get database session"""
    return next(get_db())

def update_statistics():
    """Update statistics from database"""
    db = get_db_session()
    try:
        statistics['total_chats'] = db.query(Chat).count()
        statistics['total_messages'] = db.query(Message).count()
        
        query_logs = db.query(QueryLog).all()
        if query_logs:
            total_time = sum(q.response_time or 0 for q in query_logs)
            statistics['average_response_time'] = round(total_time / len(query_logs), 2) if query_logs else 0.0
            # Count vector searches: Any query that used vector search (has vector_results_count > 0)
            # This includes 'vector', 'hybrid' types, and any query that got vector results
            statistics['vector_searches'] = sum(1 for q in query_logs if (q.vector_results_count and q.vector_results_count > 0) or q.query_type == 'vector')
            # Count graph searches: Any query that used graph search (has graph_results_count > 0 or is 'graph'/'hybrid' type)
            statistics['graph_searches'] = sum(1 for q in query_logs if (q.graph_results_count and q.graph_results_count > 0) or q.query_type in ['graph', 'hybrid'])
    finally:
        db.close()

@app.route('/')
def index():
    """Main page"""
    update_statistics()
    db = get_db_session()
    try:
        chats = db.query(Chat).order_by(Chat.updated_at.desc()).all()
        chats_data = [{
            'id': chat.id,
            'title': chat.title,
            'created_at': chat.created_at.isoformat(),
            'updated_at': chat.updated_at.isoformat(),
            'messages': [{
                'id': msg.id,
                'content': msg.content,
                'role': msg.role,
                'timestamp': msg.timestamp.isoformat()
            } for msg in chat.messages]
        } for chat in chats]
    finally:
        db.close()
    
    return render_template('index.html', chats=chats_data, statistics=statistics)

@app.route('/api/chats', methods=['GET'])
def get_chats():
    """Get all chats"""
    db = get_db_session()
    try:
        chats = db.query(Chat).order_by(Chat.updated_at.desc()).all()
        return jsonify([{
            'id': chat.id,
            'title': chat.title,
            'created_at': chat.created_at.isoformat(),
            'updated_at': chat.updated_at.isoformat(),
            'messages': [{
                'id': msg.id,
                'content': msg.content,
                'role': msg.role,
                'timestamp': msg.timestamp.isoformat()
            } for msg in chat.messages]
        } for chat in chats])
    finally:
        db.close()

@app.route('/api/chats', methods=['POST'])
def create_chat():
    """Create new chat"""
    try:
        data = request.json or {}
        chat_id = str(uuid.uuid4())
        
        db = get_db_session()
        try:
            chat = Chat(
                id=chat_id,
                title=data.get('title', 'New Chat'),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(chat)
            db.commit()
            
            # Verify chat was saved
            saved_chat = db.query(Chat).filter(Chat.id == chat_id).first()
            if not saved_chat:
                logger.error(f"Chat {chat_id} was not saved to database!")
                return jsonify({'error': 'Failed to save chat to database'}), 500
            
            logger.info(f"Successfully created and saved chat {chat_id} to database")
            
            return jsonify({
                'id': saved_chat.id,
                'title': saved_chat.title,
                'created_at': saved_chat.created_at.isoformat(),
                'updated_at': saved_chat.updated_at.isoformat(),
                'messages': []
            }), 201
        except Exception as db_error:
            logger.error(f"Database error creating chat: {db_error}", exc_info=True)
            db.rollback()
            return jsonify({'error': f'Database error: {str(db_error)}'}), 500
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error creating chat: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    """Get a single chat by ID"""
    try:
        db = get_db_session()
        try:
            chat = db.query(Chat).filter(Chat.id == chat_id).first()
            if not chat:
                return jsonify({'error': 'Chat not found'}), 404
            
            return jsonify({
                'id': chat.id,
                'title': chat.title,
                'created_at': chat.created_at.isoformat(),
                'updated_at': chat.updated_at.isoformat(),
                'messages': [{
                    'id': msg.id,
                    'content': msg.content,
                    'role': msg.role,
                    'timestamp': msg.timestamp.isoformat()
                } for msg in chat.messages]
            })
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error getting chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete chat"""
    try:
        db = get_db_session()
        try:
            chat = db.query(Chat).filter(Chat.id == chat_id).first()
            if chat:
                # Delete query logs first (they have foreign key to messages)
                messages = db.query(Message).filter(Message.chat_id == chat_id).all()
                for msg in messages:
                    db.query(QueryLog).filter(QueryLog.message_id == msg.id).delete()
                
                # Delete agent memory for this chat
                from src.database.db_models import AgentMemory
                db.query(AgentMemory).filter(AgentMemory.chat_id == chat_id).delete()
                
                # Now delete the chat (messages will cascade)
                db.delete(chat)
                db.commit()
                return jsonify({'success': True})
            return jsonify({'error': 'Chat not found'}), 404
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error deleting chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chats/<chat_id>/messages', methods=['POST'])
def add_message(chat_id):
    """Add message and get AI response"""
    try:
        # Validate request
        msg_request = MessageRequest(**request.json)
        
        db = get_db_session()
        try:
            # Check if chat exists
            chat = db.query(Chat).filter(Chat.id == chat_id).first()
            if not chat:
                return jsonify({'error': 'Chat not found'}), 404
            
            # Auto-rename chat if it's "New Chat" and this is the first user message
            message_count = db.query(Message).filter(Message.chat_id == chat_id, Message.role == 'user').count()
            if chat.title == 'New Chat' and message_count == 0:
                # Generate a title from the first question (max 50 chars)
                new_title = msg_request.content[:50].strip()
                if not new_title:
                    new_title = 'New Chat'
                chat.title = new_title
                logger.info(f"Auto-renamed chat {chat_id} to: {new_title}")
                # Commit title change immediately
                db.commit()
            
            # Save user message
            user_msg = Message(
                id=str(uuid.uuid4()),
                chat_id=chat_id,
                content=msg_request.content,
                role='user',
                timestamp=datetime.utcnow()
            )
            db.add(user_msg)
            db.commit()
            
            # Get chat history
            chat_history = db.query(Message).filter(
                Message.chat_id == chat_id
            ).order_by(Message.timestamp.asc()).all()
            
            history_data = [{
                'role': msg.role,
                'content': msg.content
            } for msg in chat_history]
            
            # Process with Hybrid RAG Agent
            try:
                agent = HybridRAGAgent(db)
                result = agent.process_query(
                    user_query=msg_request.content,
                    chat_id=chat_id,
                    chat_history=history_data
                )
            except Exception as agent_error:
                logger.error(f"Agent error: {agent_error}", exc_info=True)
                # Return a helpful error message
                error_content = f"I'm sorry, I encountered an error while processing your request: {str(agent_error)}. Please check the server logs for more details."
                assistant_msg = Message(
                    id=str(uuid.uuid4()),
                    chat_id=chat_id,
                    content=error_content,
                    role='assistant',
                    timestamp=datetime.utcnow()
                )
                db.add(assistant_msg)
                db.commit()
                return jsonify({
                    'id': assistant_msg.id,
                    'content': error_content,
                    'role': 'assistant',
                    'timestamp': assistant_msg.timestamp.isoformat(),
                    'sources': [],
                    'confidence_score': 0.0,
                    'error': True
                }), 201
            
            # Save assistant response
            assistant_msg = Message(
                id=str(uuid.uuid4()),
                chat_id=chat_id,
                content=result['response']['content'],
                role='assistant',
                timestamp=datetime.utcnow()
            )
            db.add(assistant_msg)
            
            # Save query log
            query_type_value = result['query_type']
            logger.info(f"Saving query log with query_type: {query_type_value}, graph_count: {result['graph_count']}, vector_count: {result['vector_count']}")
            
            query_log = QueryLog(
                id=str(uuid.uuid4()),
                message_id=assistant_msg.id,
                query_text=msg_request.content,
                query_type=query_type_value,
                response_time=result['response_time'],
                vector_results_count=result['vector_count'],
                graph_results_count=result['graph_count'],
                web_results_count=result.get('web_count', 0),
                confidence_score=result['response'].get('confidence_score', 0.0),
                timestamp=datetime.utcnow()
            )
            db.add(query_log)
            
            # Update chat
            chat.updated_at = datetime.utcnow()
            db.commit()
            
            # Update statistics
            update_statistics()
            logger.info(f"Updated stats - graph_searches: {statistics['graph_searches']}, vector_searches: {statistics['vector_searches']}")
            
            # Return response with updated stats and chat title
            return jsonify({
                'id': assistant_msg.id,
                'content': result['response']['content'],
                'role': 'assistant',
                'timestamp': assistant_msg.timestamp.isoformat(),
                'sources': result['response'].get('sources', []),
                'confidence_score': result['response'].get('confidence_score'),
                'completeness_score': result['response'].get('completeness_score'),
                'recommendations': result.get('recommendations', {}),
                'statistics': statistics.copy(),  # Include updated stats
                'chat_title': chat.title  # Include updated chat title
            }), 201
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chats/<chat_id>/messages', methods=['GET'])
def get_messages(chat_id):
    """Get messages for a chat"""
    try:
        db = get_db_session()
        try:
            messages = db.query(Message).filter(
                Message.chat_id == chat_id
            ).order_by(Message.timestamp.asc()).all()
            
            result = []
            for msg in messages:
                msg_data = {
                    'id': msg.id,
                    'content': msg.content,
                    'role': msg.role,
                    'timestamp': msg.timestamp.isoformat(),
                    'sources': [],
                    'confidence_score': None,
                    'completeness_score': None
                }
                
                # Get scores from QueryLog for assistant messages
                if msg.role == 'assistant' and msg.query_log:
                    msg_data['confidence_score'] = msg.query_log.confidence_score
                    # Calculate completeness from response (we don't store it, so estimate)
                    if msg.query_log.confidence_score is not None:
                        # Estimate completeness based on content length and sources
                        content_len = len(msg.content)
                        length_score = min(content_len / 500, 1.0)
                        # Assume some sources if confidence is high
                        source_score = 0.2 if msg.query_log.confidence_score > 0.5 else 0.0
                        msg_data['completeness_score'] = min(length_score + source_score, 1.0)
                
                result.append(msg_data)
            
            return jsonify(result)
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Upload and process PDF"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        processor = PDFProcessor()
        file_path = processor.save_pdf(file.read(), filename)
        
        # Process PDF
        result = processor.process_pdf(file_path)
        
        if result['success']:
            # Add to database
            db = get_db_session()
            try:
                pdf_doc = PDFDocument(
                    id=str(uuid.uuid4()),
                    filename=filename,
                    file_path=file_path,
                    uploaded_at=datetime.utcnow(),
                    page_count=result['page_count'],
                    extracted_text=result['text'][:10000],  # Store first 10k chars
                    processed=0
                )
                db.add(pdf_doc)
                db.commit()
                
                # Index in vector store - use chat_id from form or create new chat
                chat_id_for_pdf = request.form.get('chat_id')
                if not chat_id_for_pdf:
                    # Create a new chat for this PDF
                    new_chat = Chat(
                        id=str(uuid.uuid4()),
                        title=f"PDF: {filename}",
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    db.add(new_chat)
                    db.commit()
                    chat_id_for_pdf = new_chat.id
                
                agent = HybridRAGAgent(db)
                index_result = agent.process_pdf_and_index(file_path, chat_id_for_pdf, filename=filename)
                
                if index_result['success']:
                    pdf_doc.processed = 1
                    db.commit()
                
                # Validate with Pydantic
                pdf_response = PDFUploadResponse(
                    id=pdf_doc.id,
                    filename=pdf_doc.filename,
                    page_count=pdf_doc.page_count,
                    processed=bool(pdf_doc.processed),
                    uploaded_at=pdf_doc.uploaded_at
                )
                
                return jsonify(pdf_response.dict()), 201
            finally:
                db.close()
        else:
            return jsonify({'error': result.get('error', 'PDF processing failed')}), 500
            
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/teach', methods=['POST'])
def teach_agent():
    """Teach agent new knowledge"""
    try:
        data = request.json or {}
        knowledge = data.get('knowledge', '').strip()
        
        if not knowledge:
            return jsonify({'error': 'Knowledge text is required'}), 400
        
        # Get chat_id if provided (optional)
        chat_id = data.get('chat_id')
        
        db = get_db_session()
        try:
            agent = HybridRAGAgent(db)
            result = agent.teach_knowledge(knowledge, chat_id)
            
            if result['success']:
                return jsonify({
                    'success': True,
                    'message': result['message'],
                    'vector_id': result.get('vector_id'),
                    'graph_nodes': result.get('graph_nodes', 0),
                    'graph_relationships': result.get('graph_relationships', 0)
                }), 201
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'Failed to store knowledge')
                }), 500
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error teaching agent: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get statistics"""
    update_statistics()
    return jsonify(statistics)

@app.route('/api/patterns/<chat_id>', methods=['GET'])
def get_patterns(chat_id):
    """Get detected patterns for a chat"""
    try:
        db = get_db_session()
        try:
            from src.database.db_models import Message
            user_messages_count = db.query(Message).filter(
                Message.chat_id == chat_id,
                Message.role == 'user'
            ).count()
            
            agent = HybridRAGAgent(db)
            patterns = agent.detect_patterns(chat_id)
            
            return jsonify({
                'patterns': patterns,
                'total_user_messages': user_messages_count,
                'message': 'No patterns found' if not patterns and user_messages_count > 0 else 'No user messages found in this chat' if user_messages_count == 0 else f'Found {len(patterns)} pattern(s)'
            })
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error detecting patterns: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/anomalies/<chat_id>', methods=['GET'])
def get_anomalies(chat_id):
    """Get detected anomalies for a chat"""
    try:
        db = get_db_session()
        try:
            agent = HybridRAGAgent(db)
            anomalies = agent.detect_anomalies(chat_id)
            
            # Get query logs count for debugging
            from src.database.db_models import QueryLog, Message
            query_logs_count = db.query(QueryLog).filter(
                QueryLog.message_id.in_(
                    db.query(Message.id).filter(Message.chat_id == chat_id)
                )
            ).count()
            
            return jsonify({
                'anomalies': anomalies,
                'total_queries': query_logs_count,
                'message': 'No anomalies found' if not anomalies and query_logs_count > 0 else 'Need at least 2 queries to detect anomalies' if query_logs_count < 2 else 'No queries found in this chat'
            })
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trends', methods=['GET'])
@app.route('/api/trends/<chat_id>', methods=['GET'])
def get_trends(chat_id=None):
    """Get trend analysis: aggregate embeddings over time, graph query frequency, rising topic detection"""
    try:
        db = get_db_session()
        try:
            agent = HybridRAGAgent(db)
            days = request.args.get('days', 30, type=int)
            trends = agent.analyze_trends(chat_id=chat_id, days=days)
            
            return jsonify({
                'trends': trends,
                'chat_id': chat_id,
                'days': days,
                'message': 'Trend analysis completed'
            })
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations/<chat_id>', methods=['GET'])
def get_recommendations(chat_id):
    """Get recommendations for a chat - uses last query or provided query"""
    try:
        db = get_db_session()
        try:
            # Get last user message as query
            last_message = db.query(Message).filter(
                Message.chat_id == chat_id,
                Message.role == 'user'
            ).order_by(Message.timestamp.desc()).first()
            
            if not last_message:
                return jsonify({
                    'error': 'No user messages found in this chat',
                    'recommendations': {}
                }), 404
            
            # Get optional query parameter
            query = request.args.get('query', last_message.content)
            
            agent = HybridRAGAgent(db)
            recommendations = agent.recommendation_service.get_recommendations(query, chat_id)
            
            return jsonify({
                'recommendations': recommendations,
                'query': query,
                'chat_id': chat_id,
                'message': 'Recommendations generated successfully'
            })
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(Config.PDF_UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SQLITE_DB_PATH), exist_ok=True)
    os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)
    
    app.run(debug=True, port=5000)