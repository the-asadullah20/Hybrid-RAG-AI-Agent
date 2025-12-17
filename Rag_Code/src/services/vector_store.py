"""ChromaDB Vector Store Service"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import uuid
from src.config import Config

class VectorStore:
    def __init__(self):
        import logging
        logger = logging.getLogger(__name__)
        
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        # Load embedding model FIRST
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embedding function for ChromaDB
        def embedding_function(texts):
            return self.embedder.encode(texts).tolist()
        
        # Get or create collection with embedding function
        # This ensures consistent embeddings for both storage and querying
        try:
            # Check if collection already exists
            try:
                existing = self.client.get_collection(name=Config.CHROMA_COLLECTION_NAME)
                # If exists, check if it has embedding function
                if not hasattr(existing, '_embedding_function') or existing._embedding_function is None:
                    logger.warning("Existing collection found without embedding function. Recreating with embedding function...")
                    # Get all existing documents first
                    try:
                        existing_docs = existing.get()
                        logger.info(f"Found {len(existing_docs.get('ids', []))} existing documents")
                    except:
                        existing_docs = None
                    
                    # Delete and recreate
                    self.client.delete_collection(name=Config.CHROMA_COLLECTION_NAME)
                    self.collection = self.client.create_collection(
                        name=Config.CHROMA_COLLECTION_NAME,
                        embedding_function=embedding_function,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info("Created new collection with embedding function")
                    
                    # Re-add existing documents if any
                    if existing_docs and existing_docs.get('ids'):
                        logger.info(f"Re-indexing {len(existing_docs['ids'])} existing documents with new embedding function...")
                        self.collection.add(
                            documents=existing_docs['documents'],
                            metadatas=existing_docs['metadatas'],
                            ids=existing_docs['ids']
                        )
                        logger.info("Re-indexing complete")
                else:
                    self.collection = existing
                    logger.info("Using existing collection with embedding function")
            except:
                # Collection doesn't exist, create new one
                self.collection = self.client.create_collection(
                    name=Config.CHROMA_COLLECTION_NAME,
                    embedding_function=embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Created new collection with embedding function")
            
            collection_count = self.collection.count()
            logger.info(f"ChromaDB collection '{Config.CHROMA_COLLECTION_NAME}' initialized with {collection_count} documents")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB collection: {e}", exc_info=True)
            # Fallback: try without embedding function (for compatibility)
            try:
                self.collection = self.client.get_or_create_collection(
                    name=Config.CHROMA_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.warning("Using ChromaDB collection without explicit embedding function (may cause search issues)")
            except Exception as e2:
                logger.error(f"Failed to create collection even without embedding function: {e2}")
                raise
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        """Add documents to vector store"""
        import logging
        logger = logging.getLogger(__name__)
        
        if not documents:
            logger.warning("No documents provided to add_documents")
            return []
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        try:
            # Ensure we have the same number of items
            if len(documents) != len(metadatas) or len(documents) != len(ids):
                logger.error(f"Mismatch: {len(documents)} docs, {len(metadatas)} metadatas, {len(ids)} ids")
                return []
            
            logger.info(f"Adding {len(documents)} documents to ChromaDB collection")
            
            # Add documents - ChromaDB will use the embedding function we set
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Verify they were added
            new_count = self.collection.count()
            logger.info(f"Successfully added documents. Collection now has {new_count} total documents")
            
            return ids
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}", exc_info=True)
            raise
    
    def search(self, query: str, top_k: int = Config.VECTOR_SEARCH_TOP_K, chat_id: Optional[str] = None) -> List[Dict]:
        """Enhanced search - searches all PDFs and documents comprehensively"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Check collection count first
        try:
            collection_count = self.collection.count()
            logger.info(f"ChromaDB collection '{Config.CHROMA_COLLECTION_NAME}' has {collection_count} documents")
        except Exception as e:
            logger.warning(f"Could not get collection count: {e}")
        
        # Get more results to ensure we find all relevant content - especially for PDFs
        max_results = max(top_k * 5, 30)  # Get at least 30 results or 5x top_k to ensure PDF chunks are found
        
        try:
            # Try query_texts first (if collection has embedding function)
            results = None
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=max_results
                )
            except Exception as e1:
                # If query_texts fails, manually embed and use query_embeddings
                logger.warning(f"query_texts failed ({e1}), trying manual embedding...")
                try:
                    query_embedding = self.embedder.encode([query])[0].tolist()
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=max_results
                    )
                except Exception as e2:
                    logger.error(f"Both query methods failed: {e2}")
                    raise
            
            raw_count = len(results.get('documents', [[]])[0]) if results.get('documents') and len(results.get('documents', [[]])[0]) > 0 else 0
            logger.info(f"ChromaDB query returned {raw_count} raw results for query: '{query[:50]}...'")
            
            # If no results but collection has documents, try with more results
            if raw_count == 0 and collection_count > 0:
                logger.warning(f"No results found but collection has {collection_count} documents. Trying with more results...")
                try:
                    max_retry = min(collection_count, 50)
                    # Try again with more results
                    if hasattr(self.collection, '_embedding_function') and self.collection._embedding_function:
                        results = self.collection.query(
                            query_texts=[query],
                            n_results=max_retry
                        )
                    else:
                        query_embedding = self.embedder.encode([query])[0].tolist()
                        results = self.collection.query(
                            query_embeddings=[query_embedding],
                            n_results=max_retry
                        )
                    raw_count = len(results.get('documents', [[]])[0]) if results.get('documents') and len(results.get('documents', [[]])[0]) > 0 else 0
                    logger.info(f"Retry with {max_retry} max results: {raw_count} results")
                except Exception as retry_error:
                    logger.error(f"Retry also failed: {retry_error}")
        except Exception as e:
            logger.error(f"ChromaDB query error: {e}", exc_info=True)
            return []
        
        documents = []
        if results.get('documents') and len(results['documents'][0]) > 0:
            # Collect all results with distance threshold
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 1.0
                
                # VERY lenient distance threshold - include results up to 0.98 distance for PDFs
                # This ensures we don't miss relevant content, especially for PDFs
                # For PDFs, be even more lenient (0.98), for general docs use 0.95
                is_pdf = metadata.get('source') == 'pdf'
                threshold = 0.98 if is_pdf else 0.95
                if distance <= threshold:
                    documents.append({
                        'content': doc,
                        'id': results['ids'][0][i] if results['ids'] else None,
                        'metadata': metadata,
                        'distance': distance
                    })
            
            # Sort by distance (lower is better) and prioritize PDFs
            documents.sort(key=lambda x: x['distance'])
            
            # Separate PDFs and general results
            pdf_docs = [d for d in documents if d.get('metadata', {}).get('source') == 'pdf']
            general_docs = [d for d in documents if d.get('metadata', {}).get('source') != 'pdf']
            
            # Log PDF results found
            logger.info(f"PDF documents found: {len(pdf_docs)}")
            if pdf_docs:
                pdf_filenames = set([d.get('metadata', {}).get('filename', 'unknown') for d in pdf_docs])
                logger.info(f"PDF files in results: {pdf_filenames}")
                # Log sample PDF content
                for i, pdf_doc in enumerate(pdf_docs[:3]):  # Log first 3 PDF chunks
                    filename = pdf_doc.get('metadata', {}).get('filename', 'unknown')
                    content_preview = pdf_doc.get('content', '')[:100]
                    distance = pdf_doc.get('distance', 0)
                    logger.info(f"PDF chunk {i+1}: {filename} (distance: {distance:.3f}) - {content_preview}...")
            else:
                logger.warning("No PDF documents found in search results!")
                # Check if PDFs exist in collection
                try:
                    all_docs = self.collection.get(limit=1000)
                    pdf_count = 0
                    pdf_filenames_in_db = set()
                    pdf_chunks_to_add = []
                    if all_docs.get('ids'):
                        for i, metadata in enumerate(all_docs.get('metadatas', [])):
                            if metadata and metadata.get('source') == 'pdf':
                                pdf_count += 1
                                pdf_filenames_in_db.add(metadata.get('filename', 'unknown'))
                                # Store PDF chunks for fallback
                                if all_docs.get('documents') and i < len(all_docs['documents']):
                                    pdf_chunks_to_add.append({
                                        'content': all_docs['documents'][i],
                                        'id': all_docs['ids'][i],
                                        'metadata': metadata,
                                        'distance': 0.95  # Default distance for fallback
                                    })
                    logger.info(f"Total PDF chunks in ChromaDB: {pdf_count} from files: {pdf_filenames_in_db}")
                    if pdf_count > 0 and len(pdf_docs) == 0:
                        logger.error("PDF chunks exist in DB but were not retrieved by search! Adding PDF chunks as fallback.")
                        # Add PDF chunks as fallback - include them even if search didn't find them
                        # This ensures PDF content is always available
                        pdf_docs = pdf_chunks_to_add[:10]  # Add up to 10 PDF chunks as fallback
                        logger.info(f"Added {len(pdf_docs)} PDF chunks as fallback to ensure PDF content is available")
                except Exception as check_error:
                    logger.error(f"Error checking PDF chunks in collection: {check_error}")
            
            # PRIORITIZE PDFs - Add PDF results FIRST, then general results
            # This ensures PDF data is always included if available
            combined = []
            seen_ids = set()
            
            # FIRST: Add ALL PDF results (they're most important for user queries)
            for doc in pdf_docs:
                if doc['id'] not in seen_ids:
                    combined.append(doc)
                    seen_ids.add(doc['id'])
                    logger.debug(f"Added PDF chunk: {doc.get('metadata', {}).get('filename', 'unknown')} (distance: {doc.get('distance', 0):.3f})")
            
            # THEN: Add general results (up to top_k)
            for doc in general_docs[:top_k]:
                if doc['id'] not in seen_ids:
                    combined.append(doc)
                    seen_ids.add(doc['id'])
            
            # If we still have space, add more general results
            for doc in general_docs[top_k:]:
                if doc['id'] not in seen_ids and len(combined) < top_k * 2:
                    combined.append(doc)
                    seen_ids.add(doc['id'])
            
            # Re-sort by distance (but PDFs are already included)
            combined.sort(key=lambda x: x['distance'])
            
            final_pdf_count = len([d for d in combined if d.get('metadata', {}).get('source') == 'pdf'])
            logger.info(f"Vector search: {len(combined)} total results ({final_pdf_count} PDF, {len(combined) - final_pdf_count} general) after filtering (distance <= 0.95)")
            return combined[:top_k * 2]  # Return up to 2x top_k for comprehensive results
        else:
            logger.warning(f"Vector search returned 0 results for query: '{query[:50]}...'")
            # Try to get all documents to verify collection has data
            try:
                all_docs = self.collection.get(limit=100)
                total_docs = len(all_docs.get('ids', [])) if all_docs else 0
                logger.info(f"Collection has {total_docs} total documents")
                if total_docs == 0:
                    logger.error("ChromaDB collection is EMPTY - no documents indexed!")
                else:
                    # If we have documents but no search results, try keyword matching as fallback
                    logger.warning("No semantic search results, but collection has documents. This might indicate an embedding mismatch.")
                    # Show sample document IDs for debugging
                    sample_ids = all_docs.get('ids', [])[:5]
                    logger.info(f"Sample document IDs in collection: {sample_ids}")
            except Exception as e:
                logger.error(f"Error checking collection: {e}")
        
        return documents
    
    def delete(self, ids: List[str]):
        """Delete documents by IDs"""
        self.collection.delete(ids=ids)
    
    def get_all(self) -> List[Dict]:
        """Get all documents"""
        results = self.collection.get()
        documents = []
        if results['ids']:
            for i, doc_id in enumerate(results['ids']):
                documents.append({
                    'id': doc_id,
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i] if results['metadatas'] else {}
                })
        return documents

