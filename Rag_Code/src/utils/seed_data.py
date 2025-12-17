"""Seed data for Neo4j and ChromaDB"""
from src.services.vector_store import VectorStore
from src.services.graph_store import GraphStore
import logging

logger = logging.getLogger(__name__)

def seed_vector_store():
    """Seed ChromaDB with initial data"""
    vector_store = VectorStore()
    
    documents = [
        "Data structures are fundamental building blocks in computer science. They organize and store data efficiently for algorithms to process.",
        "Arrays are linear data structures that store elements in contiguous memory locations. They provide O(1) access time but fixed size.",
        "Linked lists are dynamic data structures where elements are connected via pointers. They allow efficient insertion and deletion.",
        "Trees are hierarchical data structures with a root node and child nodes. Binary trees have at most two children per node.",
        "Graphs represent relationships between entities using nodes and edges. They are essential for modeling networks and relationships.",
        "Hash tables provide O(1) average-case lookup time using hash functions to map keys to values.",
        "Stacks follow LIFO (Last In First Out) principle. Operations include push, pop, and peek.",
        "Queues follow FIFO (First In First Out) principle. Operations include enqueue and dequeue.",
        "Artificial Intelligence (AI) is the simulation of human intelligence by machines. It includes learning, reasoning, and self-correction.",
        "Machine Learning is a subset of AI that enables systems to learn from data without explicit programming.",
        "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons).",
        "Deep Learning uses neural networks with multiple layers to learn complex patterns in data.",
        "Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language.",
        "Computer Vision allows machines to interpret and understand visual information from images and videos.",
        "Reinforcement Learning is a type of ML where agents learn by interacting with an environment and receiving rewards.",
        "Retrieval-Augmented Generation (RAG) combines information retrieval with language generation for accurate responses.",
        "RAG systems retrieve relevant documents from a knowledge base and use them as context for generating answers.",
        "Vector databases store embeddings of documents for semantic search. They enable finding similar content based on meaning.",
        "ChromaDB is an open-source vector database designed for AI applications and semantic search.",
        "Embeddings are dense vector representations of text that capture semantic meaning. Similar texts have similar embeddings.",
        "LangChain is a framework for building applications with Large Language Models (LLMs).",
        "LangChain provides tools for chaining LLM calls, managing prompts, and integrating with data sources.",
        "LangChain supports multiple LLM providers including OpenAI, Anthropic, and Google's Gemini.",
        "Agents in LangChain can use tools and make decisions based on user queries and available data.",
        "Chains in LangChain connect multiple components to create complex workflows for LLM applications.",
        "Memory in LangChain allows agents to maintain conversation history and context across interactions.",
        "Vector stores in LangChain integrate with databases like ChromaDB, Pinecone, and Weaviate for semantic search.",
        "Document loaders in LangChain can load data from various sources including PDFs, web pages, and databases.",
        "Text splitters in LangChain break documents into chunks suitable for embedding and retrieval.",
        "Prompt templates in LangChain help structure inputs to LLMs for consistent and effective responses."
    ]
    
    metadatas = [
        {"topic": "data_structures", "category": "fundamentals"},
        {"topic": "data_structures", "category": "arrays"},
        {"topic": "data_structures", "category": "linked_lists"},
        {"topic": "data_structures", "category": "trees"},
        {"topic": "data_structures", "category": "graphs"},
        {"topic": "data_structures", "category": "hash_tables"},
        {"topic": "data_structures", "category": "stacks"},
        {"topic": "data_structures", "category": "queues"},
        {"topic": "ai", "category": "introduction"},
        {"topic": "ai", "category": "machine_learning"},
        {"topic": "ai", "category": "neural_networks"},
        {"topic": "ai", "category": "deep_learning"},
        {"topic": "ai", "category": "nlp"},
        {"topic": "ai", "category": "computer_vision"},
        {"topic": "ai", "category": "reinforcement_learning"},
        {"topic": "rag", "category": "introduction"},
        {"topic": "rag", "category": "concepts"},
        {"topic": "rag", "category": "vector_databases"},
        {"topic": "rag", "category": "chromadb"},
        {"topic": "rag", "category": "embeddings"},
        {"topic": "langchain", "category": "introduction"},
        {"topic": "langchain", "category": "framework"},
        {"topic": "langchain", "category": "llm_providers"},
        {"topic": "langchain", "category": "agents"},
        {"topic": "langchain", "category": "chains"},
        {"topic": "langchain", "category": "memory"},
        {"topic": "langchain", "category": "vector_stores"},
        {"topic": "langchain", "category": "document_loaders"},
        {"topic": "langchain", "category": "text_splitters"},
        {"topic": "langchain", "category": "prompt_templates"}
    ]
    
    try:
        vector_store.add_documents(documents, metadatas)
        logger.info(f"Seeded {len(documents)} documents to ChromaDB")
        return True
    except Exception as e:
        logger.error(f"Error seeding vector store: {e}")
        return False

def seed_graph_store():
    """Seed Neo4j with initial data - same content as vector DB"""
    try:
        graph_store = GraphStore()
        logger.info("Neo4j connection successful")
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        logger.error("Please check Neo4j is running and credentials are correct")
        return False
    
    try:
        # First, clear existing data (optional - comment out if you want to keep existing data)
        try:
            graph_store.execute_cypher("MATCH (n) DETACH DELETE n")
            logger.info("Cleared existing Neo4j data")
        except Exception as e:
            logger.warning(f"Could not clear existing data: {e}")
        
        # Same documents as vector DB
        documents = [
            "Data structures are fundamental building blocks in computer science. They organize and store data efficiently for algorithms to process.",
            "Arrays are linear data structures that store elements in contiguous memory locations. They provide O(1) access time but fixed size.",
            "Linked lists are dynamic data structures where elements are connected via pointers. They allow efficient insertion and deletion.",
            "Trees are hierarchical data structures with a root node and child nodes. Binary trees have at most two children per node.",
            "Graphs represent relationships between entities using nodes and edges. They are essential for modeling networks and relationships.",
            "Hash tables provide O(1) average-case lookup time using hash functions to map keys to values.",
            "Stacks follow LIFO (Last In First Out) principle. Operations include push, pop, and peek.",
            "Queues follow FIFO (First In First Out) principle. Operations include enqueue and dequeue.",
            "Artificial Intelligence (AI) is the simulation of human intelligence by machines. It includes learning, reasoning, and self-correction.",
            "Machine Learning is a subset of AI that enables systems to learn from data without explicit programming.",
            "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons).",
            "Deep Learning uses neural networks with multiple layers to learn complex patterns in data.",
            "Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language.",
            "Computer Vision allows machines to interpret and understand visual information from images and videos.",
            "Reinforcement Learning is a type of ML where agents learn by interacting with an environment and receiving rewards.",
            "Retrieval-Augmented Generation (RAG) combines information retrieval with language generation for accurate responses.",
            "RAG systems retrieve relevant documents from a knowledge base and use them as context for generating answers.",
            "Vector databases store embeddings of documents for semantic search. They enable finding similar content based on meaning.",
            "ChromaDB is an open-source vector database designed for AI applications and semantic search.",
            "Embeddings are dense vector representations of text that capture semantic meaning. Similar texts have similar embeddings.",
            "LangChain is a framework for building applications with Large Language Models (LLMs).",
            "LangChain provides tools for chaining LLM calls, managing prompts, and integrating with data sources.",
            "LangChain supports multiple LLM providers including OpenAI, Anthropic, and Google's Gemini.",
            "Agents in LangChain can use tools and make decisions based on user queries and available data.",
            "Chains in LangChain connect multiple components to create complex workflows for LLM applications.",
            "Memory in LangChain allows agents to maintain conversation history and context across interactions.",
            "Vector stores in LangChain integrate with databases like ChromaDB, Pinecone, and Weaviate for semantic search.",
            "Document loaders in LangChain can load data from various sources including PDFs, web pages, and databases.",
            "Text splitters in LangChain break documents into chunks suitable for embedding and retrieval.",
            "Prompt templates in LangChain help structure inputs to LLMs for consistent and effective responses."
        ]
        
        metadatas = [
            {"topic": "data_structures", "category": "fundamentals"},
            {"topic": "data_structures", "category": "arrays"},
            {"topic": "data_structures", "category": "linked_lists"},
            {"topic": "data_structures", "category": "trees"},
            {"topic": "data_structures", "category": "graphs"},
            {"topic": "data_structures", "category": "hash_tables"},
            {"topic": "data_structures", "category": "stacks"},
            {"topic": "data_structures", "category": "queues"},
            {"topic": "ai", "category": "introduction"},
            {"topic": "ai", "category": "machine_learning"},
            {"topic": "ai", "category": "neural_networks"},
            {"topic": "ai", "category": "deep_learning"},
            {"topic": "ai", "category": "nlp"},
            {"topic": "ai", "category": "computer_vision"},
            {"topic": "ai", "category": "reinforcement_learning"},
            {"topic": "rag", "category": "introduction"},
            {"topic": "rag", "category": "concepts"},
            {"topic": "rag", "category": "vector_databases"},
            {"topic": "rag", "category": "chromadb"},
            {"topic": "rag", "category": "embeddings"},
            {"topic": "langchain", "category": "introduction"},
            {"topic": "langchain", "category": "framework"},
            {"topic": "langchain", "category": "llm_providers"},
            {"topic": "langchain", "category": "agents"},
            {"topic": "langchain", "category": "chains"},
            {"topic": "langchain", "category": "memory"},
            {"topic": "langchain", "category": "vector_stores"},
            {"topic": "langchain", "category": "document_loaders"},
            {"topic": "langchain", "category": "text_splitters"},
            {"topic": "langchain", "category": "prompt_templates"}
        ]
        
        # Create Document nodes with full content (same as vector DB)
        document_ids = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            # Extract title from first sentence
            title = doc.split('.')[0][:50] if doc else f"Document {i+1}"
            
            node_props = {
                "title": title,
                "content": doc,
                "topic": meta.get("topic", "general"),
                "category": meta.get("category", "general"),
                "doc_id": i
            }
            
            node_id = graph_store.create_node("Document", node_props)
            if node_id:
                document_ids.append((int(node_id), meta, i))
        
        # Create topic nodes and link documents to topics
        topic_map = {}
        unique_topics = set(meta.get("topic", "general") for meta in metadatas)
        
        for topic_name in unique_topics:
            topic_node = {
                "name": topic_name.replace("_", " ").title(),
                "topic_id": topic_name
            }
            node_id = graph_store.create_node("Topic", topic_node)
            if node_id:
                topic_map[topic_name] = int(node_id)
        
        # Link documents to topics
        for doc_id, meta, _ in document_ids:
            topic_name = meta.get("topic", "general")
            topic_id = topic_map.get(topic_name)
            if topic_id:
                graph_store.create_relationship(topic_id, doc_id, "CONTAINS")
        
        # Create concept nodes from key terms
        concepts = {
            "Array": ["arrays", "array"],
            "Linked List": ["linked list", "linked lists"],
            "Tree": ["trees", "tree", "hierarchical"],
            "Graph": ["graphs", "graph", "relationships"],
            "Hash Table": ["hash table", "hash tables"],
            "Stack": ["stacks", "stack", "LIFO"],
            "Queue": ["queues", "queue", "FIFO"],
            "AI": ["artificial intelligence", "AI"],
            "Machine Learning": ["machine learning", "ML"],
            "Neural Network": ["neural network", "neural networks"],
            "Deep Learning": ["deep learning"],
            "NLP": ["natural language processing", "NLP"],
            "Computer Vision": ["computer vision"],
            "Reinforcement Learning": ["reinforcement learning"],
            "RAG": ["retrieval-augmented generation", "RAG"],
            "ChromaDB": ["chromadb", "chroma"],
            "Embedding": ["embeddings", "embedding"],
            "LangChain": ["langchain", "lang chain"],
            "Vector Database": ["vector database", "vector databases"]
        }
        
        concept_ids = {}
        for concept_name, keywords in concepts.items():
            concept_node = {
                "name": concept_name,
                "keywords": ", ".join(keywords)
            }
            node_id = graph_store.create_node("Concept", concept_node)
            if node_id:
                concept_ids[concept_name] = int(node_id)
        
        # Link documents to concepts based on content matching
        for doc_id, meta, doc_index in document_ids:
            doc_content_lower = documents[doc_index].lower()
            for concept_name, concept_id in concept_ids.items():
                keywords = concepts[concept_name]
                if any(keyword.lower() in doc_content_lower for keyword in keywords):
                    graph_store.create_relationship(concept_id, doc_id, "DESCRIBES")
        
        # Create relationships between concepts
        concept_relationships = [
            ("Data Structures", "Array", "CONTAINS"),
            ("Data Structures", "Linked List", "CONTAINS"),
            ("Data Structures", "Tree", "CONTAINS"),
            ("Data Structures", "Graph", "CONTAINS"),
            ("Data Structures", "Hash Table", "CONTAINS"),
            ("Data Structures", "Stack", "CONTAINS"),
            ("Data Structures", "Queue", "CONTAINS"),
            ("AI", "Machine Learning", "CONTAINS"),
            ("Machine Learning", "Neural Network", "USES"),
            ("Machine Learning", "Deep Learning", "INCLUDES"),
            ("Machine Learning", "NLP", "INCLUDES"),
            ("Machine Learning", "Computer Vision", "INCLUDES"),
            ("Machine Learning", "Reinforcement Learning", "INCLUDES"),
            ("RAG", "Vector Database", "USES"),
            ("RAG", "Embedding", "USES"),
            ("RAG", "ChromaDB", "USES"),
            ("LangChain", "RAG", "SUPPORTS"),
            ("Vector Database", "ChromaDB", "INCLUDES"),
            ("Vector Database", "Embedding", "STORES")
        ]
        
        for from_name, to_name, rel_type in concept_relationships:
            from_id = concept_ids.get(from_name)
            to_id = concept_ids.get(to_name)
            if from_id and to_id:
                graph_store.create_relationship(from_id, to_id, rel_type)
        
        logger.info(f"Seeded Neo4j with {len(document_ids)} documents, {len(topic_map)} topics, and {len(concept_ids)} concepts")
        return True
    except Exception as e:
        logger.error(f"Error seeding graph store: {e}", exc_info=True)
        return False

def seed_all():
    """Seed both vector and graph stores"""
    logger.info("Starting data seeding...")
    vector_success = seed_vector_store()
    graph_success = seed_graph_store()
    
    if vector_success and graph_success:
        logger.info("Data seeding completed successfully")
    else:
        logger.warning("Data seeding completed with some errors")
    
    return vector_success and graph_success

