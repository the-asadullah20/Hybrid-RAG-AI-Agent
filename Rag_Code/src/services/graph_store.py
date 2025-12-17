"""Neo4j Graph Database Service"""
from neo4j import GraphDatabase
from typing import List, Dict, Optional
from src.config import Config
import logging

logger = logging.getLogger(__name__)

class GraphStore:
    def __init__(self):
        # Use bolt:// for single instance (neo4j:// is for clustering)
        uri = Config.NEO4J_URI
        if uri.startswith('neo4j://'):
            uri = uri.replace('neo4j://', 'bolt://')
            logger.warning(f"Changed neo4j:// to bolt:// (use bolt:// for single instance Neo4j)")
        
        self.driver = GraphDatabase.driver(
            uri,
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Neo4j connection"""
        try:
            uri = Config.NEO4J_URI.replace('neo4j://', 'bolt://') if Config.NEO4J_URI.startswith('neo4j://') else Config.NEO4J_URI
            logger.info(f"Attempting to connect to Neo4j at {uri}")
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record:
                    logger.info("Neo4j connection verified successfully")
                else:
                    raise Exception("Connection test returned no result")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            uri = Config.NEO4J_URI.replace('neo4j://', 'bolt://') if Config.NEO4J_URI.startswith('neo4j://') else Config.NEO4J_URI
            logger.error(f"URI: {uri}, User: {Config.NEO4J_USER}")
            logger.error("Please ensure Neo4j is running and use bolt:// for single instance (not neo4j://)")
            raise
    
    def execute_cypher(self, cypher_query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute Cypher query and return results"""
        if parameters is None:
            parameters = {}
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, parameters)
                records = []
                for record in result:
                    records.append(dict(record))
                return records
        except Exception as e:
            logger.error(f"Cypher query error: {e}")
            return []
    
    def create_node(self, label: str, properties: Dict) -> str:
        """Create a node with label and properties (uses MERGE to avoid duplicates)"""
        # Use title or name as unique identifier if available
        key_prop = None
        if 'title' in properties:
            key_prop = 'title'
        elif 'name' in properties:
            key_prop = 'name'
        elif 'doc_id' in properties:
            key_prop = 'doc_id'
        
        if key_prop:
            # Use MERGE to avoid duplicates based on unique key
            set_clauses = ', '.join([f"n.{k} = ${k}" for k in properties.keys()])
            cypher = f"""
            MERGE (n:{label} {{{key_prop}: ${key_prop}}})
            ON CREATE SET {set_clauses}
            ON MATCH SET {set_clauses}
            RETURN id(n) as node_id
            """
        else:
            # Fallback to CREATE if no unique key
            props_str = ', '.join([f"{k}: ${k}" for k in properties.keys()])
            cypher = f"CREATE (n:{label} {{{props_str}}}) RETURN id(n) as node_id"
        
        result = self.execute_cypher(cypher, properties)
        return str(result[0]['node_id']) if result else None
    
    def create_relationship(self, from_id: int, to_id: int, rel_type: str, properties: Optional[Dict] = None):
        """Create relationship between nodes"""
        if properties is None:
            properties = {}
        
        props_str = ', '.join([f"{k}: ${k}" for k in properties.keys()]) if properties else ""
        props_part = f" {{{props_str}}}" if props_str else ""
        
        cypher = f"""
        MATCH (a), (b)
        WHERE id(a) = $from_id AND id(b) = $to_id
        CREATE (a)-[r:{rel_type}{props_part}]->(b)
        RETURN r
        """
        params = {'from_id': from_id, 'to_id': to_id, **properties}
        return self.execute_cypher(cypher, params)
    
    def search_by_keyword(self, keyword: str, limit: int = Config.GRAPH_SEARCH_TOP_K) -> List[Dict]:
        """Enhanced search - searches more comprehensively across all node types"""
        # Extract all meaningful keywords (not just 3)
        words = keyword.lower().split()
        keywords = [w for w in words if len(w) > 2]  # Include words longer than 2 chars
        if not keywords:
            keywords = words[:3]  # Fallback to first 3 words
        
        if not keywords:
            return []
        
        results = []
        seen_nodes = set()
        
        # Enhanced search: Try full query first, then individual keywords
        search_terms = [keyword] + keywords[:5]  # Full query + up to 5 keywords
        
        for search_term in search_terms:
            if len(results) >= limit * 2:  # Get more results to filter
                break
            
            # Search Document nodes with more flexible matching
            cypher = """
            MATCH (n:Document)
            WHERE toLower(n.content) CONTAINS toLower($keyword)
               OR toLower(n.title) CONTAINS toLower($keyword)
               OR toLower(n.topic) CONTAINS toLower($keyword)
               OR toLower(n.category) CONTAINS toLower($keyword)
               OR toLower(n.filename) CONTAINS toLower($keyword)
            RETURN n, labels(n) as labels, id(n) as node_id
            LIMIT $limit
            """
            doc_results = self.execute_cypher(cypher, {'keyword': search_term, 'limit': limit * 2})
            
            for r in doc_results:
                node_id = r.get('node_id') or str(id(r.get('n', {})))
                if node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    results.append(r)
        
        # Search Concept and Topic nodes if we need more results
        if len(results) < limit:
            remaining = limit * 2 - len(results)
            for search_term in search_terms[:3]:  # Use first 3 terms
                cypher2 = """
                MATCH (n)
                WHERE (n:Concept OR n:Topic)
                  AND (toLower(n.name) CONTAINS toLower($keyword) 
                   OR toLower(toString(n.keywords)) CONTAINS toLower($keyword))
                RETURN n, labels(n) as labels, id(n) as node_id
                LIMIT $remaining
                """
                additional = self.execute_cypher(cypher2, {'keyword': search_term, 'remaining': remaining})
                for r in additional:
                    node_id = r.get('node_id') or str(id(r.get('n', {})))
                    if node_id not in seen_nodes:
                        seen_nodes.add(node_id)
                        results.append(r)
                if len(results) >= limit * 2:
                    break
        
        logger.info(f"Enhanced keyword search for '{keyword}' returned {len(results)} results")
        return results[:limit * 2]  # Return more results for better coverage
    
    def get_related_nodes(self, node_id: int, relationship_type: Optional[str] = None, depth: int = 1) -> List[Dict]:
        """Get related nodes"""
        rel_filter = f":{relationship_type}" if relationship_type else ""
        cypher = f"""
        MATCH (n)-[r{rel_filter}*1..{depth}]-(related)
        WHERE id(n) = $node_id
        RETURN DISTINCT related, labels(related) as labels
        LIMIT 20
        """
        return self.execute_cypher(cypher, {'node_id': node_id})
    
    def close(self):
        """Close database connection"""
        self.driver.close()

