from neo4j import GraphDatabase
from config import config
import logging
import time
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class Neo4jGraph:
    def __init__(self):
        self.driver = None
        self.database = config.NEO4J_DATABASE
        max_retries = 5
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to Neo4j at {config.NEO4J_URI} with user {config.NEO4J_USER}")
                uri = config.NEO4J_URI.replace("localhost", "127.0.0.1")
                self.driver = GraphDatabase.driver(
                    uri,
                    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
                    connection_timeout=30,
                )
                with self.driver.session(database=self.database) as session:
                    session.run("CALL dbms.components() YIELD name, versions RETURN name, versions LIMIT 1")
                logger.info(f"Neo4j driver initialized successfully for database '{self.database}'")
                self._create_indexes()
                break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed to initialize Neo4j driver: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        if not self.driver:
            raise ConnectionError("Failed to connect to Neo4j after multiple attempts")

    def get_session(self):
        return self.driver.session(database=self.database)

    def _create_indexes(self):
        with self.get_session() as session:
            session.run(
                """
                CREATE VECTOR INDEX embeddingIndex IF NOT EXISTS
                FOR (n:Chunk) ON (n.embedding)
                OPTIONS {
                  indexProvider: 'vector-2.0',
                  indexConfig: {
                    `vector.dimensions`: $dimension,
                    `vector.similarity_function`: 'cosine'
                  }
                }
                """,
                dimension=config.EMBEDDING_DIMENSION,
            )
            logger.info("Vector index on :Chunk(embedding) created or already exists in Neo4j.")

            session.run(
                """
                CREATE FULLTEXT INDEX entityNamesIndex IF NOT EXISTS
                FOR (n:Entity) ON EACH [n.name]
                OPTIONS {
                  indexProvider: 'fulltext-1.0',
                  indexConfig: {
                    `fulltext.analyzer`: 'standard'
                  }
                }
                """
            )
            logger.info("Full-text index on :Entity(name) created or already exists in Neo4j.")

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed")

    def _sanitize_label(self, label: str) -> str:
        if not label or not isinstance(label, str):
            return "UNKNOWN"
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", str(label).strip())
        if not sanitized:
            return "UNKNOWN"
        if sanitized[0].isdigit():
            sanitized = "NODE_" + sanitized
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        return sanitized.upper()

    def add_node(self, label: str, id: str, properties: dict):
        try:
            sanitized_label = self._sanitize_label(label)
            if not id or not isinstance(id, str):
                id = f"node_{hash(str(properties))}"[:16]
            clean_properties = {}
            if properties:
                for k, v in properties.items():
                    if isinstance(k, str) and k.strip() and v is not None:
                        clean_key = re.sub(r"[^a-zA-Z0-9_]", "_", k.strip())[:50]
                        if isinstance(v, (str, int, float, bool)):
                            clean_properties[clean_key] = v
                        elif isinstance(v, list) and all(isinstance(item, (str, int, float)) for item in v):
                            clean_properties[clean_key] = v
                        else:
                            clean_properties[clean_key] = str(v)[:500]
            with self.get_session() as session:
                q = f"MERGE (n:{sanitized_label} {{id: $id}}) SET n += $properties"
                session.run(q, id=str(id), properties=clean_properties)
            logger.debug(f"Added node: {sanitized_label} with id {id}")
        except Exception as e:
            logger.error(f"Error adding node with label '{label}' and id '{id}': {str(e)}")
            raise

    def add_relationship(self, source_id: str, target_id: str, rel_type: str, properties: dict = None):
        try:
            if not all([source_id, target_id, rel_type]):
                logger.warning(f"Invalid relationship parameters: {source_id} -[{rel_type}]-> {target_id}")
                return
            sanitized_rel_type = re.sub(r"[^a-zA-Z0-9_]", "_", str(rel_type).strip()).upper()
            sanitized_rel_type = sanitized_rel_type or "RELATED_TO"
            if len(sanitized_rel_type) > 50:
                sanitized_rel_type = sanitized_rel_type[:50]
            query = """
            MATCH (source {id: $source_id}), (target {id: $target_id})
            MERGE (source)-[r:%s]->(target)
            """ % sanitized_rel_type
            params = {"source_id": str(source_id), "target_id": str(target_id)}
            if properties:
                set_clause = " SET " + ", ".join([f"r.{k} = ${k}" for k in properties])
                query += set_clause
                params.update(properties)
            with self.get_session() as session:
                session.run(query, **params)
            logger.debug(f"Added relationship: {source_id} -[{sanitized_rel_type}]-> {target_id}")
        except Exception as e:
            logger.error(f"Error adding relationship {source_id} -[{rel_type}]-> {target_id}: {str(e)}")

    def query_graph(self, cypher_query: str):
        with self.get_session() as session:
            try:
                results = session.run(cypher_query)
                context = []
                for record in results:
                    record_dict = dict(record)
                    context.append(str(record_dict))
                logger.debug(f"Graph query returned {len(context)} results for query: {cypher_query[:50]}...")
                return context
            except Exception as e:
                logger.error(f"Failed to execute Cypher query '{cypher_query[:100]}...': {e}")
                return []

    def health_check(self):
        try:
            with self.get_session() as session:
                session.run("MATCH (n) RETURN n LIMIT 1")
            logger.info("Neo4j health check passed")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    # Compact helpers to avoid Cypher errors seen in logs:

    def get_chunk_neighborhood_compact(self, chunk_id: str) -> Dict:
        cypher = """
        MATCH (c:Chunk {id: $chunk_id})
        OPTIONAL MATCH (c)-[:HAS_MENTION]->(m:Mention)-[:IN_CHUNK]->(c2:Chunk)
        OPTIONAL MATCH (c)-[:PART_OF]->(p:PDF)
        OPTIONAL MATCH (c)<-[:HAS_CHUNK]-(s:Section)
        OPTIONAL MATCH (c)<-[:IN_CHUNK]-(:Mention)<-[:HAS_MENTION]-(e:Entity)
        WITH c, p, s, collect(DISTINCT e) AS ents, collect(DISTINCT c2) AS related_chunks
        RETURN {
          source_chunk: { id: c.id, doc_id: c.doc_id, chunk_index: c.chunk_index, content: c.content },
          pdf: CASE WHEN p IS NULL THEN null ELSE {doc_id: p.doc_id, title: p.title} END,
          section: CASE WHEN s IS NULL THEN null ELSE {name: s.name} END,
          entities: [e IN ents | {id: e.id, name: e.name, type: e.type}],
          related_chunks: [rc IN related_chunks | {id: rc.id, doc_id: rc.doc_id, chunk_index: rc.chunk_index}]
        } AS neighborhood
        """
        try:
            with self.get_session() as session:
                rec = session.run(cypher, chunk_id=chunk_id).single()
                if rec:
                    return rec["neighborhood"]
                return {}
        except Exception as e:
            logger.error(f"Error get_chunk_neighborhood_compact: {e}")
            return {}

    def get_document_title(self, doc_id: str) -> Optional[str]:
        cypher = """
        MATCH (p:PDF {doc_id: $doc_id})
        RETURN p.title AS title
        """
        try:
            with self.get_session() as session:
                rec = session.run(cypher, doc_id=doc_id).single()
                if rec and rec.get("title"):
                    return rec["title"]
                return None
        except Exception as e:
            logger.error(f"Error get_document_title: {e}")
            return None


graph_db = Neo4jGraph()
