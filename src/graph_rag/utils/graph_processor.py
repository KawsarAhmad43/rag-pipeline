import logging
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.schema import Document
from spacy import load
from app.llm.llm import get_llm
from app.graph.graph import graph_db
from config import config
from langchain.prompts import PromptTemplate
from typing import List, Tuple
import hashlib
import ast
import re
import json
from app.embedding.embedding import get_embedding_model
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

try:
    nlp = load(config.SPACY_MODEL)
except OSError:
    logger.error(f"Spacy model '{config.SPACY_MODEL}' not found. Please run 'python -m spacy download {config.SPACY_MODEL}'")
    import sys
    sys.exit(1)

def extract_entities(text: str) -> List[dict]:
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "NORP", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"] and len(ent.text.strip()) > 2:
            clean_name = re.sub(r'[^\w\s\-\.]', ' ', ent.text).strip()
            clean_name = re.sub(r'\s+', ' ', clean_name)
            if clean_name and len(clean_name) > 2:
                entities.append({
                    "type": ent.label_,
                    "name": clean_name,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 1.0
                })
    return entities

def infer_entity_relationships(text: str, entities: List[dict]) -> List[dict]:
    if not entities or len(entities) < 2:
        logger.debug("No entities or insufficient entities for relationship inference")
        return []

    llm = get_llm()
    entity_names = [e["name"] for e in entities[:10]]

    prompt_template = """Given the following text and list of entities, identify valid relationships between the entities.
    Text: "{text}"
    Entities: {entity_names}
    Return a JSON list of dictionaries, each containing 'source', 'target', 'type', and 'confidence' (0-1).
    Valid relationship types: RELATED_TO, PART_OF, USES, CREATED_BY, LOCATED_IN, WORKS_FOR.
    'source' and 'target' MUST be exact matches to entity names from the provided list and MUST NOT include external knowledge.
    Example: [{"source": "Entity1", "target": "Entity2", "type": "RELATED_TO", "confidence": 0.9}]
    If no valid relationships are found, return an empty list: []
    Return ONLY the JSON, no extra text.
    """

    prompt = PromptTemplate.from_template(prompt_template)
    truncated_text = text[:1000]
    try:
        raw_response = llm.invoke(prompt.format(text=truncated_text, entity_names=entity_names)).content.strip()
        logger.debug(f"Raw LLM response for relationships: {raw_response[:200]}...")
    except Exception as e:
        logger.error(f"LLM invocation failed: {str(e)}")
        return []

    relations = parse_llm_response(raw_response)

    valid_relations = []
    valid_entity_names = set(entity_names)
    metadata_fields = {'source', 'doc_id', 'chunk_index', 'title', 'section', 'confidence', 'type'}

    for rel in relations:
        if (isinstance(rel, dict) and
            'source' in rel and 'target' in rel and 'type' in rel and 'confidence' in rel and
            isinstance(rel['source'], str) and isinstance(rel['target'], str) and
            rel['source'].strip(' "\'') in valid_entity_names and
            rel['target'].strip(' "\'') in valid_entity_names and
            rel['source'].strip(' "\'') != rel['target'].strip(' "\'') and
            rel['source'].lower() not in metadata_fields and
            rel['target'].lower() not in metadata_fields and
            rel['type'] in {"RELATED_TO", "PART_OF", "USES", "CREATED_BY", "LOCATED_IN", "WORKS_FOR"}):
            rel['source'] = rel['source'].strip(' "\'')
            rel['target'] = rel['target'].strip(' "\'')
            valid_relations.append(rel)
        else:
            logger.warning(f"Skipping invalid relationship: {rel}")
    return valid_relations

def parse_llm_response(response: str) -> List[dict]:
    if not response:
        logger.warning("Empty LLM response received")
        return []

    try:
        cleaned = re.sub(r'```(?:python|json)?\n?', '', response)
        cleaned = re.sub(r'\n?```', '', cleaned).strip()
        cleaned = re.sub(r'[\u2018\u2019]', "'", cleaned)
        cleaned = re.sub(r'[\u201c\u201d]', '"', cleaned)

        list_pattern = r'\[.*?\]'
        matches = re.findall(list_pattern, cleaned, re.DOTALL)
        list_str = matches[0] if matches else (cleaned if cleaned.startswith('[') and cleaned.endswith(']') else f"[{cleaned}]")

        try:
            parsed = json.loads(list_str)
            if not isinstance(parsed, list):
                logger.warning(f"Parsed LLM response is not a list: {list_str[:200]}...")
                return []
            valid_rels = [r for r in parsed if isinstance(r, dict) and all(k in r for k in ['source', 'target', 'type', 'confidence']) and
                          all(isinstance(r[k], (str, int, float)) for k in ['source', 'target', 'type', 'confidence'])]
            if not valid_rels:
                logger.warning(f"No valid relationships found in parsed response: {list_str[:200]}...")
            return valid_rels
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(list_str)
                if not isinstance(parsed, list):
                    logger.warning(f"AST parsed LLM response is not a list: {list_str[:200]}...")
                    return []
                valid_rels = [r for r in parsed if isinstance(r, dict) and all(k in r for k in ['source', 'target', 'type', 'confidence']) and
                              all(isinstance(r[k], (str, int, float)) for k in ['source', 'target', 'type', 'confidence'])]
                if not valid_rels:
                    logger.warning(f"No valid relationships found in AST parsed response: {list_str[:200]}...")
                return valid_rels
            except (ValueError, SyntaxError):
                logger.warning(f"Could not parse LLM response with ast.literal_eval: {list_str[:200]}...")
                return []

    except Exception as e:
        logger.error(f"Error parsing LLM response: {str(e)}, response: {response[:200]}...")
        return []

def extract_graph_from_text(texts: List[str], doc_id: str):
    llm = get_llm()
    transformer = LLMGraphTransformer(llm=llm)
    embedding_model = get_embedding_model()

    all_documents = []
    for text in texts:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        for sentence in sentences:
            if sentence and len(sentence.strip()) > 5:
                doc = Document(page_content=sentence, metadata={"doc_id": doc_id, "is_chunk": True})
                all_documents.append(doc)
        # Add full text as a document for comprehensive coverage
        if text.strip() and len(text.strip()) > 50:
            doc = Document(page_content=text, metadata={"doc_id": doc_id, "is_full": True})
            all_documents.append(doc)

    if not all_documents:
        logger.warning(f"No valid documents for graph extraction for doc_id {doc_id}")
        return []

    entity_map = {}
    try:
        graph_documents = transformer.convert_to_graph_documents(all_documents)
    except Exception as e:
        logger.error(f"Failed to convert to graph documents for doc_id {doc_id}: {str(e)}")
        graph_documents = []

    for graph_doc in graph_documents:
        for node in graph_doc.nodes:
            node_id = str(node.id) if node.id else f"node_{hashlib.md5(str(node).encode()).hexdigest()[:8]}"
            node_type = str(node.type) if node.type else "Entity"
            node_properties = {}
            if hasattr(node, 'properties') and isinstance(node.properties, dict):
                for k, v in node.properties.items():
                    if isinstance(k, str) and v is not None:
                        clean_k = re.sub(r'[^\w]', '_', k)[:50]
                        if isinstance(v, (str, int, float, bool)):
                            node_properties[clean_k] = v
            if 'name' not in node_properties:
                node_properties['name'] = node_id
            graph_db.add_node(node_type, node_id, node_properties)
            with graph_db.get_session() as session:
                session.run(
                    """
                    MATCH (n {id: $node_id})
                    MATCH (p:PDF {doc_id: $doc_id})
                    MERGE (n)-[:PART_OF]->(p)
                    """,
                    node_id=node_id,
                    doc_id=doc_id
                )
            entity_map[str(node.id)] = node_id

        for rel in graph_doc.relationships:
            source_id = str(rel.source.id) if rel.source and rel.source.id else None
            target_id = str(rel.target.id) if rel.target and rel.target.id else None
            rel_type = str(rel.type) if rel.type else "RELATED_TO"
            if source_id and target_id and source_id in entity_map and target_id in entity_map:
                graph_db.add_relationship(entity_map[source_id], entity_map[target_id], rel_type)
            else:
                logger.warning(f"Skipping invalid relationship from LLMGraphTransformer: source={source_id}, target={target_id}")

    for i, doc in enumerate(all_documents):
        entities = extract_entities(doc.page_content)
        if not entities:
            logger.debug(f"No entities extracted from document {i} for doc_id {doc_id}")
            continue
        chunk_id = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()[:8]
        for ent in entities:
            ent_name = ent["name"]
            ent_id = hashlib.md5(ent_name.encode('utf-8')).hexdigest()[:8]
            ment_id = hashlib.md5((ent_name + str(ent["start"])).encode('utf-8')).hexdigest()[:8]
            graph_db.add_node("Mention", ment_id, {"text": ent_name, "start": ent["start"], "end": ent["end"], "chunk_id": chunk_id, "confidence": ent["confidence"]})
            graph_db.add_node("Entity", ent_id, {"name": ent_name, "type": ent["type"], "doc_id": doc_id})
            graph_db.add_relationship(ent_id, ment_id, "HAS_MENTION")
            graph_db.add_relationship(ment_id, chunk_id, "IN_CHUNK")
            entity_map[ent_name] = ent_id

            ent_emb = embedding_model.embed_query(ent_name)
            cypher_sim = """
            MATCH (c:CanonicalEntity)
            WHERE vector.similarity.cosine(c.embedding, $ent_emb) > 0.95
            RETURN c.canonical_id, vector.similarity.cosine(c.embedding, $ent_emb) AS sim
            ORDER BY sim DESC LIMIT 1
            """
            similar = graph_db.query_graph(cypher_sim.replace("$ent_emb", str(ent_emb)))
            if similar:
                can_id = similar[0]['canonical_id']
                graph_db.add_relationship(ent_id, can_id, "SAME_AS", properties={"confidence": similar[0]['sim']})
            else:
                can_id = f"CE_{hashlib.md5(ent_name.encode('utf-8')).hexdigest()[:8]}"
                graph_db.add_node("CanonicalEntity", can_id, {"label": ent_name, "embedding": ent_emb})
                graph_db.add_relationship(ent_id, can_id, "SAME_AS", properties={"confidence": 1.0})

        if len(entities) > 1:
            try:
                relations = infer_entity_relationships(doc.page_content, entities)
                for rel in relations:
                    rel_id = hashlib.md5(f"{rel['source']}_{rel['type']}_{rel['target']}".encode('utf-8')).hexdigest()[:8]
                    source_id = entity_map.get(rel["source"])
                    target_id = entity_map.get(rel["target"])
                    if source_id and target_id:
                        graph_db.add_node("Relation", rel_id, {"type": rel["type"], "confidence": rel["confidence"]})
                        graph_db.add_relationship(source_id, rel_id, "RELATION")
                        graph_db.add_relationship(rel_id, target_id, "RELATION_TO")
                        graph_db.add_relationship(rel_id, chunk_id, "SUPPORTED_BY")
                    else:
                        logger.warning(f"Skipping relationship due to missing entity IDs: {rel}")
            except Exception as e:
                logger.error(f"Failed to infer relationships for document {i} (doc_id: {doc_id}): {str(e)}")
                continue
