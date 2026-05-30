import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


@dataclass
class KGNode:
    entity_id: str
    label: str
    source_document: str
    source_chunk: str
    confidence_score: float
    created_at: str
    metadata: Dict


@dataclass
class KGRelation:
    relation_id: str
    source_id: str
    target_id: str
    relation_type: str
    source_document: str
    source_chunk: str
    confidence_score: float
    created_at: str
    metadata: Dict


class BaseKnowledgeGraph(ABC):
    @abstractmethod
    def add_node(self, node: KGNode):
        pass

    @abstractmethod
    def add_edge(self, relation: KGRelation):
        pass

    @abstractmethod
    def query(self, query_text: str, top_n: int = 10) -> List[Dict]:
        pass


class NetworkXKnowledgeGraph(BaseKnowledgeGraph):
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node: KGNode):
        self.graph.add_node(node.entity_id, **asdict(node))

    def add_edge(self, relation: KGRelation):
        self.graph.add_edge(relation.source_id, relation.target_id, **asdict(relation))

    def query(self, query_text: str, top_n: int = 10) -> List[Dict]:
        query_lower = query_text.lower()
        results = []
        for node_id, data in self.graph.nodes(data=True):
            label = str(data.get("label", "")).lower()
            if query_lower in label:
                results.append({"entity_id": node_id, **data})
        return results[:top_n]

    def save(self, path: Path):
        data = {
            "nodes": [asdict(KGNode(**self.graph.nodes[node])) for node in self.graph.nodes],
            "edges": [dict(self.graph.edges[edge]) for edge in self.graph.edges],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for node in payload.get("nodes", []):
            self.graph.add_node(node["entity_id"], **node)
        for edge in payload.get("edges", []):
            self.graph.add_edge(edge["source_id"], edge["target_id"], **edge)


class Neo4jKnowledgeGraph(BaseKnowledgeGraph):
    def __init__(self, uri: str, user: str, password: str):
        if GraphDatabase is None:
            raise RuntimeError("neo4j package is not installed")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def add_node(self, node: KGNode):
        with self.driver.session() as session:
            session.run(
                "MERGE (e:Entity {entity_id: $entity_id}) "
                "SET e.label = $label, e.source_document = $source_document, "
                "e.source_chunk = $source_chunk, e.confidence_score = $confidence_score, "
                "e.created_at = $created_at, e.metadata = $metadata",
                asdict(node),
            )

    def add_edge(self, relation: KGRelation):
        with self.driver.session() as session:
            session.run(
                "MATCH (s:Entity {entity_id: $source_id}), (t:Entity {entity_id: $target_id}) "
                "MERGE (s)-[r:RELATION {relation_id: $relation_id}]->(t) "
                "SET r.relation_type = $relation_type, r.source_document = $source_document, "
                "r.source_chunk = $source_chunk, r.confidence_score = $confidence_score, "
                "r.created_at = $created_at, r.metadata = $metadata",
                asdict(relation),
            )

    def query(self, query_text: str, top_n: int = 10) -> List[Dict]:
        statement = (
            "MATCH (e:Entity) WHERE toLower(e.label) CONTAINS toLower($query_text) "
            "RETURN e LIMIT $top_n"
        )
        with self.driver.session() as session:
            result = session.run(statement, query_text=query_text, top_n=top_n)
            return [record["e"] for record in result]


class KnowledgeGraphBuilder:
    def __init__(self, kg: BaseKnowledgeGraph):
        self.kg = kg

    def add_fact(self, entity_label: str, target_label: str, relation_type: str, source_document: str, source_chunk: str, confidence_score: float = 0.85, metadata: Optional[Dict] = None):
        metadata = metadata or {}
        source_id = f"entity-{entity_label}-{source_chunk}"
        target_id = f"entity-{target_label}-{source_chunk}"
        now = datetime.utcnow().isoformat() + "Z"
        self.kg.add_node(KGNode(
            entity_id=source_id,
            label=entity_label,
            source_document=source_document,
            source_chunk=source_chunk,
            confidence_score=confidence_score,
            created_at=now,
            metadata=metadata,
        ))
        self.kg.add_node(KGNode(
            entity_id=target_id,
            label=target_label,
            source_document=source_document,
            source_chunk=source_chunk,
            confidence_score=confidence_score,
            created_at=now,
            metadata=metadata,
        ))
        self.kg.add_edge(KGRelation(
            relation_id=f"rel-{source_id}-{target_id}",
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            source_document=source_document,
            source_chunk=source_chunk,
            confidence_score=confidence_score,
            created_at=now,
            metadata=metadata,
        ))

    def extract_entities(self, text: str) -> List[str]:
        candidates = []
        for chunk in text.split(". "):
            for token in chunk.split():
                if token.istitle() and len(token) > 2:
                    candidates.append(token)
        return list(dict.fromkeys(candidates))

    def extract_relations(self, text: str) -> List[Dict]:
        relations = []
        entity_list = self.extract_entities(text)
        if len(entity_list) >= 2:
            relations.append({
                "source": entity_list[0],
                "target": entity_list[1],
                "type": "related_to",
            })
        return relations

    def build_graph(self, chunks: List[Dict]):
        for chunk in chunks:
            entities = self.extract_entities(chunk["chunk_text"])
            relations = self.extract_relations(chunk["chunk_text"])
            for relation in relations:
                self.add_fact(
                    entity_label=relation["source"],
                    target_label=relation["target"],
                    relation_type=relation["type"],
                    source_document=chunk["metadata"].get("source_file", "unknown"),
                    source_chunk=chunk["chunk_id"],
                    confidence_score=0.8,
                    metadata=chunk["metadata"],
                )
