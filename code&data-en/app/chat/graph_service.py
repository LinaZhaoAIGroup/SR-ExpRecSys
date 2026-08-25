from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Callable

import numpy as np
import pandas as pd
from django.conf import settings
from py2neo import Graph, Node
from zhipuai import ZhipuAI
logger = logging.getLogger(__name__)

class Py2NeoGraphRAG:

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        triples_path: Optional[Path] = None,
        embeddings_batch_size: int = 64,
        api_key: Optional[str] = None,
    ) -> None:
        self.graph = Graph(uri, auth=(user, password))
        if not api_key:
            raise ValueError("GraphRAG requires a ZhipuAI API key.")
        self.client = ZhipuAI(api_key=api_key)
        self.triples_path = triples_path
        self.embeddings_batch_size = embeddings_batch_size
        self.st: List[str] = []
        self.embeddings_dict: Dict[str, Sequence[float]] = {}
        self._graph_built = False

    # ------------------------------------------------------------------
    # build phase
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        if not self.triples_path or not self.triples_path.exists():
            logger.warning("Knowledge graph triple file not found, skip building: %s", self.triples_path)
            return

        try:
            self.create_graph()
            self.create_embeddings()
            self._graph_built = True
            logger.info("The construction of the knowledge graph is completed, and the number of entities: %s", len(self.st))
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Knowledge graph construction failed: %s", exc)
            self._graph_built = False

    def create_graph(self) -> None:
        logger.info("Start buildingNeo4j Knowledge graph")
        data = pd.read_excel(self.triples_path).dropna()
        self.graph.delete_all()
        created_nodes: set[str] = set()

        # Create entity node
        for _, row in data.iterrows():
            sub_type, sub_name = row["sub_type"], row["sub_name"]
            obj_type, obj_name = row["obj_type"], row["obj_name"]

            if sub_name not in created_nodes:
                self.graph.create(Node(sub_type, name=sub_name))
                created_nodes.add(sub_name)
            if obj_name not in created_nodes:
                self.graph.create(Node(obj_type, name=obj_name))
                created_nodes.add(obj_name)

        # Create relationship
        for _, row in data.iterrows():
            query = (
                "MATCH (p {name: $sub_name}), (q {name: $obj_name}) "
                "CREATE (p)-[:`" + row["rel_name"] + "`]->(q)"
            )
            try:
                self.graph.run(query, sub_name=row["sub_name"], obj_name=row["obj_name"])
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("Failed to create relationship: %s", exc)

        self.st = list(created_nodes)

    def create_embeddings(self) -> None:
        if not self.st:
            logger.warning("No entities found, vector creation skipped")
            return

        embeddings: List[Sequence[float]] = []
        for start in range(0, len(self.st), self.embeddings_batch_size):
            batch = self.st[start : start + self.embeddings_batch_size]
            response = self.client.embeddings.create(model="embedding-3", input=batch)
            embeddings.extend(item.embedding for item in response.data)

        self.embeddings_dict = {name: emb for name, emb in zip(self.st, embeddings)}

    # ------------------------------------------------------------------
    # query tool
    # ------------------------------------------------------------------
    def query_related_entities(self, entity_name: str) -> List[Dict[str, str]]:
        # Query both outgoing and incoming relations for the target entity.
        query_out = (
            "MATCH (e {name: $entity_name})-[r]->(related) "
            "RETURN e.name AS source, type(r) AS relation, related.name AS target"
        )
        query_in = (
            "MATCH (related)-[r]->(e {name: $entity_name}) "
            "RETURN related.name AS source, type(r) AS relation, e.name AS target"
        )
        return self.graph.run(query_out, entity_name=entity_name).data() + \
            self.graph.run(query_in, entity_name=entity_name).data()

    def query_relation_between_entities(self, entity1: str, entity2: str) -> List[Dict[str, str]]:
        query1 = (
            "MATCH (e1 {name: $entity1})-[r]->(e2 {name: $entity2}) "
            "RETURN type(r) AS relation, e1.name AS source, e2.name AS target"
        )
        query2 = (
            "MATCH (e2 {name: $entity2})-[r]->(e1 {name: $entity1}) "
            "RETURN type(r) AS relation, e2.name AS source, e1.name AS target"
        )
        return self.graph.run(query1, entity1=entity1, entity2=entity2).data() + \
            self.graph.run(query2, entity1=entity1, entity2=entity2).data()

    def query_entities_by_relation(self, entity_name: str, relation_type: str) -> List[Dict[str, str]]:
        query_out = (
            f"MATCH (e {{name: $entity_name}})-[r:`{relation_type}`]->(related) "
            "RETURN e.name AS source, type(r) AS relation, related.name AS target"
        )
        query_in = (
            f"MATCH (related)-[r:`{relation_type}`]->(e {{name: $entity_name}}) "
            "RETURN related.name AS source, type(r) AS relation, e.name AS target"
        )
        return self.graph.run(query_out, entity_name=entity_name).data() + \
            self.graph.run(query_in, entity_name=entity_name).data()

    def get_all_relation_types(self) -> List[str]:#How to obtain all non-duplicate relationship types in the knowledge graph
        result = self.graph.run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS relation_type").data()
        return [record["relation_type"] for record in result]

    def _extract_entities_from_question(self, question: str) -> List[str]:
        try:
            result = self.graph.run("MATCH (e) RETURN e.name AS name").data()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to extract entity: %s", exc)
            return []

        question_lower = question.lower()
        entities = [record["name"] for record in result if record["name"].lower() in question_lower]
        return entities

    # ------------------------------------------------------------------
    # Q&A main process
    # ------------------------------------------------------------------
    def answer_question(self, question: str, llm_callable: Callable[[str], str]) -> Optional[str]:
        if not self._graph_built:
            logger.debug("The knowledge graph has not been initialized and cannot be answered.")
            return None

        entities = self._extract_entities_from_question(question)

        if len(entities) >= 2:
            relations = self.query_relation_between_entities(entities[0], entities[1])
            if relations:
                raw_answer = [
                    f"{rel['source']} {rel['relation']} {rel['target']}"
                    for rel in relations
                ]
                prompt = self._build_prompt(question, raw_answer)
                return llm_callable(prompt)

        if entities:
            relations = self.get_all_relation_types()
            question_lower = question.lower()
            found_relations = [rel for rel in relations if rel.lower() in question_lower]

            if found_relations:
                results = self.query_entities_by_relation(entities[0], found_relations[0])
                if results:
                    raw_answer = [
                        f"{item['source']} {item['relation']} {item['target']}"
                        for item in results
                    ]
                    prompt = self._build_prompt(question, raw_answer)
                    return llm_callable(prompt)
            else:
                related = self.query_related_entities(entities[0])
                if related:
                    raw_answer = [
                        f"{item['source']} {item['relation']} {item['target']}"
                        for item in related
                    ]
                    prompt = self._build_prompt(question, raw_answer)
                    return llm_callable(prompt)
        else:
            match = self._find_similar_entity(question)
            if match:
                entity_name, _ = match
                related = self.query_related_entities(entity_name)
                raw_answer = [f"{item['source']} {item['relation']} {item['target']}" for item in related]
                prompt = self._build_prompt(question, raw_answer)
                return llm_callable(prompt)

        return None

    # ------------------------------------------------------------------
    # Tool method
    # ------------------------------------------------------------------
    def _build_prompt(self, question: str, facts: Sequence[str]) -> str:
        facts_text = "\n".join(f"- {fact}" for fact in facts)
        print('facts_text',facts_text)
        return (
            "Answer the user's question only from the following knowledge-graph facts. "
            "State clearly when the facts are insufficient.\n"
            f"Knowledge Graph Facts:\n{facts_text}\n"
            f"User question: {question}"
        )

    def _find_similar_entity(self, question: str) -> Optional[Tuple[str, float]]:
        if not self.embeddings_dict:
            return None

        response = self.client.embeddings.create(model="embedding-3", input=[question])
        question_embedding = response.data[0].embedding

        best_match: Optional[Tuple[str, float]] = None
        for name, embedding in self.embeddings_dict.items():
            score = self._cosine_similarity(question_embedding, embedding)
            if not best_match or score > best_match[1]:
                best_match = (name, score)

        if best_match and best_match[1] > 0.5:
            return best_match
        return None

    @staticmethod
    def _cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


class GraphRAGManager:
    """Singleton manager for knowledge graph question and answer service."""

    _instance: Optional["GraphRAGManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.config = getattr(settings, "GRAPH_RAG_CONFIG", {})
        self.enabled = self.config.get("ENABLED", False)
        self._rag: Optional[Py2NeoGraphRAG] = None
        self._initialized = False
        self._init_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "GraphRAGManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def answer(self, question: str, llm_callable: Callable[[str], str]) -> Optional[str]:
        if not self.enabled:
            return None

        rag = self._ensure_initialized()
        if not rag:
            return None

        try:
            return rag.answer_question(question, llm_callable)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Knowledge graph question and answer failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    def _ensure_initialized(self) -> Optional[Py2NeoGraphRAG]:
        if self._initialized:
            return self._rag

        with self._init_lock:
            if self._initialized:
                return self._rag

            if not self.enabled:
                return None

            try:
                uri = self.config.get("NEO4J_URI", "bolt://localhost:7687")
                user = self.config.get("NEO4J_USER", "neo4j")
                password = self.config.get("NEO4J_PASSWORD", "neo4j")
                triples_path = self.config.get("TRIPLES_PATH")
                triples_file = Path(triples_path) if triples_path else None

                api_key = self.config.get("ZHIPU_API_KEY")

                if not api_key:
                    raise ValueError("GRAPH_RAG_CONFIG is missing ZHIPU_API_KEY.")

                self._rag = Py2NeoGraphRAG(
                    uri=uri,
                    user=user,
                    password=password,
                    triples_path=triples_file,
                    embeddings_batch_size=self.config.get("EMBEDDINGS_BATCH", 64),
                    api_key=api_key,
                )
                self._rag.initialize()
                self._initialized = True
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Failed to initialize knowledge graph: %s", exc)
                self._rag = None
                self._initialized = False

        return self._rag
