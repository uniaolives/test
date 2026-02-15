# ArkheOS Geodesic Memory (Π_4) - Persistent Extension
# The Hipocampo of the Geodesic Arch with pgvector Support

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import numpy as np
import json
import logging
from typing import List, Optional, Tuple, Any
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel
from arkhe.registry import Entity, EntityType, EntityState

logger = logging.getLogger("arkhe.memory")

class GeodesicMemory:
    """
    Manages long-term persistence using pgvector for semantic retrieval.
    Includes support for few-shot learning and automated conflict resolution.
    """
    def __init__(self, db_config: Optional[dict] = None):
        self.config = db_config or {
            "host": "localhost",
            "dbname": "arkheos",
            "user": "postgres",
            "password": "password"
        }
        self.conn = None
        self.simulated_storage = {} # Fallback storage

    def _get_connection(self):
        if not self.conn:
            try:
                self.conn = psycopg2.connect(**self.config)
                register_vector(self.conn)
                self._initialize_db()
            except Exception as e:
                logger.warning(f"Database connection failed: {e}. Using simulated storage.")
                return None
        return self.conn

    def _initialize_db(self):
        conn = self.conn
        if not conn: return
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id UUID PRIMARY KEY,
                    entity_name TEXT,
                    entity_type TEXT,
                    value JSONB,
                    confidence FLOAT,
                    last_seen TIMESTAMPTZ,
                    embedding vector(384)
                )
            """)
            conn.commit()

    def store_entity(self, entity: Entity, embedding: List[float]):
        """Persists a confirmed entity and its embedding."""
        conn = self._get_connection()
        if not conn:
            self.simulated_storage[str(entity.id)] = {
                "entity_name": entity.name,
                "entity_type": entity.entity_type.value,
                "value": entity.value,
                "confidence": entity.confidence,
                "last_seen": entity.last_seen,
                "embedding": embedding
            }
            return

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO semantic_memory (id, entity_name, entity_type, value, confidence, last_seen, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    confidence = (semantic_memory.confidence + EXCLUDED.confidence) / 2,
                    last_seen = EXCLUDED.last_seen,
                    value = EXCLUDED.value
            """, (
                str(entity.id),
                entity.name,
                entity.entity_type.value,
                json.dumps(entity.value),
                entity.confidence,
                entity.last_seen,
                embedding
            ))
            conn.commit()

    def semantic_recall(self, query_embedding: List[float], limit: int = 5) -> List[Tuple]:
        """Retrieves similar entities for few-shot learning."""
        conn = self._get_connection()
        if not conn:
            # Simulated search
            results = []
            for item in self.simulated_storage.values():
                results.append((item["entity_name"], item["entity_type"], item["value"], item["confidence"], 1.0))
            return results[:limit]

        with conn.cursor() as cur:
            cur.execute("""
                SELECT entity_name, entity_type, value, confidence, 1 - (embedding <=> %s) as similarity
                FROM semantic_memory
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (query_embedding, query_embedding, limit))
            return cur.fetchall()

    def resolve_conflict(self, name: str, value: Any) -> Tuple[bool, Any]:
        """
        Resolves conflict by name.
        Mock implementation for tests when DB is not available.
        """
        for item in self.simulated_storage.values():
            if item["entity_name"] == name:
                return True, item["value"]
        return False, value

    def resolve_conflict_semantically(self, entity: Entity, embedding: List[float], threshold: float = 0.95) -> Tuple[bool, Any]:
        """
        Resolves conflict by checking for semantically similar entities
        with high confidence in historical records.
        """
        similar = self.semantic_recall(embedding, limit=1)
        if similar:
            name, etype, value, confidence, similarity = similar[0]
            if similarity > threshold and confidence > 0.9:
                logger.info(f"Semantically resolved conflict for {entity.name} using {name} (sim={similarity:.4f})")
                return True, value
        return False, entity.value

    def get_few_shot_examples(self, embedding: List[float], limit: int = 3) -> str:
        """Generates a prompt string with past successful extractions."""
        similar = self.semantic_recall(embedding, limit=limit)
        examples = []
        for name, etype, value, confidence, sim in similar:
            examples.append(f"Input text similar to: '{name}' -> Output: {json.dumps(value)}")
        return "\n".join(examples)
