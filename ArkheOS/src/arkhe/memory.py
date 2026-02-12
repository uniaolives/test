# ArkheOS Geodesic Memory (Π_4) - Persistent Extension
# The Hipocampo of the Geodesic Arch with pgvector Support

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import numpy as np
import json
from typing import List, Optional, Tuple, Any
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel
from arkhe.registry import Entity, EntityType, EntityState

class GeodesicMemory:
    """
    Manages long-term persistence using pgvector for semantic retrieval.
    Includes a simulation fallback for environments without Postgres.
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
            except Exception:
                # Silent fallback to simulation in sandbox
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
            # Simulation fallback
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
                    last_seen = EXCLUDED.last_seen
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

    def semantic_recall(self, query_embedding: List[float], limit: int = 5):
        """Retrieves similar entities."""
        conn = self._get_connection()
        if not conn:
            # Simulated search
            results = []
            for item in self.simulated_storage.values():
                sim = 1.0 # Mock similarity
                results.append((item["entity_name"], item["entity_type"], item["value"], item["confidence"], sim))
            return results[:limit]

        with conn.cursor() as cur:
            cur.execute("""
                SELECT entity_name, entity_type, value, confidence, 1 - (embedding <=> %s) as similarity
                FROM semantic_memory
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (query_embedding, query_embedding, limit))
            return cur.fetchall()

    def resolve_conflict(self, entity_name: str, proposed_value: Any) -> Tuple[bool, Any]:
        """Resolves conflict by checking historical records."""
        conn = self._get_connection()
        if not conn:
            # Simulated resolution
            for item in self.simulated_storage.values():
                if item["entity_name"] == entity_name and item["confidence"] > 0.9:
                    return True, item["value"]
            return False, proposed_value

        with conn.cursor() as cur:
            cur.execute("""
                SELECT value, confidence FROM semantic_memory
                WHERE entity_name = %s
                ORDER BY confidence DESC
                LIMIT 1
            """, (entity_name,))
            row = cur.fetchone()
            if row and row[1] > 0.9:
                return True, row[0]
        return False, proposed_value

    def get_stats(self):
        conn = self._get_connection()
        if not conn:
            return {"total_entities": len(self.simulated_storage)}
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM semantic_memory")
            return {"total_entities": cur.fetchone()[0]}
