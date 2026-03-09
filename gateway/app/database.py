from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import Engine
import os
import time
import logging
from .models.database import Base

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/arkhe_dmr")

# Slow query logging threshold in seconds
SLOW_QUERY_THRESHOLD = 0.5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arkhe_db")

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
)

@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._query_start_time = time.time()

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - context._query_start_time
    if total > SLOW_QUERY_THRESHOLD:
        logger.warning(f"Slow Query ({total:.4f}s): {statement}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def check_db_health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False

def get_db_stats():
    try:
        with engine.connect() as conn:
            # PostgreSQL specific stats
            result = conn.execute(text("SELECT count(*) FROM pg_stat_activity"))
            active_connections = result.scalar()
            return {
                "status": "online",
                "active_connections": active_connections,
                "engine": engine.name,
                "pool_size": engine.pool.size(),
                "checkedin": engine.pool.checkedin(),
                "checkedout": engine.pool.checkedout(),
            }
    except Exception as e:
        return {"status": "offline", "error": str(e)}
