"""
PostgreSQL database connection management.
- SQLAlchemy engine + SessionLocal factory, aligned with SuperMew pattern.
- init_db() creates tables on startup.
- pool_pre_ping=True ensures connections are alive after idle periods.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/mragagent",
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


def init_db() -> None:
    """Create all tables. Delayed import avoids circular dependency."""
    import backend.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
