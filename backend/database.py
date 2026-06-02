"""
PostgreSQL 数据库连接管理。
- SQLAlchemy 引擎 + SessionLocal 工厂，对齐 SuperMew 模式。
- init_db() 在启动时创建表。
- pool_pre_ping=True 确保空闲后连接仍存活。
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
    """创建所有表。延迟导入避免循环依赖。"""
    import backend.models  # noqa: F401（忽略未使用导入警告）

    Base.metadata.create_all(bind=engine)
