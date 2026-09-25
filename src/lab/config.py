"""Environment and connections.

Every connection this lab opens is READ ONLY: the lab reads what kratikos-ai-backend produced
(Postgres rows, Qdrant vectors) and writes only local files under data/ and models/.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sqlalchemy import Engine, create_engine, event

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

load_dotenv(ROOT / ".env")


def _require(name: str) -> str:

    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set — see .env.example")
    return value


def _sync_url(url: str) -> str:
    """kratikos-ai-backend stores an asyncpg URL; the lab uses sync psycopg."""

    scheme, rest = url.split("://", 1)
    if scheme.startswith("postgres"):
        return f"postgresql+psycopg://{rest}"
    return url


def get_engine() -> Engine:
    """A Postgres engine whose every transaction is read only, with a statement timeout."""

    engine = create_engine(_sync_url(_require("DATABASE_URL")), pool_pre_ping=True)

    @event.listens_for(engine, "connect")
    def _read_only(dbapi_connection, _record) -> None:

        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
            cursor.execute("SET statement_timeout = '120s'")
        dbapi_connection.commit()

    return engine


def get_qdrant() -> QdrantClient:

    return QdrantClient(
        url=_require("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
        timeout=60,
    )
