from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from pathlib import Path

# SQLite file lives at E:\ORIAN\orianmed.db — no server needed
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "orianmed.db"

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False}  # needed for FastAPI
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session then closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist yet."""
    from backend.models import Patient, Session, Message  # noqa
    Base.metadata.create_all(bind=engine)
    print(f"[DB] SQLite ready at {DB_PATH}")