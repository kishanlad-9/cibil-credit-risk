"""
auth.py — JWT + bcrypt + SQLite (via SQLAlchemy)
No passlib dependency. Works with Python 3.12+
"""

import os
import bcrypt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from jose import JWTError, jwt
from sqlalchemy import create_engine, Column, Integer, String, DateTime, text
from sqlalchemy.orm import DeclarativeBase, Session

# ── Config ────────────────────────────────────────────────────
SECRET_KEY  = os.getenv("SECRET_KEY", "cibil-credit-risk-secret-change-in-production")
ALGORITHM   = "HS256"
TOKEN_TTL   = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 h

DB_FILE = Path(__file__).parent / "users.db"
ENGINE  = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})


# ── ORM model ─────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    email         = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    full_name     = Column(String, nullable=False)
    created_at    = Column(String, nullable=False)


def init_db():
    """Create tables and seed demo user. Called once at startup."""
    Base.metadata.create_all(ENGINE)
    try:
        create_user("demo@cibil.ai", hash_password("demo1234"), "Demo User")
    except UserExists:
        pass  # already seeded


# ── Password ──────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── JWT ───────────────────────────────────────────────────────
def create_access_token(data: dict) -> str:
    payload = {**data, "exp": datetime.utcnow() + timedelta(minutes=TOKEN_TTL)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ── User CRUD ─────────────────────────────────────────────────
class UserExists(Exception):
    pass


class UserNotFound(Exception):
    pass


def create_user(email: str, password_hash: str, full_name: str) -> dict:
    with Session(ENGINE) as s:
        if s.query(User).filter_by(email=email).first():
            raise UserExists(f"{email} already registered")
        u = User(
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            created_at=datetime.utcnow().isoformat(),
        )
        s.add(u)
        s.commit()
        s.refresh(u)
        return _to_dict(u)


def get_user(email: str) -> dict:
    with Session(ENGINE) as s:
        u = s.query(User).filter_by(email=email).first()
    if u is None:
        raise UserNotFound(f"{email} not found")
    return _to_dict(u)


def get_all_users() -> list:
    with Session(ENGINE) as s:
        rows = s.query(User).order_by(User.id).all()
    return [{"id": r.id, "email": r.email, "full_name": r.full_name,
             "created_at": r.created_at} for r in rows]


def _to_dict(u: User) -> dict:
    return {
        "id": u.id, "email": u.email,
        "password_hash": u.password_hash,
        "full_name": u.full_name,
        "created_at": u.created_at,
    }
