"""
database.py — SQLite persistence layer using SQLAlchemy
Handles all read/write for enrolled faces and attendance logs.
"""

import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, String,
    DateTime, Date, Float, Text, Boolean, func
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


# ─── Base & Engine ────────────────────────────────────────────────────────────

DB_PATH = os.getenv("DB_PATH", "data/attendance.db")
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


# ─── Models ───────────────────────────────────────────────────────────────────

class Person(Base):
    """A registered person with an enrolled face."""
    __tablename__ = "persons"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    name        = Column(String(100), nullable=False)
    employee_id = Column(String(50), unique=True, nullable=True)
    department  = Column(String(100), nullable=True)
    email       = Column(String(150), nullable=True)
    face_image  = Column(String(255), nullable=False)   # path to stored image
    encoding    = Column(Text, nullable=True)            # JSON-serialised numpy array
    registered_at = Column(DateTime, default=datetime.utcnow)
    is_active   = Column(Boolean, default=True)

    def __repr__(self):
        return f"<Person id={self.id} name={self.name!r}>"


class AttendanceLog(Base):
    """One recognition event = one row."""
    __tablename__ = "attendance_logs"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    person_id    = Column(Integer, nullable=True)        # NULL = unknown
    person_name  = Column(String(100), nullable=False)
    confidence   = Column(Float, nullable=True)
    log_date     = Column(Date, default=date.today)
    log_time     = Column(DateTime, default=datetime.utcnow)
    status       = Column(String(20), default="Present") # Present | Unknown
    snapshot     = Column(String(255), nullable=True)    # path to captured frame

    def __repr__(self):
        return f"<Log {self.person_name} @ {self.log_time}>"


# ─── Init ─────────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(engine)


# ─── Person CRUD ──────────────────────────────────────────────────────────────

def add_person(
    name: str,
    face_image: str,
    encoding: Optional[str] = None,
    employee_id: Optional[str] = None,
    department: Optional[str] = None,
    email: Optional[str] = None,
) -> Person:
    with SessionLocal() as db:
        person = Person(
            name=name,
            face_image=face_image,
            encoding=encoding,
            employee_id=employee_id,
            department=department,
            email=email,
        )
        db.add(person)
        db.commit()
        db.refresh(person)
        return person


def get_all_persons(active_only: bool = True) -> list[dict]:
    with SessionLocal() as db:
        q = db.query(Person)
        if active_only:
            q = q.filter(Person.is_active == True)
        persons = q.order_by(Person.name).all()
        return [_person_to_dict(p) for p in persons]


def get_person_by_id(person_id: int) -> Optional[dict]:
    with SessionLocal() as db:
        p = db.query(Person).filter(Person.id == person_id).first()
        return _person_to_dict(p) if p else None


def delete_person(person_id: int) -> bool:
    with SessionLocal() as db:
        p = db.query(Person).filter(Person.id == person_id).first()
        if not p:
            return False
        p.is_active = False          # soft delete
        db.commit()
        return True


def update_person_encoding(person_id: int, encoding: str):
    with SessionLocal() as db:
        p = db.query(Person).filter(Person.id == person_id).first()
        if p:
            p.encoding = encoding
            db.commit()


# ─── Attendance CRUD ──────────────────────────────────────────────────────────

def log_attendance(
    person_name: str,
    confidence: Optional[float] = None,
    person_id: Optional[int] = None,
    status: str = "Present",
    snapshot: Optional[str] = None,
) -> AttendanceLog:
    with SessionLocal() as db:
        log = AttendanceLog(
            person_id=person_id,
            person_name=person_name,
            confidence=confidence,
            status=status,
            snapshot=snapshot,
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return log


def already_logged_today(person_name: str) -> bool:
    """Return True if this person already has an entry for today."""
    with SessionLocal() as db:
        count = (
            db.query(func.count(AttendanceLog.id))
            .filter(
                AttendanceLog.person_name == person_name,
                AttendanceLog.log_date == date.today(),
            )
            .scalar()
        )
        return count > 0


def get_attendance_logs(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    person_name: Optional[str] = None,
) -> list[dict]:
    with SessionLocal() as db:
        q = db.query(AttendanceLog)
        if start_date:
            q = q.filter(AttendanceLog.log_date >= start_date)
        if end_date:
            q = q.filter(AttendanceLog.log_date <= end_date)
        if person_name:
            q = q.filter(AttendanceLog.person_name == person_name)
        logs = q.order_by(AttendanceLog.log_time.desc()).all()
        return [_log_to_dict(l) for l in logs]


def get_today_summary() -> dict:
    with SessionLocal() as db:
        today = date.today()
        total = (
            db.query(func.count(AttendanceLog.id))
            .filter(AttendanceLog.log_date == today, AttendanceLog.status == "Present")
            .scalar()
        )
        unknown = (
            db.query(func.count(AttendanceLog.id))
            .filter(AttendanceLog.log_date == today, AttendanceLog.status == "Unknown")
            .scalar()
        )
        return {"present": total, "unknown": unknown, "date": str(today)}


# ─── Serialisers ──────────────────────────────────────────────────────────────

def _person_to_dict(p: Person) -> dict:
    return {
        "id": p.id,
        "name": p.name,
        "employee_id": p.employee_id,
        "department": p.department,
        "email": p.email,
        "face_image": p.face_image,
        "encoding": p.encoding,
        "registered_at": str(p.registered_at),
        "is_active": p.is_active,
    }


def _log_to_dict(l: AttendanceLog) -> dict:
    return {
        "id": l.id,
        "person_id": l.person_id,
        "person_name": l.person_name,
        "confidence": l.confidence,
        "log_date": str(l.log_date),
        "log_time": str(l.log_time),
        "status": l.status,
        "snapshot": l.snapshot,
    }