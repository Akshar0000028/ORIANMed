from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from backend.database import Base


def new_id():
    return str(uuid.uuid4())


class Patient(Base):
    __tablename__ = "patients"

    id         = Column(String, primary_key=True, default=new_id)
    name       = Column(String, nullable=False)
    age        = Column(Integer, nullable=True)
    gender     = Column(String, nullable=True)        # male / female / other
    conditions = Column(Text, nullable=True)          # comma-separated: diabetes, hypertension
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions   = relationship("Session", back_populates="patient")


class Session(Base):
    __tablename__ = "sessions"

    id         = Column(String, primary_key=True, default=new_id)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at   = Column(DateTime, nullable=True)
    summary    = Column(Text, nullable=True)          # AI-generated session summary

    patient    = relationship("Patient", back_populates="sessions")
    messages   = relationship("Message", back_populates="session")


class Message(Base):
    __tablename__ = "messages"

    id          = Column(String, primary_key=True, default=new_id)
    session_id  = Column(String, ForeignKey("sessions.id"), nullable=False)
    role        = Column(String, nullable=False)       # user / assistant
    content     = Column(Text, nullable=False)
    confidence  = Column(Float, nullable=True)         # 0.0 – 1.0
    is_emergency = Column(Boolean, default=False)
    backend_used = Column(String, nullable=True)       # NVIDIA NIM / Groq / etc
    timestamp   = Column(DateTime, default=datetime.utcnow)

    session     = relationship("Session", back_populates="messages")