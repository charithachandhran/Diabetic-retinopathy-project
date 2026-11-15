from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from datetime import datetime

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)

    screening_id = Column(String, nullable=True, index=True)

    filename = Column(String, nullable=False)
    eye = Column(String, nullable=True)
    predicted_grade = Column(Integer, nullable=False)
    stage_name = Column(String, nullable=False)
    continuous_score = Column(Float, nullable=False)
    referable = Column(Boolean, nullable=False)
    urgency = Column(String, nullable=False)
    clinical_action = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)