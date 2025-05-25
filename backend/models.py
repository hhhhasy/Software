import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from datetime import datetime

# ============= 数据库配置（MySQL） =============
DATABASE_URL = "mysql+pymysql://root:123456@localhost:3306/software"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 依赖项：获取 DB 会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============= SQLAlchemy 数据库模型 =============
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)  # 明文存储
    role = Column(String(20), nullable=False, default="user")

class UserMemory(Base):
    __tablename__ = "user_chat_memory"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, unique=True, nullable=False)
    content = Column(String(length=10000), nullable=False)  # JSON格式
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# ============= Pydantic 验证模型 =============
class UserLogin(BaseModel):
    username: str = Field(..., max_length=50)
    password: str

class UserRegister(UserLogin):
    confirm_password: str

class LoginResponse(BaseModel):
    id: int
    role: str

class MessageResponse(BaseModel):
    message: str

class SpeechResponse(BaseModel):
    command: str
    text: str

class VideoResponse(BaseModel):
    message: str
    alert: bool = False

class GestureResponse(BaseModel):
    gesture: Optional[str] = None
    resp_text: str

class LogEntry(BaseModel):
    timestamp: str
    level: str
    source: str
    message: str

class LogResponse(BaseModel):
    logs: List[LogEntry]
    total_entries: int

class AIResponse(BaseModel):
    chatmessage: str

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]  # e.g. [{"role": "user", "content": "你是谁？"}]

class UserUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    username: str
    password: str
    role: str

# 创建所有表（如果它们不存在）
def create_tables():
    Base.metadata.create_all(bind=engine)
