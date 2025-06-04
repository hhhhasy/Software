"""
数据库模型和Pydantic验证模型
包含系统所有数据结构定义和数据库相关配置
"""
import os
from typing import Optional, Dict, Any, List,Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.pool import QueuePool
from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy import Text, Boolean # 用于存储JSON等复杂数据

# 加载环境变量
load_dotenv()

# ============= 数据库配置 =============
def get_database_url():
    """从环境变量获取数据库连接信息，如果不存在则使用默认值"""
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "041202")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "software")
    
    return f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# 数据库连接配置
DATABASE_URL = get_database_url()
ENGINE_CONFIG = {
    "pool_size": int(os.getenv("DB_POOL_SIZE", "5")),         # 连接池大小
    "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "10")),  # 最大溢出连接数
    "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "30")),  # 连接池超时（秒）
    "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "3600")), # 连接回收时间（秒）
    "pool_pre_ping": True,                                    # 连接前发送ping以验证连接有效性
}

# 创建引擎实例
engine = create_engine(
    DATABASE_URL, 
    poolclass=QueuePool,
    **ENGINE_CONFIG
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基类
Base = declarative_base()

# ============= 数据库依赖项 =============
def get_db():
    """依赖项：获取数据库会话
    
    在FastAPI路由函数中使用 db: Session = Depends(get_db) 注入数据库会话
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============= SQLAlchemy 数据库模型 =============
class User(Base):
    """用户模型
    
    存储系统用户信息，包括凭据和角色
    """
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    # TODO: 未来需要实现密码哈希加密，暂时按要求使用明文存储
    password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="user")
    
    # 关系: 一个用户有一个聊天记忆
    memory = relationship("UserMemory", back_populates="user", uselist=False, 
                         cascade="all, delete-orphan")
    
    preference = relationship("UserPreference", back_populates="user", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        """对象的字符串表示"""
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"

class UserMemory(Base):
    """用户聊天记忆模型
    
    存储用户与AI助手的对话历史记录
    """
    __tablename__ = "user_chat_memory"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    content = Column(String(length=10000), nullable=False, default="[]")  # JSON格式
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # 关系: 属于某个用户
    user = relationship("User", back_populates="memory")
    
    def __repr__(self):
        """对象的字符串表示"""
        return f"<UserMemory(user_id={self.user_id}, updated_at='{self.updated_at}')>"

# ============= Pydantic 验证模型 =============
class UserBase(BaseModel):
    """用户基本信息验证模型"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")

class UserLogin(UserBase):
    """用户登录验证模型"""
    password: str = Field(..., min_length=6, description="密码")
    # role:str = Field(..., min_length=5, description="身份")

class UserRegister(UserLogin):
    """用户注册验证模型"""
    confirm_password: str = Field(..., min_length=6, description="确认密码")
    # identify:str = Field(..., min_length=5, description="身份")
    role:Literal["admin", "user", "driver", "maintenance_personne"]

    @field_validator('confirm_password')
    def passwords_match(cls, v, info):
        """验证两次输入的密码是否匹配"""
        if 'password' in info.data and v != info.data['password']:
            raise ValueError('两次输入的密码不匹配')
        return v

class UserUpdate(BaseModel):
    """用户信息更新验证模型"""
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="用户名")
    password: Optional[str] = Field(None, min_length=6, description="密码")
    role: Optional[str] = Field(None, description="用户角色")
    
    @field_validator('role')
    def validate_role(cls, v):
        """验证角色值是否有效"""
        valid_roles = ["admin", "user", "driver", "maintenance_personne"]
        if v and v not in valid_roles:
            raise ValueError(f"无效的角色，有效值为: {', '.join(valid_roles)}")
        return v

# ============= 响应模型 =============
class LoginResponse(BaseModel):
    """登录响应模型"""
    id: int
    role: str

class UserResponse(BaseModel):
    """用户信息响应模型"""
    id: int
    username: str
    password: str  # TODO: 未来改为不返回密码
    role: str

    # Pydantic配置
    model_config = ConfigDict(from_attributes=True)

class MessageResponse(BaseModel):
    """通用消息响应模型"""
    message: str

class SpeechResponse(BaseModel):
    """语音识别响应模型"""
    command: str
    text: str

class VideoResponse(BaseModel):
    """视频处理响应模型"""
    message: str
    alert: bool = False

class GestureResponse(BaseModel):
    """手势识别响应模型"""
    gesture: Optional[str] = None
    resp_text: str

class LogEntry(BaseModel):
    """日志条目模型"""
    timestamp: str
    level: str
    source: str
    message: str

class LogResponse(BaseModel):
    """日志响应模型"""
    logs: List[LogEntry]
    total_entries: int

class AIMessage(BaseModel):
    """AI消息模型"""
    role: str
    content: str

class ChatRequest(BaseModel):
    """聊天请求模型"""
    messages: List[Dict[str, str]]  # e.g. [{"role": "user", "content": "你是谁？"}]

class AIResponse(BaseModel):
    """AI响应模型"""
    chatmessage: str


class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False) # 每个用户一份配置

    # 存储常用指令 (可以是JSON格式，例如 {"指令文本": 频率, ...} 或列表)
    common_commands = Column(Text, nullable=True, default="{}")

    # 存储交互习惯 (可以是JSON格式，例如 {"preferred_confirmation_modality": "voice", ...})
    interaction_habits = Column(Text, nullable=True, default="{}")

    # 快捷指令/别名 (JSON格式，例如 {"回家": "导航到家庭住址", "安静": "暂停音乐并静音"})
    command_aliases = Column(Text, nullable=True, default="{}")

    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="preference") # 假设User模型中添加反向关系


class CommonCommandEntry(BaseModel):
    command: str
    frequency: int

class InteractionHabits(BaseModel):
    preferred_confirmation_modality: Optional[str] = None # e.g., "voice", "gesture", "visual"
    # 可以根据需要添加更多习惯字段
    # e.g., call_handling_preference: Optional[str] = None # "answer", "mute", "reject"

class CommandAliasEntry(BaseModel):
    alias: str
    original_command: str

class UserPreferenceData(BaseModel):
    common_commands: Optional[List[CommonCommandEntry]] = None
    interaction_habits: Optional[InteractionHabits] = None
    command_aliases: Optional[List[CommandAliasEntry]] = None

class UserPreferenceResponse(BaseModel):
    user_id: int
    username: str
    common_commands: Dict[str, int] = Field(default_factory=dict)  # 常用指令及其频率
    interaction_habits: Dict[str, Any] = Field(default_factory=dict)  # 交互习惯
    command_aliases: Dict[str, str] = Field(default_factory=dict)  # 指令别名
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)