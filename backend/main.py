"""
FastAPI 应用主入口
定义路由和应用配置
"""
import os
import time
import logging
import tempfile
import re
import json
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, File, UploadFile, Request, Depends, status, Query, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
from security import encrypt_password,decrypt_password

# 导入本地模块
from models import (
    User, UserMemory, UserPreference, get_db,
    UserLogin, UserRegister, LoginResponse, MessageResponse, 
    SpeechResponse, VideoResponse, GestureResponse, LogEntry,
    LogResponse, AIResponse, ChatRequest, UserUpdate, UserResponse,
    UserPreferenceResponse
)
import services

# ============= 配置 =============
# 确保所有路径使用绝对路径，避免当前工作目录影响
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "api.log")

# ============= 日志配置 =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api")

# ============= 应用初始化 =============

# 预加载模型
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动前执行
    try:
        logger.info("FastAPI 启动，预加载模型中...")
        services.get_whisper_model()
    except Exception as e:
        logger.error(f"预加载失败: {str(e)}")
        raise RuntimeError(f"模型预加载失败: {e}")
    
    yield  # 应用运行期间
    
    # 关闭前执行
    logger.info("FastAPI 关闭中...")

app = FastAPI(
    title="智能驾驶助手 API",
    description="提供语音识别、头部姿态监测和手势识别功能",
    version="1.0.0",
    lifespan=lifespan # 使用自定义生命周期管理器
)

# CORS配置
origins = [
    "http://localhost:3000",     # React 开发服务器
    "http://localhost:8080",     # Vue 开发服务器
    "http://127.0.0.1:5500",    # Live Server
    "http://localhost:5500",     # Live Server 备用
]

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600,               # 预检请求缓存时间
)

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} 处理时间: {process_time:.4f}秒 状态码: {response.status_code}")
    return response

# ============= API路由定义 =============

# API 路由：登录
@app.post("/api/login", response_model=LoginResponse)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    """用户登录API，返回用户角色"""
    try:
        db_user = db.query(User).filter(User.username == user.username).first()
        
        try:
            decrypted_pwd = decrypt_password(db_user.password)
        except ValueError:
            raise HTTPException(status_code=500, detail="密码解密失败")

        if not db_user or decrypted_pwd != user.password:
            logger.warning(f"登录失败: 用户 {user.username} 用户名或密码错误")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="用户名或密码错误"
            )

        
        # if not db_user or db_user.password != user.password:
        #     logger.warning(f"登录失败: 用户 {user.username} 用户名或密码错误")
        #     raise HTTPException(
        #         status_code=status.HTTP_401_UNAUTHORIZED, 
        #         detail="用户名或密码错误"
        #     )
        
        logger.info(f"用户 {user.username} 角色 {db_user.role} 登录成功")
        return {"id": db_user.id,"role": db_user.role}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"登录时发生错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器内部错误"
        )

# API 路由：注册
@app.post("/api/register", response_model=MessageResponse)
async def register(user: UserRegister, db: Session = Depends(get_db)):
    """用户注册API"""
    try:
        if user.password != user.confirm_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="密码不一致"
            )
        
        if db.query(User).filter(User.username == user.username).first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已存在"
            )

        encrypted_pwd = encrypt_password(user.password)
        new_user = User(username=user.username, password=encrypted_pwd, role=user.role)
        
        # new_user = User(username=user.username, password=user.password, role=user.role)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"用户 {user.username} 注册成功")
        return {"message": "注册成功"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()  # 确保事务回滚
        logger.error(f"注册时发生错误:str{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="注册失败，请稍后再试"
        )

@app.post("/api/speech-to-text", response_model=SpeechResponse)
async def speech_to_text(request: Request, audio: UploadFile = File(...), db: Session = Depends(get_db)) -> Dict[str, str]:
    """语音转文本API，识别语音指令并执行相应操作"""
    try:
        # 验证用户ID
        user_id_str = request.headers.get("X-User-ID")
        if not user_id_str or not user_id_str.isdigit():
            raise HTTPException(status_code=400, detail="缺少或非法的用户 ID")
        user_id = int(user_id_str)
        
        # 保存临时音频文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info("临时音频文件已保存，开始进行语音识别")
        
        try:
            # 调用服务层处理语音
            result = await services.process_speech(tmp_path, user_id, db)
            return result
        finally:
            # 确保临时文件被删除
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.warning(f"删除临时文件失败: {str(e)}")
    except Exception as e:
        logger.error(f"语音识别错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"语音识别错误: {str(e)}"
        )

@app.post('/api/process-video', response_model=VideoResponse)
async def process_video():
    """驾驶员状态姿态监测API，检测驾驶员是否有分心驾驶"""
    return await services.process_video()

@app.post('/api/process-gesture', response_model=GestureResponse)
async def process_gesture(request: Request):
    """手势识别API，识别并响应用户的手势控制"""
    # 读取用户 ID
    user_id_str = request.headers.get("X-User-ID") 
    user_id = int(user_id_str) if user_id_str and user_id_str.isdigit() else 0
    return await services.process_gesture(user_id)

@app.post("/api/users")
async def get_users(db: Session = Depends(get_db)):
    """获取所有用户信息"""
    try:
        users = db.query(User).all()
        logger.info(f"返回 {len(users)} 个用户信息")
        return [{"id": user.id, "username": user.username, "role": user.role} for user in users]
    except Exception as e:
        logger.error(f"获取用户信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户信息失败"
        )

@app.get("/api/users/{user_id}", response_model=UserResponse)
async def read_user(user_id: int, db: Session = Depends(get_db)):
    """获取单个用户信息"""
    try:
        db_user = db.query(User).filter(User.id == user_id).first()
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        return db_user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户信息时发生错误"
        )

@app.put("/api/users/{user_id}", response_model=MessageResponse)
async def update_user(user_id: int, user_data: UserUpdate, db: Session = Depends(get_db)):
    """更新用户信息"""
    try:
        db_user = db.query(User).filter(User.id == user_id).first()
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        # 更新用户信息
        if user_data.username:
            # 检查用户名是否已存在
            existing_user = db.query(User).filter(
                User.username == user_data.username,
                User.id != user_id
            ).first()
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="用户名已存在"
                )
            db_user.username = user_data.username
        if user_data.password:
            db_user.password = user_data.password  
        if user_data.role:
            if user_data.role not in ["admin", "user", "driver", "maintenance_personne"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="无效的角色值"
                )
            db_user.role = user_data.role
        
        db.commit()
        db.refresh(db_user)
        
        logger.info(f"用户ID: {user_id} 信息已更新")
        return {"message": "用户信息更新成功"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"更新用户信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新用户信息时发生错误"
        )

@app.delete("/api/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int, request: Request, db: Session = Depends(get_db)):
    """删除用户"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )

        # 检查是否删除自己
        user_id_str = request.headers.get("X-User-ID")
        current_user_id = int(user_id_str) if user_id_str and user_id_str.isdigit() else 0
        
        if user.id == current_user_id:
            logger.warning(f"用户ID: {user_id} 尝试删除自己")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="不能删除自己"
            )
        db.delete(user)
        db.commit()
        logger.info(f"用户ID: {user_id}, 用户名: {user.username} 已被删除")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"删除用户失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="删除用户失败"
        )

@app.post("/api/logs", response_model=LogResponse)
async def get_logs(limit: int = Body(100, ge=1, le=1000), level: Optional[str] = Body(None)):
    """
    获取系统日志API

    参数:
    - limit: 返回的最大日志条数，默认100，最大1000
    - level: 过滤日志级别 (INFO, WARNING, ERROR)，不指定则返回所有级别

    返回:
    - 按时间倒序排列的日志条目
    """
    try:
        if not os.path.exists(LOG_PATH):
            logger.warning(f"日志文件不存在: {LOG_PATH}")
            return {"logs": [], "total_entries": 0}

        logs = []
        valid_levels = ["INFO", "WARNING", "ERROR", "DEBUG", "CRITICAL"]

        # 验证日志级别参数
        if level and level.upper() not in valid_levels:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无效的日志级别。有效值为: {', '.join(valid_levels)}"
            )

        # 尝试不同的编码方式读取日志文件
        encodings_to_try = ['utf-8', 'gbk', 'gb18030', 'gb2312', 'latin1']
        log_lines = []

        for encoding in encodings_to_try:
            try:
                with open(LOG_PATH, "r", encoding=encoding) as f:
                    log_lines = f.readlines()
                logger.info(f"使用 {encoding} 编码成功读取日志文件")
                break  # 如果成功读取，跳出循环
            except UnicodeDecodeError:
                logger.warning(f"尝试使用 {encoding} 编码读取日志文件失败")
                continue  # 尝试下一种编码

        if not log_lines:
            logger.error("无法用任何已知编码读取日志文件")
            raise ValueError("无法用任何已知编码读取日志文件")

        # 日志解析正则表达式 - 匹配标准日志格式
        log_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ([^-]+) - ([^-]+) - (.*)"

        parsed_logs = []
        for line in log_lines:
            line = line.strip()
            if not line:
                continue

            match = re.match(log_pattern, line)
            if match:
                timestamp, source, log_level, message = match.groups()

                # 过滤日志级别
                if level and log_level.strip() != level.upper():
                    continue

                parsed_logs.append({
                    "timestamp": timestamp.strip(),
                    "source": source.strip(),
                    "level": log_level.strip(),
                    "message": message.strip()
                })

        # 按时间戳倒序排序并限制数量
        parsed_logs.reverse()
        limited_logs = parsed_logs[:limit]

        logger.info(f"返回 {len(limited_logs)} 条日志记录")
        return {
            "logs": limited_logs,
            "total_entries": len(parsed_logs)
        }

    except Exception as e:
        logger.error(f"读取日志文件失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"无法读取日志: {str(e)}"
        )

@app.get("/api/user-preferences/{user_id}", response_model=UserPreferenceResponse)
async def get_user_preference(user_id: int, db: Session = Depends(get_db)):
    """获取用户偏好信息API"""
    try:
        # 检查用户是否存在
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        # 获取用户偏好信息
        preference = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
        
        # 如果用户没有偏好信息，返回空数据
        if not preference:
            return {
                "user_id": user_id,
                "username": user.username,
                "common_commands": {},
                "interaction_habits": {},
                "command_aliases": {},
                "updated_at": None
            }
        
        # 解析JSON字符串为Python字典
        common_commands = json.loads(preference.common_commands or "{}")
        interaction_habits = json.loads(preference.interaction_habits or "{}")
        command_aliases = json.loads(preference.command_aliases or "{}")
        
        logger.info(f"返回用户ID: {user_id} 的偏好信息")
        return {
            "user_id": user_id,
            "username": user.username,
            "common_commands": common_commands,
            "interaction_habits": interaction_habits,
            "command_aliases": command_aliases,
            "updated_at": preference.updated_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户偏好信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取用户偏好信息时发生错误"
        )

@app.get("/api/user-preferences")
async def get_all_user_preferences(db: Session = Depends(get_db)):
    """获取所有用户的偏好信息API"""
    try:
        # 获取所有用户
        users = db.query(User).all()
        
        # 获取所有用户的偏好信息
        result = []
        for user in users:
            preference = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
            
            # 如果用户没有偏好信息，添加空数据
            if not preference:
                result.append({
                    "user_id": user.id,
                    "username": user.username,
                    "common_commands": {},
                    "interaction_habits": {},
                    "command_aliases": {},
                    "updated_at": None
                })
                continue
            
            # 解析JSON字符串为Python字典
            common_commands = json.loads(preference.common_commands or "{}")
            interaction_habits = json.loads(preference.interaction_habits or "{}")
            command_aliases = json.loads(preference.command_aliases or "{}")
            
            result.append({
                "user_id": user.id,
                "username": user.username,
                "common_commands": common_commands,
                "interaction_habits": interaction_habits,
                "command_aliases": command_aliases,
                "updated_at": preference.updated_at
            })
        
        logger.info(f"返回 {len(result)} 个用户的偏好信息")
        return result
    except Exception as e:
        logger.error(f"获取所有用户偏好信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取所有用户偏好信息时发生错误"
        )
    



@app.post("/api/text-command", response_model=SpeechResponse) # 可以复用 SpeechResponse
async def handle_text_command(
    request: Request,
    # 方案1: 通过 Form data 接收 (与前端发送 FormData('text_command', commandText) 对应)
    command_input: str = Form(..., alias="text_command"), # 使用 alias 匹配前端的字段名
    db: Session = Depends(get_db)
):
    """处理来自文本框的指令输入"""
    user_id_str = request.headers.get("X-User-ID")
    if not user_id_str or not user_id_str.isdigit():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="缺少或非法的用户 ID")
    user_id = int(user_id_str)

    # 如果使用方案1 (Form data):
    text_to_process = command_input

    if not text_to_process or not text_to_process.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="指令文本不能为空")

    try:
        logger.info(f"API /api/text-command received: '{text_to_process}' for user_id: {user_id}")
        result = await services.process_text_command(text_to_process, user_id, db)
        return result
    except HTTPException: # 直接 re-raise services 层抛出的 HTTPException
        raise
    except Exception as e:
        logger.error(f"处理文本指令 API 调用时发生意外错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理文本指令时发生服务器内部错误: {str(e)}"
        )

# ============= 应用启动 =============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)