import os
import time
import logging
import tempfile
import re
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, File, UploadFile, Request, Depends, status, Query, Body
from sqlalchemy import Column, Integer, String, DateTime, create_engine
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import whisper
import pyttsx3
import cv2
import json
import numpy as np
import joblib
import mediapipe as mp
from datetime import datetime
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from fastapi.middleware.cors import CORSMiddleware  # ✅ 必须有这行
from zhipu import call_zhipu_chat  # 确保模块路径正确
from sqlalchemy.orm.exc import NoResultFound
from openvino.runtime import Core
import asyncio
# ============= 配置 =============
# 确保所有路径使用绝对路径，避免当前工作目录影响
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "api.log")
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)  # 确保模型目录存在

# ============= 日志配置 =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api")
mmlog  = logging.getLogger("multimodal")        # 多模态行为的日志

# ---------- 多模态日志工具 ----------
def log_multimodal(user_id: int,
                   modality: str,
                   content: str,
                   response: str):
    """
    结构化写多模态交互日志
    - user_id : 用户 ID（0 表示匿名）
    - modality: speech / gesture / vision
    - content : 用户输入 (“播放音乐”, “fist”…)
    - response: 系统反馈 (“为您播放默认播放列表”…)
    """
    mmlog.info(
        f"user:{user_id} | modality:{modality} | content:{content} | response:{response}"
    )


# ============= 数据库配置（MySQL） =============
# 请替换 user、password、host、port、dbname 为你的 MySQL 信息
DATABASE_URL = "mysql+pymysql://root:123456@localhost:3306/software"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ============= 数据模型 =============
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)  # 明文存储
    role = Column(String(20), nullable=False, default="user")

# ============= Pydantic 模型 =============
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
    chatmessage:str


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]  # e.g. [{"role": "user", "content": "你是谁？"}]


class UserMemory(Base):
    __tablename__ = "user_chat_memory"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, unique=True, nullable=False)
    content = Column(String(length=10000), nullable=False)  # JSON格式
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    username: str
    password: str
    role: str

# ============= 应用初始化 =============
app = FastAPI(
    title="智能驾驶助手 API",
    description="提供语音识别、头部姿态监测和手势识别功能",
    version="1.0.0"
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

# 延迟加载Whisper模型
whisper_model = None
def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        try:
            logger.info("正在加载Whisper模型...")
            whisper_model = whisper.load_model("small")  # 使用small模型提高速度
            logger.info("Whisper模型加载完成")
        except Exception as e:
            logger.error(f"加载Whisper模型失败: {e}")
            raise RuntimeError(f"无法加载语音识别模型: {e}")
    return whisper_model

# 初始化语音引擎
tts_engine = pyttsx3.init()

# ========== 持续警报相关 ==========
alert_active = False
alert_task = None

async def continuous_warning():
    global alert_active
    num = 0
    while alert_active and num < 50:
        num += 1
        try:
            tts_engine.say("注意力偏离，请集中注意力！如需解除警报，请说‘解除警报’ 或者 做 ‘OK’ 手势。")
            tts_engine.runAndWait()
        except Exception as e:
            logger.warning(f"持续警报播报失败: {str(e)}")
            break
        await asyncio.sleep(2)


# 预加载模型
@app.on_event("startup")
def preload_whisper_model():
    try:
        logger.info("FastAPI 启动，预加载 Whisper 模型中...")
        get_whisper_model()
    except Exception as e:
        logger.error(f"预加载失败: {str(e)}")
        raise RuntimeError(f"Whisper 模型预加载失败: {e}")

# 依赖项：获取 DB 会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} 处理时间: {process_time:.4f}秒 状态码: {response.status_code}")
    return response

# ============= 工具函数 =============
def compute_ear(eye_points, landmarks, img_w, img_h):
    """计算眼睛纵横比(EAR)"""
    pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_points]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

def get_command_response(text: str) -> Optional[str]:
    """根据识别文本获取对应的指令响应"""
    command_mapping = {
        "打开空调": "已经打开空调",
        "关闭空调": "已经关闭空调",
        "把温度调高": "已经调高温度",
        "调低温度": "已经调低温度",
        "把风速调到中档": "风速已调至中档",
        "最大风速": "已将风速调至最大档位",
        "把风向调到吹向前排": "风向已调整为前排吹风模式",
        "打开内循环": "已切换到内循环模式",
        "切换外循环": "已切换到外循环模式",
        "开启前挡风玻璃除雾": "前挡除雾功能已启动",
        "关闭后挡风玻璃除雾": "已关闭后挡除雾",
        "打开司机座椅加热": "司机座椅加热已开启",
        "关闭副驾驶座椅加热": "已关闭副驾驶座椅加热",
        "播放音乐": "为您播放默认播放列表",
        "播放周杰伦的歌": "正在播放周杰伦的热门歌曲",
        "暂停音乐": "音乐播放已暂停",
        "下一首歌": "已切换到下一首歌曲",
        "上一首歌": "已播放上一首歌曲",
        "导航到最近的加油站": "已规划到最近加油站的路线，开始导航",
        "取消导航": "导航已取消",
        "查询剩余油量": "当前油箱剩余约40%",
        "打开左后车窗": "左后车窗已打开",
        "关闭所有车窗": "所有车窗已关闭并锁定",
        "检查胎压": "所有轮胎胎压正常，前轮36 PSI，后轮35 PSI",
        "切换到运动模式": "已切换至运动模式",
        "切换到舒适模式": "已切换至舒适模式"
    }
    
    for command, response in command_mapping.items():
        if command in text:
            logger.info(f"匹配到指令: {command}")
            return response
    
    return None


import subprocess
import os

def convert_to_whisper_format(input_path, output_path=None):
    """
    使用 ffmpeg 将音频文件转换为 Whisper 推荐的格式（16kHz、mono、wav）

    参数:
    - input_path: 原始音频文件路径
    - output_path: 转换后文件保存路径，默认在原路径后添加 '_converted.wav'

    返回:
    - output_path: 转换后的音频文件路径
    """
    if output_path is None:
        output_path = input_path.replace('.wav', '_converted.wav')

    # 确保输入文件存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到音频文件: {input_path}")

    command = [
        'ffmpeg', '-y',                # -y 自动覆盖
        '-i', input_path,              # 输入文件
        '-ar', '16000',                # 设置采样率 16kHz
        '-ac', '1',                    # 设置为单声道
        '-f', 'wav',                   # 输出格式为 WAV
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"音频转换失败: {e}")
    return output_path


# ============= API路由定义 =============

# API 路由：登录
@app.post("/api/login", response_model=LoginResponse)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    """用户登录API，返回用户角色"""
    try:
        db_user = db.query(User).filter(User.username == user.username).first()
        if not db_user or db_user.password != user.password:
            logger.warning(f"登录失败: 用户 {user.username} 用户名或密码错误")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="用户名或密码错误"
            )
        
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

        new_user = User(username=user.username, password=user.password, role="user")
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"用户 {user.username} 注册成功")
        return {"message": "注册成功"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()  # 确保事务回滚
        logger.error(f"注册时发生错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="注册失败，请稍后再试"
        )

@app.post("/api/speech-to-text", response_model=SpeechResponse)
async def speech_to_text(request: Request, audio: UploadFile = File(...), db: Session = Depends(get_db)) -> Dict[str, str]:
    """
    语音转文本API，识别语音指令并执行相应操作
    """
    global alert_active
    try:
        # 保存临时音频文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info("临时音频文件已保存，开始进行语音识别")
        try:
            # 获取模型并进行语音识别
            model = get_whisper_model()
            converted_path = convert_to_whisper_format(tmp_path)
            result = model.transcribe(converted_path, language='zh')
            recognized_text = result["text"]
            logger.info(f"语音识别结果: {recognized_text}")

            # 删除临时文件
            os.unlink(tmp_path)

            # 解除警报关键词
            if alert_active:
                if "解除警报" in recognized_text or "取消警报" in recognized_text:
                    alert_active = False
                    try:
                        # 使用语音引擎进行语音输出
                        tts_engine.say("警报已解除")
                        tts_engine.runAndWait()
                    except Exception as e:
                        logger.warning(f"语音输出失败: {str(e)}")
                    return {"command": recognized_text, "text": "警报已解除"}
                else:
                    return {"command": recognized_text, "text": "警报未解除"}

            # 检查是否匹配预设指令
            response = get_command_response(recognized_text)
            if response:
                try:
                    # 使用语音引擎进行语音输出
                    tts_engine.say(response)
                    tts_engine.runAndWait()
                except Exception as e:
                    logger.warning(f"语音输出失败: {str(e)}")
                
                user_id_str = request.headers.get("X-User-ID")
                if not user_id_str or not user_id_str.isdigit():
                    raise HTTPException(status_code=400, detail="缺少或非法的用户 ID")
                user_id = int(user_id_str)
                log_multimodal(user_id, "speech", recognized_text, response)
                return {
                    "command": recognized_text,
                    "text": response
                        }
            
            # 如果未匹配任何预设指令，则调用大模型进行回答
            logger.info("未匹配到预设指令，交给大模型处理")
            user_id_str = request.headers.get("X-User-ID")
            if not user_id_str or not user_id_str.isdigit():
                raise HTTPException(status_code=400, detail="缺少或非法的用户 ID")
            user_id = int(user_id_str)
            logger.info(f"用户 {user_id} 发起大模型请求")
            
            # 使用 zhipu_chat 的逻辑来读取和更新对话历史
            try:
                memory = db.query(UserMemory).filter(UserMemory.user_id == user_id).one()
                history = json.loads(memory.content)
            except NoResultFound:
                history = []
                memory = UserMemory(user_id=user_id, content="[]")

            # 将当前请求的对话追加到历史中
            history.append({"role": "user", "content": recognized_text})

            MAX_HISTORY = 9
            if len(history) > MAX_HISTORY:
                history = history[-MAX_HISTORY:]

            # 调用大模型
            ai_reply = await call_zhipu_chat(history)

            # 添加 assistant 回复
            history.append({"role": "assistant", "content": ai_reply})

            # 写回数据库
            memory.content = json.dumps(history, ensure_ascii=False)
            db.merge(memory)
            db.commit()
            logger.info(f"用户 {user_id} 对话历史已更新: {history}")
            
            # 可选：语音播报大模型回答
            tts_engine.say(ai_reply)
            tts_engine.runAndWait()

            log_multimodal(user_id, "speech", recognized_text, ai_reply)
            return {
                    "command": recognized_text,
                    "text": ai_reply
            }

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
'''
模型配置如果有问题，请删除intel文件夹，然后参考模型下载.md内容下载模型
'''
# 模型路径
VINO_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intel")

# 加载 OpenVINO 模型
core = Core()
face_det_model = core.read_model(f"{VINO_MODEL_DIR}/face-detection-adas-0001/FP16/face-detection-adas-0001.xml")
face_det = core.compile_model(face_det_model, "CPU")
face_output_layer = face_det.output(0)

landmarks_model = core.read_model(f"{VINO_MODEL_DIR}/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml")
landmarks_det = core.compile_model(landmarks_model, "CPU")
landmarks_output_layer = landmarks_det.output(0)

head_pose_model = core.read_model(f"{VINO_MODEL_DIR}/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml")
head_pose_det = core.compile_model(head_pose_model, "CPU")

gaze_model = core.read_model(f"{VINO_MODEL_DIR}/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml")
gaze_det = core.compile_model(gaze_model, "CPU")

def preprocess_for_openvino(image, target_shape):
    h, w = target_shape
    resized = cv2.resize(image, (w, h))
    return resized.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32)

# 安全裁剪函数，确保眼睛区域裁剪在图像边界内
def safe_crop(image, center_x, center_y, size):
    h, w = image.shape[:2]
    half = size // 2
    x1 = max(int(center_x) - half, 0)
    y1 = max(int(center_y) - half, 0)
    x2 = min(int(center_x) + half, w)
    y2 = min(int(center_y) + half, h)
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return None
    return cropped
@app.post('/api/process-video', response_model=VideoResponse)
async def process_video():
    """
    驾驶员状态姿态监测API，检测驾驶员是否有分心驾驶
    """
    global alert_active, alert_task
    logger.info("启动驾驶员状态监测")
    mp_face_mesh = None
    face_mesh = None
    cap = None
    
    try:
        # 初始化 MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="无法访问摄像头"
            )

        # 阈值和索引定义
        LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
        EAR_THRESH = 0.25
        CLOSED_SEC_THRESHOLD = 5.0

        # 打哈欠
        UPPER_LIP_IDX = 13
        LOWER_LIP_IDX = 14
        LEFT_MOUTH_IDX = 78
        RIGHT_MOUTH_IDX = 308
        YAWN_MAR_THRESH = 0.5
        YAWN_FRAMES_THRESH = 15

        # 点头/摇头
        nod_counter = 0
        shake_counter = 0
        NOD_THRESH_COUNT = 5
        SHAKE_THRESH_COUNT = 5
        PITCH_THRESH_DEG = 10
        YAW_THRESH_DEG = 10

        # 状态变量
        eye_closed_start = None
        yawn_frames = 0
        warnings = []
        start_time = time.time()
        
        # 最多运行30秒或者收集到3个警告
        while (time.time() - start_time < 30) and len(warnings) < 3:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # OpenVINO 人脸检测
            face_input = preprocess_for_openvino(frame, (384, 672))  # 模型默认输入大小
            result = face_det(face_input)[face_output_layer]
            for det in result[0][0]:
                if det[2] > 0.6:
                    xmin = int(det[3] * w)
                    ymin = int(det[4] * h)
                    xmax = int(det[5] * w)
                    ymax = int(det[6] * h)
                    face_roi = frame[ymin:ymax, xmin:xmax]

                    # 面部关键点
                    lm_input = preprocess_for_openvino(face_roi, (60, 60))
                    lm_result = landmarks_det(lm_input)[landmarks_output_layer][0]
                    left_eye = lm_result[0:2] * [face_roi.shape[1], face_roi.shape[0]]
                    right_eye = lm_result[2:4] * [face_roi.shape[1], face_roi.shape[0]]

                    # 安全裁剪眼睛区域
                    left_eye_img = safe_crop(face_roi, left_eye[0], left_eye[1], 30)
                    right_eye_img = safe_crop(face_roi, right_eye[0], right_eye[1], 30)

                    # 如果任意一个为空则跳过本轮
                    if left_eye_img is None or right_eye_img is None:
                        logger.warning("无法提取眼睛图像，跳过该帧")
                        continue

                    # 头部姿态
                    head_pose_input = preprocess_for_openvino(face_roi, (60, 60))
                    yaw = head_pose_det(head_pose_input)[head_pose_det.output("angle_y_fc")][0][0]
                    pitch = head_pose_det(head_pose_input)[head_pose_det.output("angle_p_fc")][0][0]
                    roll = head_pose_det(head_pose_input)[head_pose_det.output("angle_r_fc")][0][0]

                    # 视线估计
                    gaze_input = {
                        "left_eye_image": preprocess_for_openvino(left_eye_img, (60, 60)),
                        "right_eye_image": preprocess_for_openvino(right_eye_img, (60, 60)),
                        "head_pose_angles": np.array([[yaw, pitch, roll]], dtype=np.float32)
                    }
                    gaze_vector = gaze_det(gaze_input)[gaze_det.output(0)][0]

                    # 判断是否注视前方（阈值可调）
                    if abs(gaze_vector[0]) > 0.4 or abs(gaze_vector[1]) > 0.4:
                        warning = "注意力偏离前方，请集中注意力！"
                        if not alert_active:
                            alert_active = True
                            if alert_task is None or alert_task.done():
                                alert_task = asyncio.create_task(continuous_warning())
                        return {"message": warning, "alert": True}

                    # 画框显示
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f"Gaze: ({gaze_vector[0]:.2f}, {gaze_vector[1]:.2f})", (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                # 闭眼检测
                left_ear = compute_ear(LEFT_EYE_IDX, lm, w, h)
                right_ear = compute_ear(RIGHT_EYE_IDX, lm, w, h)
                ear = (left_ear + right_ear) / 2.0

                if ear < EAR_THRESH:
                    if eye_closed_start is None:
                        eye_closed_start = time.time()
                    else:
                        elapsed = time.time() - eye_closed_start
                        if elapsed >= CLOSED_SEC_THRESHOLD:
                            warning = "请注意，您已经闭眼超过5秒钟！"
                            if warning not in warnings:
                                warnings.append(warning)
                                try:
                                    tts_engine.say(warning)
                                    tts_engine.runAndWait()
                                except Exception as e:
                                    logger.warning(f"语音输出失败: {str(e)}")
                else:
                    eye_closed_start = None

                # 打哈欠检测
                ul = np.array([lm[UPPER_LIP_IDX].x * w, lm[UPPER_LIP_IDX].y * h])
                ll = np.array([lm[LOWER_LIP_IDX].x * w, lm[LOWER_LIP_IDX].y * h])
                lm_pt = np.array([lm[LEFT_MOUTH_IDX].x * w, lm[LEFT_MOUTH_IDX].y * h])
                rm_pt = np.array([lm[RIGHT_MOUTH_IDX].x * w, lm[RIGHT_MOUTH_IDX].y * h])

                mar = np.linalg.norm(ul - ll) / np.linalg.norm(lm_pt - rm_pt)
                if mar > YAWN_MAR_THRESH:
                    yawn_frames += 1
                else:
                    yawn_frames = 0

                if yawn_frames >= YAWN_FRAMES_THRESH:
                    warning = "请注意，您正在打哈欠！"
                    if warning not in warnings:
                        warnings.append(warning)
                        try:
                            tts_engine.say(warning)
                            tts_engine.runAndWait()
                        except Exception as e:
                            logger.warning(f"语音输出失败: {str(e)}")

                # 点头/摇头检测
                face_2d, face_3d = [], []
                for idx, lm_pt in enumerate(lm):
                    if idx in [1, 33, 263, 61, 291, 199]:
                        x, y = int(lm_pt.x * w), int(lm_pt.y * h)
                        if idx == 1:
                            nose_2d = (x, y)
                            nose_3d = np.array([x, y, lm_pt.z * 3000], dtype=np.float64)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm_pt.z])
                
                if len(face_2d) > 0:  # 确保有足够的点
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # 相机参数
                    focal_length = w
                    cam_matrix = np.array([[focal_length, 0, w/2],
                                        [0, focal_length, h/2],
                                        [0, 0, 1]], dtype=np.float64)
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    try:
                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                        if success:
                            rmat, _ = cv2.Rodrigues(rot_vec)
                            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                            pitch = angles[0] * 360
                            yaw = angles[1] * 360

                            # 累计计数
                            if pitch < -PITCH_THRESH_DEG:
                                nod_counter += 1
                            else:
                                nod_counter = 0

                            if abs(yaw) > YAW_THRESH_DEG:
                                shake_counter += 1
                            else:
                                shake_counter = 0

                            if nod_counter >= NOD_THRESH_COUNT:
                                warning = "请注意，您正在打瞌睡！"
                                if warning not in warnings:
                                    warnings.append(warning)
                                    try:
                                        tts_engine.say(warning)
                                        tts_engine.runAndWait()
                                    except Exception as e:
                                        logger.warning(f"语音输出失败: {str(e)}")

                            if shake_counter >= SHAKE_THRESH_COUNT:
                                warning = "请注意，您正在摇头！"
                                if warning not in warnings:
                                    warnings.append(warning)
                                    try:
                                        tts_engine.say(warning)
                                        tts_engine.runAndWait()
                                    except Exception as e:
                                        logger.warning(f"语音输出失败: {str(e)}")
                    except Exception as e:
                        logger.warning(f"头部姿势检测计算错误: {str(e)}")
            
            # 显示结果
            cv2.imshow('Head Pose Detection', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        # 记录检测到的警告
        if warnings:
            logger.info(f"检测到以下警告: {', '.join(warnings)}")
        else:
            logger.info("未检测到异常情况")
        
        return {"message": "视频处理完成", "alert": False}
    except Exception as e:
        logger.error(f"视频处理错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"视频处理错误: {str(e)}"
        )
    finally:
        # 确保资源释放
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

@app.post('/api/process-gesture', response_model=GestureResponse)
async def process_gesture(request: Request):
    """
    手势识别API，识别并响应用户的手势控制
    """
    global alert_active
    mp_hands = None
    hands = None
    cap = None

    # 读取用户 ID
    user_id_str = request.headers.get("X-User-ID") 
    user_id = int(user_id_str) if user_id_str and user_id_str.isdigit() else 0
    
    try:
        GESTURES = ['fist', 'palm', 'thumbs_up', 'OK']
        logger.info("启动手势识别")
        
        # 加载手势识别模型
        model_path = os.path.join(MODEL_DIR, 'gesture_model.pkl')
        
        try:
            clf = joblib.load(model_path)
        except FileNotFoundError:
            logger.error("手势识别模型不存在")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="手势识别模型不存在，请先训练模型"
            )

        # 初始化MediaPipe手部检测
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        mp_draw = mp.solutions.drawing_utils

        # 打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("无法访问摄像头")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="无法访问摄像头"
            )

        recognized_label = None
        display_start = None
        start_time = time.time()
        
        # 新增稳定性检测相关变量
        stable_threshold = 5  # 需要连续5帧相同手势
        current_stable_count = 0
        last_gesture = None

        # 最多运行10秒或者识别到手势
        while (time.time() - start_time < 10) and recognized_label is None:
            ret, img = cap.read()
            if not ret:
                break

            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # 如果检测到手并且还未识别过手势
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                row = []
                for p in lm.landmark:
                    row += [p.x, p.y, p.z]
                
                # 预测手势
                try:
                    pred = clf.predict([row])[0]
                    current_gesture = GESTURES[pred]

                    # 稳定性检测逻辑
                    if current_gesture == last_gesture:
                        current_stable_count += 1
                    else:
                        current_stable_count = 1
                        last_gesture = current_gesture

                    # 达到稳定阈值后触发识别
                    if current_stable_count >= stable_threshold:
                        recognized_label = current_gesture
                        logger.info(f"稳定识别到手势: {recognized_label}")
                        display_start = time.time()
                        mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)
                        # 重置检测状态
                        current_stable_count = 0
                        last_gesture = None
                        
                except Exception as e:
                    logger.error(f"手势预测错误: {str(e)}")

            # 如果已经识别到手势
            if recognized_label is not None:
                cv2.putText(
                    img,
                    recognized_label,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    3
                )
                # 根据手势决定反馈文本
                gesture_resp_map = {
                    'fist': "检测到拳，音乐已暂停",
                    'thumbs_up': "收到确认",
                    'palm': "检测到手展开",
                    'OK': "检测到OK"
                }
                resp_text = gesture_resp_map.get(recognized_label, "")
                log_multimodal(user_id, "gesture", recognized_label, resp_text)

                if alert_active and recognized_label == 'OK':
                    alert_active = False
                    resp_text = '警报已解除'
                    try:
                        # 使用语音引擎进行语音输出
                        tts_engine.say("警报已解除")
                        tts_engine.runAndWait()
                    except Exception as e:
                        logger.warning(f"语音输出失败: {str(e)}")

                # 显示1秒后退出
                if time.time() - display_start > 1.0:
                    break

            cv2.imshow('Gesture Recognition', img)
            # 按Esc提前退出
            if cv2.waitKey(1) & 0xFF == 27:
                break

        return {'gesture': recognized_label,
                'resp_text': resp_text }
    except Exception as e:
        logger.error(f"手势识别错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"手势识别错误: {str(e)}"
        )
    finally:
        # 确保资源释放
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

@app.post("/api/users")
async def get_users(db: Session = Depends(get_db)):
    """
    获取所有用户信息
    """
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
async def update_user(user_id: int,user_data: UserUpdate,db: Session = Depends(get_db)):
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

# ============= 应用启动 =============
if __name__ == "__main__":
    import uvicorn
    logger.info("创建数据库表（如果不存在）")
    Base.metadata.create_all(bind=engine)
    logger.info("启动API服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000)