import os
import time
import logging
import tempfile
import re
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, File, UploadFile, Request, Depends, status, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import whisper
import pyttsx3
import cv2
import numpy as np
import joblib
import mediapipe as mp
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session


# ============= 日志配置 =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api")

# ============= 配置 =============
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(MODEL_DIR, exist_ok=True)  # 确保模型目录存在

# 数据库配置（MySQL）
DATABASE_URL = "mysql+pymysql://root:Aaa041082@localhost:3306/software"
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
    role: str

class MessageResponse(BaseModel):
    message: str

class SpeechResponse(BaseModel):
    text: str

class VideoResponse(BaseModel):
    message: str

class GestureResponse(BaseModel):
    gesture: Optional[str] = None

class LogEntry(BaseModel):
    timestamp: str
    level: str
    source: str
    message: str

class LogResponse(BaseModel):
    logs: List[LogEntry]
    total_entries: int

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
        return {"role": db_user.role}
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
async def speech_to_text(audio: UploadFile = File(...)):
    """
    语音转文本API，识别语音指令并执行相应操作
    """
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
            result = model.transcribe(tmp_path, language='zh')
            recognized_text = result["text"]
            logger.info(f"语音识别结果: {recognized_text}")
            
            # 删除临时文件
            os.unlink(tmp_path)
            
            # 检查是否匹配预设指令
            response = get_command_response(recognized_text)
            if response:
                try:
                    # 使用语音引擎进行语音输出
                    tts_engine.say(response)
                    tts_engine.runAndWait()
                except Exception as e:
                    logger.warning(f"语音输出失败: {str(e)}")
                
                return {"text": response}
            
            # 未匹配到指令，返回原始识别文本
            return {"text": recognized_text}
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
    """
    驾驶员状态姿态监测API，检测驾驶员是否有分心驾驶
    """
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
        
        return {"message": "视频处理完成"}
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
async def process_gesture():
    """
    手势识别API，识别并响应用户的手势控制
    """
    mp_hands = None
    hands = None
    cap = None
    
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
                    recognized_label = GESTURES[pred]
                    logger.info(f"识别到手势: {recognized_label}")
                    display_start = time.time()
                    mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)
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
                # 显示1秒后退出
                if time.time() - display_start > 1.0:
                    break

            cv2.imshow('Gesture Recognition', img)
            # 按Esc提前退出
            if cv2.waitKey(1) & 0xFF == 27:
                break

        return {'gesture': recognized_label}
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

@app.delete("/api/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    """删除用户"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
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
        log_path = os.path.join(os.path.dirname(__file__), "api.log")
        if not os.path.exists(log_path):
            logger.warning(f"日志文件不存在: {log_path}")
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
                with open(log_path, "r", encoding=encoding) as f:
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