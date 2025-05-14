import os
import time
import tempfile
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware  # ✅ 必须有这行
from pydantic import BaseModel
import whisper
import pyttsx3
import cv2
import numpy as np
import joblib
import mediapipe as mp
from flask import Flask, jsonify, request
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from fastapi import Body
from zhipu import call_zhipu_chat  # 确保模块路径正确


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]  # e.g. [{"role": "user", "content": "你是谁？"}]


# 数据库配置（MySQL）
# 请替换 user、password、host、port、dbname 为你的 MySQL 信息
DATABASE_URL = "mysql+pymysql://root:041202@localhost:3306/soft"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy 用户模型
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)  # 明文存储
    role = Column(String(20), nullable=False, default="user")

# Pydantic 模式
class UserLogin(BaseModel):
    username: str = Field(..., max_length=50)
    password: str

class UserRegister(UserLogin):
    confirm_password: str

# ============= 应用初始化 =============

app = FastAPI(
    title="智能驾驶助手 API",
    description="提供语音识别、头部姿态监测和手势识别功能",
    version="1.0.0"
)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 或 ["http://127.0.0.1:8080"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# 加载Whisper模型
model = whisper.load_model("turbo")
# model = whisper.load_model("base")
# model=whisper.load_model("D:/anaconda/envs/soft/Lib/site-packages/whisper/whisper_model/base.pt")

# 初始化语音引擎
engine = pyttsx3.init()


# 依赖项：获取 DB 会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 计算 EAR 的函数
def compute_ear(eye_points, landmarks, img_w, img_h):
    pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_points]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)


# ============= API路由定义 =============

# API 路由：登录
@app.post("/api/login")
async def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or db_user.password != user.password:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    print(f"用户 {user.username} 登录成功")
    return {"role": db_user.role}

# API 路由：注册
@app.post("/api/register")
async def register(user: UserRegister, db: Session = Depends(get_db)):
    if user.password != user.confirm_password:
        raise HTTPException(status_code=400, detail="密码不一致")
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="用户名已存在")

    new_user = User(username=user.username, password=user.password, role="user")
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "注册成功"}

@app.post("/api/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)) -> Dict[str, str]:
    """
    语音转文本API，识别语音指令并执行相应操作
    
    接受音频文件，返回识别的文本或指令响应
    """
    try:
        # 保存临时音频文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # 语音识别配置
        default_options = {
            "initial_prompt": "这是普通话语音识别"
        }
        
        # 使用Whisper进行语音识别
        result = model.transcribe(tmp_path, language='zh', **default_options)
        os.unlink(tmp_path)  # 删除临时文件
        
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
        
        # 检查识别结果是否匹配预设指令
        recognized_text = result["text"]
        for command, response in command_mapping.items():
            if command in recognized_text:
                print(f"匹配到指令: {command}")
                engine.say(response)  # 使用语音引擎进行语音输出
                engine.runAndWait()
                return {"text": response}
        
        # 未匹配到指令，返回原始识别文本
        return {"text": recognized_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音识别错误: {str(e)}")

@app.post('/api/process-video')
async def process_video():
    """
    头部姿态监测API，检测驾驶员是否有摇头行为
    
    打开摄像头，实时监测驾驶员头部姿态
    """
    # -------------------- 初始化 MediaPipe Face Mesh --------------------
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # -------------------- 打开摄像头 --------------------
    cap = cv2.VideoCapture(0)

    # -------------------- 阈值和索引定义 --------------------
    # 眼睛
    LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
    EAR_THRESH = 0.25             # EAR 阈值
    CLOSED_SEC_THRESHOLD = 5.0    # 闭眼超过 5 秒报警

    # 打哈欠
    UPPER_LIP_IDX    = 13
    LOWER_LIP_IDX    = 14
    LEFT_MOUTH_IDX   = 78
    RIGHT_MOUTH_IDX  = 308
    YAWN_MAR_THRESH      = 0.5   # MAR 阈值
    YAWN_FRAMES_THRESH   = 15    # 连续帧数阈值 (~0.5 秒 @30fps)

    # 点头 / 摇头
    nod_counter = 0
    shake_counter = 0
    NOD_THRESH_COUNT   = 5       # 连续 5 帧视为持续点头
    SHAKE_THRESH_COUNT = 5       # 连续 5 帧视为持续摇头
    PITCH_THRESH_DEG   = 10      # 俯仰角阈值（度）
    YAW_THRESH_DEG     = 10      # 偏航角阈值（度）

    # 状态变量
    eye_closed_start = None
    yawn_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # ----------- 1. 闭眼检测 -----------
            left_ear  = compute_ear(LEFT_EYE_IDX, lm, w, h)
            right_ear = compute_ear(RIGHT_EYE_IDX, lm, w, h)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESH:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                else:
                    elapsed = time.time() - eye_closed_start
                    if elapsed >= CLOSED_SEC_THRESHOLD:
                        engine.say("请注意，您已经闭眼超过5秒钟！")
                        engine.runAndWait()
                        
                        
            else:
                eye_closed_start = None


            # ----------- 2. 打哈欠检测 -----------
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
                engine.say("请注意，您正在打哈欠！")
                engine.runAndWait()
                
            # ----------- 3. 点头/摇头检测 -----------
            # 提取用于 PnP 的 2D/3D 点
            face_2d, face_3d = [], []
            for idx, lm_pt in enumerate(lm):
                if idx in [1, 33, 263, 61, 291, 199]:
                    x, y = int(lm_pt.x * w), int(lm_pt.y * h)
                    if idx == 1:
                        nose_2d = (x, y)
                        nose_3d = np.array([x, y, lm_pt.z * 3000], dtype=np.float64)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm_pt.z])
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # 相机参数
            focal_length = w
            cam_matrix = np.array([[focal_length, 0, w/2],
                                [0, focal_length, h/2],
                                [0, 0, 1]], dtype=np.float64)
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            if success:
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                pitch = angles[0] * 360   # 单位：度
                yaw   = angles[1] * 360

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
                    engine.say("请注意，您正在打瞌睡！")
                    engine.runAndWait()
                    
                    

                if shake_counter >= SHAKE_THRESH_COUNT:
                    engine.say("请注意，您正在摇头！")
                    engine.runAndWait()
                    
                    

                # 绘制头部朝向线
                nose_proj, _ = cv2.projectPoints(
                    nose_3d.reshape(-1, 3), rot_vec, trans_vec, cam_matrix, dist_matrix
                )
                p1 = nose_2d
                p2 = (int(p1[0] + yaw * 0.5), int(p1[1] - pitch * 0.5))
                cv2.line(frame, p1, p2, (255, 0, 0), 2)
        
        

        # 显示结果
        cv2.imshow('Head Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    return {"message": "视频处理完成"}


@app.post('/api/process-gesture')
async def process_gesture():
    """
    手势识别API，识别并响应用户的手势控制
    
    打开摄像头，识别预定义的手势
    """
    try:
        GESTURES = ['fist', 'palm', 'thumbs_up', 'OK']
        
        # 加载手势识别模型
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'gesture_model.pkl')
        
        try:
            clf = joblib.load(model_path)
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="手势识别模型不存在，请先训练模型")

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
            raise HTTPException(status_code=500, detail="无法访问摄像头")

        recognized_label = None
        display_start = None

        while True:
            ret, img = cap.read()
            if not ret:
                break

            h, w = img.shape[:2]  # 获取图像尺寸，避免MediaPipe警告
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # 如果检测到手势并且还未识别过
            if results.multi_hand_landmarks and recognized_label is None:
                lm = results.multi_hand_landmarks[0]
                row = []
                for p in lm.landmark:
                    row += [p.x, p.y, p.z]
                pred = clf.predict([row])[0]
                recognized_label = GESTURES[pred]
                display_start = time.time()  # 记录开始显示文字的时间
                mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)

            # 如果已经识别到手势，则持续在窗口中显示 1 秒
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
                # 显示超过 1 秒后退出
                if time.time() - display_start > 1.0:
                    break

            cv2.imshow('Gesture Recognition', img)
            # 按 Esc 也可以提前退出
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        return {'gesture': recognized_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"手势识别错误: {str(e)}")
    
@app.post("/api/users")
async def get_users(db: Session = Depends(get_db)):
    """
    获取所有用户信息
    """
    users = db.query(User).all()
    return [{"id": user.id, "username": user.username, "role": user.role} for user in users]


@app.post("/api/zhipu-chat")
async def zhipu_chat(req: ChatRequest):
    """
    接收聊天消息，转发给智谱AI返回响应
    """
    try:
        reply = await call_zhipu_chat(req.messages)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"调用大模型出错: {str(e)}")

# ============= 应用启动 =============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





