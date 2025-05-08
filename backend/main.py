import os
import time
import tempfile
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi import Depends
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

class HeadShakeDetector:
    def __init__(self, shake_threshold=15, buffer_len=10):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.shake_threshold = shake_threshold
        self.buffer_len = buffer_len
        self.yaw_buffer = []

    def get_head_yaw(self, landmarks, image_shape):
        """
        计算头部的偏航角度
        """
        # 使用左右太阳穴点位置估计偏航角
        left = landmarks[234]
        right = landmarks[454]
        dx = left.x - right.x
        dy = left.y - right.y
        angle = np.degrees(np.arctan2(dx, dy))
        return angle
    
    def detect(self, image):
        """
        检测图像中是否有摇头行为
        """
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return False, image

        lm = results.multi_face_landmarks[0].landmark
        yaw = self.get_head_yaw(lm, (h, w))
        
        # 维护角度缓冲区
        self.yaw_buffer.append(yaw)
        if len(self.yaw_buffer) > self.buffer_len:
            self.yaw_buffer.pop(0)

        # 检测角度变化是否超过阈值
        min_yaw, max_yaw = min(self.yaw_buffer), max(self.yaw_buffer)
        shake_detected = (max_yaw - min_yaw) > self.shake_threshold

        # 在图像上添加调试信息
        cv2.putText(image, f"Yaw: {yaw:.1f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        status = "Shake" if shake_detected else "Still"
        cv2.putText(image, f"Status: {status}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        return shake_detected, image


# 数据库配置（MySQL）
# 请替换 user、password、host、port、dbname 为你的 MySQL 信息
DATABASE_URL = "mysql+pymysql://root:Aaa041082@localhost:3306/software"
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

# 加载Whisper模型
model = whisper.load_model("turbo")

# 初始化语音引擎
engine = pyttsx3.init()


# 依赖项：获取 DB 会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



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
    try:
        # 创建摇头检测器
        detector = HeadShakeDetector(shake_threshold=15, buffer_len=15)

        # 初始化摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="无法访问摄像头")
            
        shake_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            shake, annotated_frame = detector.detect(frame)
            cv2.imshow('Head Shake Detection', annotated_frame)

            if shake:
                print("🚨 检测到摇头行为！")
                shake_detected = True
                break

            if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
                break

        cap.release()
        cv2.destroyAllWindows()

        if shake_detected:
            print("⚠️ 提示：请勿在驾驶时摇头晃脑！")
            engine.say("请勿在驾驶时摇头晃脑！请集中注意力。")
            engine.runAndWait()
            return {'warning': '请集中注意力！'}
        else:
            return {'status': '正常'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"视频处理错误: {str(e)}")


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

# ============= 应用启动 =============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





