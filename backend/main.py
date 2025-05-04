from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional
import whisper
import tempfile
import os
import pyttsx3
import mediapipe as mp
import cv2
import numpy as np

class HeadShakeDetector:
    def __init__(self, shake_threshold=15, buffer_len=10):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.shake_threshold = shake_threshold  # degrees of yaw change to count as shake
        self.buffer_len = buffer_len
        self.yaw_buffer = []

    def get_head_yaw(self, landmarks, image_shape):
        # Use left (234) and right (454) temple points for yaw approximation
        left = landmarks[234]
        right = landmarks[454]
        # Compute vector difference
        dx = left.x - right.x
        dy = left.y - right.y
        # Yaw angle approximation via arctan2
        angle = np.degrees(np.arctan2(dx, dy))
        return angle

    def detect(self, image):
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return False, image

        lm = results.multi_face_landmarks[0].landmark
        yaw = self.get_head_yaw(lm, image.shape)
        # maintain buffer
        self.yaw_buffer.append(yaw)
        if len(self.yaw_buffer) > self.buffer_len:
            self.yaw_buffer.pop(0)

        # Check change in yaw over buffer
        min_yaw, max_yaw = min(self.yaw_buffer), max(self.yaw_buffer)
        shake_detected = (max_yaw - min_yaw) > self.shake_threshold

        # Draw indicators
        cv2.putText(image, f"Yaw: {yaw:.1f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        status = "Shake" if shake_detected else "Still"
        cv2.putText(image, f"Status: {status}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return shake_detected, image
        

app = FastAPI()

# 加载Whisper模型
model = whisper.load_model("turbo")

# 初始化语音引擎
engine = pyttsx3.init()

# 硬编码用户数据
users = [
    {"username": "admin", "password": "admin123", "role": "admin"},
    {"username": "user1", "password": "user123", "role": "user"}
]

class UserLogin(BaseModel):
    username: str
    password: str

class UserRegister(UserLogin):
    confirm_password: str

@app.post("/api/login")
async def login(user: UserLogin):
    for u in users:
        if u["username"] == user.username and u["password"] == user.password:
            return {"role": u["role"]}
    raise HTTPException(status_code=401, detail="用户名或密码错误")

@app.post("/api/register")
async def register(user: UserRegister):
    if user.password != user.confirm_password:
        raise HTTPException(status_code=400, detail="密码不一致")
    
    if any(u["username"] == user.username for u in users):
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    users.append({
        "username": user.username,
        "password": user.password,
        "role": "user"
    })
    return {"message": "注册成功"}

@app.post("/api/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)) -> dict:
    try:
        # 保存临时音频文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        

        default_options = {
            "initial_prompt": "这是普通话语音识别"
        }
            
        
        # 使用Whisper进行语音识别
        result = model.transcribe(tmp_path, language='zh',**default_options)
        os.unlink(tmp_path)  # 删除临时文件
        
        # 定义指令映射字典
        command_mapping = {
            "打开空调": "已经打开空调",
            "关闭空调": "已经关闭空调",
            "调高温度": "已经调高温度",
            "调低温度": "已经调低温度"
        }
        
        # 检查识别结果是否匹配预设指令
        recognized_text = result["text"]
        for command, response in command_mapping.items():
            if command in recognized_text:
                print(f"匹配到指令: {command}")
                engine.say("%s" % response)  # 使用语音引擎进行语音输出
                engine.runAndWait()
                return {"text": response}
        
        # 未匹配到指令，返回原始识别文本
        return {"text": recognized_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音识别错误: {str(e)}")

@app.post('/api/process-video')
async def process_video():
    # 创建摇头检测器
    detector = HeadShakeDetector(shake_threshold=15, buffer_len=15)

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
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

        if cv2.waitKey(1) & 0xFF == 27:
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)