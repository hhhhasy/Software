from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional
import whisper
import tempfile
import os
import pyttsx3

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)