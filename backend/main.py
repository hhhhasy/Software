from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional
import whisper
import tempfile
import os

app = FastAPI()

# 加载Whisper模型
model = whisper.load_model("base")

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
async def speech_to_text(audio: UploadFile = File(...)):
    try:
        # 保存临时音频文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        
        # 使用Whisper进行语音识别
        result = model.transcribe(tmp_path, language='zh')
        os.unlink(tmp_path)  # 删除临时文件
        
        return {"text": result["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音识别错误: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)