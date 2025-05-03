from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)