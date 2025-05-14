# zhipu_api.py
import httpx
from dotenv import load_dotenv
load_dotenv()
import os
ZHIPU_API_KEY =  os.getenv("ZHIPU_API_KEY")
ZHIPU_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


HEADERS = {
    "Authorization": f"Bearer {ZHIPU_API_KEY}",
    "Content-Type": "application/json"
}

async def call_zhipu_chat(messages, model="glm-4-flash-250414", temperature=0.95):
    """
    加入系统提示词作为第一条
    """
    system_prompt = {
        "role": "system",
        "content": (
            "你是 NeuronDrive 智能驾驶系统助手，嵌入在车载系统中，提供语音控制、手势识别、导航、车辆状态反馈、音乐播放等功能。"
            "请以专业、简洁、友好的风格回答用户，不要自称 AI、大模型、ChatGLM，也不提及智谱。"
            "始终以“智能驾驶助手”身份回应。"
        )
    }
    payload = {
        "model": model,
        "messages":[system_prompt] + messages,
        "temperature": temperature,
        "stream": False
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(ZHIPU_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
