"""
智谱AI API调用模块
提供与智谱大语言模型交互的功能
"""
import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 配置日志
logger = logging.getLogger("zhipu_api")

# 加载环境变量
load_dotenv()

# 配置信息
class ZhipuConfig(BaseModel):
    """智谱API配置类"""
    api_key: str = Field(default_factory=lambda: os.getenv("ZHIPU_API_KEY", ""))
    api_url: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    default_model: str = "glm-4-flash-250414"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 200
    
    @property
    def headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

# 消息模型
class Message(BaseModel):
    """聊天消息模型"""
    role: str
    content: str

class ZhipuAPI:
    """智谱API客户端类"""
    
    def __init__(self, config: Optional[ZhipuConfig] = None):
        """初始化智谱API客户端
        
        Args:
            config: API配置，如果为None则使用默认配置
        """
        self.config = config or ZhipuConfig()
        if not self.config.api_key:
            logger.warning("未设置ZHIPU_API_KEY环境变量，API调用可能失败")
    
    def _get_system_prompt(self) -> Message:
        """获取系统提示词"""
        return Message(
            role="system",
            content="""你是 NeuronDrive 智能驾驶系统助手，嵌入在车载系统中，提供语音控制、手势识别、导航、
    车辆状态反馈、音乐播放等功能。请以专业、简洁、友好的风格回答用户，不要自称 AI、大模型、
    ChatGLM，也不提及智谱。始终以"智能驾驶助手"身份回应。根据聊天记录及用户最新的问题，
    该问题可能涉及聊天记录中的上下文信息。
    无论用户用什么语言问你，请使用简体中文回答用户。"""
        )
    
    async def call_chat(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: float = 0.95,
        include_system_prompt: bool = True
    ) -> str:
        """调用智谱AI聊天接口
        
        Args:
            messages: 聊天消息列表
            model: 模型名称，默认使用配置中的默认模型
            temperature: 温度参数，控制随机性
            include_system_prompt: 是否包含系统提示词
            
        Returns:
            AI回复的内容
            
        Raises:
            ValueError: 输入参数错误
            httpx.HTTPError: HTTP请求错误
            RuntimeError: API调用出错
        """
        # 验证消息格式
        if not messages:
            raise ValueError("消息列表不能为空")
            
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(f"消息格式错误: {msg}")
        
        # 准备请求数据
        api_messages = []
        if include_system_prompt:
            system_prompt = self._get_system_prompt()
            api_messages.append(system_prompt.dict())
        
        api_messages.extend(messages)
        
        payload = {
            "model": model or self.config.default_model,
            "messages": api_messages,
            "temperature": temperature,
            "stream": False
        }
        
        # 记录请求信息（排除敏感内容）
        log_payload = payload.copy()
        if len(log_payload["messages"]) > 0:
            log_payload["messages"] = f"[{len(log_payload['messages'])} messages]"
        logger.info(f"调用智谱API: {json.dumps(log_payload)}")
        
        # 发送请求（带重试机制）
        for attempt in range(self.config.max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.config.api_url, 
                        headers=self.config.headers,
                        json=payload,
                        timeout=self.config.timeout
                    )
                    
                    # 检查HTTP状态码
                    if response.status_code != 200:
                        error_msg = f"API调用失败 (HTTP {response.status_code}): {response.text}"
                        logger.error(error_msg)
                        if attempt == self.config.max_retries - 1:  # 最后一次尝试
                            raise RuntimeError(error_msg)
                        continue
                    
                    # 解析响应数据
                    try:
                        data = response.json()
                        if "choices" in data and len(data["choices"]) > 0:
                            content = data["choices"][0]["message"]["content"]
                            logger.debug(f"智谱API响应成功，内容长度: {len(content)}")
                            return content
                        else:
                            error_msg = f"API响应格式异常: {data}"
                            logger.error(error_msg)
                            if attempt == self.config.max_retries - 1:
                                raise RuntimeError(error_msg)
                    except (KeyError, json.JSONDecodeError) as e:
                        error_msg = f"解析API响应失败: {str(e)}, 响应内容: {response.text[:200]}"
                        logger.error(error_msg)
                        if attempt == self.config.max_retries - 1:
                            raise RuntimeError(error_msg)
                            
            except httpx.HTTPError as e:
                error_msg = f"HTTP请求错误: {str(e)}"
                logger.error(error_msg)
                if attempt == self.config.max_retries - 1:
                    raise
            
            # 重试前等待
            if attempt < self.config.max_retries - 1:
                logger.warning(f"尝试 {attempt+1}/{self.config.max_retries} 失败，{self.config.retry_delay}秒后重试")
                await asyncio.sleep(self.config.retry_delay)
        
        # 这里不应该被执行，因为重试失败会在循环中抛出异常
        raise RuntimeError("所有API调用尝试均失败")


# 创建默认API实例
api = ZhipuAPI()

# 向后兼容的调用函数
async def call_zhipu_chat(messages, model=None, temperature=0.95):
    """兼容旧版API调用函数
    
    使用默认API实例调用智谱聊天接口
    """
    return await api.call_chat(messages, model, temperature)
