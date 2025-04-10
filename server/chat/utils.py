from pydantic import BaseModel, Field
from langchain.prompts.chat import ChatMessagePromptTemplate
from configs import logger, log_verbose
from typing import List, Tuple, Dict, Union
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import httpx
import traceback

class History(BaseModel):
    """
    对话历史
    可从dict生成，如
    h = History(**{"role":"user","content":"你好"})
    也可转换为tuple，如
    h.to_msy_tuple = ("human", "你好")
    """
    role: str = Field(...)
    content: str = Field(...)

    def to_msg_tuple(self):
        return "ai" if self.role=="assistant" else "human", self.content

    def to_msg_template(self, is_raw=True) -> ChatMessagePromptTemplate:
        role_maps = {
            "ai": "assistant",
            "human": "user",
        }
        role = role_maps.get(self.role, self.role)
        if is_raw: # 当前默认历史消息都是没有input_variable的文本。
            content = "{% raw %}" + self.content + "{% endraw %}"
        else:
            content = self.content

        return ChatMessagePromptTemplate.from_template(
            content,
            "jinja2",
            role=role,
        )

    @classmethod
    def from_data(cls, h: Union[List, Tuple, Dict]) -> "History":
        if isinstance(h, (list,tuple)) and len(h) >= 2:
            h = cls(role=h[0], content=h[1])
        elif isinstance(h, dict):
            h = cls(**h)

        return h

class LogRequestMiddleware(BaseHTTPMiddleware):
    def __init__(
            self,
            app,
            some_attribute: str,
    ):
        super().__init__(app)
        self.some_attribute = some_attribute
    async def dispatch(self, request: Request, call_next):
        # 在请求处理之前运行的代码
        try:
            logger.info(f"URL: {request.url}, Method: {request.method}, PARAMS: {None if request.method != 'POST' else await request.json()}")
        except:
            logger.info(f"URL: {request.url}, Method: {request.method}")

        response = await call_next(request)

        # 在请求处理之后运行的代码（如果需要）
        return response

