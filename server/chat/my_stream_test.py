
import json

import asyncio
from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from configs import LLM_MODELS, TEMPERATURE
import asyncio
import json
from typing import List, Optional, Union
from server.chat.utils import History


import time
from sse_starlette.sse import EventSourceResponse

from server.chat.chat import chat, chat_for_pipe
# app = FastAPI()

llm_model = "Qwen2-72B-Instruct-GPTQ-Int8"
# api = ApiRequest(base_url="http://10.108.4.3:7971")

import json
from webui import api

async def use_openai_completion():
    # chat_test的返回不再使用EventSourceResponse包装
    # 拆解子文题
    a = await chat_for_pipe(query="你好", model_name=llm_model, stream=False)
    
    async for chunk in a:
        chunk_dict = json.loads(chunk)
        yield json.dumps({"subproblem": chunk_dict['text'], "message_id": chunk_dict['message_id']})
        
    # 知识库  原有使用的doc
    
    # 思考过程
    a = await chat_for_pipe(query="你是谁", model_name=llm_model, stream=False)
    async for chunk in a:
        chunk_dict = json.loads(chunk)
        yield json.dumps({"reflect": chunk_dict['text'], "message_id": chunk_dict['message_id']})
    
    # 验证问题
    a = await chat_for_pipe(query="天王盖地虎", model_name=llm_model, stream=False)
    async for chunk in a:
        chunk_dict = json.loads(chunk)
        yield json.dumps({"validation": chunk_dict['text'], "message_id": chunk_dict['message_id']})
    
    # 最终回答
    a = await chat_for_pipe(query="请使用python实现一个递归函数，用于计算嵌套列表，直到遇到dict才会处理", model_name=llm_model, stream=True)
    async for chunk in a:
        yield chunk
        


async def warp_test(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
               conversation_id: str = Body("", description="对话框ID"),
               history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
               history: Union[int, List[History]] = Body([],
                                                         description="历史对话，设为一个整数可以从数据库中读取历史消息",
                                                         examples=[[
                                                             {"role": "user",
                                                              "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                             {"role": "assistant", "content": "虎头虎脑"}]]
                                                         ),
               stream: bool = Body(False, description="流式输出"),
               model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
               max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
               # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
               prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
               ):
    
    return EventSourceResponse(use_openai_completion())
