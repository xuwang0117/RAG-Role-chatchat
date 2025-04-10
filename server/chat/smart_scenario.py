from fastapi import Body, Request, File, UploadFile
from sse_starlette.sse import EventSourceResponse
from fastapi.concurrency import run_in_threadpool
from configs import (LLM_MODELS, 
                     VECTOR_SEARCH_TOP_K, 
                     SCORE_THRESHOLD, 
                     TEMPERATURE,
                     USE_RERANKER,
                     RERANKER_MODEL,
                     RERANKER_MAX_LENGTH,
                     logger)
from server.utils import wrap_done, get_ChatOpenAI, get_model_path
from server.utils import BaseResponse, get_prompt_template
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio, json, re, hashlib, time, requests, urllib
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.document_transformers import (
    LongContextReorder,
)
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
from server.reranker.reranker import LangchainReranker
from server.utils import embedding_device
from server.chat.chat import chat_for_pipe
from datetime import datetime
async def smart_scenario(title: str= Body(..., description="名称", examples=["日本与国足的对抗"]),
                    restriction: str= Body(..., description="要求", examples=["生成一份300字报告"]),
                    outline: str = Body(None, description="大纲", examples=["# 标题一\n## 标题1.1\n### 标题1.1.1\n# 标题二\n## 标题2.1\n### 标题2.1.1"]),
                    company: str = Body(None, description="单位", examples=["中国足球研究中心"]),
                    stream: bool = Body(False, description="流式输出"),
                    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                    temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                    max_tokens: Optional[int] = Body(
                        None,
                        description="限制LLM生成Token数量，默认None代表模型最大值"
                    ),
                    request: Request = None,
                    ):

    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        
        nonlocal max_tokens
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )

        all_prompt = (
            """
            **任务**：请根据方案名称、方案要求、方案大纲，以{{ company }}的名义生成一份详细的方案。
            **限制**：请确保方案内容准确、逻辑清晰。符合方案要求。严格按照方案大纲结构以 markdown 格式生成。
            只输出方案的正文主体内容，不能输出方案名称、单位、日期等其他额外信息。
            **你的任务**：
            '<方案名称>{{ title }}</方案名称>\n'
            '<方案要求>{{ restriction }}</方案要求>\n'
            '<方案大纲>{{ outline }}</方案大纲>\n'
            """
        )
        input_msg = History(role="user", content=all_prompt).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)
        
        task = asyncio.create_task(wrap_done(
            chain.acall({"title": title, "restriction": restriction, "outline": outline, "company": company}),
            callback.done),
        )

        if stream:
            async for token in callback.aiter():
                yield json.dumps({"report": token}, ensure_ascii=False)
        else:
            buffer = ""
            async for token in callback.aiter():
                buffer += token
            yield json.dumps({"report": buffer}, ensure_ascii=False)

        await task

    return EventSourceResponse(knowledge_base_chat_iterator())
