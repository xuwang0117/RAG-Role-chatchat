'''
Author: xuwang xuwang
Date: 2024-11-22 20:36:25
LastEditors: xuwang kuien@iscas.ac.cn
LastEditTime: 2024-12-02 16:58:20
FilePath: /Langchain-Chatchat/server/chat/official_statement.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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
async def official_statement(title: str= Body(..., description="名称", examples=["日本与国足的对抗"]),
                    restriction: str= Body(..., description="要求", examples=["生成一份300字报告"]),
                    company: str = Body(None, description="单位", examples=["中国足球研究中心"]),
                    statement_type: str = Body(None, description="发声类型", examples=["声明"]),
                    word_limit: int = Body(500, description="字数限制", examples=[500]),
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
            **任务**：请根据官方发声名称、官方发声要求，以{{ company }}的名义生成一份{{ word_limit }}字的{{ statement_type }}类型的官方发声。
            **限制**：请确保官方发声内容准确、逻辑清晰。符合官方发声要求。
            只输出官方发声的正文主体内容，不能输出标题、发布单位、发布日期等其他额外信息。请使用中文。
            **你的任务**：
            '<官方发声名称>{{ title }}</官方发声名称>\n'
            '<官方发声要求>{{ restriction }}</官方发声要求>\n'
            """
        )
        input_msg = History(role="user", content=all_prompt).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)
        
        task = asyncio.create_task(wrap_done(
            chain.acall({"title": title, "restriction": restriction, "company": company, "statement_type": statement_type, "word_limit": word_limit}),
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
