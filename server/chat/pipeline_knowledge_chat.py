import asyncio
import json
import re
import streamlit as st

from server.chat.chat import chat_for_pipe
from server.chat.knowledge_base_chat import knowledge_base_chat_for_pipe

from fastapi import Body
# from sse_starlette.sse import EventSourceResponse
from configs import (LLM_MODELS, 
                     VECTOR_SEARCH_TOP_K, 
                     SCORE_THRESHOLD, 
                     TEMPERATURE,
                     logger)
# from server.utils import wrap_done, get_ChatOpenAI
# from langchain.chains import LLMChain
# from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
# from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional, Union
from server.chat.utils import History
# from langchain.prompts import PromptTemplate
from server.utils import get_prompt_template
# from server.memory.conversation_db_buffer_memory import ConversationBufferDBMemory
# from server.db.repository import add_message_to_db
# from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler
from webui_pages.utils import *
from sse_starlette.sse import EventSourceResponse



async def to_be_decorated_pipe_knowledge_chat(prompt: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                              conversation_id: str = Body("", description="对话框ID"),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                              score_threshold: float = Body(
                                                            SCORE_THRESHOLD,
                                                            description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                                            ge=0,
                                                            le=2
                                                            ),
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
                              prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                              request: Request = None,
                            ):
    
    text = ""
    # query rewrite before:
    query = ""
    # # prompt1 = "改写下面这句话：" + prompt + "。只给出改写的内容，不需要回答："
    # QUERY_PROMPT = """你是一个 AI 语言助手，你的任务是将用户的问题拆解成多个子问题便于检索，直接生成子问题，多个子问题以换行分割，保证每行一个。用户的原始问题为： """ + prompt
    QUERY_PROMPT = """**角色**：你是一个世界知识专家。你的任务是退后一步，将问题解释为多个更通用的阶梯问题，这更容易回答。
                    **任务**：你的具体任务是基于我给出的原始问题生成多个后退问题。每个后退问题相比原始问题具有更高级别的概念或原则，从而提高解决复杂问题的效果。
                    **限制**：你只需要生成后退问题文本，不需要生成其他多余的信息。按照后退问题与原始问题的相关性对后退问题排序，并需要严格按照XML格式输出，不超过五条。你需要结合下列例子理解我的上述要求，并且按照要求完成任务。
                    **举例**：
                    样例1：
                    ------------------------------------
                    原始问题：台湾有哪些重要政治人物发表过明显的台独言论
                    后退问题：
                    <question>哪些台湾政治人物曾发表过支持台独的言论？</question>
                    <question>台湾的哪些重要政治人物有台独立场？</question>
                    ------------------------------------
                    样例2：
                    ------------------------------------
                    原始问题：张三出生在哪个国家
                    后退问题：
                    <question>张三的出生地是哪个国家？</question>
                    <question>张三是在哪个国家出生的？</question>
                    <question>张三出生于哪个国家？</question>
                    ------------------------------------      
                    样例3:
                    ------------------------------------
                    原始问题：北约、美国、台湾之间有哪些关系
                    后退问题：
                    <question>北约与台湾之间有什么联系？</question>
                    <question>美国在北约和台湾的关系中扮演什么角色？</question>
                    <question>北约与美国之间的关系是怎样的？</question>
                    <question>北约、美国和台湾之间的关系是怎样的？</question>
                    <question>北约由哪些国家组成？</question>
                    ------------------------------------
                    以下是我给出的原始问题： """ + prompt
    # print(QUERY_PROMPT)
    # first chat with llm, 拆解子问题
    outputs_from_first_chat = await chat_for_pipe(QUERY_PROMPT, 
                                                  model_name=model_name, 
                                                  stream=False, 
                                                  temperature= temperature, 
                                                  max_tokens=max_tokens, 
                                                  prompt_name=prompt_name,
                                                  history_len=history_len,
                                                  history=history)
    
    # 流式输出 for XXX
    ts_outputs_from_first_chat = []
    async for chunk in outputs_from_first_chat:
        chunk_dict = json.loads(chunk)
        ts_outputs_from_first_chat.append(chunk_dict)
        first_message_id = chunk_dict['message_id']
    
    for t in ts_outputs_from_first_chat:
        query += t.get("text", "")
        # print(query)
        query = query.strip().split("\n")
        query1 = list(filter(lambda x: x != '', query))
        # print(query1)
        query2 = "\n".join([question.strip() for question in query1])
        pattern = r'<question>(.*?)[</question></s>]'
        query3 = re.findall(pattern, query2, re.MULTILINE)
        query = "\n".join([question.strip() for question in query3])
        query = prompt + "\n" + query
        yield json.dumps({"subproblem": query, "message_id": first_message_id})
        #拆分子问题
        # query_list = query.strip().split("\n")
        
        if query == "":
            query = prompt
        # query rewrite after:
        thinking_undate = True #第一次得出思考过程
        # second chat with knowledge_base_chat, 思考过程
    outputs_from_second_chat = await knowledge_base_chat_for_pipe(prompt, 
                                                                  query, 
                                                                  knowledge_base_name=knowledge_base_name, 
                                                                  top_k=top_k, 
                                                                  score_threshold=score_threshold, 
                                                                  history=history, 
                                                                  model_name=model_name, 
                                                                  prompt_name=prompt_name,
                                                                  temperature=temperature, 
                                                                  stream=False,
                                                                  max_tokens=max_tokens,
                                                                  request=request
                                                            )
    # 流式输出 for XXX
    ts_outputs_from_second_chat = []
    async for chunk in outputs_from_second_chat:
        chunk_dict = json.loads(chunk)
        ts_outputs_from_second_chat.append(chunk_dict)
        yield json.dumps({"docs": chunk_dict['docs']})
        yield json.dumps({"thinking": chunk_dict['thinking'].rstrip("</s>")})
        
    for d in ts_outputs_from_second_chat:
                if thinking_undate:
                    thinking_undate = False
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)
                elif chunk := d.get("answer"):
                    text += chunk
                    # chat_box.update_msg(text, element_index=0)
    logger.info("############回答初稿###################:\n" + text)
    #获取文档内容信息
    docs_context = d.get("context")
    # print("docs_context:",docs_context)
    #拿到回答后开始进入验证链-提出验证问题
    cove1_result = ""
    cove1_prompt = """
        **角色**：你是一个世界知识验证专家，你可以提出验证问题，从而减少回答初稿中产生的幻觉，解决合理但不正确事实信息的生成。
        **任务**：针对原始问题以及已知信息产生的回答初稿，提出验证问题。如果回答初稿没有需要验证的信息，则回答“无验证问题。”
        **举例**：
        样例1：
        ------------------------------------
        原始问题：列出20世纪的著名发明。
        回答初稿：互联网，量子力学，DNA结构发现。
        验证问题：1.互联网是在20世纪发明的吗？
        2.量子力学是在20世纪发展起来的吗？
        3.DNA的结构是在20世纪发现的吗？
        ------------------------------------
        样例2：
        ------------------------------------
        原始问题：提供一个非洲国家的列表。
        回答初稿：尼日利亚，埃塞俄比亚，埃及，南非，苏丹。
        验证问题：1.尼日利亚在非洲吗？
        2.埃塞俄比亚在非洲吗？
        3.埃及在非洲吗？
        4.南非在非洲吗？
        5.苏丹在非洲吗？
        ------------------------------------      
        **你的任务**：
        问题：{原始问题}
        回答初稿：{回答初稿}
        验证问题：
        """
    cove1_prompt = cove1_prompt.replace("{原始问题}", prompt).replace("{回答初稿}", text)
    # third chat with llm, 验证问题
    outputs_from_third_chat = await chat_for_pipe(cove1_prompt, model_name=model_name, 
                                                  stream=False, 
                                                  temperature= temperature, 
                                                  max_tokens=max_tokens, 
                                                  prompt_name=prompt_name,
                                                  history=history, history_len=history_len)
    
    ts_outputs_from_third_chat = []
    async for chunk in outputs_from_third_chat:
        chunk_dict = json.loads(chunk)
        ts_outputs_from_third_chat.append(chunk_dict)
        yield json.dumps({"validation": chunk_dict['text'].rstrip("</s>"), "message_id": chunk_dict['message_id']})
    
    for t in ts_outputs_from_third_chat:
        cove1_result += t.get("text", "")
    logger.info("############验证问题###################:\n" + cove1_result)
#拿到回答后开始进入验证链-提出验证问题
    # cove2_result = ""
    cove2_prompt = """
        **角色**：你是一个世界知识验证专家，你擅长使用验证问题以及原始问题、已知信息，来对回答初稿进行事实验证，从而减少回答初稿中产生的幻觉，修正合理但不正确事实信息的内容。
        **任务**：通过使用原始问题、已知信息、验证问题，来修正回答初稿,以生成更精准更正确的回答。同时你必须满足一下要求和限制。
        **要求**：你的最终回答要基于回答初稿的格式以及内容，只对验证问题相关的地方进行内容修正。如果无对应的验证问题，则输出回答初稿的内容。
        **限制**：你的最终回答应该避免被看出是经过验证环节的，所以你的最终回答中应避免生成“确实”、“验证”等相关字样，最终回答的内容不能生成举例里面的内容。
        **举例**：
        样例1：
        ------------------------------------
        原始问题：列出20世纪的著名发明。
        已知信息：（一些相关文档信息，文档中均为真实信息）。
        回答初稿：互联网，量子力学，DNA结构发现。
        验证问题：1.互联网是在20世纪发明的吗？
        2.量子力学是在20世纪发展起来的吗？
        3.DNA的结构是在20世纪发现的吗？
        最终回答：互联网，青霉素的发现，DNA结构的发现。
        ------------------------------------
        样例2：
        ------------------------------------
        原始问题：提供一个非洲国家的列表。
        已知信息：（一些相关文档信息，文档中均为真实信息）。
        回答初稿：尼日利亚，埃塞俄比亚，埃及，南非，苏丹。
        验证问题：1.尼日利亚在非洲吗？
        2.埃塞俄比亚在非洲吗？
        3.埃及在非洲吗？
        4.南非在非洲吗？
        5.苏丹在非洲吗？
        最终回答：尼日利亚，埃塞俄比亚，埃及，南非，苏丹。
        ------------------------------------      
        **你的任务**：
        问题：{原始问题}
        已知信息：{文档内容}
        回答初稿：{回答初稿}
        验证问题：{验证问题}
        最终回答：
        """
    cove2_prompt = cove2_prompt.replace("{原始问题}", prompt).replace("{文档内容}", docs_context).replace("{回答初稿}", text).replace("{验证问题}", cove1_result)
    # final chat with llm, 最终回答
    final_output_from_chat =  await chat_for_pipe(cove2_prompt, 
                                                  model_name=model_name, 
                                                  history_len=history_len,
                                                  history=history,
                                                  stream=stream, 
                                                  temperature= temperature, 
                                                  max_tokens=max_tokens, 
                                                  prompt_name=prompt_name)
    
    async for chunk in final_output_from_chat:
        yield chunk


def pipe_knowledge_chat(prompt: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                              conversation_id: str = Body("", description="对话框ID"),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                              score_threshold: float = Body(
                                                    SCORE_THRESHOLD,
                                                    description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                                    ge=0,
                                                    le=2
                                                ),
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
                              prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                              request: Request = None,
                              ):
    """
    This function is used to decorated the `pipe_knowledge_chat`
    """
    return EventSourceResponse(to_be_decorated_pipe_knowledge_chat(prompt=prompt, 
                                                   conversation_id=conversation_id, 
                                                   knowledge_base_name=knowledge_base_name, 
                                                   top_k=top_k,
                                                   score_threshold=score_threshold,
                                                   history_len=history_len,
                                                   history=history,
                                                   stream=stream,
                                                   model_name=model_name,
                                                   temperature=temperature,
                                                   max_tokens=max_tokens,
                                                   prompt_name=prompt_name,
                                                   request=request,
                                                   ))
