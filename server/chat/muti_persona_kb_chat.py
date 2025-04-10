import asyncio
import hashlib
import json
import re
from urllib.parse import urlencode
from fastapi.concurrency import run_in_threadpool
import streamlit as st
from configs.model_config import RERANKER_MAX_LENGTH, RERANKER_MODEL, USE_RERANKER
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler
from server.chat.chat import chat, chat_for_pipe
from langchain_community.document_transformers import (
    LongContextReorder,
)
from fastapi import Body

from configs import (LLM_MODELS, 
                     VECTOR_SEARCH_TOP_K, 
                     SCORE_THRESHOLD, 
                     TEMPERATURE,
                     logger)

from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
# from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional, Union
from server.chat.utils import History
# from langchain.prompts import PromptTemplate
from server.db.repository.message_repository import add_message_to_db
from server.knowledge_base.kb_doc_api import search_docs
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.reranker.reranker import LangchainReranker
from server.utils import BaseResponse, embedding_device, get_ChatOpenAI, get_model_path, get_prompt_template, wrap_done
from webui_pages.utils import *
from sse_starlette.sse import EventSourceResponse
import requests
from requests.auth import HTTPBasicAuth



#职业匹配接口
async def personas_select(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                            count: int = Body(3, description="自动选取角色的个数"),
                            conversation_id: str = Body("", description="对话框ID"),
                            stream: bool = Body(False, description="流式输出"),
                            model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                            temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
                            max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
                            prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                            ):
    QUERY_PROMPT = """
                    你需要选择一组行业专家，他们将一起回答一个问题。每个人代表与该问题相关的不同视角、角色或背景。
                    请从以下列表中选择出{count}个最适合回答该问题的角色：{personas}
                    格式：1. ...\n 2. ...\n 3. ...\n
                    以下是要回答的问题： {question}
                    """
    #获取职业库所有信息

    # 获取 access_token
    token_url = 'http://10.106.51.170:19988/auth/oauth2/token?grant_type=client_credentials'
    username = 'test'
    password = 'test'

    # 使用 Basic Auth 获取 token
    token_response = requests.post(token_url, auth=HTTPBasicAuth(username, password))

    if token_response.status_code == 200:
        token_data = token_response.json()
        access_token = token_data['access_token']
    else:
        print(f'获取 token 失败，状态码: {token_response.status_code}')
        exit()

    # 使用 access_token 进行后续请求
    url = 'http://10.106.51.138:11094/algo/api/profession/fetchAll'
    headers = {
        'Connection': 'keep-alive',
        'Accept': 'application/json, text/plain, */*',
        'Authorization': f'Bearer {access_token}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
        'Referer': 'http://10.106.51.138:11094/',
        'Accept-Language': 'zh-CN,zh;q=0.9,en-CA;q=0.8,en;q=0.7'
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()  # 解析返回的 JSON 数据
        # print(data)  # 打印或处理获取的数据
    else:
        print(f'请求失败，状态码: {response.status_code}')

    persona, persona_name, persona_description, persona_resume, persona_kb_name = [], [], [], [], []
    # 提取每个对象中的信息
    for item in data.get('data'):
        name_value = item.get('name')
        description_value = item.get('description')
        resume_value = item.get('resume')
        kb_name_value = item.get('knowledgeName')
        persona.append(f"{name_value}：{description_value}")
        persona_name.append(name_value)
        persona_description.append(description_value)
        persona_resume.append(resume_value)
        persona_kb_name.append(kb_name_value)
    personas = "\n".join(persona)

    QUERY_PROMPT = QUERY_PROMPT.replace("{question}",query).replace("{personas}",personas).replace("{count}",str(count))

    outputs_from_first_chat = await chat_for_pipe(QUERY_PROMPT, 
                                                  model_name=model_name, 
                                                  stream=False, 
                                                  temperature= temperature, 
                                                  max_tokens=max_tokens, 
                                                  prompt_name=prompt_name)
    persona_list = []
    # 流式输出 for XXX
    ts_outputs_from_first_chat = []
    async for chunk in outputs_from_first_chat:
        chunk_dict = json.loads(chunk)
        ts_outputs_from_first_chat.append(chunk_dict)
        first_message_id = chunk_dict['message_id']
    outputs = ""
    for t in ts_outputs_from_first_chat:
        outputs += t.get("text", "")
    #count个不同角度的专家persona
    persona_select = []
    for s in outputs.split('\n'):
        match = re.search(r'\d+\.\s*(.*)', s)
        if match:
            persona_select.append(match.group(1).split("：")[0])
    if len(persona_select) > count:
        persona_select = persona_select[:count]
    for per in persona_select:
        index = persona_name.index(per.split("：")[0])
        persona_list.append({"persona_name":persona_name[index],"persona_description":persona_description[index],"persona_resume":persona_resume[index],"persona_kb_name":persona_kb_name[index]}) 

    # yield json.dumps({"text": persona_list, "message_id": first_message_id})
    async def generate():
        yield json.dumps({"text": persona_list, "message_id": first_message_id})

    return EventSourceResponse(generate(), media_type="text/event-stream")
    










#职业/名人模型问答接口
async def muti_persona_kb_chat(persona_name: str = Body(..., description="角色名称", examples=["大数据分析师"]),
                               persona_description: str = Body(..., description="角色描述", examples=["负责处理与分析大量数据，提取有效认知战策略。技能包括数据挖掘、建模，研究敌方行为模式，知识范围涵盖大数据、机器学习、信息科学。"]),
                               persona_resume: str = Body(..., description="角色简历", examples=["..."]),
                               persona_kb_name: str = Body(..., description="角色知识库名称", examples=["大数据分析师"]),
                               prompt: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
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
                            ): 

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        raise ModuleNotFoundError
    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator(
            prompt: str,
            top_k: int,
            history: Optional[List[History]],
            model_name: str = model_name,
            prompt_name: str = prompt_name,
        ) -> AsyncIterable[str]:


        # 拆解子问题
        QUERY_PROMPT = """
        **你的简历**：'''{角色及角度}'''
        **任务**：你的具体任务是基于我给出的原始问题生成多个子问题，子问题通过对原始问题进行拆分，从而提高解决复杂问题的能力。
        **限制**：生成与原始问题有较高相关性的三个子问题，你的回答是基于你的口吻以及政治立场的。
        **请严格按照以下格式输出**：
        <question>子问题1</question>
        <question>子问题2</question>
        <question>子问题3</question>
        以下是我给出的原始问题：{原始问题}
        """
        QUERY_PROMPT = QUERY_PROMPT.replace("{角色及角度}", persona_resume).replace("{原始问题}", prompt)

        outputs_from_first_chat = await chat_for_pipe(QUERY_PROMPT, 
                                                    model_name=model_name, 
                                                    stream=False, 
                                                    temperature= temperature, 
                                                    prompt_name=prompt_name,
                                                    history_len=history_len,
                                                    history=history)
        
        # 流式输出 for XXX
        ts_outputs_from_first_chat = []
        async for chunk in outputs_from_first_chat:
            chunk_dict = json.loads(chunk)
            ts_outputs_from_first_chat.append(chunk_dict)
            first_message_id = chunk_dict['message_id']
        query = ""
        for t in ts_outputs_from_first_chat:
            query += t.get("text", "")

        query = query.strip().split("\n")
        query1 = list(filter(lambda x: x != '', query))
        query2 = "\n".join([question.strip() for question in query1])
        pattern = r'<question>(.*?)[</question></s>]'
        query3 = re.findall(pattern, query2, re.MULTILINE)
        sub_question = "\n".join([question.strip() for question in query3])
        sub_question = prompt + "\n" + sub_question
        yield json.dumps({"subproblem": sub_question, "message_id": first_message_id})

        #思考、验证、回答环节
        
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
        
##jskb
        #角色知识库匹配
        sub_question_list = sub_question.strip().split("\n")
        person_kb_docs = []

        # doc = search_docs(query=prompt,
        #             knowledge_base_name=persona_kb_name,
        #             top_k=6,
        #             score_threshold=score_threshold)
        for q in sub_question_list:
            if q == prompt:
                top_k_ls = top_k*2
            else:
                top_k_ls = top_k
            doc = await run_in_threadpool(search_docs,
                                        query=q,
                                        knowledge_base_name=persona_kb_name,
                                        top_k=top_k_ls,
                                        score_threshold=score_threshold)
            person_kb_docs.append(doc)
        person_kb_docs = reciprocal_rank_fusion(person_kb_docs)
        person_kb_context = "\n".join([doc.page_content for doc in person_kb_docs])

##jskb
        docs_temp = []

        
        for q in sub_question_list:
            if q == prompt:
                top_k_ls = top_k*2
            else:
                top_k_ls = top_k
            doc = await run_in_threadpool(search_docs,
                                        query=q,
                                        knowledge_base_name=knowledge_base_name,
                                        top_k=top_k_ls,
                                        score_threshold=score_threshold)
            docs_temp.append(doc)

        docs = reciprocal_rank_fusion(docs_temp)
  
        if USE_RERANKER:
            reranker_model_path = get_model_path(RERANKER_MODEL)
            reranker_model = LangchainReranker(top_n=top_k*(len(sub_question_list)+1)*2,
                                            device=embedding_device(),
                                            max_length=RERANKER_MAX_LENGTH,
                                            model_name_or_path=reranker_model_path)
            person_kb_docs = reranker_model.compress_documents(documents=person_kb_docs, query=prompt)
            docs = reranker_model.compress_documents(documents=docs, query=prompt)

        if "9g" in model_name.lower():
            if len(docs) > 3:
                docs = docs[:3]
                person_kb_docs = person_kb_docs[:3]
        else:
            if len(docs) > 20:
                docs = docs[:20]
                person_kb_docs = person_kb_docs[:20]
        reordering = LongContextReorder()
        reorder_docs = reordering.transform_documents(docs)
        context = "\n".join([doc.page_content for doc in reorder_docs])

        all_prompt = (
            """
            **任务**：根据已知信息和你的简历，首先生成你对问题的思考过程。思考过程是一种指引性质的内容，可以让别人看到问题和已知信息时立马抓住它们的重点。
            思考过程生成后，紧接着根据已知信息和思考过程，准确、全面地给出回答初稿。
            在生成答案后，提出验证问题以检验答案的准确性，并基于这些验证问题修正回答初稿，最终生成更精准的答案。
            **限制**：回答要条理清晰，不允许在答案中添加编造成分。回答初稿不输出。请使用中文。 你的思考、验证及回答均是基于你的口吻以及政治立场。你应该模仿“你的已知信息”中你语录的口吻，或者引用其中你说过的话。
            **请严格按照以下格式输出**：
            ### 思考过程 ###
            1. （第一步的思考）
            2. （第二步的思考）
            ...
            ### 验证问题 ###
            1. （验证初稿答案的第一个问题）
            2. （验证初稿答案的第二个问题）
            ...
            ### 最终答案 ###
            （基于验证问题修正回答初稿，给出最终回答）
            **你的任务**：
            '<问题>结合你的已知信息和立场，你认为{{ question }}</问题>\n'
            '<已知信息>{{ context }}</已知信息>\n'
            '<你的已知信息>{{ person_kb_context }}</你的已知信息>\n'
            '<你的简历>{{ persona }}</你的简历>\n'
            """
        )
        input_msg = History(role="user", content=all_prompt).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)
        
        task = asyncio.create_task(wrap_done(
            chain.acall({ "persona": persona_resume, "question": prompt, "context": context,"person_kb_context":person_kb_context}),
            callback.done),
        )
        # combain four step, end

##jskb
        person_source_documents = []

        for inum, doc in enumerate(person_kb_docs):
            filename = doc.metadata.get("source")
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
            base_url = Request.base_url
            url = f"{base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            person_source_documents.append(text)
        if len(person_source_documents) == 0:
            person_source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")
##jskb

        source_documents = []

        for inum, doc in enumerate(docs):
            filename = doc.metadata.get("source")
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
            base_url = Request.base_url
            url = f"{base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)
        if len(source_documents) == 0:
            source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

        #流式输出参考文档
        yield json.dumps({"person_source_documents":person_source_documents}, ensure_ascii=False)
        yield json.dumps({"docs": source_documents}, ensure_ascii=False)

        think_over = False
        valid_over = False
        if stream:
            # async for token in callback.aiter():
            #     yield json.dumps({"text": token}, ensure_ascii=False)
            # yield json.dumps({"docs": source_documents,"person_source_documents":person_source_documents}, ensure_ascii=False)
            buffer = ""
            async for token in callback.aiter():
                if think_over and valid_over:
                    yield json.dumps({"text": token}, ensure_ascii=False)
                else:
                    buffer += token

                if "### 验证问题 ###" in buffer:
                    think_part = buffer.split("### 思考过程 ###", 1)[1].split("### 验证问题 ###")[0].strip()
                    yield json.dumps({"thinking": think_part}, ensure_ascii=False)
                    buffer = ""  # 重置缓冲区
                    think_over = True

                elif "### 最终答案 ###" in buffer:
                    eval_part = buffer.split("### 最终答案 ###")[0].strip()
                    yield json.dumps({"validation": eval_part}, ensure_ascii=False)
                    buffer = ""  # 重置缓冲区
                    valid_over = True

        else:
            buffer = ""
            async for token in callback.aiter():
                buffer += token

                if "### 验证问题 ###" in buffer:
                    think_part = buffer.split("### 思考过程 ###", 1)[1].split("### 验证问题 ###")[0].strip()
                    yield json.dumps({"thinking": think_part}, ensure_ascii=False)
                    buffer = ""  # 重置缓冲区
                    think_over = True
                elif "### 最终答案 ###" in buffer:
                    eval_part = buffer.split("### 最终答案 ###")[0].strip()
                    yield json.dumps({"validation": eval_part}, ensure_ascii=False)
                    buffer = ""  # 重置缓冲区
                    valid_over = True
            yield json.dumps({"text": buffer}, ensure_ascii=False)

        await task

    return EventSourceResponse(knowledge_base_chat_iterator(prompt, top_k, history, model_name, prompt_name))



def reciprocal_rank_fusion(search_results, k=60):
    fused_scores = {}
    for sr in search_results:
        for doc in sr:
            doc_id = doc.id
            fused_scores[doc_id] = 0

        # Calculate fused score based on reciprocal rank fusion
    for sr in search_results:
        for query_rank, doc in enumerate(sorted(sr, key=lambda x: x.score, reverse=True)):
            doc_id = doc.id
            fused_scores[doc_id] += 1 / (query_rank + k)
            # print(f"Updating score for {doc_id} to {fused_scores[doc_id]} based on rank {query_rank + 1}")

    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    search_results = [item for sublist in search_results for item in sublist]
    reranked_results = [next(doc for doc in search_results if doc.id == doc_id) for doc_id, score in sorted_results]
    # print("Final reranked results:", reranked_results)
    ids = []
    for doc in reranked_results:
        sha256 = hashlib.sha256()
        sha256.update(doc.page_content.encode('utf-8'))
        doc_id = sha256.hexdigest()
        if doc_id in ids:
            reranked_results.remove(doc)
        else:
            ids.append(doc_id)
        
    return reranked_results






#总结回答接口
async def summary_chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                        persona_answers: list = Body(..., description="不同专家的回答"),
                        conversation_id: str = Body("", description="对话框ID"),
                        stream: bool = Body(False, description="流式输出"),
                        model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                        temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
                        max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
                        prompt_name: str = Body("default", description="使用的prompt模板名称")):

    summary_prompt = f"""你是一个世界知识专家，现在你需要总结以下几个专家的回答，生成原始问题最终的回答，回答要全面。原始问题：{query}/n"""
    for per in persona_answers:
        name = per.get("persona_name")
        des = per.get("persona_description")
        answer = per.get("answer")
        summary_prompt += f"专家名称及介绍：{name}，{des}。专家{name}的回答：{answer}/n"

    async def chat_iterator() -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            callbacks=callbacks,
        )

        input_msg = History(role="user", content=summary_prompt).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        if stream:
            async for token in callback.aiter():
                yield json.dumps({"text": token}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"text": answer}, ensure_ascii=False)

        await task

    return EventSourceResponse(chat_iterator())
