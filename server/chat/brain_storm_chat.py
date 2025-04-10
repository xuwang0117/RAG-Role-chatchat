import asyncio
import hashlib
import json
import re
import time
from urllib.parse import urlencode
from fastapi.concurrency import run_in_threadpool
import streamlit as st
import urllib
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
from server.chat.knowledge_base_chat import reciprocal_rank_fusion
from server.chat.utils import History
# from langchain.prompts import PromptTemplate
from server.db.repository.message_repository import add_message_to_db
from server.knowledge_base.kb_doc_api import search_docs
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.reranker.reranker import LangchainReranker
from server.utils import BaseResponse, embedding_device, get_ChatOpenAI, get_model_path, get_prompt_template, wrap_done
from webui_pages.dialogue.read_persona import read_persona
from webui_pages.utils import *
from sse_starlette.sse import EventSourceResponse
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate

from server.person_info_util import get_user_info, get_expert_list_from_db, get_info_with_name_from_db, get_special_num_experts_from_db
from server.chat.discuss_tree import  get_topic_from_outlint
from server.utils import get_respond_httpx
# import logging

# # 创建一个 logger 对象
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# # 创建一个文件处理器，用于将日志信息输出到文件
# file_handler = logging.FileHandler('xw-1206-3.log')
# file_handler.setLevel(logging.INFO)

# # 创建一个格式化器，用于设置日志信息的格式
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)

# # 将文件处理器添加到 logger 对象
# logger.addHandler(file_handler)

#切块+rerank模块
def cut_and_rerank(text,origin_query,n):
    source_documents = []
    num_chunks = (len(text) + 499) // 500  # 向上取整
    for i in range(num_chunks):
        chunk = text[i * 500: (i + 1) * 500]
        source_documents.append(chunk)
    #加载rerank模型
    if USE_RERANKER:
        reranker_model_path = get_model_path(RERANKER_MODEL)
        reranker_model = LangchainReranker(top_n=num_chunks,
                                        device=embedding_device(),
                                        max_length=RERANKER_MAX_LENGTH,
                                        model_name_or_path=reranker_model_path)
    if USE_RERANKER:
        search_documents = reranker_model.compress_search_documents(documents=source_documents, query=origin_query)
    if len(search_documents) > n:
        search_documents = search_documents[:n]
    reordering = LongContextReorder()
    reorder_docs = reordering.transform_documents(search_documents)
    search_context = '\n'.join(reorder_docs)
    
    return search_context

#搜索模块 q:搜索的文本；n:帖文个数。
def get_search_text(q,n):
    # url = "http://10.106.51.159:19999/search-service/search/v1/searchy3"
    # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # time = urllib.parse.quote(f"2024-01-01 10:00:00~{current_time}")
    
    url = "http://10.120.6.24:7778/search/api/search/v1/searchy3"
    # 定义查询参数
    params = {
        "q": q,        # 搜索词
        "start_index": 0,     # 帖源数据起始序号
        "rn": n,           # 请求帖源数据的数量
    }

    # 发送请求
    response = requests.get(url, params=params)

    data = response.json()
    
    # 获取帖源结果
    search_results = data.get('data').get('search_results', [])

    #显示搜索结果
    source_documents = []
    for doc in search_results:
        if "content" not in doc.keys():
            continue
        text = doc['content']
        source_documents.append(text)
    search_results = "\n\n".join(source_documents)
    
    return search_results


#生成推荐专家
async def recommend_expert(topic: str = Body(..., description="群智主题名", examples=["美国大选"]),
                           topic_description: str = Body(..., description="主题描述", examples=["美国大选的具体描述"]),
                           model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                           temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
                           max_tokens: Optional[int] = Body(200, description="限制LLM生成Token数量，默认None代表模型最大值")
                            ):
    async def recommend_expert_iterator(
            topic: str,
            topic_description: str,
            model_name: str = model_name,
            temperature: float = temperature,
            max_tokens = max_tokens
        ) -> AsyncIterable[str]:
        
        personas,personas_name,personas_description, personas_resume = read_persona("./webui_pages/dialogue/personas_occupation.xlsx")
        query = f"""
        你需要选择一组行业专家，他们将一起回答关于这个主题的一个问题。每个人代表与该主题相关的不同视角、角色或背景。
        请从以下列表中选择出五个最适合回答该问题的角色：{personas}
        示例格式：1. **大数据分析师**：...\n 2. **反认知战顾问**：...\n 3. **军事战略家**：...\n 4. **决策心理战顾问**：...\n 5. **情报分析师**：...\n
        以下是我给出的主题： {topic}：{topic_description}
        """
        outputs_from_first_chat = await chat_for_pipe(query=query,
                stream=False,
                model_name=model_name)
        ts_outputs_from_first_chat = []
        async for chunk in outputs_from_first_chat:
            chunk_dict = json.loads(chunk)
            ts_outputs_from_first_chat.append(chunk_dict)
        persona_text = ""
        for t in ts_outputs_from_first_chat:
            persona_text += t.get("text", "")
            
        # print("###################专家生成################:\n",query)
        #三个不同角度的专家persona
        print("persona_text:",persona_text)
        persona = []
        for s in persona_text.split('\n'):
            match = re.search(r'\d+\.\s*(.*)', s)
            if match:
                persona.append(match.group(1).replace("**", ""))
        persona = [item.split("：")[0] for item in persona if item.split("：")[0] in personas_name]

        if len(persona) > 3:
            persona = persona[:3]
        recommend_experts = []
        for per in persona:
            recommend_experts.append({"name":per,"resume":personas_resume[personas_name.index(per)]})
        
        yield json.dumps({"recommend_experts": recommend_experts})
            

    return EventSourceResponse(recommend_expert_iterator(topic,topic_description,model_name, temperature, max_tokens))


#生成大纲
async def generate_outline(topic: str = Body(..., description="群智主题名", examples=["美国大选"]),
                    topic_description: str = Body(..., description="主题描述", examples=["美国大选的具体描述"]),
                    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                            ):
    async def generate_outline_iterator(
            topic: str,
            topic_description: str,
            round: int,
            model_name: str = model_name,
        ) -> AsyncIterable[str]:
        #搜索模块
        search_results = get_search_text(q=topic,n=20)
        # logger.info(f"生成大纲中的搜索信息：{search_results}")
        search_results = cut_and_rerank(search_results,topic,20)
        # logger.info(f"生成大纲中的rerank后的搜索信息：{search_results}")
        # print("search_results:",search_results)
        
        query = f"""**任务**：你是一个资深的事件新闻报告专家，你的任务是根据议题生成一个思维导图，这个思维导图是用于指导多个不同专家和主持人对这个议题进行有深度和广度的讨论，有助于深层次多维度理解这个议题。你可以参考搜索信息来进行生成，如果搜索信息与议题相关性不高则不用参考。如果议题有提及到先验知识的内容，请参考先验知识回答。
        **先验知识**：
        **搜索信息**：{search_results}
        **议题**：{topic}：{topic_description}
        **格式**：1.使用“#标题”表示节标题，使用“##标题”表示小节标题，使用“###标题”表示子小节标题，依此类推。2.请勿包含其他信息。3.不要在大纲中包含主题名称本身。
        **样例**：# 2024美国大选概述\n## 选情关键\n## 民意分析\n## 摇摆州情
        **限制**：1.使用中文回答。生成的每个子议题是10个字以内。2.不需要生成具体内容，只需要生成各级标题即可。3.只需要生成一个一级标题和三到五个二级标题。
        """
        outputs_from_first_chat = await chat_for_pipe(query=query,
                stream=False,
                model_name=model_name)
        ts_outputs_from_first_chat = []
        async for chunk in outputs_from_first_chat:
            chunk_dict = json.loads(chunk)
            ts_outputs_from_first_chat.append(chunk_dict)
        text = ""
        for t in ts_outputs_from_first_chat:
            text += t.get("text", "")
        # logger.info(f"生成大纲prompt：{query}")
        # logger.info(f"生成大纲结果：{text}")
        yield json.dumps({"outline": text})

    return EventSourceResponse(generate_outline_iterator(topic,topic_description,round,model_name))

#生成议题背景
async def generate_background(topic: str = Body(..., description="群智主题名", examples=["美国大选"]),
                    topic_description: str = Body(..., description="主题描述", examples=["美国大选的具体描述"]),
                    outline: str = Body(..., description="群智主题名", examples=["# 2024美国大选概述\n## 选情关键\n### 民意分析\n### 摇摆州情"]),
                    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                    stream: bool = Body(True, description="流式输出"),
                            ):
    async def generate_background_iterator(
            topic: str,
            topic_description: str,
            outline: str,
            model_name: str = model_name,
        ) -> AsyncIterable[str]:
        
        #搜索模块
        search_results = get_search_text(q=topic,n=20)
        # logger.info(f"生成议题背景中的搜索信息：{search_results}")
        search_results = cut_and_rerank(search_results,topic,20)
        # logger.info(f"生成议题背景中的rerank后的搜索信息：{search_results}")
        
        #LLM初始化
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
            max_tokens=None,
            callbacks=callbacks,
        )
                    
        #进大模型模块
        chat_prompt = """
        **任务**：你是一个世界知识专家，你的任务是根据思维导图来生成整个议题背景，确保让别人对议题有更好的理解。你可以参考搜索信息来进行生成，如果搜索信息与议题相关性不高则不用参考。如果议题有提及到先验知识的内容，请参考先验知识回答。
        **先验知识**：
        **搜索信息**：{{search_results}}
        **议题**：{{topic}}：{{topic_description}}
        **思维导图**：{{outline}}
        **限制**：1.使用中文回答。只回答议题背景文本即可。2.不需要任何markdown类似的格式，不要生成以“#”区别的分级标题。3.思维导图是为了给你参考，不要对其进行填充。
        """
        # logger.info(f"生成议题背景prompt：{chat_prompt}")
        input_msg = History(role="user", content=chat_prompt).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([input_msg])
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"search_results": search_results,"topic": topic,"topic_description": topic_description,"outline": outline}),
            callback.done),
        )
        
        answer = ""
        if stream:
            async for token in callback.aiter():
                # answer += token
                # Use server-sent-events to stream the response
                yield json.dumps(
                    {"background": token},
                    ensure_ascii=False)
        else:
            async for token in callback.aiter():
                answer += token
            yield json.dumps(
                {"background": answer},
                ensure_ascii=False)
        # logger.info(f"生成议题背景结果：{answer}")
        yield json.dumps({"conversation_turn": "background<end>"})

    return EventSourceResponse(generate_background_iterator(topic,topic_description,outline,model_name))


#改进大纲
async def improve_outline(topic: str = Body(..., description="群智主题名", examples=["美国大选"]),
                    topic_description: str = Body(..., description="主题描述", examples=["美国大选的具体描述"]),
                    old_outline: str = Body(..., description="群智主题名", examples=["# 2024美国大选概述\n## 选情关键\n### 民意分析\n### 摇摆州情\n### 选战动态\n## 候选人背景\n### 特朗普\n### 哈里斯\n## 选举资金\n### 亿万富翁\n### 超级PAC\n## 选民关注\n### 经济状况\n### 失业率\n### 最低工资\n### 所得税率\n### 无家可归\n## 人口与地理\n### 人口分布\n### 经济实力\n### 地理面积\n### 人口密度\n### 移民状况\n## 社会问题\n### 种族分布\n### 青年选民\n### 产妇死亡率\n### 自杀率\n### 暴力犯罪\n### 枪击事件\n## 政策议题\n### 死刑执行\n### 堕胎法规\n### 移民政策\n## 历史与联邦\n### 各州加入\n### 历史背景\n### 宪法影响"]),
                    history: list = Body([], description="群智讨论历史", examples=[[{"role": "expert1", "content": "expert1发言"},{"role": "expert2", "content": "expert2发言"}]]),
                    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                            ):
    async def improve_outline_iterator(
            topic: str,
            topic_description: str,
            old_outline: str,
            model_name: str = model_name,
        ) -> AsyncIterable[str]:

        #历史发言模块（获取最新一轮的发言）
        history_content = ""
        for item in reversed(history):
            if item['role'] == "moderator":
                history_content = item['content'] + "\n" + history_content
                break
            else:
                history_content = item['content'] + "\n" + history_content
        # logger.info(f"改进大纲中的历史发言：{history_content}")
        # history_content = "\n".join([f"{item['content']}" for item in history])
        
        query = f"""**任务**：1.你是一个世界知识专家，你已经有了涵盖一般信息的大纲草案，现在希望你根据各位专家的历史发言学到的信息来改进它，使其提供更多信息。2.在大纲草案的基础上新添加一到三个与历史发言相关的标题即可。3.新添加的标题不能是一级标题，其他等级不限，也可以是四级、五级、六级等更多级别标题。4.如果历史发言中没有提到大纲草案以外的内容，可以不新添加内容，返回大纲草案即可。5.如果议题有提及到先验知识的内容，请参考先验知识回答。
        **先验知识**：
        **历史发言**：{history_content}
        **议题**：{topic}：{topic_description}
        **大纲草案**：{old_outline}
        **格式**：1.使用“#标题”表示一级标题，使用“##标题”表示二级标题，使用“###标题”表示三级标题，使用“####标题”表示四级标题，使用“#####标题”表示五级标题，使用“#####标题”表示六级标题，依此类推。2.请勿包含其他信息。3.不要在大纲中包含主题名称本身。
        **限制**：1.使用中文回答。生成的每个子议题是10个字以内。2.不需要生成具体内容，只需要生成各级标题即可。3.每次要根据历史发言来进行新增子标题，新添加的子标题要和历史发言强相关。4.改进大纲时要采用广度优先遍历的方式来新添加子标题。5.广度优先遍历的方式：当二级标题够五个后，就要开始生成三级标题；当每个二级标题下的三级标题至少有两个后，就要开始生成四级标题；当每个三级标题下的四级标题至少有两个后，就要开始生成五级标题；依此类推。6.新添加的标题必须在同一个标题下。
        """
        outputs_from_first_chat = await chat_for_pipe(query=query,
                stream=False,
                model_name=model_name)
        ts_outputs_from_first_chat = []
        async for chunk in outputs_from_first_chat:
            chunk_dict = json.loads(chunk)
            ts_outputs_from_first_chat.append(chunk_dict)
        text = ""
        for t in ts_outputs_from_first_chat:
            text += t.get("text", "")
        # logger.info(f"改进大纲prompt：{query}")
        # logger.info(f"改进大纲结果：{text}")
        yield json.dumps({"new_outline": text})

    return EventSourceResponse(improve_outline_iterator(topic,topic_description,old_outline,model_name))


#讨论
async def discuss(topic: str = Body(..., description="群智主题名", examples=["美国大选"]),
                    topic_description: str = Body(..., description="主题描述", examples=["美国大选的具体描述"]),
                    experts: list = Body([{"id":"Q5562913","name":"吉娜·雷蒙多","icon_url":"https://present.a.wkycloud.com/gene/image/Q5562913.png"},{"id":"Q766866","name":"凯文·麦卡锡 ","icon_url":"https://present.a.wkycloud.com/gene/image/Q766866.png"},{"id":"Q380900","name":"查克·舒默","icon_url":"https://present.a.wkycloud.com/gene/image/Q380900.png"},{"id":"Q7821917","name":"珍妮特·耶伦","icon_url":"https://present.a.wkycloud.com/gene/image/Q7821917.png"}], description="选择的专家", examples=[[{"id":"Q5562913","name":"吉娜·雷蒙多","icon_url":"https://present.a.wkycloud.com/gene/image/Q5562913.png"},{"id":"Q766866","name":"凯文·麦卡锡 ","icon_url":"https://present.a.wkycloud.com/gene/image/Q766866.png"},{"id":"Q380900","name":"查克·舒默","icon_url":"https://present.a.wkycloud.com/gene/image/Q380900.png"},{"id":"Q7821917","name":"珍妮特·耶伦","icon_url":"https://present.a.wkycloud.com/gene/image/Q7821917.png"}]]),
                    outline: str = Body("# 当前大纲", description="当前大纲", examples=["# 当前大纲"]),
                    history: list = Body([], description="群智讨论历史", examples=[[{"role": "expert1", "content": "expert1发言"},{"role": "expert2", "content": "expert2发言"}]]),
                    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                    stream: bool = Body(True, description="流式输出"),
                            ):
    async def discuss_iterator(
            topic: str,
            topic_description: str,
            history: list,
            experts: list,
            outline:str,
            model_name: str = model_name,
        ) -> AsyncIterable[str]:
        #历史发言模块（获取所有主持人的提问发言）
        history_content = ""
        for index, item in enumerate(history):
            if item['role'] == "moderator" and index > 0:
                history_content += item['content'] + "\n"
        # logger.info(f"讨论中主持人的历史发言模块：{history_content}")
        # history_content = "\n".join([f"{item['role']}：{item['content']}" for item in history])
        
        #LLM模块
        #LLM初始化
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
            max_tokens=None,
            callbacks=callbacks,
        )

        
        chat_prompt = """**任务**： 
        1.现在你是一个资深的圆桌论坛主持人，你的任务是根据大纲来对专家们进行提问，你提出的问题不仅要有引导性，还要能够激发深入讨论，促进专家之间的互动与观点碰撞。例如，在探讨“数字转型对企业的影响”这一主题时，您可以这样提问：“在数字化转型的大潮中，不同企业所面临的挑战和机遇各有不同。请问各位嘉宾，您认为在这一过程中，有哪些关键因素是企业必须重视的？”，“同时，能否分享一下您所在企业在数字转型过程中遇到的最具挑战性的时刻，以及是如何克服这些挑战的？",“此外，数字转型对企业的组织结构、文化以及人才管理带来了哪些深远影响，您又是如何应对这些变化的？”等
        2.提问的议题从大纲中选取，并根据选取的议题生成问题
        3.如果议题有提及到先验知识的内容，请参考先验知识回答。
        **先验知识**：
        **限制**：
        1.使用中文回答。你的提问要求是50个字以内。要简短精炼。
        2.生成的回答直接是你的发言，不要生成其他无关信息。
        3.发言开头不要生成类似“主持人：”的文本。
        4.严格按照样例的格式生成。
        5.要回避已提问问题。
        **已提问问题**： 
        {{ history_content }}
        **总议题**： 
        {{ topic }}:{{ topic_description }}
        **大纲**： 
        {{ question_outline }}
        """
        question_outline = get_topic_from_outlint(outline)
        # logger.info(f"讨论中主持人的提问大纲：{question_outline}")
        # logger.info(f"讨论中主持人的prompt：{chat_prompt}")
        input_msg = History(role="user", content=chat_prompt).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([input_msg])
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            # chain.acall({"history_content": history_content,"topic": topic,"topic_description": topic_description,"outline": outline}),
            chain.acall({"history_content": history_content,"topic": topic,"topic_description": topic_description,"question_outline": question_outline}),
            callback.done),
        )
        #为了提醒后端，每次主持人的发言开始标志。返回id=00。
        yield json.dumps({"moderator": "00"})
        yield json.dumps({"moderator": "主持人："})
        
        moderator_answer = ""
        count_iter = 0 #添加迭代次计数 为了去除发言中的 “人名：”
        if stream:
            async for token in callback.aiter():
                count_iter += 1
                if count_iter > len("主持人：") or token not in "主持人：":
                    moderator_answer += token
                    # Use server-sent-events to stream the response
                    yield json.dumps(
                        {"moderator": token},
                        ensure_ascii=False)
            history.append({"role":"moderator","content":"主持人："+ moderator_answer})
            # logger.info(f"讨论中主持人提问结果：{moderator_answer}")
        else:
            async for token in callback.aiter():
                count_iter += 1
                if count_iter > len("主持人：") or token not in "主持人：":
                    moderator_answer += token
            yield json.dumps(
                {"moderator": moderator_answer},
                ensure_ascii=False)
            history.append({"role":"moderator","content":"主持人："+ moderator_answer})
        yield json.dumps({"moderator": "00<end>"})
        await task

        #获取会议专家列表
        experts_list = []
        for ex in experts:
            experts_list.append(ex["name"])
        experts_list_str = "、".join(experts_list)
        for i in range(len(experts)):
            #角色知识库匹配
            person_kb_docs = []

            doc = search_docs(query=moderator_answer,
                        knowledge_base_name=experts[i]["name"],
                        top_k=5,
                        score_threshold=0.7)
        
            person_kb_docs.append(doc)
            person_kb_docs = reciprocal_rank_fusion(person_kb_docs)
            person_kb_context = "\n".join([doc.page_content for doc in person_kb_docs])
            # name = experts[i]["name"]
            # logger.info(f"讨论中{name}的知识库：{person_kb_context}")
            
            #人物画像模块
            expert_resume = ""
            # expert_id = experts[i]["id"]
            # data = await get_user_info(id=expert_id)
            # expert_resume = ',\n'.join([f'"{key}": "{value}"' for key, value in data.items()])
            # # logger.info(f"讨论中{name}的人物画像：{expert_resume}")
        
            
            #搜索模块
            search_results = ""
            # search_results = get_search_text(q=experts[i]["name"]  +"，"+ topic +"，"+ moderator_answer,n=10)
            # # logger.info(f"讨论中{name}的搜索内容：{search_results}")
            # search_results = cut_and_rerank(search_results,topic,5)
            # # logger.info(f"讨论中{name}的rerank后的搜索内容：{search_results}")
            
            #历史发言模块（获取最新一轮中从主持人开始的发言）
            history_content = "" 
            for item in reversed(history):
                if item['role'] == "moderator":
                    history_content = item['content'] + "\n" + history_content
                    break
                else:
                    history_content = item['content'] + "\n" + history_content
            # logger.info(f"讨论中{name}的历史发言：{history_content}")
            # history_content = "\n".join([f"{item['role']}：{item['content']}" for item in history[-len(experts)-1:]])
            
            #LLM模块
            #LLM初始化
            callback = AsyncIteratorCallbackHandler()
            callbacks = [callback]

            model = get_ChatOpenAI(
                model_name=model_name,
                temperature=0.7,
                max_tokens=None,
                callbacks=callbacks,
            )
            chat_prompt = """**任务**： 1.现在你是{{expert}}，请根据你的人物介绍和搜索信息，同时根据其他专家们的历史发言和总议题，以你在人物介绍中的相关立场、视角以及你擅长的领域知识，来回答问题，回答要口语化。
            2.你现在参加的会议中，专家以及发言顺序为：{{experts_list_str}}；同时参考历史发言时，如果你认同前面其他人的观点时就要表示出赞成以及不发表重复意见，如果不认同就要表示出不认同以及进行反驳，如果有补充的就表示出补充观点。
            3.你的发言要和你在人物介绍中的相关立场强结合。
            4.如果议题有提及到先验知识的内容，请参考先验知识回答。
            **先验知识**： 
            **限制**： 1.使用中文回答，回答要口语化。你的发言要求是100个字以内。要简短精炼。
            2.生成的回答直接是你的发言，不要生成其他无关信息。
            3.模仿帖文例子中的口吻以第一人称进行回答。
            4.参考历史发言，其他人说过的话就不要再重复。
            5.内容中不要生成以“#”开始的标签或任何提示性语句。
            6.不要生成如“作为商务部长，”、“马斯克：”等类似开头
            **人物介绍**： {{expert_resume}}
            **搜索信息**： {{search_results}}
            **历史发言**： {{history_content}}
            **问题**： {{moderator_answer}}
            """
            # logger.info(f"讨论中{name}的prompt：{chat_prompt}")
            # print("expert_resume:",expert_resume)
            input_msg = History(role="user", content=chat_prompt).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages([input_msg])
            chain = LLMChain(prompt=chat_prompt, llm=model)

            # Begin a task that runs in the background.
            task = asyncio.create_task(wrap_done(
                chain.acall({"expert": experts[i]["name"],"experts_list_str": experts_list_str,"expert_resume": expert_resume,"search_results": search_results+"\n"+person_kb_context,"history_content": history_content,"moderator_answer": moderator_answer}),
                callback.done),
            )

            #为了提醒后端，每个专家的发言开始标志。返回id。
            yield json.dumps({"{}".format(experts[i]["name"]): experts[i]["id"]})
            yield json.dumps({"{}".format(experts[i]["name"]): experts[i]["name"]+"："})
                
                
            answer = ""
            count_iter = 0 #添加迭代次计数 为了去除发言中的 “人名：”
            if stream:
                async for token in callback.aiter():
                    count_iter += 1
                    if count_iter > len(experts[i]["name"]) or token not in experts[i]["name"]+"：":
                        answer += token
                        # Use server-sent-events to stream the response
                        yield json.dumps(
                            {"{}".format(experts[i]["name"]): token},
                            ensure_ascii=False)
                history.append({"role":experts[i]["name"],"content":experts[i]["name"]+"："+answer})
                # logger.info(f"讨论中{name}发言结果：{answer}")
            else:
                async for token in callback.aiter():
                    count_iter += 1
                    if count_iter > len(experts[i]["name"]) or token not in experts[i]["name"]+"：":
                        answer += token
                yield json.dumps(
                    {"{}".format(experts[i]["name"]): answer},
                    ensure_ascii=False)
                history.append({"role":experts[i]["name"],"content":experts[i]["name"]+"："+answer})
            yield json.dumps({"{}".format(experts[i]["name"]): experts[i]["id"]+"<end>"})
            
            await task  
        yield json.dumps({"conversation_turn": "<end>"})   
                

    return EventSourceResponse(discuss_iterator(topic,topic_description,history,experts,outline,model_name))


#指定发言
async def select_expert_answer(topic: str = Body(..., description="群智主题名", examples=["美国大选"]),
                    topic_description: str = Body(..., description="主题描述", examples=["美国大选的具体描述"]),
                    user_input: str = Body(..., description="用户输入", examples=["特朗普当选的原因"]),
                    expert: list = Body([{"id":"Q5562913","name":"吉娜·雷蒙多","icon_url":"https://present.a.wkycloud.com/gene/image/Q5562913.png"}], description="指定专家", examples=[[{"id":"Q5562913","name":"吉娜·雷蒙多","icon_url":"https://present.a.wkycloud.com/gene/image/Q5562913.png"}]]),
                    history: list = Body([], description="群智讨论历史", examples=[[{"role": "expert1", "content": "expert1发言"},{"role": "expert2", "content": "expert2发言"}]]),
                    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                    stream: bool = Body(True, description="流式输出"),
                            ):
    async def select_expert_answer_iterator(
            topic: str,
            topic_description: str,
            user_input: str,
            expert: str,
            history:list,
            model_name: str = model_name,
        ) -> AsyncIterable[str]:
         #角色知识库匹配
        person_kb_docs = []

        doc = search_docs(query=user_input,
                    knowledge_base_name=expert[0]["name"],
                    top_k=5,
                    score_threshold=0.7)
    
        person_kb_docs.append(doc)
        person_kb_docs = reciprocal_rank_fusion(person_kb_docs)
        person_kb_context = "\n".join([doc.page_content for doc in person_kb_docs])
        
        #人物画像模块
        expert_resume = ""
        # expert_id = expert[0]["id"]
        # data = await get_user_info(id=expert_id)
        # expert_resume = ',\n'.join([f'"{key}": "{value}"' for key, value in data.items()]) 
    
        
        #搜索模块
        search_results = ""
        # search_results = get_search_text(q=expert[0]["name"] +"，"+ topic +"，"+user_input,n=10)
        # search_results = cut_and_rerank(search_results,topic,5)

        #历史发言模块
        history_content = "\n".join([f"{item['content']}" for item in history[-5:]])
        
        #LLM初始化
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
            max_tokens=None,
            callbacks=callbacks,
        )
                    
        #进大模型模块
        chat_prompt = """**任务**： 1.现在你是{{expert}}，请根据你的人物介绍和搜索信息，同时根据其他专家们的历史发言和总议题，以你在人物介绍中的相关立场、视角以及你擅长的领域知识，来回答问题。
        2.参考历史发言时，要和其他人发言风格以及内容有明显区别。
        3.你的发言要和你在人物介绍中的相关立场强结合。
        4.如果议题有提及到先验知识的内容，请参考先验知识回答。
        **先验知识**： 
        **限制**： 1.使用中文回答。你的发言要求是100个字以内。要简短精炼。
        2.生成的回答直接是你的发言，不要生成其他无关信息。
        3.模仿帖文例子中的口吻以第一人称进行回答。
        4.参考历史发言，其他人说过的话就不要再重复。
        5.内容中不要生成以“#”开始的标签或任何提示性语句。 
        6.不要生成如“作为商务部长，”、“马斯克：”等类似开头
        **人物介绍**： {{expert_resume}}
        **搜索信息**： {{search_results}}
        **历史发言**： {{history_content}}
        **问题**： {{user_input}}
        """
        input_msg = History(role="user", content=chat_prompt).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([input_msg])
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"expert": expert[0]["name"],"expert_resume": expert_resume,"search_results": search_results+"\n"+person_kb_context,"history_content": history_content,"user_input": user_input}),
            callback.done),
        )
            
        yield json.dumps({"expert_answer": expert[0]["name"]+"："})
        
        answer = ""
        count_iter = 0 #添加迭代次计数 为了去除发言中的 “人名：”
        if stream:
            async for token in callback.aiter():
                count_iter += 1
                if count_iter > len(expert[0]["name"]) or token not in expert[0]["name"]+"：":
                    # Use server-sent-events to stream the response
                    yield json.dumps(
                        {"expert_answer": token},
                        ensure_ascii=False)
        else:
            async for token in callback.aiter():
                count_iter += 1
                if count_iter > len(expert[0]["name"]) or token not in expert[0]["name"]+"：":
                    answer += token
            yield json.dumps(
                {"expert_answer": answer},
                ensure_ascii=False)
        yield json.dumps({"expert_answer": expert[0]["id"]+"<end>"})
        yield json.dumps({"conversation_turn": "<end>"})

    return EventSourceResponse(select_expert_answer_iterator(topic,topic_description,user_input,expert,history,model_name))


#生成总结
async def summary(topic: str = Body(..., description="群智主题名", examples=["美国大选"]),
                    topic_description: str = Body(..., description="主题描述", examples=["美国大选的具体描述"]),
                    outline: str = Body("# 当前大纲", description="当前大纲", examples=["# 当前大纲"]),
                    history: list = Body([], description="群智讨论历史", examples=[[{"role": "expert1", "content": "expert1发言"},{"role": "expert2", "content": "expert2发言"}]]),
                    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                    stream: bool = Body(False, description="流式输出"),
                            ):
    async def summary_iterator(
            topic: str,
            topic_description: str,
            outline: str,
            history:list,
            model_name: str = model_name,
        ) -> AsyncIterable[str]:
        
        #历史发言
        history_content = "\n".join([f"{item['content']}" for item in history])
        #LLM初始化
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
            max_tokens=None,
            callbacks=callbacks,
        )
                    
        #进大模型模块
        chat_prompt = """**任务**： 你是一个维基百科作者，你的任务是总结前面专家的历史发言，根据大纲生成一份总结文档。
        **格式**： 使用中文回答。
        **历史发言**： {{history_content}}
        **总议题**： {{topic}}:{{topic_description}}
        **大纲**： {{outline}}
        """
        input_msg = History(role="user", content=chat_prompt).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([input_msg])
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"history_content": history_content,"topic": topic,"topic_description": topic_description,"outline": outline}),
            callback.done),
        )
            
        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps(
                    {"summary": token},
                    ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps(
                {"summary": answer},
                ensure_ascii=False)
        await task

    return EventSourceResponse(summary_iterator(topic,topic_description,outline,history,model_name))



#继续讨论
async def continue_discuss(topic: str = Body(..., description="群智主题名", examples=["美国大选"]),
                    topic_description: str = Body(..., description="主题描述", examples=["美国大选的具体描述"]),
                    user_input: str = Body(..., description="用户输入", examples=["选举人制度"]),
                    experts: list = Body([{"id":"Q5562913","name":"吉娜·雷蒙多","icon_url":"https://present.a.wkycloud.com/gene/image/Q5562913.png"},{"id":"Q766866","name":"凯文·麦卡锡 ","icon_url":"https://present.a.wkycloud.com/gene/image/Q766866.png"},{"id":"Q380900","name":"查克·舒默","icon_url":"https://present.a.wkycloud.com/gene/image/Q380900.png"},{"id":"Q7821917","name":"珍妮特·耶伦","icon_url":"https://present.a.wkycloud.com/gene/image/Q7821917.png"}], description="选择的专家", examples=[[{"id":"Q5562913","name":"吉娜·雷蒙多","icon_url":"https://present.a.wkycloud.com/gene/image/Q5562913.png"},{"id":"Q766866","name":"凯文·麦卡锡 ","icon_url":"https://present.a.wkycloud.com/gene/image/Q766866.png"},{"id":"Q380900","name":"查克·舒默","icon_url":"https://present.a.wkycloud.com/gene/image/Q380900.png"},{"id":"Q7821917","name":"珍妮特·耶伦","icon_url":"https://present.a.wkycloud.com/gene/image/Q7821917.png"}]]),
                    history: list = Body([], description="群智讨论历史", examples=[[{"role": "expert1", "content": "expert1发言"},{"role": "expert2", "content": "expert2发言"}]]),
                    outline: str = Body("# 当前大纲", description="当前大纲", examples=["# 当前大纲"]),
                    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                    stream: bool = Body(True, description="流式输出"),
                            ):
    async def continue_discuss_iterator(
            topic: str,
            topic_description: str,
            user_input: str,
            experts: list,
            history:list,
            outline:str,
            model_name: str = model_name,
        ) -> AsyncIterable[str]:
        #获取会议专家列表
        experts_list = []
        for ex in experts:
            experts_list.append(ex["name"])
        experts_list_str = "、".join(experts_list)
        for i in range(len(experts)):
            #角色知识库匹配
            person_kb_docs = []

            doc = search_docs(query=user_input,
                        knowledge_base_name=experts[i]["name"],
                        top_k=5,
                        score_threshold=0.7)
        
            person_kb_docs.append(doc)
            person_kb_docs = reciprocal_rank_fusion(person_kb_docs)
            person_kb_context = "\n".join([doc.page_content for doc in person_kb_docs])
            #人物画像模块
            expert_resume = ""
            # expert_id = experts[i]["id"]
            # data = await get_user_info(id=expert_id)
            # expert_resume = ',\n'.join([f'"{key}": "{value}"' for key, value in data.items()])
        
            
            #搜索模块
            search_results = ""
            # search_results = get_search_text(q=experts[i]["name"] +"，"+ topic +"，"+user_input,n=10)
            # search_results = cut_and_rerank(search_results,topic,5)
            
            #历史发言模块（获取最新一轮中从主持人开始的发言）
            history_content = ""
            for item in reversed(history):
                if item['role'] == "moderator":
                    history_content = item['content'] + "\n" + history_content
                    break
                else:
                    history_content = item['content'] + "\n" + history_content
            # history_content = "\n".join([f"{item['content']}" for item in history[-len(experts)-1:]])
            
            #LLM模块
            #LLM初始化
            callback = AsyncIteratorCallbackHandler()
            callbacks = [callback]

            model = get_ChatOpenAI(
                model_name=model_name,
                temperature=0.7,
                max_tokens=None,
                callbacks=callbacks,
            )
            chat_prompt = """**任务**： 1.现在你是{{expert}}，请根据你的人物介绍和搜索信息，同时根据其他专家们的历史发言和总议题，以你在人物介绍中的相关立场、视角以及你擅长的领域知识，来回答问题，回答要口语化。
            2.你现在参加的会议中，专家以及发言顺序为：{{experts_list_str}}；同时参考历史发言时，如果你认同前面其他人的观点时就要表示出赞成以及不发表重复意见，如果不认同就要表示出不认同以及进行反驳，如果有补充的就表示出补充观点。
            3.你的发言要和你在人物介绍中的相关立场强结合。
            4.如果议题有提及到先验知识的内容，请参考先验知识回答。
            **先验知识**： 
            **限制**： 1.使用中文回答，回答要口语化。你的发言要求是100个字以内。要简短精炼。
            2.生成的回答直接是你的发言，不要生成其他无关信息。
            3.模仿帖文例子中的口吻以第一人称进行回答。
            4.参考历史发言，其他人说过的话就不要再重复。
            5.内容中不要生成以“#”开始的标签或任何提示性语句。 
            6.不要生成如“作为商务部长，”、“马斯克：”等类似开头
            **人物介绍**： {{expert_resume}}
            **搜索信息**： {{search_results}}
            **历史发言**： {{history_content}}
            **问题**： {{user_input}}
            """

            input_msg = History(role="user", content=chat_prompt).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages([input_msg])
            chain = LLMChain(prompt=chat_prompt, llm=model)

            # Begin a task that runs in the background.
            task = asyncio.create_task(wrap_done(
                chain.acall({"expert": experts[i]["name"],"experts_list_str": experts_list_str,"expert_resume": expert_resume,"search_results": search_results+"\n"+person_kb_context,"history_content": history_content,"user_input": user_input}),
                callback.done),
            )

            #为了提醒后端，每个专家的发言开始标志。返回id。
            yield json.dumps({"{}".format(experts[i]["name"]): experts[i]["id"]})
            yield json.dumps({"{}".format(experts[i]["name"]): experts[i]["name"]+"："})
                
            answer = ""
            count_iter = 0 #添加迭代次计数 为了去除发言中的 “人名：”
            if stream:
                async for token in callback.aiter():
                    count_iter += 1
                    if count_iter > len(experts[i]["name"]) or token not in experts[i]["name"]+"：":
                        answer += token
                        # Use server-sent-events to stream the response
                        yield json.dumps(
                            {"{}".format(experts[i]["name"]): token},
                            ensure_ascii=False)
                history.append({"role":experts[i]["name"],"content":experts[i]["name"]+"："+answer})
            else:
                async for token in callback.aiter():
                    count_iter += 1
                    if count_iter > len(experts[i]["name"]) or token not in experts[i]["name"]+"：":
                        answer += token
                yield json.dumps(
                    {"{}".format(experts[i]["name"]): answer},
                    ensure_ascii=False)
                history.append({"role":experts[i]["name"],"content":experts[i]["name"]+"："+answer})
            yield json.dumps({"{}".format(experts[i]["name"]): experts[i]["id"]+"<end>"})
            await task
            
        yield json.dumps({"conversation_turn": "<end>"})
        
                
                

    return EventSourceResponse(continue_discuss_iterator(topic,topic_description,user_input,experts,history,outline,model_name))



#推荐感兴趣话题
async def recommend_interesting_topics(topic: str = Body(..., description="群智主题名", examples=["美国大选"]),
                    topic_description: str = Body(..., description="主题描述", examples=["美国大选的具体描述"]),
                    outline: str = Body("# 当前大纲", description="当前大纲", examples=["# 当前大纲"]),
                    history: list = Body([], description="群智讨论历史", examples=[[{"role": "expert1", "content": "expert1发言"},{"role": "expert2", "content": "expert2发言"}]]),
                    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                            ):
    async def recommend_interesting_topics_iterator(
            topic: str,
            topic_description: str,
            outline: str,
            history: list,
            model_name: str = model_name,
        ) -> AsyncIterable[str]:

        #历史发言模块（获取所有主持人的提问发言）
        history_content = ""
        for index, item in enumerate(history):
            if item['role'] == "moderator" and index > 0:
                history_content += item['content'] + "\n"
        # history_content = "\n".join([f"{item['content']}" for item in history])
        
        query = f"""**任务**：1.现在你是一个维基百科的作者，你的任务是根据当前大纲来对专家们进行提问，以得到你想要了解的信息。2.在“历史发言”中你是“主持人”，请参考历史发言，其中问过的问题不要重复提问，使用广度优先遍历方式进行覆盖式提问。
        **限制**：使用中文回答。生成的每个子议题是10个字以内。要简短精炼。
        **格式**：1. ...\n 2. ...\n 3. ...\n ......
        **历史发言**： {history_content}
        **总议题**：{topic}：{topic_description}
        **当前大纲**：{outline}
        """
        outputs_from_first_chat = await chat_for_pipe(query=query,
                stream=False,
                model_name=model_name)
        ts_outputs_from_first_chat = []
        async for chunk in outputs_from_first_chat:
            chunk_dict = json.loads(chunk)
            ts_outputs_from_first_chat.append(chunk_dict)
        text = ""
        for t in ts_outputs_from_first_chat:
            text += t.get("text", "")
            
        interesting_topics_list = []
        for s in text.split('\n'):
            match = re.search(r'\d+\.\s*(.*)', s)
            if match:
                interesting_topics_list.append(match.group(1))
            
        yield json.dumps({"interesting_topics": interesting_topics_list[:3]})

    return EventSourceResponse(recommend_interesting_topics_iterator(topic,topic_description,outline,history,model_name))