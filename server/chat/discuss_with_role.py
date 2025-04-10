from fastapi import Body, Request
from sse_starlette.sse import EventSourceResponse
from configs import (LLM_MODELS, 
                     VECTOR_SEARCH_TOP_K, 
                     SCORE_THRESHOLD, 
                     TEMPERATURE,
                     USE_RERANKER,
                     RERANKER_MODEL,
                     RERANKER_MAX_LENGTH)
from server.utils import wrap_done, get_ChatOpenAI, get_model_path
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional, Union
import asyncio, json, hashlib, time, requests
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.document_transformers import LongContextReorder
from server.chat.utils import History
from server.reranker.reranker import LangchainReranker
from server.utils import embedding_device
from server.person_info_util import get_info_with_name_from_db, get_user_info
async def new_chat_only(restriction: str= Body(..., description="限制条件", examples=[""]),
                        origin_query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                        top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                        score_threshold: float = Body(
                            SCORE_THRESHOLD,
                            description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                            ge=0,
                            le=2
                        ),
                        history: List[History] = Body(
                            [],
                            description="历史对话",
                            examples=[[
                                {"role": "user",
                                "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                {"role": "assistant",
                                "content": "虎头虎脑"}]]
                        ),
                        stream: bool = Body(False, description="流式输出"),
                        model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                        temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                        max_tokens: Optional[int] = Body(
                            None,
                            description="限制LLM生成Token数量，默认None代表模型最大值"
                        ),
                        request: Request = None,
                        ):
    
    history = [History.from_data(h) for h in history]
    if len(history) > 5:
        history = history[-5:]

    print(history)

    top_k = 3
    temperature = 0.7
    score_threshold = 0.7
    print("top_k: ", top_k)
    print("temperature: ", temperature)
    print("score_threshold: ", score_threshold)

    async def chat_iterator() -> AsyncIterable[str]:
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
            '{{ input }}'
            """
        )
        
        input_msg = History(role="user", content=all_prompt).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        task = asyncio.create_task(wrap_done(
            chain.acall({"input": origin_query + restriction}),
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



async def discuss_with_role(person: str= Body(..., description="角色", examples=[""]),
                            origin_query: str= Body(..., description="用户输入", examples=["你好"]),
                            top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                            score_threshold: float = Body(
                                SCORE_THRESHOLD,
                                description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                ge=0,
                                le=2
                            ),
                            history: List[History] = Body(
                                [],
                                description="历史对话",
                                examples=[[
                                    {"role": "user",
                                    "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                    {"role": "assistant",
                                    "content": "虎头虎脑"}]]
                            ),
                            stream: bool = Body(False, description="流式输出"),
                            model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                            temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                            max_tokens: Optional[int] = Body(
                                None,
                                description="限制LLM生成Token数量，默认None代表模型最大值"
                            ),
                            prompt_name: str = Body(
                                "default",
                                description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
                            ),
                            request: Request = None,
                            ):

    history = [History.from_data(h) for h in history]
    if len(history) > 5:
        history = history[-5:]

    print(history)

    top_k = 3
    temperature = 0.7
    score_threshold = 0.7
    print("top_k: ", top_k)
    print("temperature: ", temperature)
    print("score_threshold: ", score_threshold)

    async def discuss_with_role_iterator(
            origin_query: str,
            top_k: int,
            history: Optional[List[History]],
            model_name: str = model_name,
            prompt_name: str = prompt_name,
        ) -> AsyncIterable[str]:
        
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

        print("origin_query: ", origin_query)

        # 角色简历
        person_context = ""
        person_name = ""
        try:
            data = await get_user_info(person)
            expert_resume = ',\n'.join([f'"{key}": "{value}"' for key, value in data.items()])
            person_name = data['姓名']
            print(person_name)
            print(expert_resume)
            person_context = expert_resume
        except:
            person_name = "四季大模型"
            person_context = "你是由四季大模型团队研发的智能助手"
            print(person_name)
            print(person_context)

        # ES检索
        start_time = time.time()
        search_documents = []
        search_documents += search_func(origin_query)
        print("search_documents: ", search_documents)

        reordering = LongContextReorder()
        reorder_docs = reordering.transform_documents(search_documents)

        search_context = '\n'.join(reorder_docs)
        print("search_context: ", search_context)

        print(f"ES检索，运行时间: {time.time() - start_time} 秒")

        start_time = time.time()
        all_prompt = (
            """
            **任务**：参考已知信息中与问题相关的内容，并结合大模型本身的知识回答问题，不要简单的总结已知信息。
            如果问题是基本的社交性互动，可以不参考提供的已知信息。
            参考提供的已知信息回答问题时，要自然地过渡到分析和解释，以确保回答的完整性和准确性。
            根据提供的已知信息和你的简历，首先生成你对问题的思考过程。思考过程是一种指引性质的内容，可以让别人看到问题和已知信息时立马抓住它们的重点。
            思考过程生成后，紧接着根据已知信息和思考过程，准确、全面地给出回答。
            **限制**：不允许在答案中添加编造成分。
            你的思考及回答均是基于{{ person_name }}的口吻以及政治立场。要以“我”自称。
            只输出最终回答，不输出思考过程。请使用中文。
            **你的任务**：
            '<已知信息>{{ search_context }}</已知信息>\n'
            '<你的简历>{{ person_context }}</你的简历>\n'
            '<输出要求>回答要逻辑清晰，全面，准确。每段前添加4个空格，结构化输出。你的回答要表明对历史记录中信息的态度。</输出要求>\n'
            '<问题>你现在扮演{{ person_name }}，请以{{ person_name }}的风格和语气来回答问题。问题：{{ question }}</问题>\n'
            """
        )
        
        input_msg = History(role="user", content=all_prompt).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        task = asyncio.create_task(wrap_done(
            chain.acall({"person_name": person_name, "question": origin_query, "search_context": search_context, "person_context":person_context}),
            callback.done),
        )
            
        if stream:
            buffer = ""
            async for token in callback.aiter():
                buffer += token
                yield json.dumps({"text": token}, ensure_ascii=False)
            print(buffer)
        else:
            buffer = ""
            async for token in callback.aiter():
                buffer += token
            yield json.dumps({"text": buffer}, ensure_ascii=False)
            print(buffer)

        await task

        print(f"LLM回答，运行时间: {time.time() - start_time} 秒")

    return EventSourceResponse(discuss_with_role_iterator(origin_query, top_k, history,model_name,prompt_name))



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



def search_func(origin_query):
    #搜索模块
    url = "http://10.120.6.24:7778/search/api/search/v1/searchy3"
    # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # time = urllib.parse.quote(f"2024-01-01 10:00:00~{current_time}")
    # 定义查询参数
    params = {
        "q": origin_query,        # 搜索词
        "start_index": 0,     # 帖源数据起始序号
        "rn": 5,           # 请求帖源数据的数量
        # "advanced":"PublishTime:{}".format(time),
        "y2timeout":10000
    }
    # 发送请求
    try:
        response = requests.get(url, params=params)
        data = response.json()
        # 获取帖源结果
        search_results = data.get('data').get('search_results', [])
        #显示搜索结果
        source_documents = []
        set_source = []
        for inum, doc in enumerate(search_results):
            if doc['data_source_channel'] in ["EVENT", "POST", "NEWS"]:
                text = doc['content']
                if text not in set_source:
                    set_source.append(text)
                    # site_name = doc["site_name"]
                    # title_name = doc["title"]
                    # public_time = doc['date'].split(' ')[0]
                    # num_chunks = (len(text) + 99) // 100  # 向上取整
                    # for i in range(num_chunks):
                    #     chunk = text[i * 100: (i + 1) * 100]
                    #     source_documents.append(f"{public_time} \n\n {site_name} \n\n {title_name} \n\n {chunk}")
                    # source_documents.append(f"{public_time} \n\n {site_name} \n\n {title_name} \n\n {text}")
                    source_documents.append(f"{text}")
    except:
        source_documents = []
    return source_documents



async def extract_expert_info(name_list):
    result = []
    for name in name_list:
        try:
            name_info = await get_info_with_name_from_db(name)[0]
            result.append({
                "name": name_info.get("name"),
                "id": name_info.get("id"),
                "abstract": name_info.get("简介")
            })
        except:
            pass
    return result

am_name_list = ["乔·拜登","贝拉克·奥巴马","埃隆·马斯克","唐纳德·特朗普","南希·佩洛西","希拉里·克林顿"]
tw_name_list = ["赖清德","蔡英文","朱立伦","韩国瑜","管碧玲","王定宇","萧美琴","侯友宜","柯文哲","徐巧芯"]

async def list_am_roles():
    return await extract_expert_info(am_name_list)

async def list_tw_roles():
    return await extract_expert_info(tw_name_list)
