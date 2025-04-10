from fastapi import Body, Request
from sse_starlette.sse import EventSourceResponse
from fastapi.concurrency import run_in_threadpool
from configs import (LLM_MODELS, 
                     VECTOR_SEARCH_TOP_K, 
                     SCORE_THRESHOLD, 
                     TEMPERATURE,
                     USE_RERANKER,
                     RERANKER_MODEL,
                     RERANKER_MAX_LENGTH)
from server.utils import wrap_done, get_ChatOpenAI, get_model_path
from server.utils import BaseResponse, get_prompt_template
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio, json, re, hashlib, time, requests, urllib
from datetime import datetime
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
from server.utils import BaseResponse, ListResponse
from server.person_info_util import get_info_with_name_from_db, get_user_info
async def new_kb_chat_with_role(prompt: str = Body(..., description="官方发声 or 方案生成（已弃用）", examples=[""]),
                              restriction: str= Body(..., description="限制条件", examples=[""]),
                              search_results: list = Body(None, description="搜索结果", examples=[[]]),
                              person: str= Body(..., description="角色", examples=[""]),
                              origin_query: str= Body(..., description="用户输入", examples=["你好"]),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=[""]),
                              knowledge_base_name_list: list = Body(..., description="知识库名称", examples=[["书籍1", "书籍2"]]),
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
    if len(history) > 3:
        history = history[-3:]
    # print(history)
    print("knowledge_base_name: ", knowledge_base_name)
    print("knowledge_base_name_list1: ", knowledge_base_name_list)
    knowledge_base_name_list = [knowledge_base_name]
    print("knowledge_base_name_list2: ", knowledge_base_name_list)
    knowledge_base_name_id = knowledge_base_name

    print("restriction: ", restriction)
    print(history)

    top_k = 3
    temperature = 0.3
    score_threshold = 0.7
    print("top_k: ", top_k)
    print("temperature: ", temperature)
    print("score_threshold: ", score_threshold)

    async def knowledge_base_chat_iterator(
            origin_query: str,
            top_k: int,
            history: Optional[List[History]],
            model_name: str = model_name,
            prompt_name: str = prompt_name,
            person: str = person,
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

        start_time = time.time()
        yield json.dumps({"status_text": "问题理解中"})
        # 子问题和感兴趣的话题
        QUERY_PROMPT = """
        **任务**：你的具体任务是基于我给出的原始问题生成多个子问题，子问题通过对原始问题进行拆分，从而提高解决复杂问题的能力。根据这些问题，生成用户可能感兴趣的话题。
        **限制**：生成与原始问题有较高相关性的三个子问题，以及三个可能感兴趣的话题。
        **请严格按照以下格式输出**：
        <question>子问题1</question>
        <question>子问题2</question>
        <question>子问题3</question>
        <topic>感兴趣的话题1</topic>
        <topic>感兴趣的话题2</topic>
        <topic>感兴趣的话题3</topic>
        以下是我给出的原始问题：{原始问题}
        """
        query_prompt = QUERY_PROMPT.replace("{原始问题}", origin_query)
        outputs_from_first_chat = await chat_for_pipe(query_prompt, 
                                model_name=model_name,
                                stream=False, 
                                temperature= temperature, 
                                max_tokens=max_tokens, 
                                prompt_name=prompt_name)
        ts_outputs_from_first_chat = []
        async for chunk in outputs_from_first_chat:
            chunk_dict = json.loads(chunk)
            ts_outputs_from_first_chat.append(chunk_dict)
            first_message_id = chunk_dict['message_id']
        query0 = ""
        for t in ts_outputs_from_first_chat:
            query0 += t.get("text", "")
        query0 = query0.strip().split("\n")
        print(query0)
        query1 = list(filter(lambda x: x != '', query0))
        query2 = "\n".join([question.strip() for question in query1])
        pattern = r'<question>(.*?)</question>'
        query3 = re.findall(pattern, query2, re.MULTILINE)
        sub_question = "\n".join([question.strip() for question in query3])
        sub_question = origin_query + "\n" + sub_question
        yield json.dumps({"sub_query": sub_question})
        yield json.dumps({"status_text": "问题理解完成"})
        # yield json.dumps({"sub_query": sub_question, "message_id": first_message_id})

        pattern1 = r'<topic>(.*?)</topic>'
        query4 = re.findall(pattern1, query2, re.MULTILINE)
        interest_topic = "\n".join([q.strip() for q in query4])
        yield json.dumps({"interest_topic": interest_topic})

        query = sub_question

        query_new = []
        query_ori = query.strip().split("\n")
        for q in query_ori:
            if q != "":
                query_new.append(q)
        query_new = list(filter(lambda x: x != '', query_new))

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"生成子问题，运行时间: {execution_time} 秒")

        
        yield json.dumps({"status_text": "等待信息检索"})
        yield json.dumps({"status_text": "信息检索中"})

        start_time = time.time()
        # 角色简历
        if person == '' or knowledge_base_name_id != '':
            pattern = re.compile(r'([^\d_]+)')
            try:
                person_name = pattern.findall(knowledge_base_name_id)[0]
                print(person_name)
                person = await get_id_by_name(person_name)
                print(person)
            except:
                person = None            
        person_context = ""            
        person_name = await get_name_by_id(person)
        print(person_name)
        if person_name != None:
            data = await get_user_info(person)
            expert_resume = ',\n'.join([f'"{key}": "{value}"' for key, value in data.items()])
            print(expert_resume)
            person_context = expert_resume
        
        if USE_RERANKER:
            reranker_model_path = get_model_path(RERANKER_MODEL)
            reranker_model = LangchainReranker(top_n=len(query_new) * 10,
                                            device=embedding_device(),
                                            max_length=RERANKER_MAX_LENGTH,
                                            model_name_or_path=reranker_model_path)

        # 检索知识库
        source_documents = []
        docs_temp = []
        inum = 0
        start_time1 = time.time()
        for knowledge_base_name in knowledge_base_name_list:
            docs_temp = []
            for q in query_new:
                if q == origin_query:
                    top_k_ls = top_k*2
                else:
                    top_k_ls = top_k

                doc = await run_in_threadpool(search_docs,
                                            query=q,
                                            knowledge_base_name=knowledge_base_name,
                                            top_k=top_k_ls,
                                            score_threshold=score_threshold)
                docs_temp.append(doc)
            docs_temp = reciprocal_rank_fusion(docs_temp)
            
            for i, doc in enumerate(docs_temp):
                filename = doc.metadata.get("source")
                text = f"""{knowledge_base_name} \n\n {filename} \n\n{doc.page_content}\n\n"""
                source_documents.append(text)
            print("source_documents: ", source_documents)

        if USE_RERANKER:
            source_documents = reranker_model.compress_search_documents(documents=source_documents, query=origin_query)

        if "9g" in model_name.lower():
            if len(source_documents) > 3:
                source_documents = source_documents[:3]
        else:
            if len(source_documents) > 5:
                source_documents = source_documents[:5]
        reordering = LongContextReorder()
        reorder_docs = reordering.transform_documents(source_documents)
        context = ''
        for doc in reorder_docs:
            context += '\n\n'.join(doc.split('\n\n')[2:])
        print("source_documents_context: ", context)
        # context = "\n".join([doc for doc in reorder_docs])
        source_documents_output = []
        for i, text in enumerate(source_documents):
            parts = text.split('\n\n')
            docName = parts[0]
            title = parts[1]
            content = '\n\n'.join(parts[2:])
            source_documents_output.append({'publish_time': '', 'title': title, 'content': content, 'docName': docName})
        print("source_documents_output: ", source_documents_output)

        execution_time1 = time.time() - start_time1
        print(f"知识库检索，运行时间: {execution_time1} 秒")

        # ES检索
        search_documents = []
        start_time2 = time.time()
        # for q in query_new:
        #     search_documents += search_func(q)
        search_documents += search_func(origin_query)
        print("search_documents: ", search_documents)
        # if USE_RERANKER:
        #     search_documents = reranker_model.compress_search_documents(documents=search_documents, query=origin_query)
        # if len(search_documents) > 5:
        #     search_documents = search_documents[:5]
        reordering = LongContextReorder()
        reorder_docs = reordering.transform_documents(search_documents)
        search_context = ''
        for doc in reorder_docs:
            search_context += '\n\n'.join(doc.split('\n\n')[3:])
            if len(search_context) > 25000:
                break
        search_context = search_context[:28000]
        print("search_context: ", search_context)
        # search_context = "\n".join([doc for doc in reorder_docs])
        search_documents_output = []
        for i, text in enumerate(search_documents):
            parts = text.split('\n\n')
            publish_time = parts[0]
            docName = parts[1]
            title = parts[2]
            content = '\n\n'.join(parts[3:])
            search_documents_output.append({'publish_time': publish_time, 'title': title, 'content': content, 'docName': docName})
        print("search_documents_output: ", search_documents_output)

        execution_time2 = time.time() - start_time2
        print(f"ES检索，运行时间: {execution_time2} 秒")

        #流式输出参考文档
        yield json.dumps({"docs": source_documents_output + search_documents_output}, ensure_ascii=False)
        yield json.dumps({"status_text": "信息检索完成"})
        yield json.dumps({"status_text": "等待资源整合"})
        yield json.dumps({"status_text": "资源整合中"})

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"检索总运行时间: {execution_time} 秒")

        # # 3. 检索知识图谱
        # url = "http://10.106.1.4:8020/query"
        # payload = {
        #     "query": origin_query,
        #     "top_k": 3
        # }
        # original_response = requests.post(url, json=payload)
        # light_data = original_response.json()["data"]
        # print(light_data)

        start_time = time.time()
        if person_name != None:
            all_prompt = (
                """
                **任务**：参考已知信息中与问题相关的内容，并结合大模型本身的知识回答问题，不要简单的总结已知信息。
                如果问题是基本的社交性互动，可以不参考提供的已知信息。
                参考提供的已知信息回答问题时，要自然地过渡到分析和解释，以确保回答的完整性和准确性。
                根据提供的已知信息和你的简历，首先生成你对问题的思考过程。思考过程是一种指引性质的内容，可以让别人看到问题和已知信息时立马抓住它们的重点。
                思考过程生成后，紧接着根据已知信息和思考过程，准确、全面地给出回答。
                **限制**：不允许在答案中添加编造成分。
                你的思考及回答均是基于你的口吻以及政治立场。你应该模仿“你的已知信息”中你语录的口吻，或者引用其中你说过的话。要以“我”自称。
                只输出最终回答，不输出思考过程。请使用中文。
                **你的任务**：
                '<已知信息>{{ search_context }}\n{{ context }}</已知信息>\n'
                '<你的简历>{{ person_context }}</你的简历>\n'
                '<输出要求>回答要逻辑清晰，全面，准确。每段前添加4个空格，结构化输出。{{ restriction }}</输出要求>\n'
                '<问题>你现在扮演{{ person_name }}，请以{{ person_name }}的风格和语气来回答问题。问题：{{ question }}</问题>\n'
                """
            )
        else:
            all_prompt = (
                """
                **任务**：参考已知信息中与问题相关的内容，并结合大模型本身的知识回答问题，不要简单的总结已知信息。
                如果问题是基本的社交性互动，可以不参考提供的已知信息。
                参考提供的已知信息回答问题时，要自然地过渡到分析和解释，以确保回答的完整性和准确性。
                根据提供的已知信息，首先生成你对问题的思考过程。思考过程是一种指引性质的内容，可以让别人看到问题和已知信息时立马抓住它们的重点。
                思考过程生成后，紧接着根据已知信息和思考过程，准确、全面地给出回答。
                **限制**：不允许在答案中添加编造成分。只输出最终回答，不输出思考过程。请使用中文。站在中国共产党的立场上回答问题，不能出现“中共”等外媒报道时的字样，涉及到我方立场时，即中国大陆的立场，就以“我党”或“我方”等口吻回答问题。
                **输出格式**：你需要分析问题“{{ question }}”的类型，根据不同的问题类型的选择不同的输出格式。
                输出格式中的各级标题需要根据问题“{{ question }}”个性化生成，不要输出“引言”、“主体内容”、“总结”等字样，不输出（）中的内容。

                1.知识性问题（涉及事实、定义、理论等内容的问题）：
                （引言：简要介绍问题背景）
                （主体内容）
                    （###相关定义或理论）
                    （###例子或应用）
                （总结：概括主要观点）
                
                2.技术性问题（涉及具体技术、工具或方法的问题）：
                （引言：描述技术背景）
                （主体内容）
                    （###技术原理）
                    （###实施步骤或代码示例）
                    （###注意事项或常见问题）
                （总结：强调技术的优势和局限性）
                
                3.社会性问题（涉及社会现象、文化、心理等方面的问题）：
                （引言：介绍社会问题的背景和重要性）
                （主体内容）
                    （###问题分析：深入分析社会现象及其成因）
                    （###相关数据或案例：引用统计数据或案例研究支持论点）
                    （###各方观点：展示不同观点以提供全面视角）
                （总结：提出可能的解决方案或未来展望）
                
                4.时事新闻问题（涉及当前事件、新闻动态的问题）：
                （引言：概述事件背景及其重要性）
                （主体内容）
                    （###事件经过：详细描述事件的发展过程）
                    （###影响分析：分析不同利益相关者的反应及其影响）
                （总结：对事件未来发展的预测及可能后果）
                
                5.实用性问题（涉及日常生活、工作中的实用技巧或建议的问题）：
                （引言：说明实用问题的重要性及其普遍性）
                （主体内容）
                    （###具体建议或步骤：清晰列出可行的建议或操作步骤）
                    （###常见误区提醒：指出常见错误以帮助读者避免）
                （总结：鼓励读者实践建议并反馈效果）
                
                6.交互式对话/社交性问题（基于社交互动或自我介绍）：
                友好地回应问候。
                概述你的能力，例如“我可以回答问题、提供信息和帮助解决问题”，但是不要表明你是谁。
                鼓励进一步交流，例如“请问有什么我可以帮助你的吗？”

                **你的任务**：
                '<已知信息>{{ search_context }}\n{{ context }}</已知信息>\n'
                '<先验知识>
                问题和回答中出现的“我方”必须代表中国大陆，以“我党”或“我方”等口吻字样回答问题时要站在中国共产党的立场上。
                </先验知识>\n'
                '<输出要求>
                回答要逻辑清晰，全面，准确。回答中不要出现某个事件的具体时间，不要输出“引言”、“主体内容”、“总结”等字样。结构化输出。
                要分析已知信息中的内容是代表大陆的立场还是台湾的立场，如何信息无法识别立场，则不在回答中使用。{{ restriction }}
                </输出要求>\n'
                '<问题>{{ question }}</问题>\n'
                """
            )
        input_msg = History(role="user", content=all_prompt).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        if "反对管碧玲群体" in knowledge_base_name and "管碧玲" in origin_query:
            origin_query += "，以管碧玲是个邀功、渎职、虚伪的人为前提"
        
        if person_name != None:
            task = asyncio.create_task(wrap_done(
                chain.acall({"restriction": restriction, "person_name": person_name, "question": origin_query, "search_context": search_context, "context": context, "person_context":person_context}),
                callback.done),
            )
        else:
            task = asyncio.create_task(wrap_done(
                chain.acall({"restriction": restriction, "question": origin_query, "search_context": search_context, "context": context}),
                callback.done),
            )
            
        if stream:
            buffer = ""
            async for token in callback.aiter():
                buffer += token
                yield json.dumps({"text": token}, ensure_ascii=False)
        else:
            buffer = ""
            async for token in callback.aiter():
                buffer += token
            yield json.dumps({"text": buffer}, ensure_ascii=False)

        await task
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"LLM回答，运行时间: {execution_time} 秒")

    return EventSourceResponse(knowledge_base_chat_iterator(origin_query, top_k, history,model_name,prompt_name,person))



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
    # url = "http://10.106.51.159:19999/search-service/search/v1/searchy3"
    url = "http://10.120.6.24:7778/search/api/search/v1/searchy3"
    # url = "http://10.108.1.5:7778/search/api/search/v1/searchy3"
    # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # time = urllib.parse.quote(f"2024-01-01 10:00:00~{current_time}")
    # time = f"2024-10-01 10:00:00~2024-11-17 10:00:00"
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
                    site_name = doc["site_name"]
                    title_name = doc["title"]
                    public_time = doc['date'].split(' ')[0]
                    # num_chunks = (len(text) + 499) // 500  # 向上取整
                    # for i in range(num_chunks):
                    #     chunk = text[i * 500: (i + 1) * 500]
                    #     source_documents.append(f"{public_time} \n\n {site_name} \n\n {title_name} \n\n {chunk}")
                    source_documents.append(f"{public_time} \n\n {site_name} \n\n {title_name} \n\n {text}")
    except:
        source_documents = []
    return source_documents



Role_List = [
                {
                    "name": "赖清德",
                    "id": "Q3847080"
                },
                {
                    "name": "乔·拜登",
                    "id": "Q6279"
                },
                {
                    "name": "蔡英文",
                    "id": "Q233984"
                },
                {
                    "name": "布林肯",
                    "id": "Q7821917"
                },
                {
                    "name": "巴拉克·奥巴马",
                    "id": "Q76"
                },
                {
                    "name": "伊隆·马斯克",
                    "id": "Q317521"
                },
                {
                    "name": "特朗普",
                    "id": "Q22686"
                }
            ]

def list_roles():
    return Role_List

async def get_id_by_name(name):
    try:
        name_info = await get_info_with_name_from_db(name)
        return name_info[0].get("id")
    except:
        return None

async def get_name_by_id(expert_id):
    if expert_id == "1" or expert_id == None:
        return None
    try:
        user_info = await get_user_info(expert_id)
        return user_info.get("name")
    except:
        return None
