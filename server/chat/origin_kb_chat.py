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
import asyncio, json, re, hashlib, time
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
async def origin_kb_chat(origin_query: str= Body(..., description="用户输入", examples=["你好"]),
                              query: str = Body(..., description="用户输入", examples=["你好"]),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
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
    # kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    # if kb is None:
    #     return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator(
            query: str,
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

        #提取问题实体部分
        non_entity_corpus = [
            "你怎么看", "你觉得呢", "你如何评价", "你有什么想法", "这是真的吗", 
            "是这样吗", "你认为呢", "怎么理解", "是不是", "对吗", "怎么",
            "怎么说", "如何看待", "你同意吗", "能否解释一下", "你有何意见",
            "请问", "你能不能告诉我", "请教一下", "能解释一下吗", "能说明一下吗",
            "对吧", "是的吧", "你确定吗", "发生了什么", "为什么会这样", "具体是怎么回事",
            "能再详细一点吗", "你能简单讲一下吗", "能再说一遍吗", "你能帮我理解吗",
            "我想问一下", "我有个问题", "能打扰一下吗", "能再讲清楚些吗",
            "你怎么看待", "你觉得可行吗", "你怎么看这件事", "你怎么看这个问题",
            "对这件事你怎么想", "对这个问题你怎么看", "你有何看法", "你能分享一下看法吗",
            "能详细说明吗", "发生了什么情况", "结果会怎样", "你能说明一下吗",
            "请问是什么原因", "你有更多信息吗", "能给我多一点信息吗", "你能描述一下吗",
            "你能再讲一下吗", "能再简单说一下吗", "能稍微详细解释一下吗", "你能更清楚地解释吗",
            "你能举个例子吗", "你有更多看法吗", "能详细解释吗", "能麻烦你再解释一遍吗",
            "你能再说明一次吗", "你能提供更多细节吗", "你觉得这个怎么样", "能不能帮个忙",
            "再说一下", "能不能说详细点", "讲清楚", "解释一下", 
            "可以再解释一次吗", "可以帮我理解一下吗","你怎么想"
        ]
        # 遍历语料库，看是否匹配
        def filter_non_entity_parts(input_text, corpus):
            for phrase in corpus:
                if phrase in input_text:
                    # 删除匹配到的非实体部分
                    input_text = input_text.replace(phrase, "")
            return input_text.strip()
        #/提取问题实体部分

        query_new = []
        query_ori = query.strip().split("\n")
        for q in query_ori:
            if q != "":
                query_new.append(q)
        query_new = list(filter(lambda x: x != '', query_new))

        docs_temp = []

        start_time = time.time()

        for q in query_new:
            if q == origin_query:
                top_k_ls = top_k*2
            else:
                top_k_ls = top_k

            #提取问题实体部分
            filtered_input = filter_non_entity_parts(q, non_entity_corpus)
            #/提取问题实体部分

            doc = await run_in_threadpool(search_docs,
                                        query=filtered_input,
                                        knowledge_base_name=knowledge_base_name,
                                        top_k=top_k_ls,
                                        score_threshold=score_threshold)
            docs_temp.append(doc)


        docs = reciprocal_rank_fusion(docs_temp)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"检索知识库，运行时间: {execution_time} 秒")

        if USE_RERANKER:
            reranker_model_path = get_model_path(RERANKER_MODEL)
            reranker_model = LangchainReranker(top_n=top_k*(len(query_new)+1)*2,
                                            device=embedding_device(),
                                            max_length=RERANKER_MAX_LENGTH,
                                            model_name_or_path=reranker_model_path)
            
            #提取问题实体部分
            filtered_input = filter_non_entity_parts(origin_query, non_entity_corpus)
            #/提取问题实体部分

            person_kb_docs = reranker_model.compress_documents(documents=person_kb_docs, query=filtered_input)
            docs = reranker_model.compress_documents(documents=docs, query=filtered_input)
        
        if len(docs) > 20:
            docs = docs[:20]
        reordering = LongContextReorder()
        reorder_docs = reordering.transform_documents(docs)
        context = "\n".join([doc.page_content for doc in reorder_docs])
        
        
        source_documents = []
        for inum, doc in enumerate(docs):
            filename = doc.metadata.get("source")
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
            base_url = request.base_url
            url = f"{base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0:
            source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")


        start_time = time.time()
        all_prompt = (
                """
                **任务**：参考已知信息中与问题相关的内容，并结合大模型本身的知识回答问题，不要简单的总结已知信息。
                如果问题是基本的社交性互动，可以不参考提供的已知信息。
                参考提供的已知信息回答问题时，要自然地过渡到分析和解释，以确保回答的完整性和准确性。
                根据提供的已知信息，首先生成你对问题的思考过程。思考过程是一种指引性质的内容，可以让别人看到问题和已知信息时立马抓住它们的重点。
                思考过程生成后，紧接着根据已知信息和思考过程，准确、全面地给出回答。
                **限制**：不允许在答案中添加编造成分。只输出最终回答，不输出思考过程。请使用中文。
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
                '<已知信息>{{ context }}\n{{ sou_context }}</已知信息>\n'
                '<输出要求>
                回答要逻辑清晰，全面，准确。回答中不要出现某个事件的具体时间，不要输出“引言”、“主体内容”、“总结”等字样。结构化输出。
                </输出要求>\n'
                '<问题>{{ question }}</问题>\n'
                """
            )
        input_msg = History(role="user", content=all_prompt).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)
        
        task = asyncio.create_task(wrap_done(
            chain.acall({"question": origin_query, "context": context, "sou_context": prompt_name}),
            callback.done),
        )
        
        if stream:
            buffer = ""
            async for token in callback.aiter():
                buffer += token
                yield json.dumps({"text": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            buffer = ""
            async for token in callback.aiter():
                buffer += token
            yield json.dumps({"text": buffer}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
            

        await task
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"LLM回答，运行时间: {execution_time} 秒")

    return EventSourceResponse(knowledge_base_chat_iterator(query, top_k, history,model_name,prompt_name))



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
