from fastapi import Body, Request
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
import asyncio, json, re, hashlib
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
async def knowledge_base_chat(origin_query: str= Body(..., description="用户输入", examples=["你好"]),
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
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator(
            query: str,
            top_k: int,
            history: Optional[List[History]],
            model_name: str = model_name,
            prompt_name: str = prompt_name,
    ) -> AsyncIterable[str]:
        nonlocal max_tokens
        callback1 = AsyncIteratorCallbackHandler()
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model1 = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback1],
        )
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )
        docs_temp = []
        query = query.strip().split("\n")
        query = list(filter(lambda x: x != '', query))
        # query = query[-3:]
        for q in query:
            # print(q)
            if q == origin_query:
                top_k_ls = top_k*2
            else:
                top_k_ls = top_k
            doc = await run_in_threadpool(search_docs,
                                       query=q,
                                       knowledge_base_name=knowledge_base_name,
                                       top_k=top_k_ls,
                                       score_threshold=score_threshold)
            # print(doc)
            docs_temp.append(doc)
        # docs = [item for sublist in docs_temp for item in sublist]
        # print(docs)
        docs = reciprocal_rank_fusion(docs_temp)    #[:top_k]
        # 加入reranker
        if USE_RERANKER:
            reranker_model_path = get_model_path(RERANKER_MODEL)
            reranker_model = LangchainReranker(top_n=top_k,
                                            device=embedding_device(),
                                            max_length=RERANKER_MAX_LENGTH,
                                            model_name_or_path=reranker_model_path
                                            )
            # print("-------------before rerank-----------------")
            # print(docs)
            docs = reranker_model.compress_documents(documents=docs,
                                                     query=origin_query)
            # print("------------after rerank------------------")
            # print(docs)
        reordering = LongContextReorder()
        reorder_docs = reordering.transform_documents(docs)
        context = "\n".join([doc.page_content for doc in reorder_docs])
#生成对文档及内容的思考
        if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
            prompt_template1 = get_prompt_template("knowledge_base_chat", "empty")
        else:
            prompt_template1 = get_prompt_template("knowledge_base_chat", "thinking_prompt")
        input_msg1 = History(role="user", content=prompt_template1).to_msg_template(False)
        chat_prompt1 = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg1])

        chain1 = LLMChain(prompt=chat_prompt1, llm=model1)

        # Begin a task that runs in the background.
        task1 = asyncio.create_task(wrap_done(
            chain1.acall({"context": context, "question": origin_query}),
            callback1.done),
        )
        
        thinking = ""
        async for token in callback1.aiter():
            thinking += token
        print("############思考过程###################:\n",thinking)
#添加思考的回答
        if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
            prompt_template2 = get_prompt_template("knowledge_base_chat", "empty")
        else:
            prompt_template2 = get_prompt_template("knowledge_base_chat", "output_with_thinking_prompt")
        input_msg2 = History(role="user", content=prompt_template2).to_msg_template(False)
        chat_prompt2 = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg2])

        chain2 = LLMChain(prompt=chat_prompt2, llm=model)
        
        task = asyncio.create_task(wrap_done(
            chain2.acall({"context": context, "thinking": thinking, "question": origin_query}),
            callback.done),
        )
        
        source_documents = []
        for inum, doc in enumerate(docs):
            filename = doc.metadata.get("source")
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
            base_url = request.base_url
            url = f"{base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0:  # 没有找到相关文档
            source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")
        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response 
                yield json.dumps({"answer": token, "thinking": thinking, "context": context}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents, "thinking": thinking, "context": context}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer, 
                              "thinking": thinking, 
                              "context": context,
                              "docs": source_documents},
                             ensure_ascii=False)
        await task
        await task1

    return EventSourceResponse(knowledge_base_chat_iterator(query, top_k, history,model_name,prompt_name))

async def knowledge_base_chat_for_pipe(origin_query: str= Body(..., description="用户输入", examples=["你好"]),
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
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        # return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
        raise ModuleNotFoundError

    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator(
            query: str,
            top_k: int,
            history: Optional[List[History]],
            model_name: str = model_name,
            prompt_name: str = prompt_name,
    ) -> AsyncIterable[str]:
        nonlocal max_tokens
        callback1 = AsyncIteratorCallbackHandler()
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model1 = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback1],
        )
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )
        docs_temp = []
        # query = re.sub("\n+", "\n", query) # avoid multi \n
        query_new = []
        query_ori = query.strip().split("\n")
        for q in query_ori:
            if q != "":
                query_new.append(q)
        query_new = list(filter(lambda x: x != '', query_new))
        # query = query[-3:]
        for q in query_new:
            # print(q)
            if q == origin_query:
                top_k_ls = top_k*2
            else:
                top_k_ls = top_k
            doc = await run_in_threadpool(search_docs,
                                       query=q,
                                       knowledge_base_name=knowledge_base_name,
                                       top_k=top_k_ls,
                                       score_threshold=score_threshold)
            # print(doc)
            docs_temp.append(doc)
        # docs = [item for sublist in docs_temp for item in sublist]
        # print(docs)
        docs = reciprocal_rank_fusion(docs_temp)    #[:top_k]
        # 加入reranker
        if USE_RERANKER:
            reranker_model_path = get_model_path(RERANKER_MODEL)
            reranker_model = LangchainReranker(top_n=len(docs),
                                            device=embedding_device(),
                                            max_length=RERANKER_MAX_LENGTH,
                                            model_name_or_path=reranker_model_path
                                            )
            logger.info("-------------before rerank-----------------")
            logger.info(docs)
            docs = reranker_model.compress_documents(documents=docs,
                                                     query=origin_query)
            logger.info("------------after rerank------------------")
            logger.info(docs)
        if "9g" in model_name.lower():
            if len(docs) > 3:
                docs = docs[:3]
        else:
            if len(docs) > 20:
                docs = docs[:20]
        reordering = LongContextReorder()
        reorder_docs = reordering.transform_documents(docs)
        context = "\n".join([doc.page_content for doc in reorder_docs])
        #生成对文档及内容的思考
        if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
            prompt_template1 = get_prompt_template("knowledge_base_chat", "empty")
        else:
            prompt_template1 = get_prompt_template("knowledge_base_chat", "thinking_prompt")
        input_msg1 = History(role="user", content=prompt_template1).to_msg_template(False)
        chat_prompt1 = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg1])

        chain1 = LLMChain(prompt=chat_prompt1, llm=model1)

        # Begin a task that runs in the background.
        task1 = asyncio.create_task(wrap_done(
            chain1.acall({"context": context, "question": origin_query}),
            callback1.done),
        )
        
        thinking = ""
        async for token in callback1.aiter():
            thinking += token
        logger.info("############思考过程###################:\n" + thinking)
#添加思考的回答
        if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
            prompt_template2 = get_prompt_template("knowledge_base_chat", "empty")
        else:
            prompt_template2 = get_prompt_template("knowledge_base_chat", "output_with_thinking_prompt")
        input_msg2 = History(role="user", content=prompt_template2).to_msg_template(False)
        chat_prompt2 = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg2])

        chain2 = LLMChain(prompt=chat_prompt2, llm=model)
        
        task = asyncio.create_task(wrap_done(
            chain2.acall({"context": context, "thinking": thinking, "question": origin_query}),
            callback.done),
        )
        
        source_documents = []
        for inum, doc in enumerate(docs):
            filename = doc.metadata.get("source")
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
            base_url = request.base_url
            url = f"{base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0:  # 没有找到相关文档
            source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")
        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response 
                yield json.dumps({"answer": token, "thinking": thinking, "context": context}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents, "thinking": thinking, "context": context}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer, 
                              "thinking": thinking, 
                              "context": context,
                              "docs": source_documents},
                             ensure_ascii=False)
        await task
        await task1

    return knowledge_base_chat_iterator(query, top_k, history,model_name,prompt_name)
# def reciprocal_rank_fusion(search_results, k=60):
#     fused_scores = {}
#     for doc in search_results:
#         doc_id = doc.id
#         fused_scores[doc_id] = 0
    
#     # Calculate fused score based on reciprocal rank fusion
#     for query_rank, doc in enumerate(sorted(search_results, key=lambda x: x.score, reverse=True)):
#         doc_id = doc.id
#         fused_scores[doc_id] += 1 / (query_rank + k)
#         # print(f"Updating score for {doc_id} to {fused_scores[doc_id]} based on rank {query_rank + 1}")

#     # Sort fused_scores by score in descending order
#     sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
#     # Reconstruct the results in the original format
#     reranked_results = [next(doc for doc in search_results if doc.id == doc_id) for doc_id, score in sorted_results]
#     # print("Final reranked results:", reranked_results)
#     return reranked_results

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