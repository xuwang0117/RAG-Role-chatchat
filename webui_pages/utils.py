# 该文件封装了对api.py的请求，可以被不同的webui使用
# 通过ApiRequest和AsyncApiRequest支持同步/异步调用
from fastapi import Request
from typing import *
from pathlib import Path
# 此处导入的配置为发起请求（如WEBUI）机器上的配置，主要用于为前端设置默认值。分布式部署时可以与服务器上的不同
from configs import (
    EMBEDDING_MODEL,
    DEFAULT_VS_TYPE,
    LLM_MODELS,
    TEMPERATURE,
    SCORE_THRESHOLD,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE,
    VECTOR_SEARCH_TOP_K,
    SEARCH_ENGINE_TOP_K,
    HTTPX_DEFAULT_TIMEOUT,
    logger, log_verbose,
)
import httpx
import contextlib
import json
import os
from io import BytesIO
from configs.model_config import RERANKER_MAX_LENGTH, RERANKER_MODEL, USE_RERANKER
from server.reranker.reranker import LangchainReranker
from server.utils import embedding_device, get_model_path, set_httpx_config, api_address, get_httpx_client
from langchain_community.document_transformers import (
    LongContextReorder,
)
from pprint import pprint
from langchain_core._api import deprecated

set_httpx_config()
import requests
from bs4 import BeautifulSoup
import concurrent.futures

def cut_and_rerank(text,origin_query,n):
    """对text进行切块+rerank"""
    
    source_documents = []
    num_chunks = (len(text) + 499) // 500  # 向上取整
    for i in range(num_chunks):
        chunk = text[i * 500: (i + 1) * 500]
        source_documents.append(chunk)
    if USE_RERANKER:
        #加载rerank模型
        reranker_model_path = get_model_path(RERANKER_MODEL)
        reranker_model = LangchainReranker(top_n=num_chunks,
                                        device=embedding_device(),
                                        max_length=RERANKER_MAX_LENGTH,
                                        model_name_or_path=reranker_model_path)

        search_documents = reranker_model.compress_search_documents(documents=source_documents, query=origin_query)
        if len(search_documents) > n:
            search_documents = search_documents[:n]
        reordering = LongContextReorder()
        reorder_docs = reordering.transform_documents(search_documents)
        search_context = '\n'.join(reorder_docs)
    else:
        search_context = '\n'.join(source_documents)
        
    return search_context

def serper_search(query):
    """使用serper进行搜索，return list,str"""
    
    helper = WebPageHelper(min_char_count=150, max_thread_num=5)
    url = "https://google.serper.dev/search"

    payload = json.dumps({
    "q": query,
    "hl": "zh-cn"
    })
    headers = {
    'X-API-KEY': '9f0fe5ee8aa4e3ee6da970f1d95cc5b6c389521e',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    # api_key = "9f0fe5ee8aa4e3ee6da970f1d95cc5b6c389521e"
    # # 构建请求的 URL
    # url = f"https://google.serper.dev/search?q={query}&api_key={api_key}"
    # proxies = {
    #     "http": "http://172.16.0.5:7890",
    #     "https": "https://172.16.0.5:7890",
    # }
    # response = requests.get(url,proxies)
    source_documents = []
    search_context_list = []
    urls = []
    try:
        search_json = response.json()
        for inum, doc in enumerate(search_json.get("organic",[])):
            filename = doc.get("title","...")
            url = doc.get("link","...")
            urls.append(url)
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.get("snippet","")}\n\n"""
            source_documents.append(text)
            
        articles = helper.urls_to_articles(urls)
        for url in urls:
            search_context_list.append(articles.get(url).get("text",""))
        search_context = "\n".join([text for text in search_context_list])
    except requests.exceptions.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        search_context = ""
    
    return source_documents, search_context

class WebPageHelper:
    """获取serper搜索后网页内的具体内容"""

    def __init__(self, min_char_count: int = 150, max_thread_num: int = 10):
        self.min_char_count = min_char_count
        self.max_thread_num = max_thread_num

    def download_webpage(self, url: str):
        """Download the webpage content."""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code >= 400:
                print(f"Error: {url} returned status code {response.status_code}")
                return None
            return response.content  # Use .content instead of .text to get raw bytes
        except requests.exceptions.RequestException as e:
            print(f"Error while requesting {url}: {e}")
            return None

    def extract_text_from_html(self, html: bytes) -> str:
        """Extract readable text from HTML using BeautifulSoup."""
        soup = BeautifulSoup(html, 'html.parser', from_encoding='utf-8')
        text = soup.get_text(separator="\n", strip=True)
        return text


    def urls_to_articles(self, urls: list[str]) -> dict:
        """Download and extract articles from a list of URLs."""
        articles = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
            htmls = list(executor.map(self.download_webpage, urls))

        for url, html in zip(urls, htmls):
            if html is None:
                continue
            article_text = self.extract_text_from_html(html)  # Pass raw bytes to the extractor
            if len(article_text) > self.min_char_count:
                articles[url] = {"text": article_text}

        return articles

class ApiRequest:
    '''
    api.py调用的封装（同步模式）,简化api调用方式
    '''

    def __init__(
            self,
            base_url: str = api_address(),
            timeout: float = HTTPX_DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self._use_async = False
        self._client = None

    @property
    def client(self):
        if self._client is None or self._client.is_closed:
            self._client = get_httpx_client(base_url=self.base_url,
                                            use_async=self._use_async,
                                            timeout=self.timeout)
        return self._client

    def get(
            self,
            url: str,
            params: Union[Dict, List[Tuple], bytes] = None,
            retry: int = 3,
            stream: bool = False,
            **kwargs: Any,
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                if stream:
                    return self.client.stream("GET", url, params=params, **kwargs)
                else:
                    return self.client.get(url, params=params, **kwargs)
            except Exception as e:
                msg = f"error when get {url}: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                retry -= 1

    def post(
            self,
            url: str,
            data: Dict = None,
            json: Dict = None,
            retry: int = 3,
            stream: bool = False,
            **kwargs: Any
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                # print(kwargs)
                if stream:
                    return self.client.stream("POST", url, data=data, json=json, **kwargs)
                else:
                    return self.client.post(url, data=data, json=json, **kwargs)
            except Exception as e:
                msg = f"error when post {url}: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                retry -= 1

    def delete(
            self,
            url: str,
            data: Dict = None,
            json: Dict = None,
            retry: int = 3,
            stream: bool = False,
            **kwargs: Any
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                if stream:
                    return self.client.stream("DELETE", url, data=data, json=json, **kwargs)
                else:
                    return self.client.delete(url, data=data, json=json, **kwargs)
            except Exception as e:
                msg = f"error when delete {url}: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                retry -= 1

    def _httpx_stream2generator(
            self,
            response: contextlib._GeneratorContextManager,
            as_json: bool = False,
    ):
        '''
        将httpx.stream返回的GeneratorContextManager转化为普通生成器
        '''

        async def ret_async(response, as_json):
            try:
                async with response as r:
                    async for chunk in r.aiter_text(None):
                        if not chunk:  # fastchat api yield empty bytes on start and end
                            continue
                        if as_json:
                            try:
                                if chunk.startswith("data: "):
                                    data = json.loads(chunk[6:-2])
                                elif chunk.startswith(":"):  # skip sse comment line
                                    continue
                                else:
                                    data = json.loads(chunk)
                                yield data
                            except Exception as e:
                                msg = f"接口返回json错误： ‘{chunk}’。错误信息是：{e}。"
                                logger.error(f'{e.__class__.__name__}: {msg}',
                                             exc_info=e if log_verbose else None)
                        else:
                            # print(chunk, end="", flush=True)
                            yield chunk
            except httpx.ConnectError as e:
                msg = f"无法连接API服务器，请确认 ‘api.py’ 已正常启动。({e})"
                logger.error(msg)
                yield {"code": 500, "msg": msg}
            except httpx.ReadTimeout as e:
                msg = f"API通信超时，请确认已启动FastChat与API服务（详见Wiki '5. 启动 API 服务或 Web UI'）。（{e}）"
                logger.error(msg)
                yield {"code": 500, "msg": msg}
            except Exception as e:
                msg = f"API通信遇到错误：{e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                yield {"code": 500, "msg": msg}

        def ret_sync(response, as_json):
            try:
                with response as r:
                    for chunk in r.iter_text(None):
                        if not chunk:  # fastchat api yield empty bytes on start and end
                            continue
                        if as_json:
                            try:
                                if chunk.startswith("data: "):
                                    data = json.loads(chunk[6:-2])
                                elif chunk.startswith(":"):  # skip sse comment line
                                    continue
                                else:
                                    data = json.loads(chunk)
                                yield data
                            except Exception as e:
                                msg = f"接口返回json错误： ‘{chunk}’。错误信息是：{e}。"
                                logger.error(f'{e.__class__.__name__}: {msg}',
                                             exc_info=e if log_verbose else None)
                        else:
                            # print(chunk, end="", flush=True)
                            yield chunk
            except httpx.ConnectError as e:
                msg = f"无法连接API服务器，请确认 ‘api.py’ 已正常启动。({e})"
                logger.error(msg)
                yield {"code": 500, "msg": msg}
            except httpx.ReadTimeout as e:
                msg = f"API通信超时，请确认已启动FastChat与API服务（详见Wiki '5. 启动 API 服务或 Web UI'）。（{e}）"
                logger.error(msg)
                yield {"code": 500, "msg": msg}
            except Exception as e:
                msg = f"API通信遇到错误：{e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                yield {"code": 500, "msg": msg}

        if self._use_async:
            return ret_async(response, as_json)
        else:
            return ret_sync(response, as_json)

    def _get_response_value(
            self,
            response: httpx.Response,
            as_json: bool = False,
            value_func: Callable = None,
    ):
        '''
        转换同步或异步请求返回的响应
        `as_json`: 返回json
        `value_func`: 用户可以自定义返回值，该函数接受response或json
        '''

        def to_json(r):
            try:
                return r.json()
            except Exception as e:
                msg = "API未能返回正确的JSON。" + str(e)
                if log_verbose:
                    logger.error(f'{e.__class__.__name__}: {msg}',
                                 exc_info=e if log_verbose else None)
                return {"code": 500, "msg": msg, "data": None}

        if value_func is None:
            value_func = (lambda r: r)

        async def ret_async(response):
            if as_json:
                return value_func(to_json(await response))
            else:
                return value_func(await response)

        if self._use_async:
            return ret_async(response)
        else:
            if as_json:
                return value_func(to_json(response))
            else:
                return value_func(response)

    # 服务器信息
    def get_server_configs(self, **kwargs) -> Dict:
        response = self.post("/server/configs", **kwargs)
        return self._get_response_value(response, as_json=True)

    def list_search_engines(self, **kwargs) -> List:
        response = self.post("/server/list_search_engines", **kwargs)
        return self._get_response_value(response, as_json=True, value_func=lambda r: r["data"])

    def get_prompt_template(
            self,
            type: str = "llm_chat",
            name: str = "default",
            **kwargs,
    ) -> str:
        data = {
            "type": type,
            "name": name,
        }
        response = self.post("/server/get_prompt_template", json=data, **kwargs)
        return self._get_response_value(response, value_func=lambda r: r.text)

    # 对话相关操作
    def chat_chat(
            self,
            query: str,
            conversation_id: str = None,
            history_len: int = -1,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
            **kwargs,
    ):
        '''
        对应api.py/chat/chat接口
        '''
        data = {
            "query": query,
            "conversation_id": conversation_id,
            "history_len": history_len,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        # print(f"received input message:")
        # pprint(data)

        response = self.post("/chat/chat", json=data, stream=True, **kwargs)
        return self._httpx_stream2generator(response, as_json=True)

    @deprecated(
        since="0.3.0",
        message="自定义Agent问答将于 Langchain-Chatchat 0.3.x重写, 0.2.x中相关功能将废弃",
        removal="0.3.0")
    def agent_chat(
            self,
            query: str,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
    ):
        '''
        对应api.py/chat/agent_chat 接口
        '''
        data = {
            "query": query,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        # print(f"received input message:")
        # pprint(data)

        response = self.post("/chat/agent_chat", json=data, stream=True)
        return self._httpx_stream2generator(response, as_json=True)
# #brain_storm接口
#     def generate_outline(
#             self,
#             topic: str,
#             topic_description: str,
#             model_name: str = LLM_MODELS[0],
#     ):
#         '''
#         对应api.py/brain_storm/generate_outline接口
#         '''
#         data = {
#             "topic": topic,
#             "topic_description": topic_description,
#             "model_name": model_name,
#         }


#         response = self.post(
#             "/brain_storm/generate_outline",
#             json=data,
#         )
#         return self._httpx_stream2generator(response, as_json=True)
    
    
    def knowledge_base_chat(
            self,
            origin_query: str,
            query: str,
            knowledge_base_name: str,
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: float = SCORE_THRESHOLD,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
    ):
        '''
        对应api.py/chat/knowledge_base_chat接口
        '''
        data = {
            "origin_query": origin_query,
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        # print(f"received input message:")
        # pprint(data)

        response = self.post(
            "/chat/knowledge_base_chat",
            json=data,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True)

    def jsby_kb_chat(
            self,
            persona_prompt: str,
            origin_query: str,
            query: str,
            knowledge_base_name: str,
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: float = SCORE_THRESHOLD,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
    ):
        '''
        对应api.py/chat/jsby_kb_chat接口
        '''
        data = {
            "persona_prompt": persona_prompt,
            "origin_query": origin_query,
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        # print(f"received input message:")
        # pprint(data)

        response = self.post(
            "/chat/jsby_kb_chat",
            json=data,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True)
    
    def origin_kb_chat(
            self,
            origin_query: str,
            query: str,
            knowledge_base_name: str,
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: float = SCORE_THRESHOLD,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
    ):
        '''
        对应api.py/chat/origin_kb_chat接口
        '''
        data = {
            "origin_query": origin_query,
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        # print(f"received input message:")
        # pprint(data)

        response = self.post(
            "/chat/origin_kb_chat",
            json=data,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True)

    def pipeline_knowledge_chat(
            self,
            prompt: str,
            conversation_id: str,
            knowledge_base_name: str,
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: float = SCORE_THRESHOLD,
            history_len: int = -1,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
            request: Request = None,
            **kwargs,
    ):
        '''
        对应 api.py/chat/pipeline_knowledge_chat 接口
        '''
        data = {
            "prompt": prompt,
            "conversation_id": conversation_id,
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "history_len": history_len,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
            "request": request,
        }

        # print(f"received input message:")
        # print(data)

        response = self.post(
            "/chat/pipeline_knowledge_chat",
            json=data,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True, **kwargs)
    
    def personas_select(
            self,
            query: str,
            count: int,
            conversation_id: str,
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
            **kwargs,
    ):
        '''
        对应 api.py/chat/personas_select 接口
        '''
        data = {
            "prompt": query,
            "count": count,
            "conversation_id": conversation_id,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        response = self.post(
            "/chat/personas_select",
            json=data,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True, **kwargs)

    def upload_temp_docs(
            self,
            files: List[Union[str, Path, bytes]],
            knowledge_id: str = None,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
    ):
        '''
        对应api.py/knowledge_base/upload_tmep_docs接口
        '''

        def convert_file(file, filename=None):
            if isinstance(file, bytes):  # raw bytes
                file = BytesIO(file)
            elif hasattr(file, "read"):  # a file io like object
                filename = filename or file.name
            else:  # a local path
                file = Path(file).absolute().open("rb")
                filename = filename or os.path.split(file.name)[-1]
            return filename, file

        files = [convert_file(file) for file in files]
        data = {
            "knowledge_id": knowledge_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
        }

        response = self.post(
            "/knowledge_base/upload_temp_docs",
            data=data,
            files=[("files", (filename, file)) for filename, file in files],
        )
        return self._get_response_value(response, as_json=True)

    def file_chat(
            self,
            query: str,
            knowledge_id: str,
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: float = SCORE_THRESHOLD,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
    ):
        '''
        对应api.py/chat/file_chat接口
        '''
        data = {
            "query": query,
            "knowledge_id": knowledge_id,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        response = self.post(
            "/chat/file_chat",
            json=data,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True)

    @deprecated(
        since="0.3.0",
        message="搜索引擎问答将于 Langchain-Chatchat 0.3.x重写, 0.2.x中相关功能将废弃",
        removal="0.3.0"
    )
    def search_engine_chat(
            self,
            query: str,
            search_engine_name: str,
            top_k: int = SEARCH_ENGINE_TOP_K,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
            split_result: bool = False,
    ):
        '''
        对应api.py/chat/search_engine_chat接口
        '''
        data = {
            "query": query,
            "search_engine_name": search_engine_name,
            "top_k": top_k,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
            "split_result": split_result,
        }

        # print(f"received input message:")
        # pprint(data)

        response = self.post(
            "/chat/search_engine_chat",
            json=data,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True)

    # 知识库相关操作

    def list_knowledge_bases(
            self,
    ):
        '''
        对应api.py/knowledge_base/list_knowledge_bases接口
        '''
        response = self.get("/knowledge_base/list_knowledge_bases")
        return self._get_response_value(response,
                                        as_json=True,
                                        value_func=lambda r: r.get("data", []))

    def create_knowledge_base(
            self,
            knowledge_base_name: str,
            vector_store_type: str = DEFAULT_VS_TYPE,
            embed_model: str = EMBEDDING_MODEL,
    ):
        '''
        对应api.py/knowledge_base/create_knowledge_base接口
        '''
        data = {
            "knowledge_base_name": knowledge_base_name,
            "vector_store_type": vector_store_type,
            "embed_model": embed_model,
        }

        response = self.post(
            "/knowledge_base/create_knowledge_base",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def delete_knowledge_base(
            self,
            knowledge_base_name: str,
    ):
        '''
        对应api.py/knowledge_base/delete_knowledge_base接口
        '''
        response = self.post(
            "/knowledge_base/delete_knowledge_base",
            json=f"{knowledge_base_name}",
        )
        return self._get_response_value(response, as_json=True)

    def list_kb_docs(
            self,
            knowledge_base_name: str,
    ):
        '''
        对应api.py/knowledge_base/list_files接口
        '''
        response = self.get(
            "/knowledge_base/list_files",
            params={"knowledge_base_name": knowledge_base_name}
        )
        return self._get_response_value(response,
                                        as_json=True,
                                        value_func=lambda r: r.get("data", []))

    def search_kb_docs(
            self,
            knowledge_base_name: str,
            query: str = "",
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: int = SCORE_THRESHOLD,
            file_name: str = "",
            metadata: dict = {},
    ) -> List:
        '''
        对应api.py/knowledge_base/search_docs接口
        '''
        data = {
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "file_name": file_name,
            "metadata": metadata,
        }

        response = self.post(
            "/knowledge_base/search_docs",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def update_docs_by_id(
            self,
            knowledge_base_name: str,
            docs: Dict[str, Dict],
    ) -> bool:
        '''
        对应api.py/knowledge_base/update_docs_by_id接口
        '''
        data = {
            "knowledge_base_name": knowledge_base_name,
            "docs": docs,
        }
        response = self.post(
            "/knowledge_base/update_docs_by_id",
            json=data
        )
        return self._get_response_value(response)

    def upload_kb_docs(
            self,
            files: List[Union[str, Path, bytes]],
            knowledge_base_name: str,
            override: bool = False,
            to_vector_store: bool = True,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
            docs: Dict = {},
            not_refresh_vs_cache: bool = False,
    ):
        '''
        对应api.py/knowledge_base/upload_docs接口
        '''

        def convert_file(file, filename=None):
            if isinstance(file, bytes):  # raw bytes
                file = BytesIO(file)
            elif hasattr(file, "read"):  # a file io like object
                filename = filename or file.name
            else:  # a local path
                file = Path(file).absolute().open("rb")
                filename = filename or os.path.split(file.name)[-1]
            return filename, file

        files = [convert_file(file) for file in files]
        data = {
            "knowledge_base_name": knowledge_base_name,
            "override": override,
            "to_vector_store": to_vector_store,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
            "docs": docs,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        if isinstance(data["docs"], dict):
            data["docs"] = json.dumps(data["docs"], ensure_ascii=False)
        response = self.post(
            "/knowledge_base/upload_docs",
            data=data,
            files=[("files", (filename, file)) for filename, file in files],
        )
        return self._get_response_value(response, as_json=True)

    def delete_kb_docs(
            self,
            knowledge_base_name: str,
            file_names: List[str],
            delete_content: bool = False,
            not_refresh_vs_cache: bool = False,
    ):
        '''
        对应api.py/knowledge_base/delete_docs接口
        '''
        data = {
            "knowledge_base_name": knowledge_base_name,
            "file_names": file_names,
            "delete_content": delete_content,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        response = self.post(
            "/knowledge_base/delete_docs",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def update_kb_info(self, knowledge_base_name, kb_info):
        '''
        对应api.py/knowledge_base/update_info接口
        '''
        data = {
            "knowledge_base_name": knowledge_base_name,
            "kb_info": kb_info,
        }

        response = self.post(
            "/knowledge_base/update_info",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def update_kb_docs(
            self,
            knowledge_base_name: str,
            file_names: List[str],
            override_custom_docs: bool = False,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
            docs: Dict = {},
            not_refresh_vs_cache: bool = False,
    ):
        '''
        对应api.py/knowledge_base/update_docs接口
        '''
        data = {
            "knowledge_base_name": knowledge_base_name,
            "file_names": file_names,
            "override_custom_docs": override_custom_docs,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
            "docs": docs,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        if isinstance(data["docs"], dict):
            data["docs"] = json.dumps(data["docs"], ensure_ascii=False)

        response = self.post(
            "/knowledge_base/update_docs",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def recreate_vector_store(
            self,
            knowledge_base_name: str,
            allow_empty_kb: bool = True,
            vs_type: str = DEFAULT_VS_TYPE,
            embed_model: str = EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
    ):
        '''
        对应api.py/knowledge_base/recreate_vector_store接口
        '''
        data = {
            "knowledge_base_name": knowledge_base_name,
            "allow_empty_kb": allow_empty_kb,
            "vs_type": vs_type,
            "embed_model": embed_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
        }

        response = self.post(
            "/knowledge_base/recreate_vector_store",
            json=data,
            stream=True,
            timeout=None,
        )
        return self._httpx_stream2generator(response, as_json=True)

    # LLM模型相关操作
    def list_running_models(
            self,
            controller_address: str = None,
    ):
        '''
        获取Fastchat中正运行的模型列表
        '''
        data = {
            "controller_address": controller_address,
        }

        if log_verbose:
            logger.info(f'{self.__class__.__name__}:data: {data}')

        response = self.post(
            "/llm_model/list_running_models",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r: r.get("data", []))

    def get_default_llm_model(self, local_first: bool = True) -> Tuple[str, bool]:
        '''
        从服务器上获取当前运行的LLM模型。
        当 local_first=True 时，优先返回运行中的本地模型，否则优先按LLM_MODELS配置顺序返回。
        返回类型为（model_name, is_local_model）
        '''

        def ret_sync():
            running_models = self.list_running_models()
            if not running_models:
                return "", False

            model = ""
            for m in LLM_MODELS:
                if m not in running_models:
                    continue
                is_local = not running_models[m].get("online_api")
                if local_first and not is_local:
                    continue
                else:
                    model = m
                    break

            if not model:  # LLM_MODELS中配置的模型都不在running_models里
                model = list(running_models)[0]
            is_local = not running_models[model].get("online_api")
            return model, is_local

        async def ret_async():
            running_models = await self.list_running_models()
            if not running_models:
                return "", False

            model = ""
            for m in LLM_MODELS:
                if m not in running_models:
                    continue
                is_local = not running_models[m].get("online_api")
                if local_first and not is_local:
                    continue
                else:
                    model = m
                    break

            if not model:  # LLM_MODELS中配置的模型都不在running_models里
                model = list(running_models)[0]
            is_local = not running_models[model].get("online_api")
            return model, is_local

        if self._use_async:
            return ret_async()
        else:
            return ret_sync()

    def list_config_models(
            self,
            types: List[str] = ["local", "online"],
    ) -> Dict[str, Dict]:
        '''
        获取服务器configs中配置的模型列表，返回形式为{"type": {model_name: config}, ...}。
        '''
        data = {
            "types": types,
        }
        response = self.post(
            "/llm_model/list_config_models",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r: r.get("data", {}))

    def get_model_config(
            self,
            model_name: str = None,
    ) -> Dict:
        '''
        获取服务器上模型配置
        '''
        data = {
            "model_name": model_name,
        }
        response = self.post(
            "/llm_model/get_model_config",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r: r.get("data", {}))

    def list_search_engines(self) -> List[str]:
        '''
        获取服务器支持的搜索引擎
        '''
        response = self.post(
            "/server/list_search_engines",
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r: r.get("data", {}))

    def stop_llm_model(
            self,
            model_name: str,
            controller_address: str = None,
    ):
        '''
        停止某个LLM模型。
        注意：由于Fastchat的实现方式，实际上是把LLM模型所在的model_worker停掉。
        '''
        data = {
            "model_name": model_name,
            "controller_address": controller_address,
        }

        response = self.post(
            "/llm_model/stop",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def change_llm_model(
            self,
            model_name: str,
            new_model_name: str,
            controller_address: str = None,
    ):
        '''
        向fastchat controller请求切换LLM模型。
        '''
        if not model_name or not new_model_name:
            return {
                "code": 500,
                "msg": f"未指定模型名称"
            }

        def ret_sync():
            running_models = self.list_running_models()
            if new_model_name == model_name or new_model_name in running_models:
                return {
                    "code": 200,
                    "msg": "无需切换"
                }

            if model_name not in running_models:
                return {
                    "code": 500,
                    "msg": f"指定的模型'{model_name}'没有运行。当前运行模型：{running_models}"
                }

            config_models = self.list_config_models()
            if new_model_name not in config_models.get("local", {}):
                return {
                    "code": 500,
                    "msg": f"要切换的模型'{new_model_name}'在configs中没有配置。"
                }

            data = {
                "model_name": model_name,
                "new_model_name": new_model_name,
                "controller_address": controller_address,
            }

            response = self.post(
                "/llm_model/change",
                json=data,
            )
            return self._get_response_value(response, as_json=True)

        async def ret_async():
            running_models = await self.list_running_models()
            if new_model_name == model_name or new_model_name in running_models:
                return {
                    "code": 200,
                    "msg": "无需切换"
                }

            if model_name not in running_models:
                return {
                    "code": 500,
                    "msg": f"指定的模型'{model_name}'没有运行。当前运行模型：{running_models}"
                }

            config_models = await self.list_config_models()
            if new_model_name not in config_models.get("local", {}):
                return {
                    "code": 500,
                    "msg": f"要切换的模型'{new_model_name}'在configs中没有配置。"
                }

            data = {
                "model_name": model_name,
                "new_model_name": new_model_name,
                "controller_address": controller_address,
            }

            response = self.post(
                "/llm_model/change",
                json=data,
            )
            return self._get_response_value(response, as_json=True)

        if self._use_async:
            return ret_async()
        else:
            return ret_sync()

    def embed_texts(
            self,
            texts: List[str],
            embed_model: str = EMBEDDING_MODEL,
            to_query: bool = False,
    ) -> List[List[float]]:
        '''
        对文本进行向量化，可选模型包括本地 embed_models 和支持 embeddings 的在线模型
        '''
        data = {
            "texts": texts,
            "embed_model": embed_model,
            "to_query": to_query,
        }
        resp = self.post(
            "/other/embed_texts",
            json=data,
        )
        return self._get_response_value(resp, as_json=True, value_func=lambda r: r.get("data"))

    def chat_feedback(
            self,
            message_id: str,
            score: int,
            reason: str = "",
    ) -> int:
        '''
        反馈对话评价
        '''
        data = {
            "message_id": message_id,
            "score": score,
            "reason": reason,
        }
        resp = self.post("/chat/feedback", json=data)
        return self._get_response_value(resp)


class AsyncApiRequest(ApiRequest):
    def __init__(self, base_url: str = api_address(), timeout: float = HTTPX_DEFAULT_TIMEOUT):
        super().__init__(base_url, timeout)
        self._use_async = True


def check_error_msg(data: Union[str, dict, list], key: str = "errorMsg") -> str:
    '''
    return error message if error occured when requests API
    '''
    if isinstance(data, dict):
        if key in data:
            return data[key]
        if "code" in data and data["code"] != 200:
            return data["msg"]
    return ""


def check_success_msg(data: Union[str, dict, list], key: str = "msg") -> str:
    '''
    return error message if error occured when requests API
    '''
    if (isinstance(data, dict)
            and key in data
            and "code" in data
            and data["code"] == 200):
        return data[key]
    return ""


if __name__ == "__main__":
    api = ApiRequest()
    aapi = AsyncApiRequest()

    # with api.chat_chat("你好") as r:
    #     for t in r.iter_text(None):
    #         print(t)

    # r = api.chat_chat("你好", no_remote_api=True)
    # for t in r:
    #     print(t)

    # r = api.duckduckgo_search_chat("室温超导最新研究进展", no_remote_api=True)
    # for t in r:
    #     print(t)

    # print(api.list_knowledge_bases())
