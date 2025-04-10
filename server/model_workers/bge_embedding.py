from fastchat.conversation import Conversation
from typing import List, Literal, Dict

from fastchat import conversation as conv
from server.model_workers.base import *
from server.model_workers.base import ApiEmbeddingsParams
from configs import logger
import requests
from configs import logger
from server.utils import get_model_worker_config


class BGEModelWorker(ApiModelWorker):
    """使用openai的方式调用远程服务
    
    需要考虑以下实现方式
    ```python
    from langchain.embeddings.openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model=model,
                                  openai_api_key=get_model_path(model),
                                  chunk_size=CHUNK_SIZE)
    ```
    """
    
    DEFAULT_EMBED_MODEL: str = "bge-large-embedding"
    def __init__(
        self,
        *,
        version: Literal["qwen-turbo", "qwen-plus"] = "qwen-turbo",
        model_names: List[str] = ["bge-large-embedding"],
        controller_addr: str = None,
        worker_addr: str = None,
        **kwargs,
    ):
        logger.info(f"bgemodel: {model_names}")
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 1024)
        super().__init__(**kwargs)
        self.version = version

    def do_embeddings(self, params: ApiEmbeddingsParams) -> Dict:
        texts = params.texts
        data = {"input": texts, "model": params.embed_model}
        
        logger.info(f"embdding: {texts}")
        
        config = get_model_worker_config(params.embed_model)
        api_key = config['api_key']
        api_base_url = config["api_base_url"]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        response = requests.post(f"{api_base_url}embeddings", headers=headers, json=data)
        if response.status_code == 200:
            datas = response.json()
            respond = {"code": 200, "data": [data['embedding'] for data in datas['data']], "msg": f"use {params.embed_model} get embedding"}
            return respond
        else:
            response.raise_for_status()
    
    def get_embeddings(self, params):
        logger.info(f"params: {params}")
        
    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是一个embedding模型",
            messages=[],
            roles=["user", "assistant", "system"],
            sep="\n### ",
            stop_str="###",
        )
    

