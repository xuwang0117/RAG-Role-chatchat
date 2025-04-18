from fastchat.conversation import Conversation
from server.model_workers.base import *
from fastchat import conversation as conv
import sys
import json
from server.model_workers import SparkApi_175
import websockets
from server.utils import iter_over_async, asyncio
from typing import List, Dict
from configs import logger
import re


async def request(appid, api_key, api_secret, Spark_url, question, temperature, max_token):
    wsParam = SparkApi_175.Ws_Param(appid, api_key, api_secret, Spark_url)
    wsUrl = wsParam.create_url()
    data = SparkApi_175.gen_params(appid, question, temperature, max_token)
    logger.debug(f"data: \n{data}")
    async with websockets.connect(wsUrl) as ws:
        await ws.send(json.dumps(data, ensure_ascii=False))
        finish = False
        while not finish:
            chunk = await ws.recv()
            response = json.loads(chunk)
            if response.get("header", {}).get("status") == 2:
                finish = True
            if text := response.get("payload", {}).get("choices", {}).get("text"):
                yield text[0]["content"]


class XingHuoWorker175B(ApiModelWorker):
    def __init__(
            self,
            *,
            model_names: List[str] = ["xinghuo-api"],
            controller_addr: str = None,
            worker_addr: str = None,
            version: str = None,
            **kwargs,
    ):
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 8000)
        super().__init__(**kwargs)
        self.version = version

    def do_chat(self, params: ApiChatParams) -> Dict:
        params.load_config(self.model_names[0])

        # version_mapping = {
        #     "v3.0": {"url": "ws://10.120.3.1:9991/turing/v3/chat", "max_tokens": 8000},
        # }

        # def get_version_details(version_key):
        #     return version_mapping.get(version_key, {"url": None})

        details = params.version
        Spark_url = details["url"]
        text = ""
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
        params.max_tokens = min(details["max_tokens"], params.max_tokens or 0)
        for chunk in iter_over_async(
                request(params.APPID, params.api_key, params.APISecret, Spark_url, params.messages,
                        params.temperature, params.max_tokens),
                loop=loop,
        ):
            if chunk:
                text += self.clean_summary_xf(chunk)
                yield {"error_code": 0, "text": text}

    def clean_summary_xf(self, content):
        """清理讯飞的无意义字符"""
        return re.sub(r"<ret>|<end>", "\n", content)
    
    def get_embeddings(self, params):
        print("embedding")
        print(params)

    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是一个聪明的助手，请根据用户的提示来完成任务",
            messages=[],
            roles=["user", "assistant"],
            sep="\n### ",
            stop_str="###",
        )


if __name__ == "__main__":
    import uvicorn
    from server.utils import MakeFastAPIOffline
    from fastchat.serve.model_worker import app

    worker = XingHuoWorker175B(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21003",
    )
    sys.modules["fastchat.serve.model_worker"].worker = worker
    MakeFastAPIOffline(app)
    uvicorn.run(app, port=21003)
