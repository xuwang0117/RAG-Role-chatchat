import nltk
import sys
import os

from server.chat.origin_kb_chat import origin_kb_chat
from server.chat.jsby_kb_chat import jsby_kb_chat
from server.chat.brain_storm_chat import continue_discuss, discuss, generate_background, generate_outline, improve_outline, recommend_expert, recommend_interesting_topics, select_expert_answer, summary

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from configs import VERSION
from configs.model_config import NLTK_DATA_PATH
from configs.server_config import OPEN_CROSS_DOMAIN, OPEN_LOGING_URL_METHOD_PARAMS
import argparse
import uvicorn
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from server.chat.chat import chat, chat_with_all
from server.chat.pipeline_knowledge_chat import pipe_knowledge_chat
from server.chat.smart_reporting import smart_reporting
from server.chat.smart_scenario import smart_scenario
from server.chat.official_statement import official_statement
from server.chat.discuss_with_role import discuss_with_role, list_am_roles, list_tw_roles, new_chat_only
from server.chat.new_kb_chat_with_role import new_kb_chat_with_role, list_roles
from server.chat.muti_persona_kb_chat import muti_persona_kb_chat, personas_select, summary_chat
from server.chat.utils import LogRequestMiddleware
from server.chat.search_engine_chat import search_engine_chat
from server.chat.completion import completion
from server.chat.feedback import chat_feedback
from server.embeddings_api import embed_texts_endpoint
from server.llm_api import (list_running_models, list_config_models,
                            change_llm_model, stop_llm_model,
                            get_model_config, list_search_engines)
from server.utils import (BaseResponse, ListResponse, FastAPI, MakeFastAPIOffline,
                          get_server_configs, get_prompt_template)
from typing import List, Literal

from server.chat.my_stream_test import warp_test

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


async def document():
    return RedirectResponse(url="/docs")


def create_app(run_mode: str = None):
    app = FastAPI(
        title="Langchain-Chatchat API Server",
        version=VERSION
    )
    MakeFastAPIOffline(app)
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    if OPEN_LOGING_URL_METHOD_PARAMS:
        app.add_middleware(LogRequestMiddleware, some_attribute="请求中间件")
    mount_app_routes(app, run_mode=run_mode)
    return app


def mount_app_routes(app: FastAPI, run_mode: str = None):
    app.get("/",
            response_model=BaseResponse,
            summary="swagger 文档")(document)

    # Tag: Chat
    app.post("/chat/chat",
             tags=["Chat"],
             summary="与llm模型对话(通过LLMChain)",
             )(chat)
    
    app.post("/chat/chat_with_all",
             tags=["Chat"],
             summary="“四步走、预制、搜索”三合一接口",
             )(chat_with_all)
    
    app.post("/chat/test", tags=["Chat"], summary="流式输出测试")(warp_test)
    
    app.post("/chat/search_engine_chat",
             tags=["Chat"],
             summary="与搜索引擎对话",
             )(search_engine_chat)

    app.post("/chat/feedback",
             tags=["Chat"],
             summary="返回llm模型对话评分",
             )(chat_feedback)

    app.post("/brain_storm/recommend_expert",
             tags=["brain_storm"],
             summary="生成推荐专家",
             )(recommend_expert)
    
    app.post("/brain_storm/generate_outline",
             tags=["brain_storm"],
             summary="生成大纲",
             )(generate_outline)
    
    app.post("/chat/origin_kb_chat",
             tags=["Chat"],
             summary="原始知识库对话")(origin_kb_chat)
    
    app.post("/chat/jsby_kb_chat",
             tags=["Chat"],
             summary="角色扮演对话")(jsby_kb_chat)
    
    app.post("/brain_storm/generate_background",
             tags=["brain_storm"],
             summary="生成议题背景",
             )(generate_background)
    
    app.post("/brain_storm/improve_outline",
             tags=["brain_storm"],
             summary="改进大纲",
             )(improve_outline)
    
    # app.post("/brain_storm/sub_topic",
    #          tags=["brain_storm"],
    #          summary="生成子议题",
    #          )(sub_topic)
    
    app.post("/brain_storm/discuss",
             tags=["brain_storm"],
             summary="讨论",
             )(discuss)
    
    app.post("/brain_storm/select_expert_answer",
             tags=["brain_storm"],
             summary="指定发言",
             )(select_expert_answer)
    
    app.post("/brain_storm/summary",
             tags=["brain_storm"],
             summary="生成总结",
             )(summary)
    
    app.post("/brain_storm/continue_discuss",
             tags=["brain_storm"],
             summary="继续讨论",
             )(continue_discuss)
    
    app.post("/brain_storm/recommend_interesting_topics",
             tags=["brain_storm"],
             summary="推荐感兴趣话题",
             )(recommend_interesting_topics)
    
    
    

    # 知识库相关接口
    mount_knowledge_routes(app)
    # 摘要相关接口
    mount_filename_summary_routes(app)

    # LLM模型相关接口
    app.post("/llm_model/list_running_models",
             tags=["LLM Model Management"],
             summary="列出当前已加载的模型",
             )(list_running_models)

    app.post("/llm_model/list_config_models",
             tags=["LLM Model Management"],
             summary="列出configs已配置的模型",
             )(list_config_models)

    app.post("/llm_model/get_model_config",
             tags=["LLM Model Management"],
             summary="获取模型配置（合并后）",
             )(get_model_config)

    app.post("/llm_model/stop",
             tags=["LLM Model Management"],
             summary="停止指定的LLM模型（Model Worker)",
             )(stop_llm_model)

    app.post("/llm_model/change",
             tags=["LLM Model Management"],
             summary="切换指定的LLM模型（Model Worker)",
             )(change_llm_model)

    # 服务器相关接口
    app.post("/server/configs",
             tags=["Server State"],
             summary="获取服务器原始配置信息",
             )(get_server_configs)

    app.post("/server/list_search_engines",
             tags=["Server State"],
             summary="获取服务器支持的搜索引擎",
             )(list_search_engines)

    @app.post("/server/get_prompt_template",
             tags=["Server State"],
             summary="获取服务区配置的 prompt 模板")
    def get_server_prompt_template(
        type: Literal["llm_chat", "knowledge_base_chat", "search_engine_chat", "agent_chat"]=Body("llm_chat", description="模板类型，可选值：llm_chat，knowledge_base_chat，search_engine_chat，agent_chat"),
        name: str = Body("default", description="模板名称"),
    ) -> str:
        return get_prompt_template(type=type, name=name)

    # 其它接口
    app.post("/other/completion",
             tags=["Other"],
             summary="要求llm模型补全(通过LLMChain)",
             )(completion)

    app.post("/other/embed_texts",
            tags=["Other"],
            summary="将文本向量化，支持本地模型和在线模型",
            )(embed_texts_endpoint)


def mount_knowledge_routes(app: FastAPI):
    from server.chat.knowledge_base_chat import knowledge_base_chat
    from server.chat.file_chat import upload_temp_docs, file_chat
    from server.chat.agent_chat import agent_chat
    from server.knowledge_base.kb_api import list_kbs, create_kb, delete_kb
    from server.knowledge_base.kb_doc_api import (list_files, upload_docs, delete_docs,
                                                update_docs, download_doc, recreate_vector_store,
                                                search_docs, DocumentWithVSId, update_info,
                                                update_docs_by_id,)

    app.post("/chat/knowledge_base_chat",
             tags=["Chat"],
             summary="与知识库对话")(knowledge_base_chat)
    
    # add by chenming 20240705 18:45
    app.post("/chat/pipeline_knowledge_chat",
             tags=["Chat"],
             summary="流水线与知识库对话")(pipe_knowledge_chat)
    
    # add 2024.11.7 lightrag********************************************************************************************
    app.post("/chat/new_kb_chat_with_role",
             tags=["Chat"],
             summary="角色扮演对话")(new_kb_chat_with_role)
    
    # add 2024.12.5 群聊会话********************************************************************************************
    app.post("/chat/discuss_with_role",
             tags=["Chat"],
             summary="群聊会话")(discuss_with_role)
    
    # add 2024.12.5 大模型会话********************************************************************************************
    app.post("/chat/new_chat_only",
             tags=["Chat"],
             summary="大模型会话")(new_chat_only)
    
    # add 2024.11.22 智能筹划-方案生成********************************************************************************************
    app.post("/chat/smart_scenario",
             tags=["Chat"],
             summary="智能筹划-方案生成")(smart_scenario)
    
    # add 2024.11.22 智能筹划-官方发声********************************************************************************************
    app.post("/chat/official_statement",
             tags=["Chat"],
             summary="智能筹划-官方发声")(official_statement)
    
    # add 2024.11.8 智能出报********************************************************************************************
    app.post("/chat/smart_reporting",
             tags=["Chat"],
             summary="智能出报")(smart_reporting)
    
    app.get("/chat/list_roles",
            tags=["Roles List"],
            summary="获取角色列表")(list_roles)
    
    # add 2024.12.5 智能出报********************************************************************************************
    app.get("/chat/list_am_roles",
            tags=["Roles List"],
            summary="获取美国角色列表")(list_am_roles)
            
    # add 2024.12.5 智能出报********************************************************************************************
    app.get("/chat/list_tw_roles",
            tags=["Roles List"],
            summary="获取台湾角色列表")(list_tw_roles)
    
    # add by xuwang 2024-9-24 20:39:25
    app.post("/chat/personas_select",
             tags=["Chat"],
             summary="职业匹配接口")(personas_select)
    
    # add by xuwang 2024-9-25 10:39:25
    app.post("/chat/muti_persona_kb_chat",
             tags=["Chat"],
             summary="职业模型问答接口")(muti_persona_kb_chat)
    
    # add by xuwang 2024-9-25 20:49:00
    app.post("/chat/summary_chat",
             tags=["Chat"],
             summary="总结回答接口")(summary_chat)

    app.post("/chat/file_chat",
             tags=["Knowledge Base Management"],
             summary="文件对话"
             )(file_chat)

    app.post("/chat/agent_chat",
             tags=["Chat"],
             summary="与agent对话")(agent_chat)

    # Tag: Knowledge Base Management
    app.get("/knowledge_base/list_knowledge_bases",
            tags=["Knowledge Base Management"],
            response_model=ListResponse,
            summary="获取知识库列表")(list_kbs)

    app.post("/knowledge_base/create_knowledge_base",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="创建知识库"
             )(create_kb)

    app.post("/knowledge_base/delete_knowledge_base",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="删除知识库"
             )(delete_kb)

    app.get("/knowledge_base/list_files",
            tags=["Knowledge Base Management"],
            response_model=ListResponse,
            summary="获取知识库内的文件列表"
            )(list_files)

    app.post("/knowledge_base/search_docs",
             tags=["Knowledge Base Management"],
             response_model=List[DocumentWithVSId],
             summary="搜索知识库"
             )(search_docs)

    app.post("/knowledge_base/update_docs_by_id",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="直接更新知识库文档"
             )(update_docs_by_id)


    app.post("/knowledge_base/upload_docs",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="上传文件到知识库，并/或进行向量化"
             )(upload_docs)

    app.post("/knowledge_base/delete_docs",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="删除知识库内指定文件"
             )(delete_docs)

    app.post("/knowledge_base/update_info",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="更新知识库介绍"
             )(update_info)
    app.post("/knowledge_base/update_docs",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="更新现有文件到知识库"
             )(update_docs)

    app.get("/knowledge_base/download_doc",
            tags=["Knowledge Base Management"],
            summary="下载对应的知识文件")(download_doc)

    app.post("/knowledge_base/recreate_vector_store",
             tags=["Knowledge Base Management"],
             summary="根据content中文档重建向量库，流式输出处理进度。"
             )(recreate_vector_store)

    app.post("/knowledge_base/upload_temp_docs",
             tags=["Knowledge Base Management"],
             summary="上传文件到临时目录，用于文件对话。"
             )(upload_temp_docs)


def mount_filename_summary_routes(app: FastAPI):
    from server.knowledge_base.kb_summary_api import (summary_file_to_vector_store, recreate_summary_vector_store,
                                                      summary_doc_ids_to_vector_store)

    app.post("/knowledge_base/kb_summary_api/summary_file_to_vector_store",
             tags=["Knowledge kb_summary_api Management"],
             summary="单个知识库根据文件名称摘要"
             )(summary_file_to_vector_store)
    app.post("/knowledge_base/kb_summary_api/summary_doc_ids_to_vector_store",
             tags=["Knowledge kb_summary_api Management"],
             summary="单个知识库根据doc_ids摘要",
             response_model=BaseResponse,
             )(summary_doc_ids_to_vector_store)
    app.post("/knowledge_base/kb_summary_api/recreate_summary_vector_store",
             tags=["Knowledge kb_summary_api Management"],
             summary="重建单个知识库文件摘要"
             )(recreate_summary_vector_store)



def run_api(host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='langchain-ChatGLM',
                                     description='About langchain-ChatGLM, local knowledge based ChatGLM with langchain'
                                                 ' ｜ 基于本地知识库的 ChatGLM 问答')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # 初始化消息
    args = parser.parse_args()
    args_dict = vars(args)

    app = create_app()

    run_api(host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            )
