import os

# 可以指定一个绝对路径，统一存放所有的Embedding和LLM模型。
# 每个模型可以是一个单独的目录，也可以是某个目录下的二级子目录。
# 如果模型目录名称和 MODEL_PATH 中的 key 或 value 相同，程序会自动检测加载，无需修改 MODEL_PATH 中的路径。
#MODEL_ROOT_PATH = "/opt/models"
MODEL_ROOT_PATH = "./models"

# 选用的 Embedding 名称
EMBEDDING_MODEL = "bge-large-zh-v1.5"

# Embedding 模型运行设备。设为 "auto" 会自动检测(会有警告)，也可手动设定为 "cuda","mps","cpu","xpu" 其中之一。
EMBEDDING_DEVICE = "cuda"

# 选用的reranker模型
RERANKER_MODEL = "bge-reranker-large"
# 是否启用reranker模型
USE_RERANKER = False
RERANKER_MAX_LENGTH = 1024

# 如果需要在 EMBEDDING_MODEL 中增加自定义的关键字时配置
EMBEDDING_KEYWORD_FILE = "keywords.txt"
EMBEDDING_MODEL_OUTPUT_PATH = "output"

# 要运行的 LLM 名称，可以包括本地模型和在线模型。列表中本地模型将在启动项目时全部加载。
# 列表中第一个模型将作为 API 和 WEBUI 的默认模型。
# 在这里，我们使用目前主流的两个离线模型，其中，chatglm3-6b 为默认加载模型。
# 如果你的显存不足，可使用 Qwen-1_8B-Chat, 该模型 FP16 仅需 3.8G显存。

LLM_MODELS = ["qwen2.5:32b"]

Agent_MODEL = None

# LLM 模型运行设备。设为"auto"会自动检测(会有警告)，也可手动设定为 "cuda","mps","cpu","xpu" 其中之一。
LLM_DEVICE = "cuda"

HISTORY_LEN = 0

MAX_TOKENS = 2048

TEMPERATURE = 0.7

ONLINE_LLM_MODEL = {
    # "qwen2.5:72b": {
    #     "version": "qwen2.5:72b",
    #     "api_key": "EMPTY",
    #     "api_base_url": "http://127.0.0.1:11434/v1",
    #     "provider": "QwenWorker",
    #     "embed_model": "bge-large-zh-v1.5"  # embedding 模型名称
    # },
    
    "qwen2.5:32b": {
        "version": "qwen2.5:32b",
        "api_key": "EMPTY",
        "api_base_url": "http://127.0.0.1:11434/v1",
        "provider": "QwenWorker",
        "embed_model": "bge-large-zh-v1.5"  # embedding 模型名称
    },
    
    # "Qwen2": {
    #     "version": "Qwen2",
    #     "api_key": "EMPTY",
    #     "api_base_url": "http://127.0.0.1:12532/v1",
    #     "provider": "QwenWorker",
    #     "embed_model": "bge-large-zh-v1.5"  # embedding 模型名称
    # },

}

# 在以下字典中修改属性值，以指定本地embedding模型存储位置。支持3种设置方法：
# 1、将对应的值修改为模型绝对路径
# 2、不修改此处的值（以 text2vec 为例）：
#       2.1 如果{MODEL_ROOT_PATH}下存在如下任一子目录：
#           - text2vec
#           - GanymedeNil/text2vec-large-chinese
#           - text2vec-large-chinese
#       2.2 如果以上本地路径不存在，则使用huggingface模型

MODEL_PATH = {
    "embed_model": {
        "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
        "ernie-base": "nghuyong/ernie-3.0-base-zh",
        "text2vec-base": "shibing624/text2vec-base-chinese",
        "text2vec": "GanymedeNil/text2vec-large-chinese",
        "text2vec-paraphrase": "shibing624/text2vec-base-chinese-paraphrase",
        "text2vec-sentence": "shibing624/text2vec-base-chinese-sentence",
        "text2vec-multilingual": "shibing624/text2vec-base-multilingual",
        "text2vec-bge-large-chinese": "shibing624/text2vec-bge-large-chinese",
        "m3e-small": "moka-ai/m3e-small",
        "m3e-base": "moka-ai/m3e-base",
        "m3e-large": "moka-ai/m3e-large",

        "bge-small-zh": "BAAI/bge-small-zh",
        "bge-base-zh": "BAAI/bge-base-zh",
        "bge-large-zh": "BAAI/bge-large-zh",
        "bge-large-zh-noinstruct": "BAAI/bge-large-zh-noinstruct",
        "bge-base-zh-v1.5": "./models/bge-base-zh-v1.5",
        "bge-large-zh-v1.5": "./models/bge-large-zh-v1.5",

        "bge-m3": "BAAI/bge-m3",

        "piccolo-base-zh": "sensenova/piccolo-base-zh",
        "piccolo-large-zh": "sensenova/piccolo-large-zh",
        "nlp_gte_sentence-embedding_chinese-large": "damo/nlp_gte_sentence-embedding_chinese-large",
        "text-embedding-ada-002": "your OPENAI_API_KEY",
    },

    "llm_model": {
        # "Qwen2-1.5B-Instruct": "./models/Qwen2-1.5B-Instruct",
        "CogGPT2-0.5B": "./models/Qwen2.5-0.5B-Instruct",
        "CogGPT2-1.5B": "./models/Qwen2.5-1.5B-Instruct",
        # "chatglm2-6b": "THUDM/chatglm2-6b",
        # "chatglm2-6b-32k": "THUDM/chatglm2-6b-32k",
        # "chatglm3-6b": "THUDM/chatglm3-6b",
        # "chatglm3-6b-32k": "THUDM/chatglm3-6b-32k",

        # "Orion-14B-Chat": "OrionStarAI/Orion-14B-Chat",
        # "Orion-14B-Chat-Plugin": "OrionStarAI/Orion-14B-Chat-Plugin",
        # "Orion-14B-LongChat": "OrionStarAI/Orion-14B-LongChat",

        # "CPM-9G-8b-hf":"/lfs/models/cpm-9g-8b",
        # "CPM-9G-8b-hf-special":"/lfs/models/cpm-9g-8b-2",
        # "cpm-9g-8b-4bit":"/lfs/models/cpm-9g-8b-4bit",

        # "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
        # "Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
        # "Llama-2-70b-chat-hf": "meta-llama/Llama-2-70b-chat-hf",

        # "Qwen-1_8B-Chat": "Qwen/Qwen-1_8B-Chat",
        # "Qwen-7B-Chat": "Qwen/Qwen-7B-Chat",
        # "Qwen-14B-Chat": "Qwen/Qwen-14B-Chat",
        # "Qwen-72B-Chat": "Qwen/Qwen-72B-Chat",

        # # Qwen1.5 模型 VLLM可能出现问题
        # "Qwen1.5-0.5B-Chat": "Qwen/Qwen1.5-0.5B-Chat",
        # "Qwen1.5-1.8B-Chat": "Qwen/Qwen1.5-1.8B-Chat",
        # "Qwen1.5-4B-Chat": "Qwen/Qwen1.5-4B-Chat",
        # "Qwen1.5-7B-Chat": "Qwen/Qwen1.5-7B-Chat",
        # "Qwen1.5-14B-Chat": "Qwen/Qwen1.5-14B-Chat",
        # "Qwen1.5-72B-Chat": "Qwen/Qwen1.5-72B-Chat",
        # "Qwen1.5-32B-Chat": "/lfs/models/Qwen1.5-32B-Chat"
        
        # "Qwen2-72B-Instruct": "/lfs/.cache/modelscope/qwen/Qwen2-72B-Instruct",
        # "Qwen2-72B-Instruct-GPTQ-Int8": "/lfs/.cache/modelscope/Qwen/Qwen2-72B-Instruct-GPTQ-Int8",

        # "baichuan-7b-chat": "baichuan-inc/Baichuan-7B-Chat",
        # "baichuan-13b-chat": "baichuan-inc/Baichuan-13B-Chat",
        # "baichuan2-7b-chat": "baichuan-inc/Baichuan2-7B-Chat",
        # "baichuan2-13b-chat": "baichuan-inc/Baichuan2-13B-Chat",

        # "internlm-7b": "internlm/internlm-7b",
        # "internlm-chat-7b": "internlm/internlm-chat-7b",
        # "internlm2-chat-7b": "internlm/internlm2-chat-7b",
        # "internlm2-chat-20b": "internlm/internlm2-chat-20b",

        # "BlueLM-7B-Chat": "vivo-ai/BlueLM-7B-Chat",
        # "BlueLM-7B-Chat-32k": "vivo-ai/BlueLM-7B-Chat-32k",

        # "Yi-34B-Chat": "https://huggingface.co/01-ai/Yi-34B-Chat",

        # "agentlm-7b": "THUDM/agentlm-7b",
        # "agentlm-13b": "THUDM/agentlm-13b",
        # "agentlm-70b": "THUDM/agentlm-70b",

        # "falcon-7b": "tiiuae/falcon-7b",
        # "falcon-40b": "tiiuae/falcon-40b",
        # "falcon-rw-7b": "tiiuae/falcon-rw-7b",

        # "aquila-7b": "BAAI/Aquila-7B",
        # "aquilachat-7b": "BAAI/AquilaChat-7B",
        # "open_llama_13b": "openlm-research/open_llama_13b",
        # "vicuna-13b-v1.5": "lmsys/vicuna-13b-v1.5",
        # "koala": "young-geng/koala",
        # "mpt-7b": "mosaicml/mpt-7b",
        # "mpt-7b-storywriter": "mosaicml/mpt-7b-storywriter",
        # "mpt-30b": "mosaicml/mpt-30b",
        # "opt-66b": "facebook/opt-66b",
        # "opt-iml-max-30b": "facebook/opt-iml-max-30b",
        # "gpt2": "gpt2",
        # "gpt2-xl": "gpt2-xl",
        # "gpt-j-6b": "EleutherAI/gpt-j-6b",
        # "gpt4all-j": "nomic-ai/gpt4all-j",
        # "gpt-neox-20b": "EleutherAI/gpt-neox-20b",
        # "pythia-12b": "EleutherAI/pythia-12b",
        # "oasst-sft-4-pythia-12b-epoch-3.5": "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
        # "dolly-v2-12b": "databricks/dolly-v2-12b",
        # "stablelm-tuned-alpha-7b": "stabilityai/stablelm-tuned-alpha-7b",
    },

    "reranker": {
        "bge-reranker-large": "/lfs/models/bge-reranker-large/snapshots/55611d7bca2a7133960a6d3b71e083071bbfc312",
        "bge-reranker-base": "BAAI/bge-reranker-base",
    }
}

# 通常情况下不需要更改以下内容

# nltk 模型存储路径
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

# 使用VLLM可能导致模型推理能力下降，无法完成Agent任务
VLLM_MODEL_DICT = {
    "chatglm2-6b": "THUDM/chatglm2-6b",
    "chatglm2-6b-32k": "THUDM/chatglm2-6b-32k",
    "chatglm3-6b": "THUDM/chatglm3-6b",
    "chatglm3-6b-32k": "THUDM/chatglm3-6b-32k",

    # "CPM-9G-8b-hf":"/lfs/models/cpm-9g-8b",

    "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "Llama-2-70b-chat-hf": "meta-llama/Llama-2-70b-chat-hf",

    "Qwen-1_8B-Chat": "Qwen/Qwen-1_8B-Chat",
    "Qwen-7B-Chat": "Qwen/Qwen-7B-Chat",
    "Qwen-14B-Chat": "Qwen/Qwen-14B-Chat",
    "Qwen-72B-Chat": "Qwen/Qwen-72B-Chat",
    
    # "Qwen2-72B-Instruct-GPTQ-Int8": "/lfs/.cache/modelscope/Qwen/Qwen2-72B-Instruct-GPTQ-Int8",
    # "Qwen2-72B-Instruct": "/lfs/.cache/modelscope/qwen/Qwen2-72B-Instruct",

    "baichuan-7b-chat": "baichuan-inc/Baichuan-7B-Chat",
    "baichuan-13b-chat": "baichuan-inc/Baichuan-13B-Chat",
    "baichuan2-7b-chat": "baichuan-inc/Baichuan-7B-Chat",
    "baichuan2-13b-chat": "baichuan-inc/Baichuan-13B-Chat",

    "BlueLM-7B-Chat": "vivo-ai/BlueLM-7B-Chat",
    "BlueLM-7B-Chat-32k": "vivo-ai/BlueLM-7B-Chat-32k",

    "internlm-7b": "internlm/internlm-7b",
    "internlm-chat-7b": "internlm/internlm-chat-7b",
    "internlm2-chat-7b": "internlm/Models/internlm2-chat-7b",
    "internlm2-chat-20b": "internlm/Models/internlm2-chat-20b",

    "aquila-7b": "BAAI/Aquila-7B",
    "aquilachat-7b": "BAAI/AquilaChat-7B",

    "falcon-7b": "tiiuae/falcon-7b",
    "falcon-40b": "tiiuae/falcon-40b",
    "falcon-rw-7b": "tiiuae/falcon-rw-7b",
    "gpt2": "gpt2",
    "gpt2-xl": "gpt2-xl",
    "gpt-j-6b": "EleutherAI/gpt-j-6b",
    "gpt4all-j": "nomic-ai/gpt4all-j",
    "gpt-neox-20b": "EleutherAI/gpt-neox-20b",
    "pythia-12b": "EleutherAI/pythia-12b",
    "oasst-sft-4-pythia-12b-epoch-3.5": "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    "dolly-v2-12b": "databricks/dolly-v2-12b",
    "stablelm-tuned-alpha-7b": "stabilityai/stablelm-tuned-alpha-7b",
    "open_llama_13b": "openlm-research/open_llama_13b",
    "vicuna-13b-v1.3": "lmsys/vicuna-13b-v1.3",
    "koala": "young-geng/koala",
    "mpt-7b": "mosaicml/mpt-7b",
    "mpt-7b-storywriter": "mosaicml/mpt-7b-storywriter",
    "mpt-30b": "mosaicml/mpt-30b",
    "opt-66b": "facebook/opt-66b",
    "opt-iml-max-30b": "facebook/opt-iml-max-30b",

}

SUPPORT_AGENT_MODEL = [
    "openai-api",  # GPT4 模型
    "qwen-api",  # Qwen Max模型
    "zhipu-api",  # 智谱AI GLM4模型
    "Qwen",  # 所有Qwen系列本地模型
    "chatglm3-6b",
    "internlm2-chat-20b",
    "Orion-14B-Chat-Plugin",
]
