import requests
import streamlit as st
from webui_pages.dialogue.read_persona import read_persona
from webui_pages.utils import *
from streamlit_chatbox import *
from streamlit_modal import Modal
from datetime import datetime
from streamlit_pills import pills
import os
import re
import time
from langchain_community.document_transformers import (
    LongContextReorder,
)
from configs import (TEMPERATURE, HISTORY_LEN, PROMPT_TEMPLATES, LLM_MODELS,
                     DEFAULT_KNOWLEDGE_BASE, DEFAULT_SEARCH_ENGINE, SUPPORT_AGENT_MODEL)
from server.knowledge_base.utils import LOADER_DICT
import uuid
from typing import List, Dict
import concurrent.futures
import signal
import urllib


def stop_project():
    pid = os.getpid()
    os.kill(pid, signal.SIGTERM)

def parse_command(text: str, modal: Modal) -> bool:
    '''
    检查用户是否输入了自定义命令，当前支持：
    /new {session_name}。如果未提供名称，默认为“会话X”
    /del {session_name}。如果未提供名称，在会话数量>1的情况下，删除当前会话。
    /clear {session_name}。如果未提供名称，默认清除当前会话
    /help。查看命令帮助
    返回值：输入的是命令返回True，否则返回False
    '''
    if m := re.match(r"/([^\s]+)\s*(.*)", text):
        cmd, name = m.groups()
        name = name.strip()
        conv_names = chat_box.get_chat_names()
        if cmd == "help":
            modal.open()
        elif cmd == "new":
            if not name:
                i = 1
                while True:
                    name = f"会话{i}"
                    if name not in conv_names:
                        break
                    i += 1
            if name in st.session_state["conversation_ids"]:
                st.error(f"该会话名称 “{name}” 已存在")
                time.sleep(1)
            else:
                st.session_state["conversation_ids"][name] = uuid.uuid4().hex
                st.session_state["cur_conv_name"] = name
                #保存本地历史对话
                save_history()
        elif cmd == "del":
            name = name or st.session_state.get("cur_conv_name")
            if len(conv_names) == 1:
                st.error("这是最后一个会话，无法删除")
                time.sleep(1)
            elif not name or name not in st.session_state["conversation_ids"]:
                st.error(f"无效的会话名称：“{name}”")
                time.sleep(1)
            else:
                st.session_state["conversation_ids"].pop(name, None)
                chat_box.del_chat_name(name)
                st.session_state["cur_conv_name"] = conv_names[0]
                #保存本地历史对话
                save_history()
        elif cmd == "clear":
            chat_box.reset_history(name=name or None)
            #保存本地历史对话
            save_history()
        return True
    return False

def default_history_json():
    data = {
        "cur_chat_name": "新会话",
        "session_key": "chat_history",
        "user_avatar": "user",
        "assistant_avatar": "img/chatchat_icon_blue_square_v2.png",
        "greetings": [],
        "histories": {
            "新会话": []
        }
    }
    file_path = 'webui_pages/dialogue/history.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def modified_from_dict(
    self,
    data: Dict,
) -> "ChatBox":
    '''
    load state from dict
    '''
    self._chat_name=data["cur_chat_name"]
    self._session_key=data["session_key"]
    self._user_avatar=data["user_avatar"]
    self._assistant_avatar=data["assistant_avatar"]
    self._greetings=[OutputElement.from_dict(x) for x in data["greetings"]]
    self.init_session(clear=True)

    for name, history in data["histories"].items():
        self.reset_history(name)
        for h in history:
            msg = {
                "role": h["role"],
                "elements": [OutputElement.from_dict(y) for y in h["elements"]],
                "metadata": h["metadata"],
                }
            self.other_history(name).append(msg)

    self.use_chat_name(data["cur_chat_name"])
    return self

ChatBox.from_dict = modified_from_dict

# 定义一个将不可序列化对象转换为字符串的方法
def default_converter(o):
    if hasattr(o, '__dict__'):
        return o.__dict__
    else:
        return str(o)
    
# 将词典保存到JSON文件
def save_dict_to_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4,default=default_converter)

# 从JSON文件读取数据到词典
def read_dict_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

#读取本地历史对话
def load_history():
    loaded_data = read_dict_from_json('webui_pages/dialogue/history.json')  # 从JSON文件读取词典
    chat_box.from_dict(loaded_data)
    st.session_state.setdefault("conversation_ids", {})
    for key in loaded_data["histories"]:
        st.session_state["conversation_ids"][key] = uuid.uuid4().hex

#保存本地历史对话
def save_history():
    history_dict = chat_box.to_dict()
    save_dict_to_json('webui_pages/dialogue/history.json', history_dict)  # 保存词典到JSON文件


chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "robot.png"
    ),
    user_avatar=os.path.join(
        "img",
        "user.png"
    )
)


def get_messages_history(history_len: int, content_in_expander: bool = False) -> List[Dict]:
    '''
    返回消息历史。
    content_in_expander控制是否返回expander元素中的内容，一般导出的时候可以选上，传入LLM的history不需要
    '''

    def filter(msg):
        content = [x for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    return chat_box.filter_history(history_len=history_len, filter=filter)



def dialogue_page(api: ApiRequest, is_lite: bool = False):

    # 检查history文件是否存在
    if not os.path.exists('webui_pages/dialogue/history.json'):
        default_history_json()   
    if "conversation_ids" not in st.session_state:
        load_history()
        #初始化当前对话
        if 'cur_conv_name' not in st.session_state:
            st.session_state['cur_conv_name'] = chat_box._chat_name
    # st.session_state.setdefault("conversation_ids", {})
    # st.session_state["conversation_ids"].setdefault(chat_box.cur_chat_name, uuid.uuid4().hex)
    st.session_state.setdefault("file_chat_id", None)
    default_model = api.get_default_llm_model()[0]

    if not chat_box.chat_inited:
        st.toast(
            # f"欢迎使用 [`Langchain-Chatchat`](https://github.com/chatchat-space/Langchain-Chatchat) ! \n\n"
            f"欢迎使用 通用对话大模型 ! \n\n"
            f"当前运行的模型`{default_model}`, 您可以开始提问了."
        )
        chat_box.init_session()

    # 弹出自定义命令帮助信息
    modal = Modal("自定义命令", key="cmd_help", max_width="500")
    if modal.is_open():
        with modal.container():
            cmds = [x for x in parse_command.__doc__.split("\n") if x.strip().startswith("/")]
            st.write("\n\n".join(cmds))

    if 'new_history' not in st.session_state:
        st.session_state['new_history'] = 1
    st.session_state['new_history'] = 1

    with st.sidebar:
        # 初始化会话状态变量  
        if 'show_input' not in st.session_state:  
            st.session_state.show_input = False  
        if 'show_add_button' not in st.session_state:  
            st.session_state.show_add_button = True  
        if 'fuc_add' not in st.session_state:  
            st.session_state.fuc_add = False 
        if 'fuc_rename' not in st.session_state:  
            st.session_state.fuc_rename = False 
        if 'dialogue_mode' not in st.session_state:  
            st.session_state.dialogue_mode = "LLM 对话"

        

        st.write("")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("新建会话", type="primary"):
                name = "新会话"
                st.session_state["conversation_ids"] = {name: uuid.uuid4().hex, **st.session_state["conversation_ids"]}
                st.session_state["cur_conv_name"] = name
                save_history() 
                st.rerun() 
        with col2:
            if st.button("清空会话", type="primary"):
                chat_box.reset_history()
                st.session_state['new_history'] = 0
                st.rerun()
        with col3:
            if st.button("终止会话", type="primary"):
                save_history() 
                st.rerun()
                stop_project()
        st.write("")

        with st.expander("历史记录", expanded=True):   
            # 多会话
            conv_names = list(st.session_state["conversation_ids"].keys())
            index = 0
            if st.session_state.get("cur_conv_name") in conv_names:
                index = conv_names.index(st.session_state.get("cur_conv_name"))
            def format_option_full_width(option):
                st.session_state['new_history'] = 0
                return f'{option}'
            conversation_name = pills("请选择会话", conv_names, index=index, label_visibility="collapsed", format_func=format_option_full_width)
            st.session_state['new_history'] = 1
            chat_box.use_chat_name(conversation_name)
            conversation_id = st.session_state["conversation_ids"][conversation_name]
            st.session_state["cur_conv_name"] = conversation_name

            # 导出对话记录
            now = datetime.now()
            cols = st.columns([1, 1])
            export_btn = cols[0]
            if cols[1].button(
                '删除', 
                key="删除"
            ):
                parse_command(text='/del', modal=modal)
                st.session_state['new_history'] = 0
                st.rerun()

        export_btn.download_button(
            "导出",
            "".join(chat_box.export2md()),
            file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
            mime="text/markdown",
            use_container_width=True,
        )
####测试动态大纲
    # 如果 session_state 中没有 outline 状态，就初始化一个
    if 'outline_content' not in st.session_state:
        st.session_state.outline_content = ""
####测试动态大纲

    dialogue_mode = st.session_state.dialogue_mode
    index_prompt = {
        "LLM 对话": "llm_chat",
        "搜索": "llm_chat",
        "搜索+知识库": "llm_chat",
        "搜索+单角色": "llm_chat",
        "搜索+多角色": "llm_chat",
        "知识库+单角色": "llm_chat",
        "知识库+多角色": "llm_chat",
        "搜索+知识库+单角色": "llm_chat",
        "搜索+知识库+多角色": "llm_chat",
        "自定义Agent问答": "agent_chat",
        "搜索引擎问答": "search_engine_chat",
        "单角色": "knowledge_base_chat",
        "知识库": "knowledge_base_chat",
        "多角色": "knowledge_base_chat",           
        "文件对话": "knowledge_base_chat",
        "storm": "llm_chat",
        "google搜索问答": "llm_chat",
        "内网搜索问答": "llm_chat",
    }
    prompt_templates_kb_list = list(PROMPT_TEMPLATES[index_prompt[dialogue_mode]].keys())
    prompt_template_name = prompt_templates_kb_list[0]
    if "prompt_template_select" not in st.session_state:
        st.session_state.prompt_template_select = prompt_templates_kb_list[0]

    prompt_template_name = st.session_state.prompt_template_select

    temperature = 0.70
    history_len = 0

    score_threshold = 0.70

    

    if 'llm_model' not in st.session_state:  
        st.session_state.llm_model = "qwen2.5:32b"
    llm_model = st.session_state.llm_model

    if 'selected_kb' not in st.session_state:  
        st.session_state.selected_kb = DEFAULT_KNOWLEDGE_BASE
    selected_kb = st.session_state.selected_kb

    if 'selected_person_kb' not in st.session_state:  
        st.session_state.selected_person_kb = "毛泽东"
    selected_person_kb = st.session_state.selected_person_kb

    if 'kb_top_k' not in st.session_state:  
        st.session_state.kb_top_k = 3
    kb_top_k = st.session_state.kb_top_k

    if 'se_top_k' not in st.session_state:  
        st.session_state.se_top_k = 3
    se_top_k = st.session_state.se_top_k    

    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = DEFAULT_SEARCH_ENGINE
    search_engine = st.session_state.search_engine

    if 'selected_if_personas' not in st.session_state:
        personas_fuc = ["自动选取（职业库）","手动选取（名人库）","自定义角色"]
        st.session_state.selected_if_personas = personas_fuc[0]



    # Display chat messages from history on app rerun
    chat_box.output_messages()
    chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter。输入/help查看自定义命令 "

    def on_feedback(
            feedback,
            message_id: str = "",
            history_index: int = -1,
    ):
        reason = feedback["text"]
        score_int = chat_box.set_feedback(feedback=feedback, history_index=history_index)
        api.chat_feedback(message_id=message_id,
                          score=score_int,
                          reason=reason)
        st.session_state["need_rerun"] = True

    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "欢迎反馈您打分的理由",
    }

    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        if parse_command(text=prompt, modal=modal):  # 用户输入自定义命令
            st.rerun()
        else:
            history = get_messages_history(history_len)
            chat_box.user_say(prompt)
            
            

#原LLM对话            
            if dialogue_mode == "LLM 对话":
                chat_box.ai_say("正在思考...")
                text = ""
                message_id = ""
                
                
                r = api.chat_chat(prompt,
                                  history=history,
                                  conversation_id=conversation_id,
                                  model=llm_model,
                                  prompt_name=prompt_template_name,
                                  temperature=temperature)
                for t in r:
                    if error_msg := check_error_msg(t):  # check whether error occured
                        st.error(error_msg)
                        break
                    text += t.get("text", "")
                    chat_box.update_msg(text)
                    message_id = t.get("message_id", "")

                metadata = {
                    "message_id": message_id,
                }
                chat_box.update_msg(text, streaming=False, metadata=metadata)  # 更新最终的字符串，去除光标
                # chat_box.show_feedback(**feedback_kwargs,
                #                        key=message_id,
                #                        on_submit=on_feedback,
                #                        kwargs={"message_id": message_id, "history_index": len(chat_box.history) - 1})

            elif dialogue_mode == "搜索":
                chat_box.ai_say([
                    f"正在搜索... ",
                    Markdown("...", in_expander=True, title="搜索结果", state="complete"),
                ])
                
                source_documents,search_results = serper_search(query=prompt)
                search_results = cut_and_rerank(search_results,prompt,5)
                chat_box.update_msg("\n\n".join(source_documents), element_index=1, streaming=False)

                prompt_search = f"""请根据搜索结果回答问题。
                                    搜索结果：{search_results}
                                    问题：{prompt}
                                    """
                text = ""
                message_id = ""
                r = api.chat_chat(prompt_search,
                                  history=history,
                                  conversation_id=conversation_id,
                                  model=llm_model,
                                  prompt_name=prompt_template_name,
                                  temperature=temperature)
                for t in r:
                    if error_msg := check_error_msg(t):  # check whether error occured
                        st.error(error_msg)
                        break
                    text += t.get("text", "")
                    chat_box.update_msg(text, element_index=0)
                    message_id = t.get("message_id", "")

                metadata = {
                    "message_id": message_id,
                }
                chat_box.update_msg(text, element_index=0, streaming=False, metadata=metadata)  # 更新最终的字符串，去除光标

            elif dialogue_mode == "知识库":

                chat_box.ai_say([
                    f"正在思考... ",
                    Markdown("...", in_expander=True, title="知识扩展", state="complete"),
                ])

                #拆分子问题
                query = origin_kb_generate_sub_questions(api, prompt, llm_model)
                # chat_box.update_msg("**理解**：\n\n"+query, element_index=0, streaming=False)

                r = api.origin_kb_chat(
                                prompt,
                                query,
                                knowledge_base_name=selected_kb,
                                top_k=kb_top_k,
                                score_threshold=score_threshold,
                                history=history,
                                model=llm_model,
                                prompt_name="", #临时使用该变量，搜索结果
                                temperature=temperature)
                text = ""
                for d in r:
                    docs = d.get("docs", "")
                    if chunk := d.get("text"):
                        text += chunk
                        chat_box.update_msg(text, element_index=0)
                chat_box.update_msg(text, element_index=0, streaming=False)  # 更新最终的字符串，去除光标
                chat_box.update_msg("\n\n".join(docs), element_index=1, streaming=False)


            elif dialogue_mode == "单角色":

                chat_box.ai_say("正在思考...")

                _,personas_name,_, personas_resume = read_persona("./webui_pages/dialogue/personas_instance.xlsx")
                persona_resume = personas_resume[personas_name.index(selected_person_kb)]
                #拆分子问题
                query = generate_sub_questions(api, persona_resume, prompt, llm_model)
                # chat_box.update_msg("**理解**：\n\n"+query, streaming=False)

                r = api.jsby_kb_chat(persona_resume,
                                prompt,
                                query,
                                knowledge_base_name="",
                                top_k=kb_top_k,
                                score_threshold=score_threshold,
                                history=history,
                                model=llm_model,
                                prompt_name="",
                                temperature=temperature)
                text = ""
                for d in r:
                    docs = d.get("docs", "")
                    if chunk := d.get("text"):
                        text += chunk
                        chat_box.update_msg(text)
                chat_box.update_msg(text, streaming=False)  # 更新最终的字符串，去除光标


            elif dialogue_mode == "多角色":

                persona = [] #角色名称和角色描述
                persona_resume = [] #角色简历（详细描述）

                if st.session_state.selected_if_personas == "自动选取（职业库）":
                    query = ""
                    QUERY_PROMPT = """
                    你需要选择一组行业专家，他们将一起回答关于这个主题的一个问题。每个人代表与该主题相关的不同视角、角色或背景。
                    请从以下列表中选择出五个最适合回答该问题的角色：{personas}
                    示例格式：1. **大数据分析师**：...\n 2. **反认知战顾问**：...\n 3. **军事战略家**：...\n 4. **决策心理战顾问**：...\n 5. **情报分析师**：...\n
                    以下是我给出的主题： {topic}
                    """
                    personas,personas_name,personas_description, personas_resume = read_persona("./webui_pages/dialogue/personas_occupation.xlsx")
                    QUERY_PROMPT = QUERY_PROMPT.replace("{topic}",prompt).replace("{personas}",personas)
                    # print("QUERY_PROMPT:",QUERY_PROMPT)
                    r = api.chat_chat(QUERY_PROMPT,
                                    model=llm_model)
                    for t in r:
                        query += t.get("text", "")
                    # print("###################专家生成################:\n",query)
                    #三个不同角度的专家persona
                    print("query:",query)
                    for s in query.split('\n'):
                        match = re.search(r'\d+\.\s*(.*)', s)
                        if match:
                            persona.append(match.group(1).replace("**", ""))
                    persona = [item for item in persona if item.split("：")[0] in personas_name]

                    if len(persona) > 3:
                        persona = persona[:3]
                    for per in persona:
                        persona_resume.append(personas_resume[personas_name.index(per.split("：")[0])])
                elif st.session_state.selected_if_personas == "手动选取（名人库）":
                    _,personas_name,personas_description, personas_resume = read_persona("./webui_pages/dialogue/personas_instance.xlsx")
                    if 'selected_personas1' in st.session_state and st.session_state.selected_personas1:
                        persona.append(st.session_state.selected_personas1 + "：" + personas_description[personas_name.index(st.session_state.selected_personas1)])
                        persona_resume.append(personas_resume[personas_name.index(st.session_state.selected_personas1)])
                    if 'selected_personas2' in st.session_state and st.session_state.selected_personas2:
                        persona.append(st.session_state.selected_personas2 + "：" + personas_description[personas_name.index(st.session_state.selected_personas2)])
                        persona_resume.append(personas_resume[personas_name.index(st.session_state.selected_personas2)])
                    if 'selected_personas3' in st.session_state and st.session_state.selected_personas3:
                        persona.append(st.session_state.selected_personas3 + "：" + personas_description[personas_name.index(st.session_state.selected_personas3)])
                        persona_resume.append(personas_resume[personas_name.index(st.session_state.selected_personas3)])
                elif st.session_state.selected_if_personas == "自定义角色":
                    if 'user_input_name1' not in st.session_state:  
                        st.session_state.user_input_name1 = "1" 
                    if 'user_input_name2' not in st.session_state:  
                        st.session_state.user_input_name2 = "2"
                    if 'user_input_name3' not in st.session_state:  
                        st.session_state.user_input_name3 = "3"
                    if 'user_input_description1' not in st.session_state:  
                        st.session_state.user_input_description1 = "1" 
                    if 'user_input_description2' not in st.session_state:  
                        st.session_state.user_input_description2 = "2"
                    if 'user_input_description3' not in st.session_state:  
                        st.session_state.user_input_description3 = "3"
                    if st.session_state.user_input_name1:
                        persona.append(st.session_state.user_input_name1 + "：" + st.session_state.user_input_description1)
                        persona_resume.append(st.session_state.user_input_description1)
                    if st.session_state.user_input_name2:
                        persona.append(st.session_state.user_input_name2 + "：" + st.session_state.user_input_description2)
                        persona_resume.append(st.session_state.user_input_description2)
                    if st.session_state.user_input_name3:
                        persona.append(st.session_state.user_input_name3 + "：" + st.session_state.user_input_description3)
                        persona_resume.append(st.session_state.user_input_description3)
                
                #开始讨论
                search_results = ""
                #生成初始大纲
                query = f"""**任务**：你是一个资深的事件新闻报告专家，你的任务是根据议题生成一个思维导图，这个思维导图是用于指导多个不同专家和主持人对这个议题进行有深度和广度的讨论，有助于深层次多维度理解这个议题。你可以参考搜索信息来进行生成，如果搜索信息与议题相关性不高则不用参考。如果议题有提及到先验知识的内容，请参考先验知识回答。
                        **搜索信息**：{search_results}
                        **议题**：{prompt}
                        **格式**：1.使用“#标题”表示节标题，使用“##标题”表示小节标题，使用“###标题”表示子小节标题，依此类推。2.请勿包含其他信息。3.不要在大纲中包含主题名称本身。
                        **样例**：# 2024美国大选概述\n## 选情关键\n## 民意分析\n## 摇摆州情
                        **限制**：1.使用中文回答。生成的每个子议题是10个字以内。2.不需要生成具体内容，只需要生成各级标题即可。3.只需要生成一个一级标题和三到五个二级标题。
                        """
                text = ""
                r = api.chat_chat(query,
                                  model=llm_model)
                for t in r:
                    text += t.get("text", "")
                st.session_state.outline_content = text
                # Streamlit 侧边栏中展示大纲
                with st.sidebar:
                    # 创建一个按钮来更新大纲内容
                    with st.expander("思维导图", expanded=True):
                        outline_display = st.empty()  # 使用 empty 保持内容更新
                outline_display.markdown(st.session_state.outline_content)
                #历史发言初始化
                history = []
                #专家头像list
                experts_img = ["img/专家1.png","img/专家2.png","img/专家3.png"]
                for turn in range(3):
                    #主持人
                    #历史发言
                    history_content = ""
                    for index, item in enumerate(history):
                        if item['role'] == "moderator" and index > 0:
                            history_content += item['content'] + "\n"
                            
                    chat_box._assistant_avatar = "img/主持人2.png"
                    chat_box.ai_say("正在思考...")
                    chat_prompt = f"""**任务**： 
                                    1.现在你是一个资深的圆桌论坛主持人，你的任务是根据大纲来对专家们进行提问，你提出的问题不仅要有引导性，还要能够激发深入讨论，促进专家之间的互动与观点碰撞。例如，在探讨“数字转型对企业的影响”这一主题时，您可以这样提问：“在数字化转型的大潮中，不同企业所面临的挑战和机遇各有不同。请问各位嘉宾，您认为在这一过程中，有哪些关键因素是企业必须重视的？”，“同时，能否分享一下您所在企业在数字转型过程中遇到的最具挑战性的时刻，以及是如何克服这些挑战的？",“此外，数字转型对企业的组织结构、文化以及人才管理带来了哪些深远影响，您又是如何应对这些变化的？”等
                                    2.提问的议题从大纲中选取，并根据选取的议题生成问题
                                    3.如果议题有提及到先验知识的内容，请参考先验知识回答。
                                    **限制**：
                                    1.使用中文回答。你只需要提出一个问题即可。
                                    2.生成的回答直接是你的发言，不要生成其他无关信息。
                                    3.发言开头不要生成类似“主持人：”的文本。
                                    4.严格按照样例的格式生成。
                                    5.要回避已提问问题。
                                    **已提问问题**： 
                                    {history_content}
                                    **总议题**： 
                                    {prompt}
                                    **大纲**： 
                                    {st.session_state.outline_content}
                                    """
                    moderator_answer = "主持人："
                    r = api.chat_chat(chat_prompt,
                                    model=llm_model)
                    for t in r:
                        moderator_answer += t.get("text", "")
                        chat_box.update_msg(moderator_answer)
                    chat_box.update_msg(moderator_answer, streaming=False)  # 更新最终的字符串，去除光标
                    history.append({"role":"moderator","content": moderator_answer})
                    #专家发言
                    for i in range(len(persona)):
                        chat_box._assistant_avatar = experts_img[i]
                        chat_box.ai_say("正在思考...")
                        #历史发言模块（获取最新一轮中从主持人开始的发言）
                        history_content = "" 
                        for item in reversed(history):
                            if item['role'] == "moderator":
                                history_content = item['content'] + "\n" + history_content
                                break
                            else:
                                history_content = item['content'] + "\n" + history_content
                                
                        #search_results
                        search_results = ""
                        
                        chat_prompt = f"""**任务**： 1.现在你是{persona[i]}，请根据你的人物介绍和搜索信息，同时根据其他专家们的历史发言和总议题，以你在人物介绍中的相关立场、视角以及你擅长的领域知识，来回答问题，回答要口语化。
                                            2.你现在参加的会议中，专家以及发言顺序为：{'、'.join(persona)}；同时参考历史发言时，如果你认同前面其他人的观点时就要表示出赞成以及不发表重复意见，如果不认同就要表示出不认同以及进行反驳，如果有补充的就表示出补充观点。
                                            3.你的发言要和你在人物介绍中的相关立场强结合。
                                            4.如果议题有提及到先验知识的内容，请参考先验知识回答。
                                            **先验知识**： 
                                            **限制**： 1.使用中文回答，回答要口语化。你的发言要求是100个字以内。要简短精炼。
                                            2.生成的回答直接是你的发言，不要生成其他无关信息。
                                            3.模仿帖文例子中的口吻以第一人称进行回答。
                                            4.参考历史发言，其他人说过的话就不要再重复。
                                            5.内容中不要生成以“#”开始的标签或任何提示性语句。
                                            6.不要生成如“作为商务部长，”、“马斯克：”等类似开头
                                            **人物介绍**： {persona_resume[i]}
                                            **搜索信息**： {search_results}
                                            **历史发言**： {history_content}
                                            **问题**： {moderator_answer}
                                            """
                        expert_answer = f"{persona[i]}："
                        r = api.chat_chat(chat_prompt,
                                        model=llm_model)
                        for t in r:
                            expert_answer += t.get("text", "")
                            chat_box.update_msg(expert_answer)
                        chat_box.update_msg(expert_answer, streaming=False)  # 更新最终的字符串，去除光标
                        history.append({"role":f"{persona[i]}","content": expert_answer})
                    #动态更新大纲
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
                    **议题**：{prompt}
                    **大纲草案**：{st.session_state.outline_content}
                    **格式**：1.使用“#标题”表示一级标题，使用“##标题”表示二级标题，使用“###标题”表示三级标题，使用“####标题”表示四级标题，使用“#####标题”表示五级标题，使用“#####标题”表示六级标题，依此类推。2.请勿包含其他信息。3.不要在大纲中包含主题名称本身。
                    **限制**：1.使用中文回答。生成的每个子议题是10个字以内。2.不需要生成具体内容，只需要生成各级标题即可。3.每次要根据历史发言来进行新增子标题，新添加的子标题要和历史发言强相关。4.改进大纲时要采用广度优先遍历的方式来新添加子标题。5.广度优先遍历的方式：当二级标题够五个后，就要开始生成三级标题；当每个二级标题下的三级标题至少有两个后，就要开始生成四级标题；当每个三级标题下的四级标题至少有两个后，就要开始生成五级标题；依此类推。6.新添加的标题必须在同一个标题下。
                    """
                    new_outline = ""
                    r = api.chat_chat(query,
                                    model=llm_model)
                    for t in r:
                        new_outline += t.get("text", "")
                    # print("new_outline:",new_outline)
                    st.session_state.outline_content = new_outline
                    outline_display.markdown(st.session_state.outline_content)
                
                # persona.append("总体回答：结合前面几位专家的回答，来总结生成的最终回答") #生成总体回答
                
                # # 用来存储每个 tab 的占位符
                # placeholders = [None] * len(persona)
                # expanders = [[None] * 1 for _ in range(len(persona))]  # 每个tab中有4个expander
                # #总结回答
                # placeholders[-1] = st.empty()
                # # 创建三个标签页
                # tab_labels = [persona[i].split("：")[0] for i in range(len(persona)-1)]
                # tabs = st.tabs(tab_labels)

                
                # # 在每个 tab 中创建一个空的占位符，后续用于动态更新内容
                # for i, tab in enumerate(tabs):
                #     with tab:
                #         # st.info(persona[i].split("：")[1])
                #         placeholders[i] = st.empty()
                #         # 为每个 tab 创建 4 个 expander
                #         for j in range(1):
                #             expanders[i][j] = st.empty()  # 占位符用于动态添加每个 expander
                            

                # # 分别从三个角度调用 API，并在对应的 tab 下进行流式输出
                # personas_results = [""] * 3

                # start_gen(persona, api, persona_resume, prompt, 
                #           llm_model, "", kb_top_k, 
                #           score_threshold, history, prompt_template_name, 
                #           temperature, expanders, 
                #           placeholders, personas_results)


                # all_prompt = f"""你是一个世界知识专家，现在你需要总结以下几个专家的回答，生成原始问题最终的回答，回答要全面。原始问题：{prompt}/n"""
                # for index in range(len(persona[:-1])):
                #     all_prompt += f"专家{index+1}：{persona[index]} 专家{index+1}的回答：{personas_results[index]}/n"

                # r = api.chat_chat(all_prompt,
                #                 model=llm_model)
                # text = ""
                # for t in r:
                #     text += t.get("text", "")
                #     placeholders[-1].write(text)  # 更新对应 tab 的内容

            elif dialogue_mode == "搜索+知识库":
                chat_box.ai_say([
                    f"正在搜索... ",
                    Markdown("...", in_expander=True, title="搜索结果", state="complete"),
                    Markdown("...", in_expander=True, title="知识扩展", state="complete"),
                ])
                source_documents,search_results = serper_search(query=prompt)
                search_results = cut_and_rerank(search_results,prompt,5)
                chat_box.update_msg("\n\n".join(source_documents), element_index=1, streaming=False)

                #拆分子问题
                query = origin_kb_generate_sub_questions(api, prompt, llm_model)
                # chat_box.update_msg("**理解**：\n\n"+query, streaming=False)

                r = api.origin_kb_chat(
                                prompt,
                                query,
                                knowledge_base_name=selected_kb,
                                top_k=kb_top_k,
                                score_threshold=score_threshold,
                                history=history,
                                model=llm_model,
                                prompt_name=search_results, #临时使用该变量，搜索结果
                                temperature=temperature)
                text = ""
                for d in r:
                    docs = d.get("docs", "")
                    if chunk := d.get("text"):
                        text += chunk
                        chat_box.update_msg(text, element_index=0)
                chat_box.update_msg(text, element_index=0, streaming=False)  # 更新最终的字符串，去除光标                
                chat_box.update_msg("\n\n".join(docs), element_index=2, streaming=False)

            elif dialogue_mode == "搜索+单角色":
                chat_box.ai_say([
                    f"正在搜索... ",
                    Markdown("...", in_expander=True, title="搜索结果", state="complete"),
                ])
                source_documents,search_results = serper_search(query=prompt)
                search_results = cut_and_rerank(search_results,prompt,5)
                chat_box.update_msg("\n\n".join(source_documents), element_index=1, streaming=False)

                _,personas_name,_, personas_resume = read_persona("./webui_pages/dialogue/personas_instance.xlsx")
                persona_resume = personas_resume[personas_name.index(selected_person_kb)]
                #拆分子问题
                query = generate_sub_questions(api, persona_resume, prompt, llm_model)
                # chat_box.update_msg("**理解**：\n\n"+query, streaming=False)

                r = api.jsby_kb_chat(persona_resume,
                                prompt,
                                query,
                                knowledge_base_name="",
                                top_k=kb_top_k,
                                score_threshold=score_threshold,
                                history=history,
                                model=llm_model,
                                prompt_name=search_results,  #临时使用该变量，搜索结果
                                temperature=temperature)
                text = ""
                for d in r:
                    docs = d.get("docs", "")
                    if chunk := d.get("text"):
                        text += chunk
                        chat_box.update_msg(text, element_index=0)
                chat_box.update_msg(text, element_index=0, streaming=False)  # 更新最终的字符串，去除光标

            elif dialogue_mode == "搜索+多角色":

                source_documents,search_results = serper_search(query=prompt)
                search_results = cut_and_rerank(search_results,prompt,5)

                persona = [] #角色名称和角色描述
                persona_resume = [] #角色简历（详细描述）

                if st.session_state.selected_if_personas == "自动选取（职业库）":
                    query = ""
                    QUERY_PROMPT = """
                    你需要选择一组行业专家，他们将一起回答关于这个主题的一个问题。每个人代表与该主题相关的不同视角、角色或背景。
                    请从以下列表中选择出五个最适合回答该问题的角色：{personas}
                    格式：1. ...\n 2. ...\n 3. ...\n 4. ...\n 5. ...\n
                    以下是我给出的主题： {topic}
                    """
                    personas,personas_name,personas_description, personas_resume = read_persona("./webui_pages/dialogue/personas_occupation.xlsx")
                    QUERY_PROMPT = QUERY_PROMPT.replace("{topic}",prompt).replace("{personas}",personas)
                    # print("QUERY_PROMPT:",QUERY_PROMPT)
                    r = api.chat_chat(QUERY_PROMPT,
                                    model=llm_model)
                    for t in r:
                        query += t.get("text", "")
                    # print("###################专家生成################:\n",query)
                    #三个不同角度的专家persona
                    print("query:",query)
                    for s in query.split('\n'):
                        match = re.search(r'\d+\.\s*(.*)', s)
                        if match:
                            persona.append(match.group(1).replace("**", ""))
                    persona = [item for item in persona if item.split("：")[0] in personas_name]
                    if len(persona) > 3:
                        persona = persona[:3]
                    for per in persona:
                        persona_resume.append(personas_resume[personas_name.index(per.split("：")[0])])
                elif st.session_state.selected_if_personas == "手动选取（名人库）":
                    _,personas_name,personas_description, personas_resume = read_persona("./webui_pages/dialogue/personas_instance.xlsx")
                    if 'selected_personas1' in st.session_state and st.session_state.selected_personas1:
                        persona.append(st.session_state.selected_personas1 + "：" + personas_description[personas_name.index(st.session_state.selected_personas1)])
                        persona_resume.append(personas_resume[personas_name.index(st.session_state.selected_personas1)])
                    if 'selected_personas2' in st.session_state and st.session_state.selected_personas2:
                        persona.append(st.session_state.selected_personas2 + "：" + personas_description[personas_name.index(st.session_state.selected_personas2)])
                        persona_resume.append(personas_resume[personas_name.index(st.session_state.selected_personas2)])
                    if 'selected_personas3' in st.session_state and st.session_state.selected_personas3:
                        persona.append(st.session_state.selected_personas3 + "：" + personas_description[personas_name.index(st.session_state.selected_personas3)])
                        persona_resume.append(personas_resume[personas_name.index(st.session_state.selected_personas3)])
                elif st.session_state.selected_if_personas == "自定义角色":
                    if 'user_input_name1' not in st.session_state:  
                        st.session_state.user_input_name1 = "1" 
                    if 'user_input_name2' not in st.session_state:  
                        st.session_state.user_input_name2 = "2"
                    if 'user_input_name3' not in st.session_state:  
                        st.session_state.user_input_name3 = "3"
                    if 'user_input_description1' not in st.session_state:  
                        st.session_state.user_input_description1 = "1" 
                    if 'user_input_description2' not in st.session_state:  
                        st.session_state.user_input_description2 = "2"
                    if 'user_input_description3' not in st.session_state:  
                        st.session_state.user_input_description3 = "3"
                    if st.session_state.user_input_name1:
                        persona.append(st.session_state.user_input_name1 + "：" + st.session_state.user_input_description1)
                        persona_resume.append(st.session_state.user_input_description1)
                    if st.session_state.user_input_name2:
                        persona.append(st.session_state.user_input_name2 + "：" + st.session_state.user_input_description2)
                        persona_resume.append(st.session_state.user_input_description2)
                    if st.session_state.user_input_name3:
                        persona.append(st.session_state.user_input_name3 + "：" + st.session_state.user_input_description3)
                        persona_resume.append(st.session_state.user_input_description3)
                        
                #开始讨论
                source_documents,search_results = serper_search(query=prompt)
                search_results = cut_and_rerank(search_results,prompt,5)
                #生成初始大纲
                query = f"""**任务**：你是一个资深的事件新闻报告专家，你的任务是根据议题生成一个思维导图，这个思维导图是用于指导多个不同专家和主持人对这个议题进行有深度和广度的讨论，有助于深层次多维度理解这个议题。你可以参考搜索信息来进行生成，如果搜索信息与议题相关性不高则不用参考。如果议题有提及到先验知识的内容，请参考先验知识回答。
                        **搜索信息**：{search_results}
                        **议题**：{prompt}
                        **格式**：1.使用“#标题”表示节标题，使用“##标题”表示小节标题，使用“###标题”表示子小节标题，依此类推。2.请勿包含其他信息。3.不要在大纲中包含主题名称本身。
                        **样例**：# 2024美国大选概述\n## 选情关键\n## 民意分析\n## 摇摆州情
                        **限制**：1.使用中文回答。生成的每个子议题是10个字以内。2.不需要生成具体内容，只需要生成各级标题即可。3.只需要生成一个一级标题和三到五个二级标题。
                        """
                text = ""
                r = api.chat_chat(query,
                                  model=llm_model)
                for t in r:
                    text += t.get("text", "")
                st.session_state.outline_content = text
                # Streamlit 侧边栏中展示大纲
                with st.sidebar:
                    # 创建一个按钮来更新大纲内容
                    with st.expander("思维导图", expanded=True):
                        outline_display = st.empty()  # 使用 empty 保持内容更新
                outline_display.markdown(st.session_state.outline_content)
                #历史发言初始化
                history = []
                #专家头像list
                experts_img = ["img/专家1.png","img/专家2.png","img/专家3.png"]
                for turn in range(3):
                    #主持人
                    #历史发言
                    history_content = ""
                    for index, item in enumerate(history):
                        if item['role'] == "moderator" and index > 0:
                            history_content += item['content'] + "\n"
                            
                    chat_box._assistant_avatar = "img/主持人2.png"
                    chat_box.ai_say("正在思考...")
                    chat_prompt = f"""**任务**： 
                                    1.现在你是一个资深的圆桌论坛主持人，你的任务是根据大纲来对专家们进行提问，你提出的问题不仅要有引导性，还要能够激发深入讨论，促进专家之间的互动与观点碰撞。例如，在探讨“数字转型对企业的影响”这一主题时，您可以这样提问：“在数字化转型的大潮中，不同企业所面临的挑战和机遇各有不同。请问各位嘉宾，您认为在这一过程中，有哪些关键因素是企业必须重视的？”，“同时，能否分享一下您所在企业在数字转型过程中遇到的最具挑战性的时刻，以及是如何克服这些挑战的？",“此外，数字转型对企业的组织结构、文化以及人才管理带来了哪些深远影响，您又是如何应对这些变化的？”等
                                    2.提问的议题从大纲中选取，并根据选取的议题生成问题
                                    3.如果议题有提及到先验知识的内容，请参考先验知识回答。
                                    **限制**：
                                    1.使用中文回答。你只需要提出一个问题即可。
                                    2.生成的回答直接是你的发言，不要生成其他无关信息。
                                    3.发言开头不要生成类似“主持人：”的文本。
                                    4.严格按照样例的格式生成。
                                    5.要回避已提问问题。
                                    **已提问问题**： 
                                    {history_content}
                                    **总议题**： 
                                    {prompt}
                                    **大纲**： 
                                    {st.session_state.outline_content}
                                    """
                    moderator_answer = "主持人："
                    r = api.chat_chat(chat_prompt,
                                    model=llm_model)
                    for t in r:
                        moderator_answer += t.get("text", "")
                        chat_box.update_msg(moderator_answer)
                    chat_box.update_msg(moderator_answer, streaming=False)  # 更新最终的字符串，去除光标
                    history.append({"role":"moderator","content": moderator_answer})
                    #专家发言
                    for i in range(len(persona)):
                        chat_box._assistant_avatar = experts_img[i]
                        chat_box.ai_say("正在思考...")
                        #历史发言模块（获取最新一轮中从主持人开始的发言）
                        history_content = "" 
                        for item in reversed(history):
                            if item['role'] == "moderator":
                                history_content = item['content'] + "\n" + history_content
                                break
                            else:
                                history_content = item['content'] + "\n" + history_content
                                
                        #search_results
                        search_results = ""
                        
                        chat_prompt = f"""**任务**： 1.现在你是{persona[i]}，请根据你的人物介绍和搜索信息，同时根据其他专家们的历史发言和总议题，以你在人物介绍中的相关立场、视角以及你擅长的领域知识，来回答问题，回答要口语化。
                                            2.你现在参加的会议中，专家以及发言顺序为：{'、'.join(persona)}；同时参考历史发言时，如果你认同前面其他人的观点时就要表示出赞成以及不发表重复意见，如果不认同就要表示出不认同以及进行反驳，如果有补充的就表示出补充观点。
                                            3.你的发言要和你在人物介绍中的相关立场强结合。
                                            4.如果议题有提及到先验知识的内容，请参考先验知识回答。
                                            **先验知识**： 
                                            **限制**： 1.使用中文回答，回答要口语化。你的发言要求是100个字以内。要简短精炼。
                                            2.生成的回答直接是你的发言，不要生成其他无关信息。
                                            3.模仿帖文例子中的口吻以第一人称进行回答。
                                            4.参考历史发言，其他人说过的话就不要再重复。
                                            5.内容中不要生成以“#”开始的标签或任何提示性语句。
                                            6.不要生成如“作为商务部长，”、“马斯克：”等类似开头
                                            **人物介绍**： {persona_resume[i]}
                                            **搜索信息**： {search_results}
                                            **历史发言**： {history_content}
                                            **问题**： {moderator_answer}
                                            """
                        expert_answer = f"{persona[i]}："
                        r = api.chat_chat(chat_prompt,
                                        model=llm_model)
                        for t in r:
                            expert_answer += t.get("text", "")
                            chat_box.update_msg(expert_answer)
                        chat_box.update_msg(expert_answer, streaming=False)  # 更新最终的字符串，去除光标
                        history.append({"role":f"{persona[i]}","content": expert_answer})
                    #动态更新大纲
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
                    **议题**：{prompt}
                    **大纲草案**：{st.session_state.outline_content}
                    **格式**：1.使用“#标题”表示一级标题，使用“##标题”表示二级标题，使用“###标题”表示三级标题，使用“####标题”表示四级标题，使用“#####标题”表示五级标题，使用“#####标题”表示六级标题，依此类推。2.请勿包含其他信息。3.不要在大纲中包含主题名称本身。
                    **限制**：1.使用中文回答。生成的每个子议题是10个字以内。2.不需要生成具体内容，只需要生成各级标题即可。3.每次要根据历史发言来进行新增子标题，新添加的子标题要和历史发言强相关。4.改进大纲时要采用广度优先遍历的方式来新添加子标题。5.广度优先遍历的方式：当二级标题够五个后，就要开始生成三级标题；当每个二级标题下的三级标题至少有两个后，就要开始生成四级标题；当每个三级标题下的四级标题至少有两个后，就要开始生成五级标题；依此类推。6.新添加的标题必须在同一个标题下。
                    """
                    new_outline = ""
                    r = api.chat_chat(query,
                                    model=llm_model)
                    for t in r:
                        new_outline += t.get("text", "")
                    # print("new_outline:",new_outline)
                    st.session_state.outline_content = new_outline
                    outline_display.markdown(st.session_state.outline_content)
                        
                # persona.append("总体回答：结合前面几位专家的回答，来总结生成的最终回答") #生成总体回答


                # # 用来存储每个 tab 的占位符
                # placeholders = [None] * len(persona)
                # expanders = [[None] * 1 for _ in range(len(persona))]  # 每个tab中有4个expander
                # #总结回答
                # placeholders[-1] = st.empty()
                # # 创建三个标签页
                # tab_labels = [persona[i].split("：")[0] for i in range(len(persona)-1)]
                # tabs = st.tabs(tab_labels)

                # # 在每个 tab 中创建一个空的占位符，后续用于动态更新内容
                # for i, tab in enumerate(tabs):
                #     with tab:
                #         # st.info(persona[i].split("：")[1])
                #         placeholders[i] = st.empty()
                #         # 为每个 tab 创建 4 个 expander
                #         for j in range(1):
                #             expanders[i][j] = st.empty()  # 占位符用于动态添加每个 expander



                # # 分别从三个角度调用 API，并在对应的 tab 下进行流式输出
                # personas_results = [""] * 3

                # start_gen(persona, api, persona_resume, prompt, 
                #           llm_model, selected_kb, kb_top_k, 
                #           score_threshold, history, search_results,  #临时使用该变量，搜索结果 
                #           temperature, expanders, 
                #           placeholders, personas_results)


                # all_prompt = f"""你是一个世界知识专家，现在你需要总结以下几个专家的回答，生成原始问题最终的回答，回答要全面。原始问题：{prompt}/n"""
                # for index in range(len(persona[:-1])):
                #     all_prompt += f"专家{index+1}：{persona[index]} 专家{index+1}的回答：{personas_results[index]}/n"


                # r = api.chat_chat(all_prompt,
                #                 model=llm_model)
                # text = ""
                # for t in r:
                #     text += t.get("text", "")
                #     placeholders[-1].write(text)  # 更新对应 tab 的内容
        
            elif dialogue_mode == "知识库+单角色":

                chat_box.ai_say([
                    f"正在思考... ",
                    Markdown("...", in_expander=True, title="知识扩展", state="complete"),
                ])

                _,personas_name,_, personas_resume = read_persona("./webui_pages/dialogue/personas_instance.xlsx")
                persona_resume = personas_resume[personas_name.index(selected_person_kb)]
                #拆分子问题
                query = generate_sub_questions(api, persona_resume, prompt, llm_model)
                # chat_box.update_msg("**理解**：\n\n"+query, streaming=False)

                r = api.jsby_kb_chat(persona_resume,
                                prompt,
                                query,
                                knowledge_base_name=selected_kb,
                                top_k=kb_top_k,
                                score_threshold=score_threshold,
                                history=history,
                                model=llm_model,
                                prompt_name="",
                                temperature=temperature)
                text = ""
                for d in r:
                    docs = d.get("docs", "")
                    if chunk := d.get("text"):
                        text += chunk
                        chat_box.update_msg(text, element_index=0)
                chat_box.update_msg(text, element_index=0, streaming=False)  # 更新最终的字符串，去除光标                
                chat_box.update_msg("\n\n".join(docs), element_index=1, streaming=False)

            elif dialogue_mode == "知识库+多角色":

                persona = [] #角色名称和角色描述
                persona_resume = [] #角色简历（详细描述）

                if st.session_state.selected_if_personas == "自动选取（职业库）":
                    query = ""
                    QUERY_PROMPT = """
                    你需要选择一组行业专家，他们将一起回答关于这个主题的一个问题。每个人代表与该主题相关的不同视角、角色或背景。
                    请从以下列表中选择出五个最适合回答该问题的角色：{personas}
                    格式：1. ...\n 2. ...\n 3. ...\n 4. ...\n 5. ...\n
                    以下是我给出的主题： {topic}
                    """
                    personas,personas_name,personas_description, personas_resume = read_persona("./webui_pages/dialogue/personas_occupation.xlsx")
                    QUERY_PROMPT = QUERY_PROMPT.replace("{topic}",prompt).replace("{personas}",personas)
                    # print("QUERY_PROMPT:",QUERY_PROMPT)
                    r = api.chat_chat(QUERY_PROMPT,
                                    model=llm_model)
                    for t in r:
                        query += t.get("text", "")
                    # print("###################专家生成################:\n",query)
                    #三个不同角度的专家persona
                    print("query:",query)
                    for s in query.split('\n'):
                        match = re.search(r'\d+\.\s*(.*)', s)
                        if match:
                            persona.append(match.group(1).replace("**", ""))
                    persona = [item for item in persona if item.split("：")[0] in personas_name]
                    if len(persona) > 3:
                        persona = persona[:3]
                    for per in persona:
                        persona_resume.append(personas_resume[personas_name.index(per.split("：")[0])])
                elif st.session_state.selected_if_personas == "手动选取（名人库）":
                    _,personas_name,personas_description, personas_resume = read_persona("./webui_pages/dialogue/personas_instance.xlsx")
                    if 'selected_personas1' in st.session_state and st.session_state.selected_personas1:
                        persona.append(st.session_state.selected_personas1 + "：" + personas_description[personas_name.index(st.session_state.selected_personas1)])
                        persona_resume.append(personas_resume[personas_name.index(st.session_state.selected_personas1)])
                    if 'selected_personas2' in st.session_state and st.session_state.selected_personas2:
                        persona.append(st.session_state.selected_personas2 + "：" + personas_description[personas_name.index(st.session_state.selected_personas2)])
                        persona_resume.append(personas_resume[personas_name.index(st.session_state.selected_personas2)])
                    if 'selected_personas3' in st.session_state and st.session_state.selected_personas3:
                        persona.append(st.session_state.selected_personas3 + "：" + personas_description[personas_name.index(st.session_state.selected_personas3)])
                        persona_resume.append(personas_resume[personas_name.index(st.session_state.selected_personas3)])
                elif st.session_state.selected_if_personas == "自定义角色":
                    if 'user_input_name1' not in st.session_state:  
                        st.session_state.user_input_name1 = "1" 
                    if 'user_input_name2' not in st.session_state:  
                        st.session_state.user_input_name2 = "2"
                    if 'user_input_name3' not in st.session_state:  
                        st.session_state.user_input_name3 = "3"
                    if 'user_input_description1' not in st.session_state:  
                        st.session_state.user_input_description1 = "1" 
                    if 'user_input_description2' not in st.session_state:  
                        st.session_state.user_input_description2 = "2"
                    if 'user_input_description3' not in st.session_state:  
                        st.session_state.user_input_description3 = "3"
                    if st.session_state.user_input_name1:
                        persona.append(st.session_state.user_input_name1 + "：" + st.session_state.user_input_description1)
                        persona_resume.append(st.session_state.user_input_description1)
                    if st.session_state.user_input_name2:
                        persona.append(st.session_state.user_input_name2 + "：" + st.session_state.user_input_description2)
                        persona_resume.append(st.session_state.user_input_description2)
                    if st.session_state.user_input_name3:
                        persona.append(st.session_state.user_input_name3 + "：" + st.session_state.user_input_description3)
                        persona_resume.append(st.session_state.user_input_description3)
                
                #开始讨论
                search_results = ""
                #生成初始大纲
                query = f"""**任务**：你是一个资深的事件新闻报告专家，你的任务是根据议题生成一个思维导图，这个思维导图是用于指导多个不同专家和主持人对这个议题进行有深度和广度的讨论，有助于深层次多维度理解这个议题。你可以参考搜索信息来进行生成，如果搜索信息与议题相关性不高则不用参考。如果议题有提及到先验知识的内容，请参考先验知识回答。
                        **搜索信息**：{search_results}
                        **议题**：{prompt}
                        **格式**：1.使用“#标题”表示节标题，使用“##标题”表示小节标题，使用“###标题”表示子小节标题，依此类推。2.请勿包含其他信息。3.不要在大纲中包含主题名称本身。
                        **样例**：# 2024美国大选概述\n## 选情关键\n## 民意分析\n## 摇摆州情
                        **限制**：1.使用中文回答。生成的每个子议题是10个字以内。2.不需要生成具体内容，只需要生成各级标题即可。3.只需要生成一个一级标题和三到五个二级标题。
                        """
                text = ""
                r = api.chat_chat(query,
                                  model=llm_model)
                for t in r:
                    text += t.get("text", "")
                st.session_state.outline_content = text
                # Streamlit 侧边栏中展示大纲
                with st.sidebar:
                    # 创建一个按钮来更新大纲内容
                    with st.expander("思维导图", expanded=True):
                        outline_display = st.empty()  # 使用 empty 保持内容更新
                outline_display.markdown(st.session_state.outline_content)
                #历史发言初始化
                history = []
                #专家头像list
                experts_img = ["img/专家1.png","img/专家2.png","img/专家3.png"]
                for turn in range(3):
                    #主持人
                    #历史发言
                    history_content = ""
                    for index, item in enumerate(history):
                        if item['role'] == "moderator" and index > 0:
                            history_content += item['content'] + "\n"
                            
                    chat_box._assistant_avatar = "img/主持人2.png"
                    chat_box.ai_say("正在思考...")
                    chat_prompt = f"""**任务**： 
                                    1.现在你是一个资深的圆桌论坛主持人，你的任务是根据大纲来对专家们进行提问，你提出的问题不仅要有引导性，还要能够激发深入讨论，促进专家之间的互动与观点碰撞。例如，在探讨“数字转型对企业的影响”这一主题时，您可以这样提问：“在数字化转型的大潮中，不同企业所面临的挑战和机遇各有不同。请问各位嘉宾，您认为在这一过程中，有哪些关键因素是企业必须重视的？”，“同时，能否分享一下您所在企业在数字转型过程中遇到的最具挑战性的时刻，以及是如何克服这些挑战的？",“此外，数字转型对企业的组织结构、文化以及人才管理带来了哪些深远影响，您又是如何应对这些变化的？”等
                                    2.提问的议题从大纲中选取，并根据选取的议题生成问题
                                    3.如果议题有提及到先验知识的内容，请参考先验知识回答。
                                    **限制**：
                                    1.使用中文回答。你只需要提出一个问题即可。
                                    2.生成的回答直接是你的发言，不要生成其他无关信息。
                                    3.发言开头不要生成类似“主持人：”的文本。
                                    4.严格按照样例的格式生成。
                                    5.要回避已提问问题。
                                    **已提问问题**： 
                                    {history_content}
                                    **总议题**： 
                                    {prompt}
                                    **大纲**： 
                                    {st.session_state.outline_content}
                                    """
                    moderator_answer = "主持人："
                    r = api.chat_chat(chat_prompt,
                                    model=llm_model)
                    for t in r:
                        moderator_answer += t.get("text", "")
                        chat_box.update_msg(moderator_answer)
                    chat_box.update_msg(moderator_answer, streaming=False)  # 更新最终的字符串，去除光标
                    history.append({"role":"moderator","content": moderator_answer})
                    #专家发言
                    for i in range(len(persona)):
                        chat_box._assistant_avatar = experts_img[i]
                        chat_box.ai_say("正在思考...")
                        #历史发言模块（获取最新一轮中从主持人开始的发言）
                        history_content = "" 
                        for item in reversed(history):
                            if item['role'] == "moderator":
                                history_content = item['content'] + "\n" + history_content
                                break
                            else:
                                history_content = item['content'] + "\n" + history_content
                                
                        #search_results
                        search_results = ""
                        
                        chat_prompt = f"""**任务**： 1.现在你是{persona[i]}，请根据你的人物介绍和搜索信息，同时根据其他专家们的历史发言和总议题，以你在人物介绍中的相关立场、视角以及你擅长的领域知识，来回答问题，回答要口语化。
                                            2.你现在参加的会议中，专家以及发言顺序为：{'、'.join(persona)}；同时参考历史发言时，如果你认同前面其他人的观点时就要表示出赞成以及不发表重复意见，如果不认同就要表示出不认同以及进行反驳，如果有补充的就表示出补充观点。
                                            3.你的发言要和你在人物介绍中的相关立场强结合。
                                            4.如果议题有提及到先验知识的内容，请参考先验知识回答。
                                            **先验知识**： 
                                            **限制**： 1.使用中文回答，回答要口语化。你的发言要求是100个字以内。要简短精炼。
                                            2.生成的回答直接是你的发言，不要生成其他无关信息。
                                            3.模仿帖文例子中的口吻以第一人称进行回答。
                                            4.参考历史发言，其他人说过的话就不要再重复。
                                            5.内容中不要生成以“#”开始的标签或任何提示性语句。
                                            6.不要生成如“作为商务部长，”、“马斯克：”等类似开头
                                            **人物介绍**： {persona_resume[i]}
                                            **搜索信息**： {search_results}
                                            **历史发言**： {history_content}
                                            **问题**： {moderator_answer}
                                            """
                        expert_answer = f"{persona[i]}："
                        r = api.chat_chat(chat_prompt,
                                        model=llm_model)
                        for t in r:
                            expert_answer += t.get("text", "")
                            chat_box.update_msg(expert_answer)
                        chat_box.update_msg(expert_answer, streaming=False)  # 更新最终的字符串，去除光标
                        history.append({"role":f"{persona[i]}","content": expert_answer})
                    #动态更新大纲
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
                    **议题**：{prompt}
                    **大纲草案**：{st.session_state.outline_content}
                    **格式**：1.使用“#标题”表示一级标题，使用“##标题”表示二级标题，使用“###标题”表示三级标题，使用“####标题”表示四级标题，使用“#####标题”表示五级标题，使用“#####标题”表示六级标题，依此类推。2.请勿包含其他信息。3.不要在大纲中包含主题名称本身。
                    **限制**：1.使用中文回答。生成的每个子议题是10个字以内。2.不需要生成具体内容，只需要生成各级标题即可。3.每次要根据历史发言来进行新增子标题，新添加的子标题要和历史发言强相关。4.改进大纲时要采用广度优先遍历的方式来新添加子标题。5.广度优先遍历的方式：当二级标题够五个后，就要开始生成三级标题；当每个二级标题下的三级标题至少有两个后，就要开始生成四级标题；当每个三级标题下的四级标题至少有两个后，就要开始生成五级标题；依此类推。6.新添加的标题必须在同一个标题下。
                    """
                    new_outline = ""
                    r = api.chat_chat(query,
                                    model=llm_model)
                    for t in r:
                        new_outline += t.get("text", "")
                    # print("new_outline:",new_outline)
                    st.session_state.outline_content = new_outline
                    outline_display.markdown(st.session_state.outline_content)
                # persona.append("总体回答：结合前面几位专家的回答，来总结生成的最终回答") #生成总体回答


                # # 用来存储每个 tab 的占位符
                # placeholders = [None] * len(persona)
                # expanders = [[None] * 1 for _ in range(len(persona))]  # 每个tab中有4个expander
                # #总结回答
                # placeholders[-1] = st.empty()
                # # 创建三个标签页
                # tab_labels = [persona[i].split("：")[0] for i in range(len(persona)-1)]
                # tabs = st.tabs(tab_labels)

                # # 在每个 tab 中创建一个空的占位符，后续用于动态更新内容
                # for i, tab in enumerate(tabs):
                #     with tab:
                #         # st.info(persona[i].split("：")[1])
                #         placeholders[i] = st.empty()
                #         # 为每个 tab 创建 4 个 expander
                #         for j in range(1):
                #             expanders[i][j] = st.empty()  # 占位符用于动态添加每个 expander



                # # 分别从三个角度调用 API，并在对应的 tab 下进行流式输出
                # personas_results = [""] * 3
                # start_gen(persona, api, persona_resume, prompt, 
                #           llm_model, selected_kb, kb_top_k, 
                #           score_threshold, history, prompt_template_name, 
                #           temperature, expanders, 
                #           placeholders, personas_results)

                # all_prompt = f"""你是一个世界知识专家，现在你需要总结以下几个专家的回答，生成原始问题最终的回答，回答要全面。原始问题：{prompt}/n"""
                # for index in range(len(persona[:-1])):
                #     all_prompt += f"专家{index+1}：{persona[index]} 专家{index+1}的回答：{personas_results[index]}/n"


                # r = api.chat_chat(all_prompt,
                #                 model=llm_model)
                # text = ""
                # for t in r:
                #     text += t.get("text", "")
                #     placeholders[-1].write(text)  # 更新对应 tab 的内容


            elif dialogue_mode == "搜索+知识库+单角色":

                chat_box.ai_say([
                    f"正在搜索... ",
                    Markdown("...", in_expander=True, title="搜索结果", state="complete"),
                    Markdown("...", in_expander=True, title="知识扩展", state="complete"),
                ])
                source_documents,search_results = serper_search(query=prompt)
                search_results = cut_and_rerank(search_results,prompt,5)
                chat_box.update_msg("\n\n".join(source_documents), element_index=1, streaming=False)

                _,personas_name,_, personas_resume = read_persona("./webui_pages/dialogue/personas_instance.xlsx")
                persona_resume = personas_resume[personas_name.index(selected_person_kb)]
                #拆分子问题
                query = generate_sub_questions(api, persona_resume, prompt, llm_model)
                # chat_box.update_msg("**理解**：\n\n"+query, streaming=False)

                r = api.jsby_kb_chat(persona_resume,
                                prompt,
                                query,
                                knowledge_base_name=selected_kb,
                                top_k=kb_top_k,
                                score_threshold=score_threshold,
                                history=history,
                                model=llm_model,
                                prompt_name=search_results,   #临时使用该变量，搜索结果
                                temperature=temperature)
                text = ""
                for d in r:
                    docs = d.get("docs", "")
                    if chunk := d.get("text"):
                        text += chunk
                        chat_box.update_msg(text, element_index=0)
                chat_box.update_msg(text, element_index=0, streaming=False)  # 更新最终的字符串，去除光标                
                chat_box.update_msg("\n\n".join(docs), element_index=2, streaming=False)

            elif dialogue_mode == "搜索+知识库+多角色":

                persona = [] #角色名称和角色描述
                persona_resume = [] #角色简历（详细描述）

                if st.session_state.selected_if_personas == "自动选取（职业库）":
                    query = ""
                    QUERY_PROMPT = """
                    你需要选择一组行业专家，他们将一起回答关于这个主题的一个问题。每个人代表与该主题相关的不同视角、角色或背景。
                    请从以下列表中选择出五个最适合回答该问题的角色：{personas}
                    格式：1. ...\n 2. ...\n 3. ...\n 4. ...\n 5. ...\n
                    以下是我给出的主题： {topic}
                    """
                    personas,personas_name,personas_description, personas_resume = read_persona("./webui_pages/dialogue/personas_occupation.xlsx")
                    QUERY_PROMPT = QUERY_PROMPT.replace("{topic}",prompt).replace("{personas}",personas)
                    # print("QUERY_PROMPT:",QUERY_PROMPT)
                    r = api.chat_chat(QUERY_PROMPT,
                                    model=llm_model)
                    for t in r:
                        query += t.get("text", "")
                    # print("###################专家生成################:\n",query)
                    #三个不同角度的专家persona
                    print("query:",query)
                    for s in query.split('\n'):
                        match = re.search(r'\d+\.\s*(.*)', s)
                        if match:
                            persona.append(match.group(1).replace("**", ""))
                    persona = [item for item in persona if item.split("：")[0] in personas_name]
                    if len(persona) > 3:
                        persona = persona[:3]
                    for per in persona:
                        persona_resume.append(personas_resume[personas_name.index(per.split("：")[0])])
                elif st.session_state.selected_if_personas == "手动选取（名人库）":
                    _,personas_name,personas_description, personas_resume = read_persona("./webui_pages/dialogue/personas_instance.xlsx")
                    if 'selected_personas1' in st.session_state and st.session_state.selected_personas1:
                        persona.append(st.session_state.selected_personas1 + "：" + personas_description[personas_name.index(st.session_state.selected_personas1)])
                        persona_resume.append(personas_resume[personas_name.index(st.session_state.selected_personas1)])
                    if 'selected_personas2' in st.session_state and st.session_state.selected_personas2:
                        persona.append(st.session_state.selected_personas2 + "：" + personas_description[personas_name.index(st.session_state.selected_personas2)])
                        persona_resume.append(personas_resume[personas_name.index(st.session_state.selected_personas2)])
                    if 'selected_personas3' in st.session_state and st.session_state.selected_personas3:
                        persona.append(st.session_state.selected_personas3 + "：" + personas_description[personas_name.index(st.session_state.selected_personas3)])
                        persona_resume.append(personas_resume[personas_name.index(st.session_state.selected_personas3)])
                elif st.session_state.selected_if_personas == "自定义角色":
                    if 'user_input_name1' not in st.session_state:  
                        st.session_state.user_input_name1 = "1" 
                    if 'user_input_name2' not in st.session_state:  
                        st.session_state.user_input_name2 = "2"
                    if 'user_input_name3' not in st.session_state:  
                        st.session_state.user_input_name3 = "3"
                    if 'user_input_description1' not in st.session_state:  
                        st.session_state.user_input_description1 = "1" 
                    if 'user_input_description2' not in st.session_state:  
                        st.session_state.user_input_description2 = "2"
                    if 'user_input_description3' not in st.session_state:  
                        st.session_state.user_input_description3 = "3"
                    if st.session_state.user_input_name1:
                        persona.append(st.session_state.user_input_name1 + "：" + st.session_state.user_input_description1)
                        persona_resume.append(st.session_state.user_input_description1)
                    if st.session_state.user_input_name2:
                        persona.append(st.session_state.user_input_name2 + "：" + st.session_state.user_input_description2)
                        persona_resume.append(st.session_state.user_input_description2)
                    if st.session_state.user_input_name3:
                        persona.append(st.session_state.user_input_name3 + "：" + st.session_state.user_input_description3)
                        persona_resume.append(st.session_state.user_input_description3)
                        
                #开始讨论
                #搜索模块
                source_documents,search_results = serper_search(query=prompt)
                search_results = cut_and_rerank(search_results,prompt,5)
                #生成初始大纲
                query = f"""**任务**：你是一个资深的事件新闻报告专家，你的任务是根据议题生成一个思维导图，这个思维导图是用于指导多个不同专家和主持人对这个议题进行有深度和广度的讨论，有助于深层次多维度理解这个议题。你可以参考搜索信息来进行生成，如果搜索信息与议题相关性不高则不用参考。如果议题有提及到先验知识的内容，请参考先验知识回答。
                        **搜索信息**：{search_results}
                        **议题**：{prompt}
                        **格式**：1.使用“#标题”表示节标题，使用“##标题”表示小节标题，使用“###标题”表示子小节标题，依此类推。2.请勿包含其他信息。3.不要在大纲中包含主题名称本身。
                        **样例**：# 2024美国大选概述\n## 选情关键\n## 民意分析\n## 摇摆州情
                        **限制**：1.使用中文回答。生成的每个子议题是10个字以内。2.不需要生成具体内容，只需要生成各级标题即可。3.只需要生成一个一级标题和三到五个二级标题。
                        """
                text = ""
                r = api.chat_chat(query,
                                  model=llm_model)
                for t in r:
                    text += t.get("text", "")
                st.session_state.outline_content = text
                # Streamlit 侧边栏中展示大纲
                with st.sidebar:
                    # 创建一个按钮来更新大纲内容
                    with st.expander("思维导图", expanded=True):
                        outline_display = st.empty()  # 使用 empty 保持内容更新
                outline_display.markdown(st.session_state.outline_content)
                #历史发言初始化
                history = []
                #专家头像list
                experts_img = ["img/专家1.png","img/专家2.png","img/专家3.png"]
                for turn in range(3):
                    #主持人
                    #历史发言
                    history_content = ""
                    for index, item in enumerate(history):
                        if item['role'] == "moderator" and index > 0:
                            history_content += item['content'] + "\n"
                            
                    chat_box._assistant_avatar = "img/主持人2.png"
                    chat_box.ai_say("正在思考...")
                    chat_prompt = f"""**任务**： 
                                    1.现在你是一个资深的圆桌论坛主持人，你的任务是根据大纲来对专家们进行提问，你提出的问题不仅要有引导性，还要能够激发深入讨论，促进专家之间的互动与观点碰撞。例如，在探讨“数字转型对企业的影响”这一主题时，您可以这样提问：“在数字化转型的大潮中，不同企业所面临的挑战和机遇各有不同。请问各位嘉宾，您认为在这一过程中，有哪些关键因素是企业必须重视的？”，“同时，能否分享一下您所在企业在数字转型过程中遇到的最具挑战性的时刻，以及是如何克服这些挑战的？",“此外，数字转型对企业的组织结构、文化以及人才管理带来了哪些深远影响，您又是如何应对这些变化的？”等
                                    2.提问的议题从大纲中选取，并根据选取的议题生成问题
                                    3.如果议题有提及到先验知识的内容，请参考先验知识回答。
                                    **限制**：
                                    1.使用中文回答。你只需要提出一个问题即可。
                                    2.生成的回答直接是你的发言，不要生成其他无关信息。
                                    3.发言开头不要生成类似“主持人：”的文本。
                                    4.严格按照样例的格式生成。
                                    5.要回避已提问问题。
                                    **已提问问题**： 
                                    {history_content}
                                    **总议题**： 
                                    {prompt}
                                    **大纲**： 
                                    {st.session_state.outline_content}
                                    """
                    moderator_answer = "主持人："
                    r = api.chat_chat(chat_prompt,
                                    model=llm_model)
                    for t in r:
                        moderator_answer += t.get("text", "")
                        chat_box.update_msg(moderator_answer)
                    chat_box.update_msg(moderator_answer, streaming=False)  # 更新最终的字符串，去除光标
                    history.append({"role":"moderator","content": moderator_answer})
                    #专家发言
                    for i in range(len(persona)):
                        chat_box._assistant_avatar = experts_img[i]
                        chat_box.ai_say("正在思考...")
                        #历史发言模块（获取最新一轮中从主持人开始的发言）
                        history_content = "" 
                        for item in reversed(history):
                            if item['role'] == "moderator":
                                history_content = item['content'] + "\n" + history_content
                                break
                            else:
                                history_content = item['content'] + "\n" + history_content
                                
                        #search_results
                        search_results = ""
                        
                        chat_prompt = f"""**任务**： 1.现在你是{persona[i]}，请根据你的人物介绍和搜索信息，同时根据其他专家们的历史发言和总议题，以你在人物介绍中的相关立场、视角以及你擅长的领域知识，来回答问题，回答要口语化。
                                            2.你现在参加的会议中，专家以及发言顺序为：{'、'.join(persona)}；同时参考历史发言时，如果你认同前面其他人的观点时就要表示出赞成以及不发表重复意见，如果不认同就要表示出不认同以及进行反驳，如果有补充的就表示出补充观点。
                                            3.你的发言要和你在人物介绍中的相关立场强结合。
                                            4.如果议题有提及到先验知识的内容，请参考先验知识回答。
                                            **先验知识**： 
                                            **限制**： 1.使用中文回答，回答要口语化。你的发言要求是100个字以内。要简短精炼。
                                            2.生成的回答直接是你的发言，不要生成其他无关信息。
                                            3.模仿帖文例子中的口吻以第一人称进行回答。
                                            4.参考历史发言，其他人说过的话就不要再重复。
                                            5.内容中不要生成以“#”开始的标签或任何提示性语句。
                                            6.不要生成如“作为商务部长，”、“马斯克：”等类似开头
                                            **人物介绍**： {persona_resume[i]}
                                            **搜索信息**： {search_results}
                                            **历史发言**： {history_content}
                                            **问题**： {moderator_answer}
                                            """
                        expert_answer = f"{persona[i]}："
                        r = api.chat_chat(chat_prompt,
                                        model=llm_model)
                        for t in r:
                            expert_answer += t.get("text", "")
                            chat_box.update_msg(expert_answer)
                        chat_box.update_msg(expert_answer, streaming=False)  # 更新最终的字符串，去除光标
                        history.append({"role":f"{persona[i]}","content": expert_answer})
                    #动态更新大纲
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
                    **议题**：{prompt}
                    **大纲草案**：{st.session_state.outline_content}
                    **格式**：1.使用“#标题”表示一级标题，使用“##标题”表示二级标题，使用“###标题”表示三级标题，使用“####标题”表示四级标题，使用“#####标题”表示五级标题，使用“#####标题”表示六级标题，依此类推。2.请勿包含其他信息。3.不要在大纲中包含主题名称本身。
                    **限制**：1.使用中文回答。生成的每个子议题是10个字以内。2.不需要生成具体内容，只需要生成各级标题即可。3.每次要根据历史发言来进行新增子标题，新添加的子标题要和历史发言强相关。4.改进大纲时要采用广度优先遍历的方式来新添加子标题。5.广度优先遍历的方式：当二级标题够五个后，就要开始生成三级标题；当每个二级标题下的三级标题至少有两个后，就要开始生成四级标题；当每个三级标题下的四级标题至少有两个后，就要开始生成五级标题；依此类推。6.新添加的标题必须在同一个标题下。
                    """
                    new_outline = ""
                    r = api.chat_chat(query,
                                    model=llm_model)
                    for t in r:
                        new_outline += t.get("text", "")
                    # print("new_outline:",new_outline)
                    st.session_state.outline_content = new_outline
                    outline_display.markdown(st.session_state.outline_content)
                  
                # persona.append("总体回答：结合前面几位专家的回答，来总结生成的最终回答") #生成总体回答


                # # 用来存储每个 tab 的占位符
                # placeholders = [None] * len(persona)
                # expanders = [[None] * 1 for _ in range(len(persona))]  # 每个tab中有4个expander
                # #总结回答
                # placeholders[-1] = st.empty()
                # # 创建三个标签页
                # tab_labels = [persona[i].split("：")[0] for i in range(len(persona)-1)]
                # tabs = st.tabs(tab_labels)

                # # 用来存储每个 tab 的占位符
                # placeholders = [None] * len(persona)
                # expanders = [[None] * 5 for _ in range(len(persona))]  # 每个tab中有4个expander

                # # 在每个 tab 中创建一个空的占位符，后续用于动态更新内容
                # for i, tab in enumerate(tabs):
                #     with tab:
                #         # st.info(persona[i].split("：")[1])
                #         placeholders[i] = st.empty()
                #         # 为每个 tab 创建 4 个 expander
                #         for j in range(1):
                #             expanders[i][j] = st.empty()  # 占位符用于动态添加每个 expander



                # # 分别从三个角度调用 API，并在对应的 tab 下进行流式输出
                # personas_results = [""] * 3

                # start_gen(persona, api, persona_resume, prompt, 
                #           llm_model, selected_kb, kb_top_k, 
                #           score_threshold, history, search_results,  #临时使用该变量，搜索结果
                #           temperature, expanders, 
                #           placeholders, personas_results)

                # all_prompt = f"""你是一个世界知识专家，现在你需要总结以下几个专家的回答，生成原始问题最终的回答，回答要全面。原始问题：{prompt}/n"""
                # for index in range(len(persona[:-1])):
                #     all_prompt += f"专家{index+1}：{persona[index]} 专家{index+1}的回答：{personas_results[index]}/n"


                # r = api.chat_chat(all_prompt,
                #                 model=llm_model)
                # text = ""
                # for t in r:
                #     text += t.get("text", "")
                #     placeholders[-1].write(text)  # 更新对应 tab 的内容


            if st.session_state.cur_conv_name == "新会话":
                origin_name = st.session_state.cur_conv_name
                i = 0
                prompt = prompt[:15]
                prompt_new = prompt
                while prompt_new in st.session_state["conversation_ids"]:
                    i = i + 1
                    prompt_new = prompt + f'({i})'
                prompt = prompt_new
                # st.session_state["chat_history"][prompt] = st.session_state["chat_history"].pop(origin_name) #继承历史对话
                # st.session_state["conversation_ids"][prompt] = st.session_state["conversation_ids"].pop(origin_name)  #继承会话ID
                st.session_state["chat_history"] = {prompt: st.session_state["chat_history"].pop(origin_name), **st.session_state["chat_history"]}
                st.session_state["conversation_ids"] = {prompt: st.session_state["conversation_ids"].pop(origin_name), **st.session_state["conversation_ids"]}
                chat_box._chat_name = prompt
                chat_box.use_chat_name(prompt)
                st.session_state.cur_conv_name = prompt
                save_history()
                st.rerun()
    #保存到本地历史对话
    save_history()

    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()

def origin_kb_generate_sub_questions(api, prompt, llm_model):
    QUERY_PROMPT = """
    **任务**：你的具体任务是基于我给出的原始问题生成多个子问题，子问题通过对原始问题进行拆分，从而提高解决复杂问题的能力。
    **限制**：生成与原始问题有较高相关性的三个子问题。
    **请严格按照以下格式输出**：
    <question>子问题1</question>
    <question>子问题2</question>
    <question>子问题3</question>
    以下是我给出的原始问题：{原始问题}
    """
    query_prompt = QUERY_PROMPT.replace("{原始问题}", prompt)
    r = api.chat_chat(query_prompt, model=llm_model)

    query = ""
    for t in r:
        query += t.get("text", "")
    query = query.strip().split("\n")
    query1 = list(filter(lambda x: x != '', query))
    query2 = "\n\n".join([question.strip() for question in query1])
    pattern = r'<question>(.*?)[</question></s>]'
    query3 = re.findall(pattern, query2, re.MULTILINE)
    query = "\n\n".join([question.strip() for question in query3])
    
    if query == "":
        query = prompt
    return prompt + "\n\n" + query

def generate_sub_questions(api, persona_prompt, prompt, llm_model):
    QUERY_PROMPT = """
    **你的简历**：'''{角色及角度}'''
    **任务**：你的具体任务是基于我给出的原始问题生成多个子问题，子问题通过对原始问题进行拆分，从而提高解决复杂问题的能力。
    **限制**：生成与原始问题有较高相关性的三个子问题，你的回答是基于你的口吻以及政治立场的。
    **请严格按照以下格式输出**：
    <question>子问题1</question>
    <question>子问题2</question>
    <question>子问题3</question>
    以下是我给出的原始问题：{原始问题}
    """
    query_prompt = QUERY_PROMPT.replace("{角色及角度}", persona_prompt).replace("{原始问题}", prompt)
    r = api.chat_chat(query_prompt, model=llm_model)

    query = ""
    for t in r:
        query += t.get("text", "")
    query = query.strip().split("\n")
    query1 = list(filter(lambda x: x != '', query))
    query2 = "\n\n".join([question.strip() for question in query1])
    pattern = r'<question>(.*?)[</question></s>]'
    query3 = re.findall(pattern, query2, re.MULTILINE)
    query = "\n\n".join([question.strip() for question in query3])
    
    if query == "":
        query = prompt
    return prompt + "\n\n" + query



def process_question(api, i, persona_prompt, prompt, llm_model, selected_kb, kb_top_k, score_threshold, history, prompt_template_name, temperature):
    
    # 生成子问题
    start_time = time.time()
    query = generate_sub_questions(api, persona_prompt, prompt, llm_model)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"生成子问题，运行时间: {execution_time} 秒")

    # 知识库检索
    # docs, text, thinking_process, docs_context = knowledge_base_retrieval(api, persona_prompt, prompt, query, llm_model, selected_kb, kb_top_k, score_threshold, history, prompt_template_name, temperature)

    # 验证问题生成和修正回答
    # validation_text, final_answer_text = validate_and_correct_answer(api, persona_prompt, prompt, docs_context, text, llm_model)

    docs,person_source_documents, thinking_process, validation_text, final_answer_text = retrieval_and_answer(api, persona_prompt, prompt, query, llm_model, selected_kb, kb_top_k, score_threshold, history, prompt_template_name, temperature)

    # 返回结果到主线程
    return query, docs, person_source_documents, validation_text, final_answer_text, thinking_process, i

# 主线程中更新 Streamlit 界面
def start_gen(persona, api, personas, prompt, llm_model, selected_kb, kb_top_k, score_threshold, history, prompt_template_name, temperature, expanders, placeholders, personas_results):
    results = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, persona_prompt in enumerate(personas):
            future = executor.submit(process_question, api, i, persona_prompt, prompt, llm_model, selected_kb, kb_top_k, score_threshold, history, prompt_template_name, temperature)
            futures.append(future)

        # 主线程中等待所有任务完成
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                query, docs, person_source_documents, validation_text, final_answer_text, thinking_process, j = future.result()
                
                # 更新 Streamlit UI，必须在主线程中执行
                with expanders[j][0].expander(persona[j].split("：")[1]):
                    st.write(final_answer_text)

                # with expanders[j][0].expander("角色知识："):
                #     st.write("\n\n".join(person_source_documents))

                # with expanders[j][1].expander("拆分问题："):
                #     st.write("\n\n".join(query.strip().split("\n")))

                # with expanders[j][2].expander("相关文档："):
                #     st.write("\n\n".join(docs))
                
                # with expanders[j][3].expander("思考过程："):
                #     st.write(thinking_process)

                # with expanders[j][4].expander("验证问题："):
                #     st.write(validation_text)

                # placeholders[j].write(final_answer_text)

                personas_results[j] = final_answer_text

            except Exception as exc:
                print(f"Generated an exception: {exc}")

def retrieval_and_answer(api, persona_prompt, prompt, query, llm_model, selected_kb, kb_top_k, score_threshold, history, prompt_template_name, temperature):
    text = ""
    r = api.knowledge_base_chat(persona_prompt,
                                prompt,
                                query,
                                knowledge_base_name=selected_kb,
                                top_k=kb_top_k,
                                score_threshold=score_threshold,
                                history=history,
                                model=llm_model,
                                prompt_name=prompt_template_name,
                                temperature=temperature)
    for d in r:
        docs = d.get("docs", "")
        person_source_documents = d.get("person_source_documents", "")
        if error_msg := check_error_msg(d):  # check whether error occured
            return error_msg, docs
        elif chunk := d.get("answer"):
            text += chunk
    # if "### 思考过程 ###" in text and "### 回答初稿 ###" in text and "### 验证问题 ###" in text and "### 最终答案 ###" in text:
    #     split_thought_validation = text.split("### 回答初稿 ###")
    #     thinking_process = split_thought_validation[0].replace("### 思考过程 ###", "").strip()
    #     split_validation_final = split_thought_validation[1].split("### 验证问题 ###")
    #     final_text = split_validation_final[1].split("### 最终答案 ###")
    #     validation_text = final_text[0].strip()
    #     final_answer = final_text[1].strip()
    print("text:",text)
    if "### 思考过程 ###" in text and "### 验证问题 ###" in text and "### 最终答案 ###" in text:
        split_thought_validation = text.split("### 验证问题 ###")
        thinking_process = split_thought_validation[0].replace("### 思考过程 ###", "").strip()
        split_validation_final = split_thought_validation[1].split("### 最终答案 ###")
        validation_text = split_validation_final[0].strip()
        final_answer = split_validation_final[1].strip()
    else:
        thinking_process = text.strip()
        validation_text = text.strip()
        final_answer = text.strip()
    
    return docs, person_source_documents, thinking_process, validation_text, final_answer

