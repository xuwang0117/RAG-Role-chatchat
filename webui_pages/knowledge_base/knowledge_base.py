import streamlit as st
from webui_pages.utils import *
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
from server.knowledge_base.utils import get_file_path, LOADER_DICT
from server.knowledge_base.kb_service.base import get_kb_details, get_kb_file_details
from typing import Literal, Dict, Tuple
from configs import (kbs_config,
                     EMBEDDING_MODEL, DEFAULT_VS_TYPE, DEFAULT_KNOWLEDGE_BASE, DEFAULT_SEARCH_ENGINE,
                     CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE)
from server.utils import list_embed_models, list_online_embed_models
import os
import time
from webui_pages.dialogue.read_persona import read_persona
import json
from server.db.repository.knowledge_file_repository import file_exists_in_db
from server.knowledge_base.kb_service.base import KnowledgeFile

cell_renderer = JsCode("""function(params) {if(params.value==true){return '✓'}else{return '×'}}""")


def config_aggrid(
        df: pd.DataFrame,
        columns: Dict[Tuple[str, str], Dict] = {},
        selection_mode: Literal["single", "multiple", "disabled"] = "single",
        use_checkbox: bool = False,
) -> GridOptionsBuilder:
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column("No", width=40)
    for (col, header), kw in columns.items():
        gb.configure_column(col, header, wrapHeaderText=True, **kw)
    gb.configure_selection(
        selection_mode=selection_mode,
        use_checkbox=use_checkbox,
        pre_selected_rows=st.session_state.get("selected_rows", [0]),
    )
    gb.configure_pagination(
        enabled=True,
        paginationAutoPageSize=False,
        paginationPageSize=10
    )
    return gb


def file_exists(kb: str, selected_rows: List) -> Tuple[str, str]:
    """
    check whether a doc file exists in local knowledge base folder.
    return the file's name and path if it exists.
    """
    if selected_rows:
        file_name = selected_rows[0]["file_name"]
        file_path = get_file_path(kb, file_name)
        if os.path.isfile(file_path):
            return file_name, file_path
    return "", ""



@st.cache_data
def upload_temp_docs(files, _api: ApiRequest) -> str:
    '''
    将文件上传到临时目录，用于文件对话
    返回临时向量库ID
    '''
    return _api.upload_temp_docs(files).get("data", {}).get("id")



def knowledge_base_page(api: ApiRequest, is_lite: bool = None):

    with st.sidebar:
        # 初始化session_state，如果之前没有创建过的话
        if 'sou_choice' not in st.session_state:
            st.session_state.sou_choice = False
        if 'kb_choice' not in st.session_state:
            st.session_state.kb_choice = False
        if 'js_if_choice' not in st.session_state:
            st.session_state.js_if_choice = False
        if 'js_choice' not in st.session_state:
            st.session_state.js_choice = '单角色'

        def update_sou():
            st.session_state.sou_choice = not st.session_state.sou_choice
        def update_kb():
            st.session_state.kb_choice = not st.session_state.kb_choice
        def update_jsif():
            st.session_state.js_if_choice = not st.session_state.js_if_choice
        def update_js():
            if st.session_state.js_choice == "单角色":
                st.session_state.js_choice = "多角色"
            elif st.session_state.js_choice == "多角色":
                st.session_state.js_choice = "单角色"

        # 创建checkbox和radio选择框
        sou_choice = st.checkbox("搜索", value=st.session_state.sou_choice, on_change=update_sou)
        kb_choice = st.checkbox("知识库", value=st.session_state.kb_choice, on_change=update_kb)
        js_if_choice = st.checkbox("角色", value=st.session_state.js_if_choice, on_change=update_jsif)

        # 如果js_if_choice被选中，则显示radio选择框
        if js_if_choice:
            js_choice = st.radio("", ['单角色', '多角色'], index=['单角色', '多角色'].index(st.session_state.js_choice), on_change=update_js)


        def on_llm_change():
            if llm_model_a:
                config = api.get_model_config(llm_model_a)
                if not config.get("online_api"):  # 只有本地model_worker可以切换模型
                    st.session_state["prev_llm_model"] = llm_model_a
                st.session_state["cur_llm_model"] = st.session_state.llm_model

        def llm_model_format_func(x):
            if x in running_models:
                return f"{x} (Running)"
            return x

        default_model = api.get_default_llm_model()[0]
        running_models = list(api.list_running_models())
        available_models = []
        config_models = api.list_config_models()
        if not is_lite:
            for k, v in config_models.get("local", {}).items():
                if (v.get("model_path_exists")
                        and k not in running_models):
                    available_models.append(k)
        for k, v in config_models.get("online", {}).items():
            if not v.get("provider") and k not in running_models and k in LLM_MODELS:
                available_models.append(k)
        llm_models = running_models + available_models
        cur_llm_model = st.session_state.get("cur_llm_model", default_model)
        if cur_llm_model in llm_models:
            index = llm_models.index(cur_llm_model)
        else:
            index = 0
        llm_model_a = st.selectbox("选择LLM模型：",
                                 llm_models,
                                 index,
                                 format_func=llm_model_format_func,
                                 on_change=on_llm_change,
                                 key="llm_model_a",
                                 )
        if (st.session_state.get("prev_llm_model") != llm_model_a
                and not is_lite
                and not llm_model_a in config_models.get("online", {})
                and not llm_model_a in config_models.get("langchain", {})
                and llm_model_a not in running_models):
            with st.spinner(f"正在加载模型： {llm_model_a}，请勿进行操作或刷新页面"):
                prev_model = st.session_state.get("prev_llm_model")
                r = api.change_llm_model(prev_model, llm_model_a)
                if msg := check_error_msg(r):
                    st.error(msg)
                elif msg := check_success_msg(r):
                    st.success(msg)
                    st.session_state["prev_llm_model"] = llm_model_a
        st.session_state.llm_model = llm_model_a



        def on_mode_change():
            mode = st.session_state.dialogue_mode
            text = f"已切换到 {mode} 模式。"
            if mode == "知识库" or mode == "多角色":
                cur_kb = st.session_state.get("selected_kb")
                if cur_kb:
                    text = f"{text} 当前知识库： `{cur_kb}`。"
            st.toast(text)

        dialogue_modes = ["LLM 对话",
                            "搜索",
                            "知识库",
                            "单角色",
                            "多角色",
                            "搜索+知识库",
                            "搜索+单角色",
                            "搜索+多角色",
                            "知识库+单角色",
                            "知识库+多角色",
                            "搜索+知识库+单角色",
                            "搜索+知识库+多角色",
                            ]
        # if "dialogue_mode" in st.session_state:
        #     id = dialogue_modes.index(st.session_state.dialogue_mode)
        # else:
        #     id = 0

        # dialogue_mode_a = st.selectbox("请选择对话模式：",
        #                                 dialogue_modes,
        #                                 index=id,
        #                                 on_change=on_mode_change,
        #                                 key="dialogue_mode_a",
        #                                 )
        dialogue_mode_a = "LLM 对话"
        if sou_choice and kb_choice and js_if_choice:
            if js_choice == "单角色":
                dialogue_mode_a = "搜索+知识库+单角色"
            elif js_choice == "多角色":
                dialogue_mode_a = "搜索+知识库+多角色"
        elif sou_choice and kb_choice:
            dialogue_mode_a = "搜索+知识库"
        elif sou_choice and js_if_choice:
            if js_choice == "单角色":
                dialogue_mode_a = "搜索+单角色"
            elif js_choice == "多角色":
                dialogue_mode_a = "搜索+多角色"
        elif kb_choice and js_if_choice:
            if js_choice == "单角色":
                dialogue_mode_a = "知识库+单角色"
            elif js_choice == "多角色":
                dialogue_mode_a = "知识库+多角色"
        elif sou_choice:
            dialogue_mode_a = "搜索"
        elif kb_choice:
            dialogue_mode_a = "知识库"
        elif js_if_choice:
            if js_choice == "单角色":
                dialogue_mode_a = "单角色"
            elif js_choice == "多角色":
                dialogue_mode_a = "多角色"

        st.write("对话模式：",dialogue_mode_a)
        st.session_state.dialogue_mode = dialogue_mode_a
        dialogue_mode = dialogue_mode_a



        def on_kb_change():
            st.toast(f"已加载知识库： {st.session_state.selected_kb}")
        def on_personas_change1():
            st.toast(f"已选择定制化角色： {st.session_state.selected_personas1}")
        def on_personas_change2():
            st.toast(f"已选择定制化角色： {st.session_state.selected_personas2}")
        def on_personas_change3():
            st.toast(f"已选择定制化角色： {st.session_state.selected_personas3}")
        def on_if_personas_change():
            st.toast(f"已选择生成角色方式： {st.session_state.selected_if_personas}")

        if "知识库" in dialogue_mode:
            with st.expander("知识库配置", True):
                kb_list = api.list_knowledge_bases()
                # index = kb_list.index('Renzhi-Dongtai-MIX')
                if DEFAULT_KNOWLEDGE_BASE in kb_list:
                    index = kb_list.index(DEFAULT_KNOWLEDGE_BASE)
                selected_kb_a = st.selectbox(
                    "请选择知识库：",
                    kb_list,
                    index=index,
                    on_change=on_kb_change,
                    key="selected_kb_a",
                )
                st.session_state.selected_kb = selected_kb_a
                # kb_top_k_a = st.number_input("匹配知识条数：", 1, 20, VECTOR_SEARCH_TOP_K)
                kb_top_k_a = 3
                st.session_state.kb_top_k = kb_top_k_a
                # ## Bge 模型会超过1
                # score_threshold = st.slider("知识匹配分数阈值：", 0.0, 2.0, float(SCORE_THRESHOLD), 0.01)
                # st.session_state.score_threshold = score_threshold
                # score_threshold = 0.70
        if "单角色" in dialogue_mode:
            _,person_name,_,_ = read_persona("./webui_pages/dialogue/personas_instance.xlsx")
            selected_kb_b = st.selectbox(
                    "请选择角色：",
                    person_name,
                    index=person_name.index("毛泽东"),
                    on_change=on_kb_change,
                    key="selected_kb_b",
                )
            st.session_state.selected_person_kb = selected_kb_b
        if "多角色" in dialogue_mode:
            with st.expander("多角色配置", True):
                personas_fuc = ["自动选取（职业库）","手动选取（名人库）","自定义角色"]
                selected_if_personas_a = st.selectbox(
                    "请选择角色生成方式：",
                    personas_fuc,
                    index=0,
                    on_change=on_if_personas_change,
                    key="selected_if_personas_a",
                )
                st.session_state.selected_if_personas = selected_if_personas_a
                if st.session_state.selected_if_personas == "手动选取（名人库）":
                    _,person_name,_,_ = read_persona("./webui_pages/dialogue/personas_instance.xlsx")
                    selected_personas1_a = st.selectbox(
                        "请选择定制化角色1：",
                        person_name,
                        index=None,
                        on_change=on_personas_change1,
                        key="selected_personas1_a",
                    )
                    st.session_state.selected_personas1 = selected_personas1_a
                    if selected_personas1_a:#条件阶梯式选择框
                        selected_personas2_a = st.selectbox(
                        "请选择定制化角色2：",
                        person_name,
                        index=None,
                        on_change=on_personas_change2,
                        key="selected_personas2_a",
                    )
                        st.session_state.selected_personas2 = selected_personas2_a
                        if selected_personas2_a:#条件阶梯式选择框
                            selected_personas3_a = st.selectbox(
                            "请选择定制化角色3：",
                            person_name,
                            index=None,
                            on_change=on_personas_change3,
                            key="selected_personas3_a",
                        )
                            st.session_state.selected_personas3 = selected_personas3_a
                elif st.session_state.selected_if_personas == "自定义角色":
                    if 'create1' not in st.session_state:  
                        st.session_state.create1 = False 
                    if 'create2' not in st.session_state:  
                        st.session_state.create2 = False 
                    user_input_name1_a = st.text_input("角色1名称:")
                    user_input_description1_a = st.text_input("角色1描述或简历:")
                    if st.button("创建1"):
                        ###角色1
                        if user_input_name1_a and user_input_description1_a:
                            st.success(f"角色1{user_input_name1_a}已创建")
                            st.session_state.create1=True
                            st.session_state.user_input_name1 = user_input_name1_a
                            st.session_state.user_input_description1 = user_input_description1_a
                        else:
                            st.error(f"角色1名称或描述为空，角色1创建失败")
                    if st.session_state.create1:
                        ###角色2
                        user_input_name2_a = st.text_input("角色2名称:")
                        user_input_description2_a = st.text_input("角色2描述或简历:")
                        if st.button("创建2"):
                            if user_input_name2_a and user_input_description2_a:
                                st.success(f"角色2{user_input_name2_a}已创建")
                                st.session_state.create2=True
                                st.session_state.user_input_name2 = user_input_name2_a
                                st.session_state.user_input_description2 = user_input_description2_a
                            else:
                                st.error(f"角色2名称或描述为空，角色2创建失败")
                    if st.session_state.create1 and st.session_state.create2:
                        ###角色3
                        user_input_name3_a = st.text_input("角色3名称:")
                        user_input_description3_a = st.text_input("角色3描述或简历:")
                        if st.button("创建3"):
                            if user_input_name3_a and user_input_description3_a:
                                st.success(f"角色3{user_input_name3_a}已创建")
                                st.session_state.user_input_name3 = user_input_name3_a
                                st.session_state.user_input_description3 = user_input_description3_a
                            else:
                                st.error(f"角色3名称或描述为空，角色3创建失败")



    try:
        kb_list = {x["kb_name"]: x for x in get_kb_details()}
    except Exception as e:
        st.error(
            "获取知识库信息错误，请检查是否已按照 `README.md` 中 `4 知识库初始化与迁移` 步骤完成初始化或迁移，或是否为数据库连接错误。")
        st.stop()
    kb_names = list(kb_list.keys())

    if "selected_kb_name" in st.session_state and st.session_state["selected_kb_name"] in kb_names:
        selected_kb_index = kb_names.index(st.session_state["selected_kb_name"])
    else:
        selected_kb_index = 0

    if "selected_kb_info" not in st.session_state:
        st.session_state["selected_kb_info"] = ""

    def format_selected_kb(kb_name: str) -> str:
        if kb := kb_list.get(kb_name):
            return f"{kb_name} ({kb['vs_type']} @ {kb['embed_model']})"
        else:
            return kb_name

    selected_kb = st.selectbox(
        "请选择或新建知识库：",
        kb_names + ["新建知识库"],
        format_func=format_selected_kb,
        index=selected_kb_index
    )

    if selected_kb == "新建知识库":
        with st.form("新建知识库"):

            kb_name = st.text_input(
                "新建知识库名称",
                placeholder="新知识库名称，不支持中文命名",
                key="kb_name",
            )
            kb_info = st.text_input(
                "知识库简介",
                placeholder="知识库简介，方便Agent查找",
                key="kb_info",
            )

            cols = st.columns(2)

            vs_types = list(kbs_config.keys())
            vs_type = cols[0].selectbox(
                "向量库类型",
                vs_types,
                index=vs_types.index(DEFAULT_VS_TYPE),
                key="vs_type",
            )

            if is_lite:
                embed_models = list_online_embed_models()
                if EMBEDDING_MODEL not in embed_models:
                    embed_models.append(EMBEDDING_MODEL)
            else:
                embed_models = list_embed_models() + list_online_embed_models()

            embed_model = cols[1].selectbox(
                "Embedding 模型",
                embed_models,
                index=embed_models.index(EMBEDDING_MODEL),
                key="embed_model",
            )

            submit_create_kb = st.form_submit_button(
                "新建",
                # disabled=not bool(kb_name),
                use_container_width=True,
            )

        if submit_create_kb:
            if not kb_name or not kb_name.strip():
                st.error(f"知识库名称不能为空！")
            elif kb_name in kb_list:
                st.error(f"名为 {kb_name} 的知识库已经存在！")
            else:
                ret = api.create_knowledge_base(
                    knowledge_base_name=kb_name,
                    vector_store_type=vs_type,
                    embed_model=embed_model,
                )
                st.toast(ret.get("msg", " "))
                st.session_state["selected_kb_name"] = kb_name
                st.session_state["selected_kb_info"] = kb_info
                st.rerun()

    elif selected_kb:
        kb = selected_kb
        st.session_state["selected_kb_info"] = kb_list[kb]['kb_info']
        # 上传文件
        files = st.file_uploader("上传知识文件：",
                                 [i for ls in LOADER_DICT.values() for i in ls],
                                 accept_multiple_files=True,
                                 )
        kb_info = st.text_area("请输入知识库介绍:", value=st.session_state["selected_kb_info"], max_chars=None,
                               key=None,
                               help=None, on_change=None, args=None, kwargs=None)

        if kb_info != st.session_state["selected_kb_info"]:
            st.session_state["selected_kb_info"] = kb_info
            api.update_kb_info(kb, kb_info)

        # with st.sidebar:
        with st.expander(
                "文件处理配置",
                expanded=True,
        ):
            cols = st.columns(3)
            chunk_size = cols[0].number_input("单段文本最大长度：", 1, 1000, CHUNK_SIZE)
            chunk_overlap = cols[1].number_input("相邻文本重合长度：", 0, chunk_size, OVERLAP_SIZE)
            cols[2].write("")
            cols[2].write("")
            zh_title_enhance = cols[2].checkbox("开启中文标题加强", ZH_TITLE_ENHANCE)

        if st.button(
                "添加文件到知识库",
                # use_container_width=True,
                disabled=len(files) == 0,
        ):
            ret = api.upload_kb_docs(files,
                                     knowledge_base_name=kb,
                                     override=True,
                                     chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap,
                                     zh_title_enhance=zh_title_enhance)
            if msg := check_success_msg(ret):
                st.toast(msg, icon="✔")
            elif msg := check_error_msg(ret):
                st.toast(msg, icon="✖")

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
            file_names = [filename for filename, file in files]
            api.update_kb_docs(kb,
                                   file_names=file_names,
                                   chunk_size=chunk_size,
                                   chunk_overlap=chunk_overlap,
                                   zh_title_enhance=zh_title_enhance)
        st.divider()

        # 知识库详情
        # st.info("请选择文件，点击按钮进行操作。")
        doc_details = pd.DataFrame(get_kb_file_details(kb))
        selected_rows = []
        if not len(doc_details):
            st.info(f"知识库 `{kb}` 中暂无文件")
        else:
            st.write(f"知识库 `{kb}` 中已有文件:")
            st.info("知识库中包含源文件与向量库，请从下表中选择文件后操作")
            doc_details.drop(columns=["kb_name"], inplace=True)
            doc_details = doc_details[[
                "No", "file_name", "document_loader", "text_splitter", "docs_count", "in_folder", "in_db",
            ]]
            doc_details["in_folder"] = doc_details["in_folder"].replace(True, "✓").replace(False, "×")
            doc_details["in_db"] = doc_details["in_db"].replace(True, "✓").replace(False, "×")
            gb = config_aggrid(
                doc_details,
                {
                    ("No", "序号"): {},
                    ("file_name", "文档名称"): {},
                    # ("file_ext", "文档类型"): {},
                    # ("file_version", "文档版本"): {},
                    ("document_loader", "文档加载器"): {},
                    ("docs_count", "文档数量"): {},
                    ("text_splitter", "分词器"): {},
                    # ("create_time", "创建时间"): {},
                    ("in_folder", "源文件"): {"cellRenderer": cell_renderer},
                    ("in_db", "向量库"): {"cellRenderer": cell_renderer},
                },
                "multiple",
            )

            doc_grid = AgGrid(
                doc_details,
                gb.build(),
                columns_auto_size_mode="FIT_CONTENTS",
                theme="alpine",
                custom_css={
                    "#gridToolBar": {"display": "none"},
                },
                allow_unsafe_jscode=True,
                enable_enterprise_modules=False
            )

            selected_rows = doc_grid.get("selected_rows", [])

            cols = st.columns(4)
            file_name, file_path = file_exists(kb, selected_rows)
            if file_path:
                with open(file_path, "rb") as fp:
                    cols[0].download_button(
                        "下载选中文档",
                        fp,
                        file_name=file_name,
                        use_container_width=True, )
            else:
                cols[0].download_button(
                    "下载选中文档",
                    "",
                    disabled=True,
                    use_container_width=True, )

            st.write()
            # 将文件分词并加载到向量库中
            if cols[1].button(
                    "重新添加至向量库" if selected_rows and (
                            pd.DataFrame(selected_rows)["in_db"]).any() else "添加至向量库",
                    disabled=not file_exists(kb, selected_rows)[0],
                    use_container_width=True,
            ):
                file_names = [row["file_name"] for row in selected_rows]
                api.update_kb_docs(kb,
                                   file_names=file_names,
                                   chunk_size=chunk_size,
                                   chunk_overlap=chunk_overlap,
                                   zh_title_enhance=zh_title_enhance)
                st.rerun()

            # 将文件从向量库中删除，但不删除文件本身。
            if cols[2].button(
                    "从向量库删除",
                    disabled=not (selected_rows and selected_rows[0]["in_db"]),
                    use_container_width=True,
            ):
                file_names = [row["file_name"] for row in selected_rows]
                api.delete_kb_docs(kb, file_names=file_names)
                st.rerun()

            if cols[3].button(
                    "从知识库中删除",
                    type="primary",
                    use_container_width=True,
            ):
                file_names = [row["file_name"] for row in selected_rows]
                api.delete_kb_docs(kb, file_names=file_names, delete_content=True)
                st.rerun()

        st.divider()

        cols = st.columns(3)

        if cols[0].button(
                "依据源文件重建向量库",
                help="无需上传文件，通过其它方式将文档拷贝到对应知识库content目录下，点击本按钮即可重建知识库。",
                use_container_width=True,
                type="primary",
        ):
            with st.spinner("向量库重构中，请耐心等待，勿刷新或关闭页面。"):
                empty = st.empty()
                empty.progress(0.0, "")
                for d in api.recreate_vector_store(kb,
                                                   chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   zh_title_enhance=zh_title_enhance):
                    if msg := check_error_msg(d):
                        st.toast(msg)
                    else:
                        empty.progress(d["finished"] / d["total"], d["msg"])
                st.rerun()

        if cols[2].button(
                "删除知识库",
                use_container_width=True,
        ):
            ret = api.delete_knowledge_base(kb)
            st.toast(ret.get("msg", " "))
            time.sleep(1)
            st.rerun()

        

        st.write("文件内文档列表。双击进行修改，在删除列填入 Y 可删除对应行。")
        docs = []
        df = pd.DataFrame([], columns=["seq", "id", "content", "source"])
        if selected_rows:
            file_name = selected_rows[0]["file_name"]
            docs = api.search_kb_docs(knowledge_base_name=selected_kb, file_name=file_name)
            data = []
            data = [
                {"seq": i + 1, "id": x["id"], "page_content": x["page_content"], "source": x["metadata"].get("source"),
                 "type": x["type"],
                 "metadata": json.dumps(x["metadata"], ensure_ascii=False),
                 "to_del": "",
                 } for i, x in enumerate(docs)]
            df = pd.DataFrame(data)

            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_columns(["id", "source", "type", "metadata"], hide=True)
            gb.configure_column("seq", "No.", width=50)
            gb.configure_column("page_content", "内容", editable=True, autoHeight=True, wrapText=True, flex=1,
                                cellEditor="agLargeTextCellEditor", cellEditorPopup=True)
            gb.configure_column("to_del", "删除", editable=True, width=50, wrapHeaderText=True,
                                cellEditor="agCheckboxCellEditor", cellRender="agCheckboxCellRenderer")
            gb.configure_selection()
            edit_docs = AgGrid(df, gb.build())
            
            if st.button("保存更改"):
                origin_docs = {
                    x["id"]: {"page_content": x["page_content"], "type": x["type"], "metadata": x["metadata"]} for x in
                    docs}
                changed_docs = []
                for index, row in edit_docs.data.iterrows():
                    origin_doc = origin_docs[row["id"]]
                    if row["page_content"] != origin_doc["page_content"]:
                        if row["to_del"] not in ["Y", "y", 1]:
                            changed_docs.append({
                                "page_content": row["page_content"],
                                "type": row["type"],
                                "metadata": json.loads(row["metadata"]),
                            })

                if changed_docs:
                    if api.update_kb_docs(knowledge_base_name=selected_kb,
                                          file_names=[file_name],
                                          docs={file_name: changed_docs}):
                        st.toast("更新文档成功")
                    else:
                        st.toast("更新文档失败")
