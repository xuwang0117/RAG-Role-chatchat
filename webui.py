import streamlit as st
from webui_pages.utils import *
from streamlit_option_menu import option_menu
from webui_pages.dialogue.dialogue import dialogue_page, chat_box
from webui_pages.knowledge_base.knowledge_base import knowledge_base_page
import os
import sys
from configs import VERSION
from server.utils import api_address


api = ApiRequest(base_url=api_address())

if __name__ == "__main__":
    is_lite = "lite" in sys.argv

    st.set_page_config(
        "通用对话大模型",
        os.path.join("img", "robot.png"),
        initial_sidebar_state="expanded",
        layout="wide",
        menu_items={
            'Get Help': 'https://github.com/chatchat-space/Langchain-Chatchat',
            'Report a bug': "https://github.com/chatchat-space/Langchain-Chatchat/issues",
            # 'About': f"""欢迎使用 Langchain-Chatchat WebUI {VERSION}！"""
            'About': f"""欢迎使用 通用对话大模型"""
        }
    )

    pages = {
        "对话": {
            "icon": "chat",
            "func": dialogue_page,
        },
        "知识库": {
            "icon": "hdd-stack",
            "func": knowledge_base_page,
        },
    }
    # 嵌入自定义 CSS 样式  
    st.markdown(  
        """  
        <style>   
            .st-emotion-cache-16txtl3 {
                padding: 3rem 1.5rem;
            }  
            .st-emotion-cache-6qob1r {
                /* 修改侧边栏内元素的样式 */  
                # background: rgba(209, 36, 36, 0.05);
            }
            .st-emotion-cache-uf99v8 {
                background: linear-gradient(180deg, #f6f3f3 0%, #f3eded 100%);
            }
            .st-emotion-cache-18ni7ap {
                display: none;
            }
            .st-emotion-cache-90vs21 {
                background: rgba(0,0,0,0)
            }
            .st-emotion-cache-18ni7ap {
                background: linear-gradient(180deg, #f6f3f3 0%, #f3eded 100%);
            }
            .st-emotion-cache-s1k4sy {
                background: #fff
            }
            .st-cb  {
                # display: none;
            }
            #id .menu {
                .nav-link-horizontal {
                    padding-top: .15rem !important;
                }
            }
            .st-emotion-cache-z5fcl4 {
                height: calc(100vh - 125px);
                overflow: auto
            }
            .st-emotion-cache-4oy321  {
                background: #fff;
                padding-right: 20px
            }
            .st-emotion-cache-s1k4sy {
                > div {
                    height: 100px
                }  
            }
            .stChatFloatingInputContainer {
                padding-bottom: 20px
            }
            .st-emotion-cache-1lypi3u {
                bacakground: #fff !important
            }
            .st-emotion-cache-f4ro0r {
                button {
                    width: 50px;
                    margin-right: 5px;
                    margin-bottom: 5px
                }
                button:hover {
                    border-radius: 1.5rem
                }
            }
            .st-emotion-cache-keje6w {
                .st-emotion-cache-yxabgg {
                    min-heihgt: none;
                    padding: 0px 15px;
                    p {
                        font-size: 14px
                    }
                }
            }
            .row-widget {
                button {
                    width: 100%                        
                }
            }
        </style>  
        """,  
        unsafe_allow_html=True,  
    )  
    with st.sidebar:
        col1, col2 = st.columns([0.2, 0.8], gap="small")
        with col1:
            st.image(
                os.path.join(
                    "img",
                    "chatchat_icon_blue_square_v2.png"
                ),
                width=60
            )
        with col2:
            st.caption(
                f"""<p style="color:#000; font-size: 22px; font-weight:500; margin: 0" align="left">通用对话大模型</p><p style="width: 190px; margin: 0" align="left">General conversational LLM</p>""",
                unsafe_allow_html=True,
            )
        # st.caption(
        #     f"""<p align="center">当前版本：{VERSION}</p>""",
        #     unsafe_allow_html=True,
        # )
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            # menu_icon="chat-quote",
            orientation="horizontal",
            default_index=default_index,
        )

    if selected_page in pages:
        pages[selected_page]["func"](api=api, is_lite=is_lite)
