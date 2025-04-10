from fastapi import Body, Request, File, UploadFile
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
import asyncio, json, re, hashlib, time, requests, urllib
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
from server.chat.chat import chat_for_pipe
from datetime import datetime
async def smart_reporting(report_name: str= Body(..., description="报告名称", examples=["北约认知战"]),
                    report_restriction: str= Body(..., description="报告要求", examples=["生成不少于800字的文档"]),
                    report_outline: list = Body(None, description="报告大纲", examples=[["一、概述","二、背景","三、详细分析","四、总结"]]),
                    begin_time: datetime = Body(..., description="起始时间2024-10-01 10:00:00", examples=["2024-10-01 10:00:00"]),
                    end_time: datetime = Body(..., description="终止时间2024-11-17 10:00:00", examples=["2024-11-17 10:00:00"]),
                    # files: List[UploadFile] = File(..., description="上传文件"),
                    files: list = Body(None, description="贴文列表", examples=[["1.","2.","3.","4."]]),
                    model_recommendation: bool = Body(True, description="大模型推荐"),
                    stream: bool = Body(False, description="流式输出"),
                    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                    temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                    max_tokens: Optional[int] = Body(
                        None,
                        description="限制LLM生成Token数量，默认None代表模型最大值"
                    ),
                    request: Request = None,
                    ):

    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        
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



    #     if "北约认知战" in report_name:
    #         pre_answer = '''北约在与南联盟的冲突中，展现了其高度发达的认知防护能力，主要体现在以下几个方面：

    # 概述
    # 北约指挥机构能够迅速向一线部队下达命令，甚至直接向导弹部队下达指令，时间分别控制在3分钟和1分钟之内。这种高效的指挥链配合GPS制导的巡航导弹、激光制导炸弹和联合直接攻击弹药，实现了信息与火力的无缝对接

    # 远程精确打击
    # 北约主要采用三种战法进行攻击，包括在距离战场上千公里处发射巡航导弹、从美国本土或盟军基地出动隐形轰炸机深入战区投射精确制导炸弹[1]，以及在掌握制空权的前提下，使用有人驾驶作战飞机从防区外发射精确制导武器。

    # 电子干扰与支援
    # 在隐形轰炸机深入战区执行任务时，北约使用电子干扰机进行伴随支援，干扰敌方雷达和通信系统，保护己方飞机免受敌方防空系统的威胁[2]，这是认知防护的重要组成部分。
    # 北约的指挥系统能够迅速响应战场变化，快速做出决策并调整作战计划，这在与南联盟的冲突中得到了充分展现，体现了北约在认知防护上的灵活性和适应性[3]。

    # 总结
    # 综上所述，北约的认知防护策略侧重于利用信息技术、快速决策和精确打击能力，形成信息与火力一体化的作战模式，以避免自身遭受有效反击，同时确保对敌方目标的精确摧毁[4]。这一策略在与南联盟的冲突中得到了有效应用，展现了北约在现代战争中的认知防护实力。'''

    #         source_documents = '''出处 [1] [科索沃战争.docx]
    # 调整策略，积极防护。北约在基本夺取战场制空权，但并未完全瘫痪南联盟军队指挥体系的情况下，持续加大了空袭的频次和强度...\n

    # 出处 [2] [科索沃战争.docx]
    # 在作战过程中，北约主要采用三种战法：一是在距离战场上千公里处发射巡航导弹进行攻击；二是从美国本土或盟军基地出动隐形轰炸机，在电子干扰机的伴随支援下，深入战区，投射精确制导炸弹；三是在掌握战区制空权的前提下，使用有人驾驶作战飞机从防区外发射精确制导武器，攻击预定目标。\n

    # 出处 [3] [北约认知攻防战-新华网]
    # 北约主要采用三种战法：一是在距离战场上千公里处发射巡航导弹进行攻击；二是从美国本土或盟军基地出动隐形轰炸机，在电...\n

    # 出处 [4] [北约认知攻防战-新华网]
    # 北约主要采用三种战法：一是在距离战场上千公里处发射巡航导弹进行攻击；二是从美国本土或盟军基地出动隐形轰炸机，在电...\n
    # '''
    #         if stream:
    #             for s in pre_answer:
    #                 yield json.dumps({"report": s}, ensure_ascii=False)
    #             yield json.dumps({"docs": source_documents}, ensure_ascii=False)
    #         else:
    #             buffer = ""
    #             for s in pre_answer:
    #                 buffer += s
    #             yield json.dumps({"report": buffer}, ensure_ascii=False)
    #             yield json.dumps({"docs": source_documents}, ensure_ascii=False)
    #     elif "美国大选" in report_name:
    #         pre_answer = "# 美国大选报告\n\n1. 概述\n\n美国大选是美国民主制度的重要组成部分，其过程复杂且严密，旨在确保每位登记选民都能公平、公正地表达自己的政治意愿。美国大选每四年举行一次，主要通过直接选举和选举人团制度来决定总统和副总统的人选。2020年的总统大选因新冠疫情的影响而显得尤为特殊，不仅选民投票方式发生了变化，选举过程和结果也引起了广泛的社会关注[1]。\n\n2. 过程\n\n2020年美国总统大选的正式投票日为11月3日，但由于新冠疫情的影响，许多州调整了选民投票的规则，增加了邮寄投票和提前投票的方式，以减少选举日当天的人群聚集，保障选民的健康安全。此次选举中，民主党候选人乔·拜登和共和党候选人唐纳德·特朗普进行了激烈的竞选活动，双方在经济、医疗、环境等多个政策领域展开了辩论[2]。选前的民意调查显示，两位候选人的支持率非常接近，这使得选民的每一票都显得尤为重要。\n\n3. 结果\n\n经过数日的计票，乔·拜登最终以306张选举人票对232张选举人票的优势胜出，当选为美国第46任总统。拜登的胜利不仅标志着美国政治的转变，也反映了选民对国家未来方向的不同期待。特朗普方面则对选举结果提出质疑，并在多个州提起诉讼，主张存在广泛的选举舞弊现象。然而，这些诉讼最终并未改变选举结果[1]。\n\n4. 总结\n\n2020年美国大选不仅是一场政治竞争，也是对美国民主制度的一次考验。尽管选举过程中出现了许多争议和挑战，但最终选举结果得到了确认，新政府也顺利就任。这次选举凸显了美国民主制度的韧性和复杂性，同时也暴露了选举制度中的一些问题，为未来的改革提供了参考[3]。随着新政府的成立，美国面临着诸多国内外的挑战，如何应对这些挑战将是拜登政府的重要任务。"
    #         source_documents = "出处 [1] [认知认知译丛015-数据战略与信息优势内文.pdf] \n\n的衡量标准和建立目标，但是会向功能提供者询问相关问题：“用户体验”和“作战韧性”的标准是什么？以及谁批准了这些标准？“用户体验”和“作战韧性”的能效如何？以及如何识别？\n\n五、三个阶段目标\n\n三个阶段目标即“优化云信息环境”、“采用企业级服务”、“实施零信任”，由总目标分解而成，并将在各自的主要设计概念中得到更详细的描述。三个阶段目标源自众多可能选项，并将统合考虑国防部战略的众多要素以形成战略杠杆，在海军部信息环境转化方面发挥最大作用，实现与业界的\n\n\n出处 [2] [2023北约缓解和应对认知战（全译本）内文印5本.pdf] \n\n可以跨媒体平台使用，作为信息描述，以及与合作者一起制定风险管理程序的手段。\n\n10.2 培训和教育\n\n在这方面，第一个问题涉及到培训的目标是什么？以及如何制定培训的规范？这不仅适用于初始培训，也适用认知战中的专家。现在，提出的问题是：需要培养哪些技能？这类培训的最佳方法和形式是什么？培训目标可能会有所不同，从最初的让人们意识到现在的问题究竟是什么？以及什么是可以实际完成的，直到在虚拟现实环境中进行专家级别\n\n\n出处 [3] [网络空间动态（2024年第5期）.pdf] \n\nConvergence Capstone 4，PC-C4）多国联合试验，美、英、澳、 加、新、法、日等多国联合部队，将在多域作战环境中试验 新兴技术的应用情况，评估新兴技术在促进跨域军事行动和 统一战略方法方面的潜力，推进美国各军种和多国联合部队 之间的数据集成可操作性和协作性。据悉，“会聚工程”是美 陆军未来司令部负责的项目，旨在以此为抓手，融入美军的 联合全域指挥控制（JADC2）概念，推进数字化转型和信息 优势。美军认为，PC-C4 积累了往届“会聚工程”大量训练\n"

    #         if stream:
    #             for s in pre_answer:
    #                 yield json.dumps({"report": s}, ensure_ascii=False)
    #             yield json.dumps({"docs": source_documents}, ensure_ascii=False)
    #         else:
    #             buffer = ""
    #             for s in pre_answer:
    #                 buffer += s
    #             yield json.dumps({"report": buffer}, ensure_ascii=False)
    #             yield json.dumps({"docs": source_documents}, ensure_ascii=False)
    #     else:
        if model_recommendation:
            #搜索模块
            # url = "http://10.106.51.159:19999/search-service/search/v1/searchy3"
            url = "http://10.120.6.24:7778/search/api/search/v1/searchy3"
            # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # time = urllib.parse.quote(f"2024-01-01 10:00:00~{current_time}")
            time = f"{begin_time}~{end_time}"
            # 定义查询参数
            params = {
                "q": report_name,        # 搜索词
                "start_index": 0,     # 帖源数据起始序号
                "rn": 5,           # 请求帖源数据的数量
                "advanced":"PublishTime:{}".format(time),
                "y2timeout":10000
            }
            # 发送请求
            try:
                response = requests.get(url, params=params)
                data = response.json()
                # 获取帖源结果
                search_results = data.get('data').get('search_results', [])
                #显示搜索结果
                source_documents = []
                for inum, doc in enumerate(search_results):
                    text = doc['content']
                    # 计算需要分块的数量
                    num_chunks = (len(text) + 299) // 300  # 向上取整
                    for i in range(num_chunks):
                        chunk = text[i * 300: (i + 1) * 300]
                        source_documents.append(chunk)
            except:
                source_documents = []
            # print("source_documents: ", source_documents)
            if USE_RERANKER:
                reranker_model_path = get_model_path(RERANKER_MODEL)
                reranker_model = LangchainReranker(top_n=len(source_documents),
                                                device=embedding_device(),
                                                max_length=RERANKER_MAX_LENGTH,
                                                model_name_or_path=reranker_model_path)
                source_documents = reranker_model.compress_search_documents(documents=source_documents, query=report_name)
            source_documents_out = []
            for inum, doc in enumerate(source_documents):
                text = f"""出处 [{inum + 1}] [] \n\n{doc}"""
                source_documents_out.append(text)
            print("source_documents_out: ", source_documents_out)
            search_results = "\n\n\n".join(source_documents_out)
        else:
            source_documents = []
            title = ""
            for inum, text in enumerate(files):
                # 找到第一个连续三个换行符的位置
                end_of_title_index = text.find('\n\n\n')
                title = text[:end_of_title_index].strip()
                text = text[end_of_title_index:].strip()
                # 计算需要分块的数量
                num_chunks = (len(text) + 299) // 300  # 向上取整
                for i in range(num_chunks):
                    chunk = text[i * 300: (i + 1) * 300]
                    source_documents.append(chunk)
            print("source_documents: ", source_documents)
            if USE_RERANKER:
                reranker_model_path = get_model_path(RERANKER_MODEL)
                reranker_model = LangchainReranker(top_n=len(source_documents),
                                                device=embedding_device(),
                                                max_length=RERANKER_MAX_LENGTH,
                                                model_name_or_path=reranker_model_path)
                source_documents = reranker_model.compress_search_documents(documents=source_documents, query=report_name)
            source_documents_out = []
            for inum, doc in enumerate(source_documents):
                text = f"""出处 [{inum + 1}] [{title}] \n\n{doc}"""
                source_documents_out.append(text)
            print("source_documents_out: ", source_documents_out)
            search_results = "\n\n\n".join(source_documents_out)
            # search_results = ""
        


        all_prompt = (
            """
            **任务**：请根据报告名称、报告要求、报告大纲和参考资料，生成一份详细的智能报告。如果没有给出具体参考资料，直接由大模型生成。
            **限制**：请确保报告内容准确、逻辑清晰。符合报告要求。严格按照报告大纲结构生成。只输出报告内容，不能输出其他额外信息。
            若参考资料不为空，报告中用[1][2]...标明参考的资料。
            **你的任务**：
            '<报告名称>{{ report_name }}</报告名称>\n'
            '<报告要求>{{ report_restriction }}</报告要求>\n'
            '<报告大纲>{{ report_outline }}</报告大纲>\n'
            '<参考资料>{{ search_results }}</参考资料>\n'
            """
        )
        input_msg = History(role="user", content=all_prompt).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)
        
        task = asyncio.create_task(wrap_done(
            chain.acall({"report_name": report_name, "report_restriction": report_restriction, "report_outline": report_outline, "search_results": search_results}),
            callback.done),
        )

        if stream:
            async for token in callback.aiter():
                yield json.dumps({"report": token}, ensure_ascii=False)
            yield json.dumps({"docs": search_results}, ensure_ascii=False)
        else:
            buffer = ""
            async for token in callback.aiter():
                buffer += token
            yield json.dumps({"report": buffer}, ensure_ascii=False)
            yield json.dumps({"docs": search_results}, ensure_ascii=False)

        await task

    return EventSourceResponse(knowledge_base_chat_iterator())