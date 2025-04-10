# prompt模板使用Jinja2语法，简单点就是用双大括号代替f-string的单大括号
# 本配置文件支持热加载，修改prompt模板后无需重启服务。

# LLM对话支持的变量：
#   - input: 用户输入内容

# 知识库和搜索引擎对话支持的变量：
#   - context: 从检索结果拼接的知识文本
#   - question: 用户提出的问题

# Agent对话支持的变量：

#   - tools: 可用的工具列表
#   - tool_names: 可用的工具名称列表
#   - history: 用户和Agent的对话历史
#   - input: 用户输入内容
#   - agent_scratchpad: Agent的思维记录

PROMPT_TEMPLATES = {
    "llm_chat": {
        
        "default":
            '{{ input }}',
        "9G8b":
            "{{ input }}",
        "with_history":
            'The following is a friendly conversation between a human and an AI. '
            'The AI is talkative and provides lots of specific details from its context. '
            'If the AI does not know the answer to a question, it truthfully says it does not know.\n\n'
            'Current conversation:\n'
            '{history}\n'
            'Human: {input}\n'
            'AI:',

        "py":
            '你是一个聪明的代码助手，请你给我写出简单的py代码。 \n'
            '{{ input }}',
    },


    "knowledge_base_chat": {
        "thinking_prompt":
            '<角色>你是一个世界知识专家，你可以从已知信息中迅速把握与问题相关的关键部分信息。</角色>\n'
            '<指令>根据已知信息以及问题，生成你对问题的思考过程，思考过程是为了帮助别人更好地回答问题，而不是让你直接回答问题。思考过程是一种指引性质的内容，可以让别人在看到问题和已知信息时立马能抓住它们的重点。'
            '你只需要生成思考过程，不需要生成答案，也不需要生成总结，也不需要生成已知信息中的内容，必要时可以用已知信息中的概括描述，生成请使用中文，你的回答中应该避免生成“思考过程：”这个开头。 </指令>\n'
            '<思考过程例子>根据已知信息分析问题：问题：张三出生在哪个国家？思考过程：1.我应该从文档中找一下有没有张三。\n2.我应该从文档中找一下张三和出生或者生日等相关的信息。\n3.我可以从文档中找一下张三和哪个国家的关系最大。\n...</思考过程例子>\n'
            '<已知信息>{{ context }}</已知信息>\n'
            '<问题>{{ question }}</问题>\n',
            
        "output_with_thinking_prompt":
            '<角色>你是一个世界知识专家，你可以根据另一个世界知识专家的思考过程来让你更快更准地回答问题。</角色>\n'
            '<指令>根据已知信息以及思考过程，简洁和专业的来回答问题。思考过程是一种指引性质的内容，可以让别人在看到问题和已知信息时立马能抓住它们的重点。其中思考过程是为了帮助你更快更准更好地回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，'
            '不允许在答案中添加编造成分，答案请使用中文。 </指令>\n'
            '<已知信息>{{ context }}</已知信息>\n'
            '<思考过程>{{ thinking }}</思考过程>\n'
            '<问题>{{ question }}</问题>\n',
            
        "default":
            '<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，'
            '不允许在答案中添加编造成分，答案请使用中文。 </指令>\n'
            '<已知信息>{{ context }}</已知信息>\n'
            '<问题>{{ question }}</问题>\n',

        "text":
            '<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，答案请使用中文。 </指令>\n'
            '<已知信息>{{ context }}</已知信息>\n'
            '<问题>{{ question }}</问题>\n',
            
        "9G8b":
            "{{ context }}\n{{ question }}",

        "empty":  # 搜不到知识库的时候使用
            '请你回答我的问题:\n'
            '{{ question }}\n\n',
            
        "9G8b-empty":  # 搜不到知识库的时候使用
            '{{ question }}',
    },


    "search_engine_chat": {
        "default":
            '<指令>这是我搜索到的互联网信息，请你根据这些信息进行提取并有调理，简洁的回答问题。'
            '如果无法从中得到答案，请说 “无法搜索到能回答问题的内容”。 </指令>\n'
            '<已知信息>{{ context }}</已知信息>\n'
            '<问题>{{ question }}</问题>\n',

        "search":
            '<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，答案请使用中文。 </指令>\n'
            '<已知信息>{{ context }}</已知信息>\n'
            '<问题>{{ question }}</问题>\n',
    },


    "agent_chat": {
        "default":
            'Answer the following questions as best you can. If it is in order, you can use some tools appropriately. '
            'You have access to the following tools:\n\n'
            '{tools}\n\n'
            'Use the following format:\n'
            'Question: the input question you must answer1\n'
            'Thought: you should always think about what to do and what tools to use.\n'
            'Action: the action to take, should be one of [{tool_names}]\n'
            'Action Input: the input to the action\n'
            'Observation: the result of the action\n'
            '... (this Thought/Action/Action Input/Observation can be repeated zero or more times)\n'
            'Thought: I now know the final answer\n'
            'Final Answer: the final answer to the original input question\n'
            'Begin!\n\n'
            'history: {history}\n\n'
            'Question: {input}\n\n'
            'Thought: {agent_scratchpad}\n',

        "ChatGLM3":
            'You can answer using the tools, or answer directly using your knowledge without using the tools. '
            'Respond to the human as helpfully and accurately as possible.\n'
            'You have access to the following tools:\n'
            '{tools}\n'
            'Use a json blob to specify a tool by providing an action key (tool name) '
            'and an action_input key (tool input).\n'
            'Valid "action" values: "Final Answer" or  [{tool_names}]'
            'Provide only ONE action per $JSON_BLOB, as shown:\n\n'
            '```\n'
            '{{{{\n'
            '  "action": $TOOL_NAME,\n'
            '  "action_input": $INPUT\n'
            '}}}}\n'
            '```\n\n'
            'Follow this format:\n\n'
            'Question: input question to answer\n'
            'Thought: consider previous and subsequent steps\n'
            'Action:\n'
            '```\n'
            '$JSON_BLOB\n'
            '```\n'
            'Observation: action result\n'
            '... (repeat Thought/Action/Observation N times)\n'
            'Thought: I know what to respond\n'
            'Action:\n'
            '```\n'
            '{{{{\n'
            '  "action": "Final Answer",\n'
            '  "action_input": "Final response to human"\n'
            '}}}}\n'
            'Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. '
            'Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.\n'
            'history: {history}\n\n'
            'Question: {input}\n\n'
            'Thought: {agent_scratchpad}',
    }
}
