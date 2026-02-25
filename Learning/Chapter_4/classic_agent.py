import os
import re
from volcenginesdkarkruntime import Ark
from dotenv import load_dotenv
from typing import List, Dict, Any
from pathlib import Path
from serpapi import SerpApiClient

from react_prompt import REACT_PROMPT_TEMPLATE

# 加载model.env中的环境变量
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


class HelloAgentsLLM:
    """
    为课程后续内容定制LLM客户端
    默认使用流式响应
    """
    def __init__(self, model: str = None, model_api_key: str = None, base_url: str = None, timeout: int = None):
        """
        初始化客户端,优先使用传入参数,如果为未提供,则从环境变量加载
        """
        self.model = model or os.getenv("MODEL_ID")
        model_api_key = model_api_key or os.getenv("MODEL_API_KEY")
        base_url = base_url or os.getenv("MODEL_BASE_URL")
        timeout = timeout or int(os.getenv("TIME_OUT_60_SECOND"))

        if not all([self.model, model_api_key, base_url]):
            raise ValueError("模型ID/API密钥和服务地址必须被提供或在.env文件中定义")

        self.client = Ark(
            # 从环境变量中读取您的方舟API Key
            base_url=base_url,
            api_key=model_api_key,
            timeout=timeout
        )

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考, 并返回其响应
        """
        print("正在调用{}模型...".format(self.model))
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                thinking={
                    "type": "disabled"  # 默认行为，不使用深度思考能力
                }
            )
            # 处理非流式响应
            print("大语言模型响应成功:")
            collected_content = response.choices[0].message.content.strip()
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print("调用LLM API时发生错误: {}".format(e))
            return None


def search(query: str) -> str:
    """
    一个基于SerApi的实战网页搜索引擎工具
    它会智能地解析搜素结果, 优先返回直接答案或知识图谱信息
    """
    print("正在执行[SerpApi]网页搜索: {}".format(query))
    try:
        serp_api_key = os.getenv("SERPAPI_API_KEY")
        if not serp_api_key:
            return "错误: SERPAPI_API_KEY未在.env文件中配置"

        params = {
            "engine": "google",
            "q": query,
            "api_key": serp_api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn",  # 语言代码
        }

        serp_api_client = SerpApiClient(params)
        serp_api_results = serp_api_client.get_dict()

        # 智能解析: 优先寻找最直接的答案
        if "answer_box_list" in serp_api_results:
            return "\n".join(serp_api_results)
        if "answer_box" in serp_api_results and "answer" in serp_api_results["answer_box"]:
            return serp_api_results["answer_box"]["answer"]
        if "knowledge_graph" in serp_api_results and "description" in serp_api_results["knowledge_graph"]:
            return serp_api_results["knowledge_graph"]["description"]
        if "organic_results" in serp_api_results and serp_api_results["organic_results"]:
            # 如果没有直接答案, 则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}" for i, res in enumerate(serp_api_results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
    except Exception as e:
        return "serpapi搜索时发生错误: {}".format(e)


class ToolExecutor:
    """
    一个工具执行器, 负责管理和执行工具
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个新工具
        """
        if name in self.tools:
            print("警告: 工具 {} 已存在, 将被覆盖".format(name))
        self.tools[name] = {"description": description, "func": func}
        print("工具 {} 已注册".format(name))

    def get_tool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数
        """
        return self.tools.get(name, {}).get("func")

    def get_available_tools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串
        """
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])


class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def _parse_output(self, text: str):
        """
        解析LLM的输出, 提取Thought和Action
        """
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """
        解析Action字符串, 提取工具名称和输入, ，例如从 Search[华为最新手机] 中提取出工具名 Search 和工具输入 华为最新手机
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题
        """
        self.history = []  # 每次运行时重置记忆
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 格式化提示词
            tools_desc = self.tool_executor.get_available_tools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("错误: LLM未能返回有效响应")
                break

            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告: 未能解析出有效的Action, 流程终止")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令, 提取最终答案并结束
                final_answer = re.match(r"Finish\[(.*)\]", action).group(1)
                print(f"最终答案: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                # ... 处理无效Action格式 ...
                continue

            print(f"行动: {tool_name}[{tool_input}]")
            tool_function = self.tool_executor.get_tool(tool_name)
            if not tool_function:
                observation = f"错误: 未找到名为 {tool_name} 的工具"
            else:
                observation = tool_function(tool_input)  # 调用真实工具

            print(f"观察: {observation}")

            # 将本轮的Action和Observation添加到记忆中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # 循环结束
        print("已达到最大步数, 流程终止")
        return None

    # ...(后续的解析 执行 整合步骤)


# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        # 1. 初始化工具执行器
        tool_executor = ToolExecutor()

        # 2. 注册实战搜索工具
        search_description = "一个网络搜索引擎, 当你需要回答关于时事、事实以及在你的知识库中找不到的信息时, 应使用该工具"
        tool_executor.registerTool("search", search_description, search)

        # 4. 智能体的Action调用，这次我们问一个实时性的问题
        print("\n--- 执行 Action: search['中国股市中商业航天板块涨势最好的股是哪个'] ---")

        tool_name = "search"
        tool_input = "中国股市中商业航天板块涨势最好的股是哪个"

        llm_client = HelloAgentsLLM()

        agent = ReActAgent(llm_client=llm_client, tool_executor=tool_executor)
        agent.run(tool_input)

        # exampleMessages = [
        #     {"role": "system", "content": "You are a helpful assistant that writes python code."},
        #     {"role": "user", "content": "写一个快速排序算法"}
        # ]

        # print("--- 调用LLM ---")
        # response_text = llm_client.think(exampleMessages)
        # if response_text:
        #     print("\n\n--- 完整模型响应 ---")
        #     print(response_text)

    except ValueError as e:
        print(e)
