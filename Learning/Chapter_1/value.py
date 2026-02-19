import requests
import os
from tavily import TavilyClient
from volcenginesdkarkruntime import Ark
import re


AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具：
- 'get_weather(city: str)': 查询指定城市的实时天气。
- 'get_attraction(city: str, weather: str)': 根据城市和天气搜索推荐的旅游景点。

# 行动格式：
你的回答必须严格遵守以下格式。首先是你的思考过程，然后是你要执行的具体行动。
Thought: [这里是你的思考过程和下一步计划]
Action: [这里是你要调用的工具，格式为 function_name(arg_name="arg_value")]

# 任务完成：
当你收集到足够的信息，能够回答用户的最终问题时，你必须在'Action:'字段后使用'finish(answer="...")'来输出最终答案。

请开始吧！
"""


def get_weather(city: str) -> str:
    """
    通过调用wttr.in API查询真实的天气信息。
    """
    # API 端点，我们请求JSON格式的数据
    url = f"https://wttr.in/{city}?format=j1"

    try:
        # 发起网络请求
        response = requests.get(url)
        # 检查响应状态码是否为200（成功）
        response.raise_for_status()
        # 解析返回的JSON数据
        data = response.json()

        # 提取当前天气状况
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']

        # 格式化成自然语言返回
        return f"{city}当前天气：{weather_desc}，气温{temp_c}摄氏度"

    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return "错误：查询天气时遇到网络问题 - {}".format(e)
    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return "错误：解析天气数据失败，可能是城市名称无效 - {}".format(e)


def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐
    """

    # 1. 从环境变量中读取API密钥
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误：未配置TAVILY_API_KEY环境变量。"

    # 2. 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)

    # 3. 构造一个精确的查询
    query = "‘{}’在‘{}’天气下最值得去的旅游景点推荐及理由".format(city, weather)

    try:
        # 4. 调用API，include_answer=True会返回一个综合性的回答
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        # 5. Tavily返回的结果已经非常干净，可以直接使用
        # response['answer']是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]

        # 如果没有综合性回答，则格式化原始结果
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append("- {}: {}".format(result['title'], result['content']))

        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐。"

        return "根据搜索，为您找到以下信息：\n" + "\n".join(formatted_results)

    except Exception as e:
        return "错误：执行Tavily搜索时出现问题 - {}".format(e)


# 将所有工具函数放入字典，方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction
}


class AskCompatibleClient:
    """
    一个用于调用任何兼容openAI接口的LLM服务的客户端。
    """
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = Ark(api_key=api_key, timeout=1800)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """
        调用LLM API来生成回应。
        """
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                thinking={
                    "type": "disabled"  # 默认行为，不使用深度思考能力
                }
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer

        except Exception as e:
            print("调用LLM API时发生错误：{}".format(e))
            return "错误：调用语言模型服务时出错。"


class MainExecute:
    """
    执行行动循环，下面的主循环将整合所有组件，并通过格式化后的Prompt驱动LLM进行决策
    """
    def __init__(self, model: str, api_key: str, tavily_api_key: str):
        # --- 1. 配置LLM客户端 ---
        # 配置个人凭证和地址
        self.api_key = api_key
        self.model_id = model
        os.environ['TAVILY_API_KEY'] = tavily_api_key

    def main_cycle(self):
        # --- 2.初始化 ---
        llm = AskCompatibleClient(
            model=self.model_id,
            api_key=self.api_key
        )

        user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
        prompt_history = ["用户请求：{}".format(user_prompt)]

        print("用户输入：{}\n".format(user_prompt) + "=" * 40)

        # --- 3. 运行主循环 ---
        for i in range(5):  # 设置最大循环次数
            print("--- 循环 {} ---\n".format(i + 1))

            # 3.1 构建Prompt
            full_prompt = "\n".join(prompt_history)

            # 3.2 调用LLM进行思考
            llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
            print("模型输出：\n{}".format(llm_output))
            prompt_history.append(llm_output)

            # 3.3 解析并执行行动
            action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
            if not action_match:
                print("解析错误：模型输出中未找到Action。")
                break
            action_str = action_match.group(1).strip()

            if action_str.startswith("finish"):
                final_answer = re.search(r'finish\(answer="(.*)"\)', action_str).group(1)
                print("任务完成，最终答案：{}".format(final_answer))
                break

            tool_name = re.search(r"(\w+)\(", action_str).group(1)
            args_str = re.search(r"\((.*)\)", action_str).group(1)
            kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

            if tool_name in available_tools:
                observation = available_tools[tool_name](**kwargs)
            else:
                observation = "错误：未定义的工具{}".format(tool_name)

            # 3.4 记录观察结果
            observation_str = "Observation: {}".format(observation)
            print("{}\n".format(observation_str) + "=" * 40)
            prompt_history.append(observation_str)


run_llm = MainExecute(
    model="deepseek-v3-2-251201",
    api_key="",
    tavily_api_key=""
)
run_llm.main_cycle()
