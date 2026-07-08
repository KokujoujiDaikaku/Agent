import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from openai import OpenAI


# 加载 .env 文件中的环境变量，处理文件不存在异常
try:
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
except FileNotFoundError:
    print("警告：未找到 .env 文件，将使用系统环境变量。")
except Exception as e:
    print(f"警告：加载 .env 文件时出错: {e}")


class MainLLM():
    def __init__(self, model: str = None, apikey: str = None, base_url: str = None, timeout: int = None):
        """
        Init openai client. Use the passed params first; if None are provided, load values from environment variables.
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        self._apikey = apikey or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.timeout = timeout or int(os.getenv("TIME_OUT_60_SECOND", 60))

        if not all([self.model, self._apikey, self.base_url, self.timeout]):
            raise ValueError("Model ID, API KEY, Base URL must be provided or defined in the .env file.")

        # Initialize the OpenAI client
        self.client = OpenAI(api_key=self._apikey, base_url=self.base_url, timeout=self.timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        Call the LLM for reasoning and return its response.
        """
        print(f"🧠 calling {self.model}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            # 处理流式响应
            print("✅ LLM response SUC:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # Add a line break after streaming output completes.
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None


# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llmClient = MainLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)
