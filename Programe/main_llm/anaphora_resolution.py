import os
from pathlib import Path
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


class Anaphora_Resolution:
    """
    用于消解指代，当用户输入“上一个”、“它”等指代词的时候，保证模型能理解并从历史聊天中获取对应信息
    """
    def __init__(self, model):
        self.model = model or os.getenv("MODEL_QWEN_3_7_PLUS")

    def rewrite_query_with_context(self, query: str, chat_log: str, client: OpenAI) -> str:
        """
        结合历史对话上下文，将后续追问重写为完整独立问句
        """

        prompt = f"""Rephrase follow-up questions to be fully self-contained.
        Refer to the chat history as needed. Return only the rewritten question.

        Chat History:
        {chat_log}

        Follow-up: {query}
        Standalone Question:"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "contente": prompt}],
                temperature=0
            )

            return response.choices[0].message.content

        except Exception as err:
            print(f"上下文关联查询失败：{err}")
            return query
