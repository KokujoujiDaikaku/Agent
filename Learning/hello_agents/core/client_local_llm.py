# pip install vllm
# 将本地模型接口设置为https://localhost:8000/v1
# python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-0.5B-Chat --host 0.0.0.0 --port 8000

# 接入vllm模型
from my_llm import MyLLM
llm_client = MyLLM(
    provider="vllm",
    model="Qwen/Qwen1.5-0.5B-Chat",
    baseUrl="https://localhost:8000/v1",
    apiKey="vllm"  # 本地模型无需真实apikey
)

messages = [{"role": "user", "content": "您好!"}]
for chunk in llm_client.think(messages):
    print(chunk, end="")
