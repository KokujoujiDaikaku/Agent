import os
from typing import Optional
from openai import OpenAI
from client_llm import HelloAgentsLLM


class MyLLM(HelloAgentsLLM):
    """
    一个自定义的LLM客户端，通过继承增加了对ModelScope的支持
    """
    def __init__(self,
                 model: Optional[str] = None,
                 apiKey: Optional[str] = None,
                 baseUrl: Optional[str] = None,
                 provider: Optional[str] = "auto",
                 **kwargs):
        # 检查provider是否为我们想处理的'modelscope'
        if provider == "modelsope":
            print("正在使用自定义的ModelScope Provider")
            self.provider = "modelscope"

            # 解析ModelScope的凭证
            self.api_key = apiKey or os.getenv("MODEL_SCOPE_API_KEY")
            self.base_url = baseUrl or "https://api-inference.modelscope.cn/v1/"

            # 验证凭证是否存在
            if not self.api_key:
                raise ValueError("ModelScope API key not found. Please set MODEL_SCOPE_API_KEY enviroment variable.")

            # 设置默认模型和其他参数
            self.model = model or os.getenv("MODEL_ID")
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)

            # 使用获取的参数创建OpenAI客户端实例
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

        else:
            # 如果不是modelscope, 则完全使用父类的原始逻辑处理
            super().__init__(model=model, apiKey=apiKey, baseUrl=baseUrl, provider=provider, **kwargs)
    pass  # 置空
