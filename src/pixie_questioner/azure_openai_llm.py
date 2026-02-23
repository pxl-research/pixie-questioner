"""Base wrapper that instantiates an Azure ChatCompletion client if available."""

from pydantic import SecretStr

from .abstractclasses.llm import LLM


class AzureOpenAILLM(LLM):
    def __init__(
        self,
        api_version: str,
        api_endpoint: str,
        api_key: str,
        model_version: str,
        temperature: float = 0.01,
    ) -> None:
        super().__init__(model_version)
        try:  # optional dependency
            from langchain_openai import AzureChatOpenAI
        except Exception:
            self.model = None
            return

        self.model = AzureChatOpenAI(
            openai_api_version=api_version,
            azure_endpoint=api_endpoint,
            openai_api_key=SecretStr(api_key),
            deployment_name=model_version,
            verbose=True,
            temperature=temperature,
            streaming=True,
        )
