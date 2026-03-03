"""Base wrapper that instantiates an Azure ChatCompletion client if available."""

from pydantic import SecretStr


class AzureOpenAILLM:
    def __init__(
        self,
        api_version: str,
        api_endpoint: str,
        api_key: str,
        model_version: str,
        temperature: float = 0.01,
        request_timeout_seconds: float = 15.0,
        max_tokens: int = 450,
    ) -> None:
        self.model = None
        try:  # optional dependency
            from langchain_openai import AzureChatOpenAI
        except Exception:
            self.model = None
            return

        normalized_max_tokens = max(int(max_tokens), 1)
        self.model = AzureChatOpenAI(
            openai_api_version=api_version,
            azure_endpoint=api_endpoint,
            openai_api_key=SecretStr(api_key),
            deployment_name=model_version,
            verbose=False,
            temperature=temperature,
            streaming=True,
            timeout=request_timeout_seconds,
            max_tokens=normalized_max_tokens,
        )
