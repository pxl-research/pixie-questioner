"""Base wrapper that instantiates an Azure ChatCompletion client if available."""

import logging
from typing import Any

from pydantic import SecretStr

logger = logging.getLogger(__name__)


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
            from langchain_openai import AzureChatOpenAI, ChatOpenAI
        except Exception:
            logger.warning(
                "Failed to import langchain_openai AzureChatOpenAI client.",
                exc_info=True,
            )
            self.model = None
            return

        normalized_max_tokens = max(int(max_tokens), 1)
        if api_version == "v1":
            # Azure v1 API: OpenAI-compatible, no dated api-version. Newer models
            # (e.g. gpt-6-luna) 404 on the dated /deployments/ route.
            self.model = ChatOpenAI(
                base_url=f"{api_endpoint.rstrip('/')}/openai/v1/",
                api_key=SecretStr(api_key),
                model=model_version,
                streaming=False,
                request_timeout=request_timeout_seconds,
                max_tokens=normalized_max_tokens,
            )
            return
        azure_client_kwargs: dict[str, Any] = {
            "openai_api_version": api_version,
            "azure_endpoint": api_endpoint,
            "openai_api_key": SecretStr(api_key),
            "deployment_name": model_version,
            "verbose": False,
            # "temperature": temperature,
            "streaming": False,
            "request_timeout": request_timeout_seconds,
            "max_tokens": normalized_max_tokens,
        }
        self.model = AzureChatOpenAI(**azure_client_kwargs)

    async def ainvoke(self, messages: list[dict[str, str]]):
        """Delegate to the underlying LangChain model."""
        if self.model is None:
            raise RuntimeError("LLM client is not configured.")
        return await self.model.ainvoke(messages)
