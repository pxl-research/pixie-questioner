# ruff: noqa: S101
from __future__ import annotations

import builtins
import logging
import sys
import types
from collections.abc import Mapping, Sequence

from pixie_questioner.azure_openai_llm import AzureOpenAILLM


def test_azure_openai_llm_disables_streaming(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAzureChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    fake_module = types.ModuleType("langchain_openai")
    setattr(fake_module, "AzureChatOpenAI", FakeAzureChatOpenAI)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)

    llm = AzureOpenAILLM(
        api_version="2024-12-01-preview",
        api_endpoint="https://example.openai.azure.com/",
        api_key="test-key",
        model_version="gpt-test",
    )

    assert llm.model is not None
    assert captured["streaming"] is False


def test_azure_openai_llm_logs_optional_dependency_import_failures(
    monkeypatch, caplog
) -> None:
    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> types.ModuleType:
        if name == "langchain_openai":
            raise ImportError("missing optional dependency")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with caplog.at_level(logging.WARNING):
        llm = AzureOpenAILLM(
            api_version="2024-12-01-preview",
            api_endpoint="https://example.openai.azure.com/",
            api_key="test-key",
            model_version="gpt-test",
        )

    assert llm.model is None
    assert "langchain_openai" in caplog.text
