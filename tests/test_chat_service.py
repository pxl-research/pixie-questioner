import asyncio

from pixie_questioner.chat_service import ChatService


class FakeLLM:
    def __init__(self) -> None:
        self.last_messages = None

    async def ainvoke(self, messages):
        self.last_messages = messages
        return "ok"

    async def astream(self, messages):
        self.last_messages = messages
        for chunk in ["o", "k"]:
            yield chunk


def test_ask_builds_prompt_and_returns_answer():
    llm = FakeLLM()
    service = ChatService(llm=llm, system_prompt="SYSTEM")

    answer = asyncio.run(service.ask("Question", "session-1", ["one", "two"]))

    assert answer == "ok"
    assert llm.last_messages[0]["role"] == "system"
    assert "Session ID: session-1" in llm.last_messages[0]["content"]
    assert "History:" in llm.last_messages[0]["content"]
    assert llm.last_messages[-1]["content"] == "Question"


def test_stream_answer_yields_chunks():
    llm = FakeLLM()
    service = ChatService(llm=llm, system_prompt="SYSTEM")

    async def _collect():
        chunks = []
        async for chunk in service.stream_answer("Question", "session-1", []):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())
    assert chunks == ["o", "k"]
