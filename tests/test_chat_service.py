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


class FakeDoneLLM:
    def __init__(self, response: str, chunks: list[str] | None = None) -> None:
        self.last_messages = None
        self.response = response
        self.chunks = chunks or [response]

    async def ainvoke(self, messages):
        self.last_messages = messages
        return self.response

    async def astream(self, messages):
        self.last_messages = messages
        for chunk in self.chunks:
            yield chunk


class FakeSummarizer:
    def __init__(self) -> None:
        self.last_inputs = None

    async def generate(self, current_question: str, current_answer: str, summary: str):
        self.last_inputs = (current_question, current_answer, summary)
        return "summary"


def test_ask_builds_prompt_and_returns_answer():
    llm = FakeLLM()
    service = ChatService(llm=llm, system_prompt="SYSTEM\n{summary}")

    result = asyncio.run(
        service.ask(
            "Question",
            "session-1",
            ["one", "two"],
            questionnaire="QUESTIONNAIRE",
        )
    )

    assert result.text == "ok"
    assert result.response_type == "answer"
    assert llm.last_messages[0]["role"] == "system"
    assert "SYSTEM\nQUESTIONNAIRE" in llm.last_messages[0]["content"]
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


def test_ask_returns_summary_when_requested():
    llm = FakeLLM()
    summarizer = FakeSummarizer()
    service = ChatService(llm=llm, system_prompt="SYSTEM", summarizer=summarizer)

    result = asyncio.run(
        service.ask(
            "Answer",
            "session-1",
            ["User: one"],
            current_question="AI: Question",
            summary="prev",
            summarize=True,
        )
    )

    assert result.text == "summary"
    assert result.response_type == "summary"
    assert summarizer.last_inputs == ("AI: Question", "Answer", "prev")


def test_stream_answer_returns_summary_when_requested():
    llm = FakeLLM()
    summarizer = FakeSummarizer()
    service = ChatService(llm=llm, system_prompt="SYSTEM", summarizer=summarizer)

    async def _collect():
        chunks = []
        async for chunk in service.stream_answer(
            "Answer",
            "session-1",
            [],
            current_question="AI: Question",
            summary="prev",
            summarize=True,
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())
    assert chunks == ["summary"]


def test_ask_strips_completion_token():
    llm = FakeDoneLLM("PIXIE_SUMMARY_READY")
    service = ChatService(
        llm=llm,
        system_prompt="SYSTEM",
        completion_token="PIXIE_SUMMARY_READY",
    )

    result = asyncio.run(service.ask("Question", "session-1", []))

    assert result.text == ""
    assert result.response_type == "answer"


def test_stream_answer_strips_completion_token():
    llm = FakeDoneLLM("PIXIE_SUMMARY_READY", chunks=["PIXIE_", "SUMMARY_READY"])
    service = ChatService(
        llm=llm,
        system_prompt="SYSTEM",
        completion_token="PIXIE_SUMMARY_READY",
    )

    async def _collect():
        chunks = []
        async for chunk in service.stream_answer("Question", "session-1", []):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())
    assert chunks == []
