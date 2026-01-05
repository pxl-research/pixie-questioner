"""FastAPI app for Pixie Questioner."""

from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

from pixie_questioner.azure_openai_llm import AzureOpenAILLM
from pixie_questioner.chat_service import ChatService
from pixie_questioner.config import get_settings
from pixie_questioner.schemas import ChatRequest, ChatResponse

app = FastAPI(title="Pixie Questioner")

settings = get_settings()
llm = AzureOpenAILLM(
    api_version=settings.advanced_azure_openai_api_version,
    api_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_key,
    model_version=settings.advanced_azure_openai_model,
    temperature=settings.advanced_azure_openai_temperature,
)
chat_service = ChatService(llm=llm, system_prompt=settings.system_prompt)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _sse_event(data: str) -> str:
    lines = data.splitlines() or [""]
    return "".join(f"data: {line}\n" for line in lines) + "\n"


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def chat(request: ChatRequest, stream: bool = Query(False)):
    if stream:
        async def event_stream():
            async for chunk in chat_service.stream_answer(
                question=request.question,
                session_id=request.session_id,
                history=request.history,
            ):
                yield _sse_event(chunk)
            yield _sse_event("[DONE]")

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    answer = await chat_service.ask(
        question=request.question,
        session_id=request.session_id,
        history=request.history,
    )
    return ChatResponse(answer=answer)


@app.post(
    "/ask",
    response_model=ChatResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def ask_alias(request: ChatRequest, stream: bool = Query(False)):
    return await chat(request=request, stream=stream)
