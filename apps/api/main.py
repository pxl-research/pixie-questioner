"""FastAPI app for Pixie Questioner."""

from __future__ import annotations

from fastapi import FastAPI

from pixie_questioner.azure_openai_llm import AzureOpenAILLM
from pixie_questioner.chat_service import ChatService
from pixie_questioner.config import get_settings
from pixie_questioner.notulist import QuestionnaireNotulist
from pixie_questioner.schemas import (
    ChatAnswerResponse,
    ChatRequest,
    ChatResponse,
    ChatSummaryResponse,
)
from pixie_questioner.summarizer import ConversationSummarizer

app = FastAPI(title="Pixie Questioner")

settings = get_settings()
llm = AzureOpenAILLM(
    api_version=settings.advanced_azure_openai_api_version,
    api_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_key,
    model_version=settings.advanced_azure_openai_model,
    temperature=settings.advanced_azure_openai_temperature,
)
summarizer_prompt = settings.summarizer_prompt
if summarizer_prompt:
    summarizer = ConversationSummarizer(
        llm=llm,
        system_prompt_template=summarizer_prompt,
    )
else:
    summarizer = ConversationSummarizer(llm=llm)
notulist_prompt = settings.questionnaire_notulist_prompt
if notulist_prompt is None:
    raise RuntimeError("QUESTIONNAIRE_NOTULIST_PROMPT_PATH must be set.")
notulist = QuestionnaireNotulist(
    llm=llm,
    system_prompt_template=notulist_prompt,
)
chat_service = ChatService(
    llm=llm,
    question_asker_prompt=settings.system_prompt,
    notulist=notulist,
    summarizer=summarizer,
    questionnaire_template=settings.questionnaire_template,
    completion_token=settings.completion_token,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(request: ChatRequest):
    result = await chat_service.ask(
        current_answer=request.current_answer,
        state=request.state,
    )
    if result.response_type == "summary":
        if result.summary is None:
            raise RuntimeError("Summary response requires a summary payload.")
        return ChatSummaryResponse(
            type="summary",
            text=result.text,
            summary=result.summary,
            conversation_id=request.session_id,
            state=result.state,
        )
    return ChatAnswerResponse(
        type="answer",
        text=result.text,
        conversation_id=request.session_id,
        state=result.state,
    )
