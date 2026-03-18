# .agents/repo-map.md

Actual file structure for `pixie-questioner`. Use this to find code quickly.

---

## Directory Layout

```
apps/api/
  main.py                    ← FastAPI entrypoint (thin, imports from api_app)

src/pixie_questioner/
  api_app.py                 ← Route handlers, response mapping, audit dispatch
  app_runtime.py             ← Runtime construction, case registration, state signers
  schemas.py                 ← Public API contracts (handle with extreme care)
  config.py                  ← Settings / env config
  chat_service.py            ← Chat orchestration (ChatService)
  cretskens_service.py       ← Cretskens case service
  llm_client.py              ← LLM client abstraction
  azure_openai_llm.py        ← Azure OpenAI implementation
  profile_prompt_loader.py   ← Profile/prompt loading
  state_signer.py            ← State signing/verification
  rate_limiter.py            ← Rate limiting
  audit_client.py            ← Audit event dispatch

  chat/                      ← Brooklyn intake flow (questionnaire, state, turns)
    flow.py, turn_handler.py, intake_turn.py, feedback_turn.py
    questionnaire.py, question_asker.py, relevance.py
    state_codec.py, types.py, messages.py, language.py
    intake_routing.py, intake_route_classifier.py, intake_outcomes.py
    intake_updates.py, summary.py, categorized_handoff.py, policy.py

  wine_advisor/              ← Cretskens wine advisor (catalog, LLM ranking, retrieval)
    service.py, llm_controller.py, llm_ranker.py
    catalog_loader.py, catalog_models.py, retrieval.py
    profile_loader.py, types.py

profiles/                    ← Case profiles and prompt templates (do not rewrite style)
  _base/                     ← Shared fallback prompts (relevance_classifier, etc.)
  brooklyn/                  ← Brooklyn-specific: domain_profile.yml + prompt .txt files
  cretskens/                 ← Cretskens-specific: domain/advisor profiles + wine catalog .xlsx

tests/                       ← Pytest suite (unit + optional live LLM smoke)
```

---

## Request Flow

1. `apps/api/main.py` → `api_app.py` receives request
2. `api_app.py` calls `ChatService` (or case-specific service)
3. Service orchestrates domain logic (chat/ or wine_advisor/) + LLM
4. Returns exactly one response message

---

## Key Boundaries

- `apps/api/*` — transport only (no business logic, no state, no prompts)
- `chat/` — owns questionnaire flow, state transitions, question ordering
- `wine_advisor/` — owns catalog retrieval and LLM-driven recommendation
- LLM layers — must not be replaced by keyword/heuristic logic

---

## Non-Obvious Conventions

- **Cases are hardcoded** in `app_runtime.py` + `api_app.py` (no plugin/registry system). Adding a case requires touching both.
- **Profile fallback**: prompts resolve case dir first, then `profiles/_base/`. Missing critical prompts = startup crash.
- **State signing**: HMAC-SHA256 with TTL. `location` is the only client-addable key (exempt from signature). `_sig`, `_ts`, `_scope` are control keys stripped before return.
- **STATE_SIGNING_KEY** must be ≥32 chars in production; <32 or empty = ephemeral key (dev only, breaks multi-worker).
- **Wine catalog** is an `.xlsx` parsed at startup — bad schema crashes immediately.
- **Rate limiting** is in-memory sliding window (per-IP: 30/60s, per-session: 100/3600s). Not shared across workers.
- **Contract version headers**: responses include `X-Pixie-Chat-Contract` and `X-Pixie-Case-Contract`.
- **Audit events** are async + best-effort; failures are logged as warnings, never block the response.
