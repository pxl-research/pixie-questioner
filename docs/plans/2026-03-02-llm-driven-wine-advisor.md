# LLM-Driven Wine Advisor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current questionnaire/categorized-handoff flow with an LLM-driven wine advisor that asks one question per turn and returns catalog-only recommendations.

**Architecture:** Add a new `WineAdvisorService` that orchestrates turn control, catalog retrieval, ranking, and state transitions. Keep all business logic in `src/` and keep `apps/api/*` transport-only. Enforce hard guarantees server-side (catalog-only results, max turn limit, one question per turn).

**Tech Stack:** Python 3.12+, FastAPI, Pydantic v2, LangChain AzureChatOpenAI wrapper, Docker Compose dev workflow.

---

### Task 1: Introduce New API Contract And Advisor State Models

**Files:**
- Modify: `src/pixie_questioner/schemas.py`
- Create: `src/pixie_questioner/wine_advisor/types.py`
- Test: `tests/test_chat_contract.py`

**Step 1: Write the failing contract tests**

Add tests asserting:
- `ChatResponse` discriminator supports `question` and `recommendations`
- recommendation payload has `name`, `price_eur`, `country`, `region`
- response no longer references old `summary`-style union

**Step 2: Run test to verify it fails**

Run:
```bash
docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_chat_contract.py -q
```
Expected: FAIL on old response-model expectations.

**Step 3: Implement minimal schema/types changes**

- Replace old response union in `schemas.py` with new response models.
- Add typed internal advisor state dataclasses/models in `wine_advisor/types.py`.

**Step 4: Run contract test to verify it passes**

Run:
```bash
docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_chat_contract.py -q
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/pixie_questioner/schemas.py src/pixie_questioner/wine_advisor/types.py tests/test_chat_contract.py
git commit -m "feat(api): introduce wine advisor response and state models"
```

### Task 2: Implement Catalog Loader (`.xlsx`) With Env-First Path Resolution

**Files:**
- Modify: `src/pixie_questioner/config.py`
- Create: `src/pixie_questioner/wine_advisor/catalog_loader.py`
- Create: `src/pixie_questioner/wine_advisor/catalog_models.py`
- Test: `tests/test_config.py`
- Test: `tests/test_wine_catalog_loader.py`

**Step 1: Write failing loader/config tests**

Add tests for:
- env var `WINE_CATALOG_PATH` has priority
- fallback repo path is used when env var absent
- missing file raises RuntimeError
- invalid/missing required columns raise RuntimeError

**Step 2: Run tests to verify they fail**

Run:
```bash
docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_config.py tests/test_wine_catalog_loader.py -q
```
Expected: FAIL because loader/config not implemented.

**Step 3: Implement minimal loader**

- Parse `.xlsx` with standard library zip+xml reader.
- Map rows into typed catalog records.
- Validate required columns and normalize `price_eur` as float.

**Step 4: Run tests to verify pass**

Run:
```bash
docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_config.py tests/test_wine_catalog_loader.py -q
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/pixie_questioner/config.py src/pixie_questioner/wine_advisor/catalog_loader.py src/pixie_questioner/wine_advisor/catalog_models.py tests/test_config.py tests/test_wine_catalog_loader.py
git commit -m "feat(wine-advisor): load and validate wine catalog from xlsx"
```

### Task 3: Add Catalog Retrieval/Scoring Layer

**Files:**
- Create: `src/pixie_questioner/wine_advisor/retrieval.py`
- Test: `tests/test_wine_retrieval.py`

**Step 1: Write failing retrieval tests**

Add tests for:
- keyword matching over pairing/taste/aroma/grape/name fields
- boosts for color/origin/price hints
- deterministic ordering
- uniqueness of candidate barcodes

**Step 2: Run tests to verify fail**

Run:
```bash
docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_wine_retrieval.py -q
```
Expected: FAIL due to missing retrieval module.

**Step 3: Implement minimal retrieval/scoring**

- Tokenize normalized text.
- Compute simple weighted overlap score.
- Apply optional boosts based on extracted constraints.
- Return top-N candidate records and barcodes.

**Step 4: Run tests to verify pass**

Run:
```bash
docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_wine_retrieval.py -q
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/pixie_questioner/wine_advisor/retrieval.py tests/test_wine_retrieval.py
git commit -m "feat(wine-advisor): add deterministic catalog retrieval scoring"
```

### Task 4: Implement LLM Turn Controller And Ranker Validation

**Files:**
- Create: `src/pixie_questioner/wine_advisor/llm_controller.py`
- Create: `src/pixie_questioner/wine_advisor/llm_ranker.py`
- Modify: `src/pixie_questioner/llm_client.py` (if small helper needed)
- Test: `tests/test_wine_llm_controller.py`

**Step 1: Write failing controller/ranker tests**

Add tests for:
- controller returns structured `action` and question text
- invalid controller payload falls back to safe question
- ranker selection must be catalog-contained and unique
- repair retry invoked once before deterministic fallback

**Step 2: Run tests to verify fail**

Run:
```bash
docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_wine_llm_controller.py -q
```
Expected: FAIL because controller/ranker modules do not exist.

**Step 3: Implement minimal controller/ranker**

- Use Pydantic models for structured output parsing.
- Validate ranker barcode output against candidate set.
- Add single retry path with validation error context.
- Add deterministic fallback selector.

**Step 4: Run tests to verify pass**

Run:
```bash
docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_wine_llm_controller.py -q
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/pixie_questioner/wine_advisor/llm_controller.py src/pixie_questioner/wine_advisor/llm_ranker.py src/pixie_questioner/llm_client.py tests/test_wine_llm_controller.py
git commit -m "feat(wine-advisor): add llm turn control and validated ranking"
```

### Task 5: Implement `WineAdvisorService` Orchestration And Replace Wiring

**Files:**
- Create: `src/pixie_questioner/wine_advisor/service.py`
- Modify: `apps/api/main.py`
- Modify: `src/pixie_questioner/config.py`
- Modify: `tests/test_api.py`
- Create/Modify: `tests/test_wine_advisor_service.py`

**Step 1: Write failing service/api tests**

Add tests for:
- one question per intake turn
- early completion before turn 6
- forced completion at turn 6
- recommendations count logic (3-5 normal, 1-2 when constrained)
- completed phase idempotency
- API response shape for `question` and `recommendations`

**Step 2: Run tests to verify fail**

Run:
```bash
docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_api.py tests/test_wine_advisor_service.py -q
```
Expected: FAIL on old `ChatService` assumptions.

**Step 3: Implement minimal orchestration and wiring**

- Add `WineAdvisorService.ask(...)` with new state transitions.
- Replace old `ChatService` usage in `apps/api/main.py`.
- Keep API layer transport-only.

**Step 4: Run tests to verify pass**

Run:
```bash
docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_api.py tests/test_wine_advisor_service.py -q
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/pixie_questioner/wine_advisor/service.py apps/api/main.py src/pixie_questioner/config.py tests/test_api.py tests/test_wine_advisor_service.py
git commit -m "feat(wine-advisor): orchestrate chat turns and catalog recommendations"
```

### Task 6: Remove Obsolete Questionnaire/Handoff Runtime Paths

**Files:**
- Modify: `src/pixie_questioner/chat_service.py` (remove or deprecate old path)
- Modify: `src/pixie_questioner/chat/*` only as needed
- Modify: `tests/test_chat_service.py`

**Step 1: Write failing deprecation/removal tests**

Add or update tests to ensure no runtime dependency remains on:
- questionnaire template filling
- categorized handoff categories payload
- summary/feedback phase logic

**Step 2: Run tests to verify fail**

Run:
```bash
docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_chat_service.py -q
```
Expected: FAIL until old assumptions are removed/updated.

**Step 3: Implement minimal cleanup**

- Remove dead runtime code paths.
- Keep only modules required by new advisor flow.
- Preserve readability and file-size constraints.

**Step 4: Run tests to verify pass**

Run:
```bash
docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_chat_service.py -q
```
Expected: PASS after test updates.

**Step 5: Commit**

```bash
git add src/pixie_questioner/chat_service.py src/pixie_questioner/chat tests/test_chat_service.py
git commit -m "refactor(chat): remove obsolete questionnaire handoff flow"
```

### Task 7: Full Validation And Documentation Update

**Files:**
- Modify: `README.md`
- Modify: profile docs/config docs as needed

**Step 1: Update docs for new behavior and config**

Document:
- new response types
- `WINE_CATALOG_PATH`
- one-question-per-turn and max-6 behavior
- catalog-only recommendation guarantee

**Step 2: Run formatting, lint, typing, tests, and security checks**

Run:
```bash
docker compose -f docker-compose.dev.yml run --rm api black .
docker compose -f docker-compose.dev.yml run --rm api ruff check .
docker compose -f docker-compose.dev.yml run --rm api uv tool run ty check
docker compose -f docker-compose.dev.yml run --rm api pytest
docker compose -f docker-compose.dev.yml run --rm api ruff check . --select S
docker compose -f docker-compose.dev.yml run --rm api uv tool run bandit -r src
docker compose -f docker-compose.dev.yml run --rm api uv tool run pip-audit
docker compose -f docker-compose.dev.yml run --rm api trivy fs --scanners vuln,misconfig,secret --severity HIGH,CRITICAL --exit-code 1 .
```
Expected: all checks PASS.

**Step 3: Commit docs and final adjustments**

```bash
git add README.md
# add any additional modified docs/config files
git commit -m "docs: describe llm-driven wine advisor behavior and config"
```

### Task 8: Final Review Gate

**Files:**
- Review-only: git diff and changed files

**Step 1: Inspect final diff for forbidden changes**

Verify no unintended modifications to:
- infra/docker tooling
- dependency lock file (`uv.lock`)
- unrelated modules

**Step 2: Re-run targeted regression tests after any final edits**

Run:
```bash
docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_api.py tests/test_wine_advisor_service.py -q
```
Expected: PASS.

**Step 3: Prepare branch for integration**

Summarize:
- behavior changes
- contract changes
- validation evidence
- known residual risks
