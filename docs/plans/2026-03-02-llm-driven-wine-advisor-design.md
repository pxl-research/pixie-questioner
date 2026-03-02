# LLM-Driven Wine Advisor Design

Date: 2026-03-02
Branch: `feat/llm-driven-wine-advisor`
Status: Approved in chat

## 1. Scope And Intent

The current categorized questionnaire flow is replaced with an LLM-driven wine advisor flow.
The core business goal remains unchanged: recommend wines for a dish.
If the customer indicates there is no food pairing context, the advisor continues in a "drinking only" mode.

## 2. Explicit Product Decisions

1. Recommendations must be 100% from the provided catalog (`cretskens_wijnen_reduced.xlsx`).
2. Backward compatibility for API/state is explicitly allowed to break for this change.
3. Catalog loading must support both:
- `WINE_CATALOG_PATH` environment variable
- fallback default file path inside the repo
4. Conversation cadence:
- exactly one question per turn while in intake
- maximum of 6 questions
- advisor may finish earlier when enough signal is available
5. Recommendation count:
- normally 3 to 5 wines
- if strict constraints produce fewer valid matches, returning 1 to 2 is allowed
6. Minimal recommendation fields per wine:
- `name`
- `price_eur`
- `country`
- `region`

## 3. Data Source Reality (validated)

Catalog file currently contains one sheet (`wijnen`) with columns:
- `barcode`
- `naam`
- `prijs`
- `land`
- `regio`
- `kleur`
- `druiven_beschrijving`
- `geur_beschrijving`
- `smaak_beschrijving`
- `foodpair_beschrijving`

## 4. Target Runtime Architecture

## 4.1 Service split

Introduce a new core service: `WineAdvisorService`.
It owns turn orchestration and replaces questionnaire/notulist/handoff behavior.

Internal collaborators:
- `WineCatalogLoader`: loads and validates catalog rows from `.xlsx`
- `WineCatalogIndex`: in-memory lookup and scoring utilities
- `LLMTurnController`: decides `question` vs `recommend` and proposes one next question
- `LLMWineRanker`: ranks final selections from server-provided candidate set

## 4.2 Enforcement boundaries

LLM controls the dialogue strategy, but server enforces hard guarantees:
- one question max per intake turn
- no more than 6 turns
- all returned wines exist in catalog
- unique selections only
- result count constrained by configured min/max and availability

## 5. API And State Changes

## 5.1 `POST /chat` response types

New discriminated response union:

1. `type: "question"`
- `text`: single next question
- `conversation_id`
- `state`

2. `type: "recommendations"`
- `text`: short lead-in message
- `recommendations`: list of recommendation objects
- `conversation_id`
- `state`

Recommendation object:
- `name: str`
- `price_eur: float`
- `country: str`
- `region: str`

## 5.2 New state model (opaque to clients)

Proposed fields:
- `phase: "intake" | "completed"`
- `turn_count: int`
- `last_question: str`
- `meal_mode: "pairing" | "drinking_only" | "unknown"`
- `working_context: str`
- `chat_history: list[{role, text}]`
- `candidate_barcodes: list[str]`
- `selected_barcodes: list[str]` (set in completed phase)

Rules:
- increment `turn_count` each intake turn
- allow completion before turn 6
- force completion attempt at turn 6 if not completed
- completed state is idempotent (return same selected wines)

## 6. LLM Flow And Retrieval

Per turn:
1. Controller call (structured output): action + optional question + query hints + meal mode.
2. Server retrieval/scoring from catalog using current context.
3. If action is `question` and turn limit not reached:
- return one question
4. Else finalize:
- ranker call selects barcodes from server candidate set
- validate output and map to canonical catalog rows

Retrieval/scoring strategy (minimal first version):
- token overlap matching over `naam`, `foodpair_beschrijving`, `geur_beschrijving`, `smaak_beschrijving`, `druiven_beschrijving`
- score boosts for color/origin/price consistency when stated
- deterministic fallback ranking when ranker output is invalid

## 7. Failure Handling

1. Missing/invalid catalog path: fail fast on startup.
2. LLM controller failure during intake: return safe fallback question.
3. LLM ranker invalid output:
- one repair retry
- then deterministic fallback over candidate ranking
4. If only 1-2 valid catalog matches remain under strict constraints, return 1-2.

## 8. Testing Strategy

1. Unit tests for catalog loader/index validation and retrieval.
2. Unit tests for turn controller orchestration and state transitions.
3. Regression tests for:
- early completion before 6 turns
- forced completion at turn 6
- one-question-per-turn behavior
- result count rules (3-5 standard, 1-2 allowed when constrained)
- strict catalog membership and uniqueness
4. API contract tests for new response union and payload shapes.

## 9. Non-Goals

1. No dependency upgrades or `uv.lock` regeneration.
2. No infrastructure changes.
3. No optimization beyond correctness/stability for first release.
