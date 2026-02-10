## Context

The current post-summary behavior mixes rendering concerns, state progression, and completion semantics in ad-hoc conditionals. This caused multiple regressions:
- summary not shown in some clients
- conversation closing before feedback can be entered
- premature summary behavior during intake turns

The system is a turn-based API where the frontend treats `type` as a strong interaction signal. Therefore, response typing and state transitions must be explicitly modeled and stable. The change must remain minimal (no infra/dependency changes) and keep prompts/questionnaire logic intact.

## Goals / Non-Goals

**Goals:**
- Introduce a deterministic post-summary state machine with explicit phases.
- Make response payload semantics unambiguous for frontend rendering.
- Ensure summary generation is one-shot and immutable during feedback collection.
- Keep feedback handling simple and machine-friendly (yes/no only).
- Preserve language consistency for handoff and feedback question.

**Non-Goals:**
- No free-text sentiment analysis for feedback.
- No analytics pipeline or persistent datastore changes.
- No backward-compatibility shims for old response contracts.
- No prompt style rewrites or questionnaire spec changes.

## Decisions

### 1) Canonical state machine in API state
Decision:
- Use two explicit state fields:
  - `phase`: `intake | awaiting_feedback | completed`
  - `feedback_status`: `pending | yes | no`

Rationale:
- This removes ambiguity from paired booleans and supports future extension safely.
- Frontend can render deterministically from one phase enum.

Alternatives considered:
- Keep `feedback_requested` + `feedback_response` strings: rejected (higher ambiguity and harder invariants).
- Derive phase from response `type`: rejected (response type is transport hint, not workflow truth).

### 2) Response contract separates render blocks
Decision:
- During `awaiting_feedback`, return:
  - `type = answer` (input remains open)
  - `text` = assistant handoff
  - `summary` = frozen final summary
  - `follow_up_question` = yes/no question
- During `completed`, return:
  - `type = summary`
  - `text` = feedback acknowledgment
  - `summary` = same frozen final summary
  - `follow_up_question = null`

Rationale:
- Matches actual UI needs: three cards in order while still allowing user input.
- Prevents client-side guesswork and premature session closure.

Alternatives considered:
- Keep `type = summary` while feedback pending: rejected (many clients stop input on summary).
- Embed follow-up question in `text`: rejected (prevents clean card separation and order guarantees).

### 3) One-shot final summary freeze
Decision:
- Generate final summary only once at intake completion transition.
- Store frozen summary in state and reuse until `completed`.

Rationale:
- Eliminates drift and inconsistent summary rendering across feedback turns.

Alternatives considered:
- Regenerate summary on each feedback turn: rejected (non-deterministic and costly).

### 4) Strict yes/no normalization
Decision:
- Normalize recognized feedback tokens to `yes` or `no` only.
- Any unrecognized input keeps `feedback_status = pending` and repeats the same question.

Rationale:
- Keeps downstream processing simple and explicit.

Alternatives considered:
- Accept free text and map later: rejected (scope creep, ambiguous semantics).

### 5) Language lock at summary transition
Decision:
- Select and persist `post_summary_language` when entering `awaiting_feedback`.
- Use that language for handoff, summary presentation context, and follow-up question.

Rationale:
- Prevents language flips between summary and feedback prompt.

Alternatives considered:
- Re-detect language on every feedback turn: rejected (may oscillate on short inputs).

## Risks / Trade-offs

- [Risk] Frontend still keys only on `type=summary` to show summary card.
  → Mitigation: document contract: render `summary` whenever field is present, independent of `type`.

- [Risk] Existing local state snapshots may miss new fields.
  → Mitigation: server-side defaults in `_parse_state` (`phase=intake`, `feedback_status=pending` only after transition).

- [Risk] Ambiguous user inputs in feedback phase could loop.
  → Mitigation: deterministic repeat question; optional client affordance for quick yes/no buttons.

- [Risk] Language detection errors on very short text.
  → Mitigation: lock post-summary language once and avoid re-detection in feedback phase.

## Migration Plan

1. Implement state model migration in `ChatState` and `_parse_state` with safe defaults:
   - map legacy fields (`feedback_requested`/`feedback_response`) to new `phase`/`feedback_status` when present.
2. Update `ChatService.ask` transition logic to use the new state machine.
3. Update API schemas (`ChatAnswerResponse`, `ChatSummaryResponse`) to match the explicit contract.
4. Update unit tests for state transitions and payload composition.
5. Update live LLM smoke test to assert:
   - pending feedback response returns `type=answer` + summary + follow_up_question
   - completion response returns `type=summary` + summary + no follow_up_question
6. Rollout in one step (no fallback mode), since deployment has not happened yet.
7. Rollback strategy: revert commit set for this change; no data migration required.

## Open Questions

- Should we expose an explicit `render_blocks` array in a later iteration to remove field-level coupling between backend and UI?
- Should `post_summary_language` be surfaced in API response for frontend telemetry/debugging, or remain internal state only?
