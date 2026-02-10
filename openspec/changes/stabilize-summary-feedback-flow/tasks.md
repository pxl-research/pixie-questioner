## 1. State Model And Contracts

- [ ] 1.1 Add `phase` and `feedback_status` to chat state handling, including explicit defaults when state is missing these fields.
- [ ] 1.2 Persist a one-shot `final_summary` and a locked `post_summary_language` at the intake-to-feedback transition.
- [ ] 1.3 Update response schemas so pending feedback responses can contain `text`, `summary`, and `follow_up_question` together.

## 2. Post-Summary Flow Logic

- [ ] 2.1 Refactor `ChatService.ask` branching to explicit `intake -> awaiting_feedback -> completed` transitions.
- [ ] 2.2 Return the first post-intake response as open input (`type=answer`) with handoff text, frozen summary, and yes/no follow-up question.
- [ ] 2.3 In `awaiting_feedback`, normalize valid feedback to `yes`/`no` and move to `completed`.
- [ ] 2.4 In `awaiting_feedback`, keep feedback pending for unrecognized input and repeat the same yes/no follow-up question while preserving the summary.
- [ ] 2.5 In `completed`, return feedback acknowledgment with `type=summary`, include the frozen summary, and omit follow-up question.

## 3. Language Consistency

- [ ] 3.1 Lock the post-summary language at the transition point and reuse it for both handoff text and feedback question.
- [ ] 3.2 Ensure feedback-phase turns do not re-detect language and cannot flip language unexpectedly.

## 4. Regression Tests

- [ ] 4.1 Add or update unit tests for phase transitions and response-type semantics (`answer` while pending, `summary` when completed).
- [ ] 4.2 Add or update unit tests that verify summary immutability across repeated pending-feedback turns.
- [ ] 4.3 Add or update unit tests for yes/no normalization and retry behavior on invalid feedback input.
- [ ] 4.4 Add or update unit tests for language consistency between handoff text and follow-up question.
- [ ] 4.5 Update live LLM smoke coverage to assert the pending-feedback and completed-feedback payload contracts.

## 5. Validation

- [ ] 5.1 Run `docker compose -f docker-compose.dev.yml run --rm api pytest` and resolve any regressions.
- [ ] 5.2 Perform a manual end-to-end chat check to confirm: handoff text first, summary visible, then follow-up question with input still open until valid yes/no.
