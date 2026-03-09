# Push Trigger Design

**Goal:** Make `Push deze branch` a complete push workflow that keeps code, documentation, and contracts aligned before anything goes to remote.

**Decision:** Keep the fixed natural-language trigger in `AGENTS.md`, but expand it into a pre-push maintenance workflow instead of a plain `git push` shortcut.

**Details:**

- `Push deze branch` is the standard explicit approval phrase for remote pushes.
- That phrase triggers a pre-push maintenance pass before any push command runs.
- The pre-push maintenance pass must review `README.md`, `AGENTS.md`, and any other relevant documentation touched by or affected by the finished work.
- The same pass must verify whether `pixie-contracts` is still respected by the current code and documentation, and whether contract or example updates are needed.
- If the required documentation or contract changes are clear, Codex should make them automatically, validate them, and commit them on the current branch before pushing.
- If the required change is ambiguous, or if `pixie-contracts` is not available locally, Codex may stop and ask the user a focused question or report the blocker.
- Only after the maintenance pass is complete should Codex push the branch to `origin`.
- If the branch has no upstream yet, Codex should use `git push -u origin <current-branch>`.
- If an upstream already exists, Codex should use `git push`.
- Any explicit user instruction naming a different remote or branch overrides the default.

**Why this design:**

- It keeps the workflow repo-local and portable across clients.
- It prevents a push-fix-push loop by moving the reflection step before the push.
- It preserves the existing rule that pushes require explicit user approval.
- It biases toward automatic maintenance while still allowing narrow user questions when the intent is genuinely unclear.
