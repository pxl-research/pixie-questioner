# Push Trigger Design

**Goal:** Make remote pushes explicit and easy to authorize in this repo without relying on client-specific slash commands.

**Decision:** Use a fixed natural-language trigger in `AGENTS.md` instead of adding shell aliases or editor-specific commands.

**Details:**

- `Push deze branch` is the standard explicit approval phrase for remote pushes.
- That phrase authorizes pushing the current local branch to `origin`.
- If the branch has no upstream yet, Codex should use `git push -u origin <current-branch>`.
- If an upstream already exists, Codex should use `git push`.
- Any explicit user instruction naming a different remote or branch overrides the default.

**Why this design:**

- It is repo-local and portable across clients.
- It preserves the existing rule that pushes require explicit user approval.
- It avoids introducing shell or editor setup that would not travel with the repository.
