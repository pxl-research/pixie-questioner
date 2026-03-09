# Push Trigger Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Document a stable repo-local shortcut that authorizes pushing the current branch to `origin`.

**Architecture:** Store the shortcut as an explicit workflow rule in `AGENTS.md`. Do not add shell aliases, slash commands, or tool-specific config because those would be less portable than repo documentation.

**Tech Stack:** Markdown, git workflow documentation

---

### Task 1: Document the Push Trigger

**Files:**
- Modify: `AGENTS.md`
- Create: `docs/plans/2026-03-09-push-trigger-design.md`
- Create: `docs/plans/2026-03-09-push-trigger.md`

**Step 1: Write the failing test**

Configuration-only change. No automated test file is added.

**Step 2: Run test to verify it fails**

Not applicable for this documentation workflow change.

**Step 3: Write minimal implementation**

Add a `Push Workflow` section to `AGENTS.md` that defines `Push deze branch` as the standard explicit push approval phrase and documents the default push target and command behavior.

**Step 4: Run test to verify it passes**

Run: `docker compose -f docker-compose.dev.yml run --rm api python - <<'PY' ...`
Expected: the script confirms the required `AGENTS.md` push trigger text is present.

**Step 5: Commit**

```bash
git add AGENTS.md docs/plans/2026-03-09-push-trigger-design.md docs/plans/2026-03-09-push-trigger.md
git commit -m "docs(workflow): add push approval shortcut"
```
