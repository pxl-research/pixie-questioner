# .agents/push.md

Push and merge workflow for `pixie-questioner`.

---

## Pre-Push Checklist

1. Branch is not `main`/`master`
2. Local validation passed (see `.agents/testing.md`)
3. No forbidden changes, temp files, or `docs/plans/` files in history
4. Diff is focused and commit history is coherent
5. `README.md` / `AGENTS.md` still accurate for the changes

---

## Push Sequence

1. Run final validation
2. Review `git diff` one last time
3. Push (`git push -u origin <branch>` if no upstream, else `git push`)
4. Wait for CI + Copilot review comments
5. Address justified comments → commit → push again
6. Merge only when CI passes and review is resolved

The trigger phrase `Push deze branch` means: run full pre-push maintenance and push.