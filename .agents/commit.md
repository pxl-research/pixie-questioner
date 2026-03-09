# Commit Agent

Use this agent to commit changes after they have been reviewed and approved. Can be invoked manually or automatically after tests pass.

## Workflow

1. Check the current branch with `git branch --show-current`
2. If the current branch is `main` or `master`, create and switch to a new branch before staging or committing
3. If the task context is clear, use a short descriptive branch name; otherwise ask the user for the branch name
4. Check for staged changes first (`git diff --cached`)
5. If no staged changes, use unstaged changes (`git diff`)
6. Analyze changes and group them logically
7. Create atomic commits for each logical unit

## Branch Guardrail

- Never create a commit on `main` or `master`.
- If a new branch is required, prefer a short name that matches the task context, for example `fix/state-signing` or `chore/codex-branch-workflow`.
- Do not discard unrelated working tree changes to satisfy this rule.

## Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**

- `feat`: new feature
- `fix`: bug fix
- `docs`: documentation only
- `style`: formatting, no code change
- `refactor`: code change without fix or feature
- `test`: adding or updating tests
- `chore`: maintenance tasks

**Rules:**

- No emojis in commit messages
- Subject line max 72 characters
- Use imperative mood ("add" not "added")
- Lowercase subject line
- No period at end of subject

## Splitting Commits

Split into multiple commits when:

- Changes affect different logical areas (e.g., API + tests)
- Unrelated fixes are bundled together
- Refactoring is mixed with new features

Keep in single commit when:

- Changes are tightly coupled
- Splitting would break the build

## Commands

```bash
# View current branch
git branch --show-current

# Create and switch to a branch
git switch -c <type/short-name>

# View staged changes
git diff --cached

# View unstaged changes
git diff

# Stage specific files
git add <file>

# Commit with message
git commit -m "type(scope): description"

# Stage and commit all
git add -A && git commit -m "type(scope): description"
```

## Auto-commit Trigger

After successful test run, the agent may auto-commit if:

- All tests pass (`pytest` exits 0)
- Linting passes (`ruff check .` exits 0)
- Formatting is clean (`black --check .` exits 0)

## Invocation

Prompt: "Use `.agents/commit.md` to commit the approved changes."
