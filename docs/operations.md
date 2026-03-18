# Operations Runbook: Direct Codex Workflow

Dit project gebruikt voorlopig een eenvoudige flow: je beschrijft het probleem direct in Codex en werkt van daaruit verder.

## Kort overzicht

1. Start in Codex (CLI, IDE extension of app).
2. Geef je probleem of wijziging direct als opdracht.
3. Laat Codex implementeren op een niet-`main` branch.
4. Push naar GitHub en werk via een PR.
5. Merge wanneer checks groen zijn.

## Waar je werkt

- Probleemomschrijving: direct in Codex
- Implementatie: lokaal in dezelfde repo
- Review/merge: GitHub PR

## Actieve GitHub workflows

- `.github/workflows/auto-pr.yml`
  - Zorgt dat een PR bestaat en projectvelden worden gezet.
- `.github/workflows/ci.yml`
  - Draait validatie voor PR's en pushes naar `main`.
- `.github/workflows/docker-publish.yml`
  - Publiceert container images alleen bij push naar `main` (of handmatig via `workflow_dispatch`).
- `.github/workflows/release-log.yml`
  - Maakt release/tag bij merge naar `main`.

## Uitgeschakelde/verwijderde flow

De volgende intake-automatisatie is verwijderd:

- `.github/workflows/issue-intake.yml`
- `.github/workflows/issue-to-agent.yml`
- `.github/workflows/copilot-handle-pr-reviews.yml`
- `.github/ISSUE_TEMPLATE/raw-intake.yml`

Gebruik dus geen `/create-ticket` of `codex:ready` issue-flow meer.

## Praktische standaardstappen

1. Maak of kies je branch (niet `main`).
2. Beschrijf taak in Codex.
3. Laat Codex wijzigen + valideren.
4. Push branch.
5. Controleer PR checks.
6. Merge als alles groen staat.

## Troubleshooting

### PR is niet automatisch aangemaakt

- Controleer branchnaam (`feat/...`, `fix/...`, `chore/...`, `test/...`, etc.).
- Controleer run van `auto-pr` in GitHub Actions.

### CI faalt

- Open de `CI` run en herstel de failing stap.
- Herpush de branch.

### Geen Docker image verwacht op feature branch

- Dat is correct: `docker-publish` draait enkel op `main` of handmatige trigger.
