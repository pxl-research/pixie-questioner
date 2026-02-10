# AGENTS.md (apps/api)

## Scope

This file applies only to `apps/api/*`.

## API Layer Rules

- This layer is **transport-only**
- No business logic allowed
- No state manipulation here
- No prompt logic here

Allowed:

- request/response wiring
- dependency injection
- auth / middleware wiring
- calling ChatService

Forbidden:

- modifying conversation rules
- editing questionnaire logic
- duplicating domain logic

## Testing

- Use FastAPI TestClient only
- No async event loop hacks
- Mock ChatService at the boundary only
