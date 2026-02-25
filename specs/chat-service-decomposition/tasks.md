## 1. Baseline Safety

- [ ] 1.1 Maak een baseline-run van bestaande tests:
  - `docker compose -f docker-compose.dev.yml run --rm -e DOMAIN_PROFILE_PATH=profiles/default/domain_profile.yml api pytest`
- [ ] 1.2 Leg een korte lijst vast van kritieke regressiescenario’s (stop, off-topic, awaiting_feedback, completed, language fallback).

## 2. Extract Types And Policy

- [ ] 2.1 Maak `src/pixie_questioner/chat/types.py` met `ChatState`, `ChatResult`, fase-constanten.
- [ ] 2.2 Maak `src/pixie_questioner/chat/policy.py` met `ChatDomainPolicy`, `ChatQuickReply`.
- [ ] 2.3 Update imports in `chat_service.py` zonder gedrag te wijzigen.
- [ ] 2.4 Run targeted tests: `tests/test_chat_service.py`.

## 3. Extract Relevance And Stop Detection

- [ ] 3.1 Maak `src/pixie_questioner/chat/relevance.py` en verplaats `LLMAnswerRelevanceClassifier`.
- [ ] 3.2 Maak `src/pixie_questioner/chat/stop_detection.py` en verplaats token/phrase matching helpers.
- [ ] 3.3 Houd exacte stop semantics identiek (context tokens + sequence matching).
- [ ] 3.4 Run targeted tests: relevance + stop tests.

## 4. Extract Language Utilities

- [ ] 4.1 Maak `src/pixie_questioner/chat/language.py` met detect/select helpers.
- [ ] 4.2 Injecteer benodigde config (supported languages, names, default language) expliciet.
- [ ] 4.3 Behoud huidige heuristische thresholds ongewijzigd.
- [ ] 4.4 Run targeted language tests.

## 5. Extract Messaging Helpers

- [ ] 5.1 Maak `src/pixie_questioner/chat/messages.py` met:
  - localized option selectie
  - quick reply fallback
  - handoff/feedback/goodbye/off-topic builders
  - term replacement logic
- [ ] 5.2 Verwijder equivalenten uit `chat_service.py`.
- [ ] 5.3 Run targeted messaging tests.

## 6. Isolate Flow Decisions

- [ ] 6.1 Evalueer na fases 2-5 of `ChatService.ask()` nog te complex is voor directe leesbaarheid.
- [ ] 6.2 Maak enkel indien nodig `src/pixie_questioner/chat/flow.py` voor pure state-transition helpers.
- [ ] 6.3 Laat `ChatService.ask()` enkel orchestreren en helpercalls doen.
- [ ] 6.4 Voeg regressietests toe voor iedere fase-transitie.

## 7. Final Validation

- [ ] 7.1 `docker compose -f docker-compose.dev.yml run --rm -e DOMAIN_PROFILE_PATH=profiles/default/domain_profile.yml api pytest`
- [ ] 7.2 `docker compose -f docker-compose.dev.yml run --rm api ruff check .`
- [ ] 7.3 `docker compose -f docker-compose.dev.yml run --rm api black .`
- [ ] 7.4 `docker compose -f docker-compose.dev.yml run --rm api uv tool run ty check`
- [ ] 7.5 `docker compose -f docker-compose.dev.yml run --rm api uv tool run pip-audit`

## 8. Done Criteria

- [ ] 8.1 Geen API contractwijzigingen (`POST /chat`, state shape).
- [ ] 8.2 `chat_service.py` is merkbaar kleiner en bevat geen losse utilityblokken meer.
- [ ] 8.3 Alle bestaande tests + nieuwe regressietests groen.
- [ ] 8.4 Duidelijke modulegrenzen gedocumenteerd in code (docstrings/type hints).
- [ ] 8.5 Geen nieuwe module die alleen pass-through wrappers bevat.
- [ ] 8.6 `flow.py` bestaat alleen als de leesbaarheid er aantoonbaar op vooruitgaat.
