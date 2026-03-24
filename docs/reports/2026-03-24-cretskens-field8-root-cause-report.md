# Cretskens Veld 8 (Geur/Smaak): Root-Cause Rapport

**Datum:** 2026-03-24  
**Status:** Forensische analyse afgerond  
**Scope:** Documentatie-only (geen codewijzigingen)

---

## 1. Context en symptomen

Tijdens de Cretskens intake zien we twee terugkerende problemen rond veld 8 (`Gewenste geur/smaak`):

1. Soms komt er een klantvraag met intern labellek, bv.  
   `Kun je je voorkeur delen voor: Gewenste geur/smaak?`
2. Soms crasht de intakeflow met:
   - log: `Question asker returned a non-question follow-up; retrying once.`
   - daarna: `RuntimeError: Question asker returned a non-question follow-up while intake still has unanswered fields.`
   - API-resultaat: `Tijdelijk niet beschikbaar.`

Deze analyse verklaart waarom net veld 8 hiervoor gevoelig is.

---

## 2. Wat maakt veld 8 bijzonder?

Veld 8 is niet gewoon "nog een veld" zoals kleur/herkomst/budget. Het combineert meerdere eigenschappen die de kans op instabiele output verhogen:

1. **Vrije, subjectieve input**
   - Veld 8 is expliciet vrije tekst en bevat nuance + negaties.
   - Referentie: `profiles/cretskens/questionnaire_notulist.txt:112-120`.

2. **Hoge promptlast in 1 vraag**
   - Voor 1 enkele vraag moeten tegelijk voorbeeldclusters, negaties, taalconsistentie, natuurlijke formulering, B1 en compactheid gerespecteerd worden.
   - Referenties:
     - `profiles/cretskens/question_asker.txt:59-67`
     - `profiles/cretskens/question_asker.txt:81-106`

3. **One-shot karakter**
   - Promptregel: voor aroma/smaak 1 gecombineerde vraag en geen extra clarificatie achteraf.
   - Referentie: `profiles/cretskens/question_asker.txt:59-60`.

Samen zorgt dit voor de moeilijkste generatie-opdracht van de hele intake.

---

## 3. Forensische bevindingen op codepad

## 3.1 Promptconflict en hoge instruction density

De `question_asker` prompt bevat tegelijk:

- verplichting tot rijke vraaginhoud voor veld 8:
  - voorbeeldclusters voor geur en smaak,
  - expliciet toelaten wat men niet wil,
  - taalconsistentie.
- algemene stijlregels:
  - geen interne labels,
  - geen carrier-zinnen zoals `Kun je je voorkeur delen voor: ...`,
  - korte, directe B1-vragen.

Concrete refs:

- aroma/smaak-specifiek: `profiles/cretskens/question_asker.txt:59-67`
- labelverbod + carrier-verbod: `profiles/cretskens/question_asker.txt:94-100`
- kort/direct/B1: `profiles/cretskens/question_asker.txt:103-106`

**Interpretatie:** deze mix verhoogt de kans dat het model ofwel terugvalt op generieke carrier-zinnen, ofwel declaratieve zinnen geeft in plaats van een vraag.

## 3.2 Outputvalidatie is vormgericht, niet kwaliteitsgericht

De runtimecheck voor een geldige vraag is momenteel vrijwel enkel:

- `endswith("?" or "？")`
- anders: retry 1x
- daarna: exception

Referenties:

- check + retry + raise:
  - `src/pixie_questioner/chat/intake_outcomes.py:191-213`
- vraagdetectie:
  - `src/pixie_questioner/chat/intake_outcomes.py:238-242`

Gevolg:

- Een **slechte vraag met `?`** (inclusief verboden label-lek) passeert.
- Een **semantisch bruikbare maar declaratieve zin zonder `?`** faalt hard.

## 3.3 Failurepad eindigt in "Tijdelijk niet beschikbaar"

Wanneer de tweede poging ook geen vraag oplevert:

1. `build_next_question_result(...)` gooit `RuntimeError`.
2. `CretskensService.ask(...)` wrapt dit naar `CretskensServiceUnavailableError`.
3. API vertaalt dit naar `Tijdelijk niet beschikbaar.`

Referenties:

- runtime error in intake outcome:
  - `src/pixie_questioner/chat/intake_outcomes.py:209-213`
- service wrapping:
  - `src/pixie_questioner/cretskens_service.py:82-88`
- API fallbacktekst (testbewijs):
  - `tests/test_api.py:762`

## 3.4 Bestaande code bevat expliciete veld-8 compensatie

Er staat al dedicated fallbackcode voor net dit scenario:

- Als veld 8 werd gevraagd maar notulist laat het blank,
  wordt field 8 rechtstreeks gevuld met het verbatim user answer.

Referenties:

- fallback callsite:
  - `src/pixie_questioner/chat/intake_turn.py:206-212`
- fallback implementatie:
  - `src/pixie_questioner/chat/intake_turn.py:343-377`

Test die dit expliciet documenteert:

- `tests/test_chat_service.py:4133-4204`

**Interpretatie:** dit is direct bewijs dat veld 8 al langer een instabiele zone is, en dat de code dit afschermt met een uitzonderingspad.

## 3.5 Retry-mechanisme helpt beperkt

Er is een regressietest voor 1 niet-vraag gevolgd door een correcte vraag, inclusief aroma/smaak-case.

Referenties:

- algemene retry test:
  - `tests/test_chat_service.py:4207-4261`
- aroma/smaak retry test:
  - `tests/test_chat_service.py:4264-4331`

Beperkingen:

- Slechts 1 retry.
- Geen inhoudsvalidatie op verboden label/cariers zodra er `?` staat.

---

## 4. Reproduceerbare observaties (uit sessie)

In deze sessie is het gedrag met gecontroleerde `FakeLLM` runs gereproduceerd:

1. **Pad A - labellek met vraagteken**
   - LLM output: `Kun je je voorkeur delen voor: Gewenste geur/smaak?`
   - Resultaat: geaccepteerd, omdat de check enkel op `?` steunt.

2. **Pad B - 2x non-question**
   - Eerste output declaratief -> retry.
   - Tweede output declaratief -> `RuntimeError` in intake outcomes.

3. **Pad C - veld 8 reeds gevraagd, notulist laat blank**
   - Question-asker wordt overgeslagen.
   - Veld 8 wordt verbatim ingevuld via fallback in `intake_turn.py`.

Deze drie paden samen verklaren het waargenomen gedrag in productie.

---

## 5. Root cause (beslissend)

De grondoorzaak is een **combinatie-effect**:

1. **Veld 8 heeft de hoogste promptcomplexiteit** (veel constraints in 1 vraag, plus natuurlijke taalverplichting).
2. **Validatie controleert vooral vorm (`?`) en niet inhoudskwaliteit** (label-lek/cariers blijven mogelijk).
3. **Retrybeleid is fragiel** (slechts 1 retry, daarna hard fail naar unavailable).
4. **Architecturale aanwijzing in code**: er bestaat al een veld-8-specifieke bypass/fallback, wat bevestigt dat dit pad structureel kwetsbaar is.

---

## 6. Korte conclusie

- Veld 8 faalt disproportioneel omdat het promptmatig de zwaarste generatie-opdracht is in de intakeflow.
- De huidige runtimeguard laat slechte vragen door zolang ze op `?` eindigen, maar crasht op niet-vragen na 1 retry.
- Het probleem is dus geen losse promptzin, maar een structurele mismatch tussen **moeilijke outputeis** en **te smalle validatie/herstelstrategie**.

---

## 7. Verouderde context in gerelateerde rapporten

> Toegevoegd op 2026-03-24 na beoordeling van alle drie de rapporten.

De twee eerdere rapporten (`aroma-taste-classifier-analysis-2026-03-23.md` en
`2026-03-23-cretskens-aroma-taste-redesign-report.md`) analyseren de
**boolean aroma/taste classifier** die in v0.2.4 (commit `114e3bd`) al volledig
is verwijderd.

Concreet:

- `aroma_taste_classifier.txt` bestaat niet meer.
- `categorized_handoff.py` bevat geen `AROMA_TASTE_KEY_ORDER`, geen parse- of
  serialisatie-code, en geen classifier-aanroep meer.
- `domain_profile.yml` heeft geen `aroma_taste_classifier_path` meer.
- `normalize_questionnaire()` overschrijft veld 8 niet meer met boolean-flags.
- `build_handoff_categories_v1()` bevat geen aroma-flags meer.

De aanbevelingen uit die rapporten zijn dus **reeds geïmplementeerd**. Het
resterende probleem (dit rapport) is de vraag*stelling*: de question_asker LLM
faalt bij het genereren van een goede vraag voor veld 8.

---

## 8. Definitieve fix

> Geïmplementeerd in branch `fix/cretskens-static-aroma-question`.

De geur/smaak-vraag is inherent statisch: de inhoud ("welke geuren/smaken
spreken je aan?", met voorbeeldclusters en negatie-toelating) varieert niet
per conversatie. Daarom wordt voor veld 8 nu een **voorgeschreven statische
vraag per taal** (NL/EN/FR) gebruikt in plaats van LLM-generatie.

Dit elimineert alle drie de failure paden uit sectie 4:

- **Pad A (label-lek)**: onmogelijk — de statische vraag bevat geen labels.
- **Pad B (non-question crash)**: onmogelijk — de statische vraag eindigt op `?`.
- **Pad C (notulist-blank fallback)**: onveranderd — de bestaande fallback
  in `_maybe_apply_aroma_taste_fallback()` vangt dit geval al op.

