# Cretskens Geur/Smaak Intake: Onderzoek En Redesignvoorstel

## Doel

Onderzoeken of de huidige `aroma_taste_classifier` nog een verdedigbare plaats heeft in de Cretskens-flow, gegeven de actuele catalogusdata in `profiles/cretskens/cretskens_wijnen_reduced.xlsx`, en een redesign uitwerken zonder runtime-aanpassingen door te voeren.

## Korte Conclusie

De huidige geur/smaak-classifier is inhoudelijk gebaseerd op een ouder datamodel dan de catalogus die vandaag effectief door de advisor gebruikt wordt.

Het huidige model kan nog "werken", maar alleen indirect:

- doordat een LLM de boolean-keys semantisch kan interpreteren;
- doordat de samenvatting soms nog natuurlijke woorden zoals `fris` of `fruitig` bewaart;
- doordat de flow failures in de classifier vaak opvangt zonder meteen te crashen.

Dat is geen robuuste architectuur. De boolean-laag is vandaag geen first-class representatie van de catalogus, maar een legacy tussenvorm die:

- vrije tekst destructief overschrijft;
- informatie dupliceert;
- niet rechtstreeks aansluit op retrieval;
- en downstream gedrag afhankelijk maakt van toevallige LLM-compensatie.

Mijn aanbeveling is daarom een redesign naar een `raw-preference-first` model: behoud geur/smaak als vrije tekst als primaire representatie, en maak eventuele afgeleide facetten optioneel en additief in plaats van destructief en leidend.

## Onderzoeksbasis

Ik heb de volgende onderdelen onderzocht:

- intake/runtime flow:
  - `src/pixie_questioner/chat/intake_updates.py`
  - `src/pixie_questioner/chat/intake_outcomes.py`
  - `src/pixie_questioner/chat/feedback_turn.py`
  - `src/pixie_questioner/chat/categorized_handoff.py`
  - `src/pixie_questioner/cretskens_service.py`
- advisor/retrieval flow:
  - `src/pixie_questioner/wine_advisor/catalog_loader.py`
  - `src/pixie_questioner/wine_advisor/retrieval.py`
  - `src/pixie_questioner/wine_advisor/service.py`
  - `src/pixie_questioner/wine_advisor/llm_controller.py`
  - `src/pixie_questioner/wine_advisor/llm_ranker.py`
- prompts en template:
  - `profiles/cretskens/questionnaire_template.txt`
  - `profiles/cretskens/questionnaire_notulist.txt`
  - `profiles/cretskens/question_asker.txt`
  - `profiles/cretskens/aroma_taste_classifier.txt`
  - `profiles/cretskens/turn_controller_prompt.txt`
  - `profiles/cretskens/ranker_prompt.txt`
- dataset:
  - `profiles/cretskens/cretskens_wijnen_reduced.xlsx`
- tests:
  - `tests/test_chat_service.py`
  - `tests/test_wine_retrieval.py`
  - `tests/test_wine_advisor_service.py`

Verificatie die effectief uitgevoerd werd:

- `RUN_LIVE_LLM_TESTS=0 docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_chat_service.py -k "categorized_handoff and aroma"`
- `RUN_LIVE_LLM_TESTS=0 docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_wine_retrieval.py`
- `RUN_LIVE_LLM_TESTS=0 docker compose -f docker-compose.dev.yml run --rm api pytest tests/test_wine_advisor_service.py -k "recommend_from_context or context_fallback"`

Alle drie de gerichte testsets slagen. Dat bevestigt dat de huidige boolean-flow bewust ondersteund wordt in de codebase, niet dat hij nog goed aansluit op de echte data.

## Wat De Dataset Vandaag Echt Is

De actuele workbook bevat exact 10 kolommen:

1. `barcode`
2. `naam`
3. `prijs`
4. `land`
5. `regio`
6. `kleur`
7. `druiven_beschrijving`
8. `geur_beschrijving`
9. `smaak_beschrijving`
10. `foodpair_beschrijving`

Belangrijke vaststelling:

- er zijn geen aparte boolean-kolommen voor geur of smaak;
- de sensorische data zit volledig in twee vrije tekstvelden:
  - `geur_beschrijving`
  - `smaak_beschrijving`

Voorbeelden uit de echte catalogus:

- `Alalia blanc`
  - geur: `Citrus, frisse agrum, witte bloesems`
  - smaak: `Wit steenfruiten met elegante, zachte zuren, sappig`
- `Amicale Wit`
  - geur: `Intens en aromatisch bouquet met tonen van tropisch fruit, abrikoos en een subtiele hint van vanille`
  - smaak: `Evenwichtig en fris in de mond, met een mooie zuurbalans en een elegante, zachte afdronk.`
- `Amicale Rood`
  - geur: `Rijp rood fruit zoals kers, braam en zwarte bes, aangevuld met subtiele kruidige tonen`
  - smaak: `Soepel en elegant, met een kruidige toets in de afdronk.`

Dit is semantisch rijke beschrijvingstekst, geen feature matrix.

## Huidige Dataflow Voor Geur/Smaak

### 1. Intake-template en notulist verwachten vrije tekst

`profiles/cretskens/questionnaire_template.txt` definieert veld 8 expliciet als:

- `8. Gewenste geur/smaak (vrije tekst): []`

`profiles/cretskens/questionnaire_notulist.txt` instrueert ook expliciet:

- veld 8 moet een korte vrije tekstsamenvatting blijven;
- expliciete negaties moeten behouden blijven;
- de laatste correctie van de gebruiker wint;
- de formulering moet dicht bij geur/smaakgroepen blijven.

Met andere woorden: de intakeprompt zelf is ontworpen rond vrije tekst, niet rond booleans.

### 2. De intake-update overschrijft die vrije tekst alsnog

In `src/pixie_questioner/chat/intake_updates.py` wordt bij categorized handoff altijd:

- eerst de questionnaire geupdatet;
- daarna `normalize_questionnaire(...)` aangeroepen.

In `src/pixie_questioner/chat/categorized_handoff.py` doet `normalize_questionnaire(...)` vervolgens dit:

- lees veld 8 uit de questionnaire;
- parse bestaande flags als er al JSON/`key=true`-achtige output staat;
- anders: stuur vrije tekst naar `aroma_taste_classifier`;
- vervang veld 8 in de questionnaire door geserialiseerde boolean-flags.

Dat is een destructieve normalisatie: de oorspronkelijke formulering van de klant verdwijnt uit de questionnaire zodra classificatie lukt.

### 3. De handoff dupliceert dezelfde info nog eens

`build_handoff_categories_v1(...)` doet daarna opnieuw:

- core categorieen classificeren;
- veld 8 nog eens lezen;
- er aroma/taste-flags uit halen of opnieuw laten classificeren;
- die flags toevoegen aan `handoff_categories_v1`.

Dus hetzelfde geur/smaak-signaal komt in de huidige flow potentieel twee keer terug:

- 1 keer als overschreven questionnaire-veld;
- 1 keer in `handoff_categories_v1`.

### 4. De advisor krijgt een gemengde contextstring

`src/pixie_questioner/cretskens_service.py` bouwt voor recommendations een `working_context` op uit:

- `summary`
- `questionnaire`
- `handoff_categories_v1`

De wine advisor krijgt dus geen nette typed preference-structuur, maar een samengestelde contextstring waarin natuurlijke tekst en machine-keys door elkaar staan.

### 5. Retrieval werkt op vrije tekst in de catalogus

`src/pixie_questioner/wine_advisor/retrieval.py` scoret records via token-overlap tussen query en cataloguscorpus.

Dat corpus bestaat uit:

- naam
- druivenbeschrijving
- geur_beschrijving
- smaak_beschrijving
- foodpair_beschrijving
- land
- regio
- kleur

Er is geen retrieval- of cataloguslaag die `geur_*` of `smaak_*` flags first-class begrijpt.

## Root Cause Analyse

### Symptoom

Het systeem geeft een fout in of vlak na de geur/smaak-vraag bij een antwoord zoals `fris en fruitig`.

### Mechanisch relevante observaties

1. De classifier draait precies op het moment dat vrije tekst in veld 8 wordt vastgelegd.
2. De classifier-output wordt niet alleen gelezen, maar ook teruggeschreven in de questionnaire.
3. De advisor krijgt daarna die genormaliseerde output als onderdeel van zijn context.
4. De catalogus zelf bevat vrije beschrijvingstekst, geen corresponderende boolean-features.

### Waarschijnlijke architecturale root cause

De huidige flow gebruikt een legacy representatie van geur/smaak als primaire tussenvorm, terwijl de rest van het systeem vandaag werkt met natuurlijke beschrijvingen en LLM-interpretatie.

Dat veroorzaakt drie structurele problemen:

1. **Model mismatch**
   - intake produceert boolean-achtige facets;
   - catalogus bewaart rijke beschrijvingstekst;
   - retrieval zoekt op tekstuele overlap in beschrijvingstekst.

2. **Destructief informatieverlies**
   - `fris en fruitig` is semantisch dicht bij de catalogus;
   - `{"smaak_fruitig": true, "smaak_fris_zuur": true}` is dat niet.

3. **Verborgen downstream afhankelijkheid**
   - om toch goed te werken moet een downstream LLM die keys weer "terugvertalen" naar natuurlijke voorkeuren;
   - als dat niet lukt, zakt de kwaliteit of faalt de flow.

### Belangrijk: waar de oude aanpak nog wel "werkt"

De oude aanpak is niet totaal dood. Ze kan nog gedeeltelijk functioneren doordat:

- de `turn_controller` een LLM is en keynamen semantisch kan lezen;
- de samenvatter soms natuurlijke smaakwoorden bewaart;
- de fallback-retrieval op de rest van de context nog iets bruikbaars kan vinden.

Maar dat is geen bewijs dat het ontwerp nog goed is. Het is bewijs dat andere lagen de mismatch soms opvangen.

## Bewijs Dat De Boolean-Keys Niet Op De Catalogus Aansluiten

Ik heb de echte catalogus getokenized en de volgende tellingen bekeken in geur+smaakbeschrijvingen:

- `fruitig`: 41 voorkomens
- `fris`: 45 voorkomens
- `citrus`: 28 voorkomens
- `smaak_fruitig`: 0 voorkomens
- `smaak_fris_zuur`: 0 voorkomens
- `geur_citrus`: 0 voorkomens

Ik heb ook retrieval vergeleken op de echte catalogus:

### Query A: natuurlijke voorkeur

- query: `fris en fruitig`
- resultaat: topmatches met expliciete overlap op `fris` en `fruitig`

### Query B: huidige genormaliseerde keys

- query: `smaak_fruitig smaak_fris_zuur`
- resultaat: geen overlap op die keys in de catalogus zelf

Dat toont aan:

- natuurlijke taal sluit direct aan op de catalogus;
- de huidige boolean-keys sluiten niet direct aan op de catalogus;
- dus de booleanlaag kan retrieval niet principieel helpen tenzij een LLM de sleutelwoorden eerst vertaalt.

## Kan De Oude Werking Nog Verdedigd Worden?

### Kort antwoord

Niet als primaire representatie. Hoogstens nog als optionele afgeleide metadata.

### Wanneer nog net wel

De oude classifier kan nog verdedigbaar zijn als aan alle onderstaande voorwaarden tegelijk voldaan wordt:

- de originele vrije tekst blijft volledig behouden;
- de booleans worden alleen additief opgeslagen;
- geen enkele retrieval- of rankingstap hangt af van die booleans;
- de booleans worden enkel gebruikt voor analytics, debugging of explainability;
- downstream prompts blijven expliciet op natuurlijke tekst steunen.

### Waarom de huidige vorm niet houdbaar is

De huidige vorm voldoet daar niet aan, omdat ze:

- vrije tekst overschrijft;
- de booleanlaag in de recommendation context injecteert;
- aroma/taste in `handoff_categories_v1` top-level maakt alsof het een first-class retrievaldimensie is;
- een legacy taxonomie bewaart zonder actieve catalogus- of retrievalkoppeling.

## Redesigndoelen

Een nieuw ontwerp moet volgens mij deze doelen halen:

1. Geur/smaak van de klant blijft primair vrije tekst.
2. De datarepresentatie sluit direct aan op de actuele catalogus.
3. Geen destructief herschrijven van questionnaire-veld 8.
4. Eventuele afgeleide structuur is additief, expliciet en optioneel.
5. Working context voor de advisor blijft natuurlijk taalgedreven.
6. Publiek API-contract blijft behouden.
7. State-evolutie blijft expliciet en backward-compatible.

## Drie Redesignopties

## Optie A: Minimal containment

### Idee

Behoud de classifier, maar maak hem volledig secundair:

- veld 8 blijft altijd ruwe vrije tekst;
- classifier schrijft niets meer terug naar de questionnaire;
- eventuele flags worden apart opgeslagen als interne afgeleide metadata;
- `handoff_categories_v1` krijgt alleen core categorieen die echt retrieval-achtig zijn:
  - foodpair
  - kleur
  - land/regio
  - prijs

### Voordelen

- laagste migratierisico;
- bestaande classifier kan nog gebruikt worden voor observability;
- weinig verandering aan prompts nodig.

### Nadelen

- legacy taxonomie blijft bestaan;
- twee modellen voor hetzelfde domein blijven naast elkaar leven;
- risico dat toekomstige code alsnog op de afgeleide flags gaat leunen.

### Beoordeling

Verdedigbaar als tussenstap, maar niet als eindbeeld.

## Optie B: Raw-preference-first met optionele facetten

### Idee

Maak vrije tekst de primaire bron en behandel afgeleide facetten als optionele verrijking:

- questionnaire veld 8 blijft letterlijk vrije tekst;
- advisor krijgt expliciet natuurlijke voorkeurstaal in de context;
- `handoff_categories_v1` of een opvolger bevat alleen structurering die echt stabiel en catalogus-relevant is;
- als facetten nuttig blijven, sla ze apart op in een interne key zoals:
  - `aroma_taste_facets`
  - of `advisor_preference_hints`
- die facetten zijn dan supplementair, niet leidend.

### Voordelen

- sluit het best aan op de actuele catalogus;
- sluit het best aan op LLM-driven ranking;
- behoudt de rijke taal van de gebruiker;
- laat nog explainability of analytics toe;
- makkelijk backward-compatible uit te voeren.

### Nadelen

- vraagt bewuste migratie van tests en state-output;
- dwingt helder onderscheid af tussen core categorieen en zachte preference-signalen.

### Beoordeling

Dit is de aanbevolen optie.

## Optie C: Volledige vereenvoudiging, classifier schrappen

### Idee

Verwijder de aroma/smaak-classifier volledig:

- veld 8 blijft vrije tekst;
- geen boolean-taxonomie meer;
- geen afgeleide aroma-flags in state of handoff;
- advisor werkt alleen met natuurlijke taal plus core constraints.

### Voordelen

- conceptueel het eenvoudigst;
- geen semantische mismatch meer;
- geen prompt- of parse-instabiliteit rond JSON booleans.

### Nadelen

- geen genormaliseerde smaakfacetten meer voor analyse of debugging;
- minder ruimte voor toekomstige hybride retrieval zonder nieuw ontwerp.

### Beoordeling

Sterk als productkeuze wanneer jullie bewust volledig op natuurlijke taal en LLM-redenering willen steunen.

## Aanbevolen Redesign

Ik raad **Optie B** aan.

### Waarom Optie B beter is dan A

Optie A stopt het ergste probleem, maar laat het oude model in leven zonder duidelijke productreden. Optie B maakt de juiste bron-van-waarheid expliciet:

- de gebruiker geeft vrije tekst;
- de catalogus bewaart vrije tekst;
- de LLM rankt op vrije tekst;
- dus vrije tekst moet de dominante representatie zijn.

### Waarom Optie B beter is dan C

Optie C is zuiverder, maar snijdt meteen alle afgeleide structuur weg. Als jullie later toch analyse, clustering of explainability op smaakprofielen willen, is het beter om facetten additief te behouden dan ze nu definitief te verwijderen.

## Concreet Doelbeeld

### 1. Questionnaire

Veld 8 blijft:

- `8. Gewenste geur/smaak (vrije tekst): fris en fruitig`

Nooit:

- `8. Gewenste geur/smaak (vrije tekst): {"smaak_fruitig":true,"smaak_fris_zuur":true}`

### 2. Handoff

Core handoff bevat alleen harde of semi-harde constraints:

- gerecht/foodpair
- kleur
- land/regio
- prijs

Geur/smaak gaat niet langer als top-level legacy booleans in dezelfde categoriecontainer zitten.

### 3. Advisor context

De advisor krijgt expliciet natuurlijke voorkeurssignalen, bijvoorbeeld:

- `Preferred aroma/taste: fris en fruitig`

Eventueel aangevuld met:

- een compacte natuurlijke parafrase;
- optionele afgeleide facetten, maar dan onder een aparte, duidelijk secundaire sleutel.

### 4. Eventuele optionele facets

Als facetten behouden blijven, maak dan expliciet dat ze:

- afgeleid zijn;
- niet de source of truth zijn;
- niet gebruikt mogen worden als exclusieve retrieval-query;
- en niet teruggeschreven worden naar veld 8.

## Voorgestelde Interne Doelsituatie

Zonder het publieke API-schema te breken zou de interne richting er conceptueel zo kunnen uitzien:

- questionnaire:
  - veld 8 blijft raw text
- summary:
  - natuurlijke taal
- handoff core:
  - alleen stabiele categorieen
- optional enrichment:
  - `aroma_taste_facets`
  - `aroma_taste_notes`
  - of `advisor_preference_hints`

Belangrijk:

- additieve state-uitbreiding is hier verdedigbaar;
- destructieve vervanging van bestaand veldgedrag niet.

## Migratie-aanpak Voor Een Toekomstige Implementatie

### Fase 1: Non-destructive alignment

- stop met veld 8 te overschrijven;
- laat `handoff_categories_v1` aroma-flags niet langer de hoofdrol spelen;
- voeg eventueel een nieuwe interne key toe voor afgeleide facetten.

### Fase 2: Advisor-context opschonen

- laat `working_context` expliciet natuurlijke geur/smaakvoorkeuren bevatten;
- gebruik de ruwe voorkeurstaal als primaire input voor `turn_controller` en `ranker`.

### Fase 3: Legacy taxonomie degraderen

- laat tests niet langer eisen dat veld 8 naar booleans wordt omgezet;
- beperk de oude aroma-keyset tot een optionele compatibiliteitslaag;
- evalueer later of die compatibiliteitslaag nog nut heeft.

## Testimpact Van Dit Redesign

De huidige tests beschermen expliciet het legacy gedrag. Voorbeelden:

- tests die verwachten dat `fruitig en fris` resulteert in:
  - `smaak_fruitig = true`
  - `smaak_fris_zuur = true`
- tests die wrapped JSON, retrygedrag en key-mapping rond de aroma-classifier verankeren

Bij implementatie van het redesign zal dus ook de teststrategie moeten verschuiven van:

- "werd de juiste boolean-key geschreven?"

naar:

- "bleef de vrije tekst behouden?"
- "werd de juiste advisor-context opgebouwd?"
- "blijft retrieval/ranking semantisch correct?"
- "blijft eventuele afgeleide metadata optioneel en non-destructive?"

## Risico's Als Jullie Niets Veranderen

1. Geur/smaakvoorkeuren blijven afhankelijk van een legacy taxonomie die niet op de catalogus rust.
2. Relevante klanttaal kan verloren gaan door overschrijven van veld 8.
3. Downstream recommendation gedrag blijft gevoelig voor hoe goed de LLM sleutelwoorden terug vertaalt.
4. Productfouten zullen moeilijk uitlegbaar blijven, omdat de "bron van waarheid" verdeeld is over:
   - summary
   - genormaliseerde questionnaire
   - handoff json
   - en impliciete LLM-interpretatie

## Eindoordeel

De huidige `aroma_taste_classifier` is geen goede primaire representatie meer voor het huidige Cretskens-systeem.

De classifier kan hoogstens nog nuttig zijn als optionele, additieve verrijking. In zijn huidige vorm is hij een legacy ontwerp dat botst met:

- de actuele catalogusstructuur;
- de LLM-driven selectieflow;
- en de expliciete promptinstructie dat veld 8 vrije tekst is.

Als jullie een robuuste volgende stap willen zetten zonder het publieke API-contract te breken, is de juiste richting:

- **vrije tekst behouden als bron van waarheid**
- **eventuele facetten alleen additief maken**
- **en de advisor-context opnieuw centreren rond natuurlijke voorkeurstaal**

## Open Vragen Voor Implementatie

1. Is er buiten deze repo nog een externe consumer die vandaag expliciet op aroma-keys in `handoff_categories_v1` leunt?
2. Willen jullie afgeleide geur/smaakfacetten nog behouden voor analytics of explainability, of mogen die volledig verdwijnen?
3. Willen jullie in de advisor-context alleen ruwe klanttaal tonen, of ook een LLM-parafrase van die voorkeur voor extra stabiliteit?
