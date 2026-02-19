# Metodologie klasifikace AI pozic

## Cíl

Identifikovat, které IT pracovní inzeráty skutečně vyžadují AI/ML dovednosti. Výsledkem je binární proměnná `has_ai` (0/1) použitá ve všech analýzách.

## Zdroje dat

Klasifikace kombinuje dva nezávislé zdroje:

| Zdroj                           | Proměnná        | Metoda                               | Vstup                                   |
| ------------------------------- | --------------- | ------------------------------------ | --------------------------------------- |
| **AI Tier**                     | `desc_tier_llm` | LLM (GPT-4o-mini)                    | Celý text popisu pozice + název         |
| **AI Skills (deterministické)** | `skills_ai_det` | Slovníkový matching (911 AI termínů) | Glassdoor metadata (`skills`)           |
| **AI Skills (LLM)**             | `desc_ai_llm`   | LLM (GPT-4o-mini)                    | Extrahováno současně s tier klasifikací |

### AI Tier klasifikace

LLM klasifikuje každou pozici do jednoho ze 4 stupňů podle **skutečné náplně práce** kandidáta:

- **core_ai** — vývoj AI modelů od základu (trénování foundation modelů, ML výzkum)
- **applied_ai** — práce s ML modely (fine-tuning, feature engineering, MLOps)
- **ai_integration** — použití AI jako black-box (volání API, integrace AI služeb)
- **none** — žádná zmínka o AI

Současně LLM vrací confidence skóre (0.0–1.0). Do analýzy vstupují pouze pozice s **confidence ≥ 0.7**.

### AI Skills

`skills_ai_det` — deterministický matching Glassdoor tagů (metadata sloupec `skills`) proti slovníku 911 AI/ML termínů.

`desc_ai_llm` — konkrétní AI technologie zmíněné v popisu pozice, extrahované LLM jako vedlejší produkt tier klasifikace (nikoliv samostatný skills task).

## Konstrukce proměnné `has_ai`

Samotný tier ani samotné skills nejsou dostatečně spolehlivé. Proto je `has_ai` definována jako **průnik obou zdrojů** s filtrací buzzwords:

### Krok 1: Sloučení skills

```
skills_combined = lower(skills_ai_det + " " + desc_ai_llm)
```

### Krok 2: Odstranění buzzwords

Z `skills_combined` se regex odstraní obecné termíny, které samy o sobě neindikují konkrétní AI dovednost:

```
buzzwords = AI, ML, artificial intelligence, machine learning, GenAI
```

Regex používá word boundaries (`\b`), aby neodstranil např. „OpenAI" nebo „MLflow".

### Krok 3: Odstranění interpunkce

Ze zbylého textu se odstraní čárky, mezery a středníky → `skills_cleaned`.

### Krok 4: Definice has_ai

```
has_ai = 1  pokud  (tier ≠ none)  ∧  (length(skills_cleaned) > 1)
```

Pozice je klasifikována jako AI job pouze pokud:

1. LLM rozpoznal AI kontext v popisu (tier ≠ none), **a zároveň**
2. existuje alespoň jeden konkrétní AI skill po odstranění buzzwords.

## Odůvodnění průnikového přístupu

### Proč nestačí samotný tier?

Tier občas nadhodnocuje — LLM přiřadí `ai_integration` pozicím kde firma jen obecně zmíní „we leverage AI", aniž by kandidát potřeboval jakékoli AI dovednosti. Analýza ukázala **208 takových případů** (5.3 % pozic s tier ≠ none), kde jedinou „AI evidencí" byly buzzwords bez konkrétní technologie.

Typické příklady: Fullstack Engineer, SOC Engineer — skills_combined obsahoval pouze `"ai"`, `"ai ai"` nebo `"machine learning"`, žádné konkrétní frameworky.

### Proč nestačí samotné skills?

`desc_ai_llm` je šumný — LLM při extrakci technologií vrací i non-AI nástroje (Kafka, Redis, Python, ETL). Analýza ukázala **1 173 pozic** s tier = none, ale neprázdným skills_cleaned. Z toho většina obsahovala nástroje jako „data science", „knowledge management", „forecasting" — nikoliv AI/ML skills.

Bez kontroly tierem by se tyto pozice staly false positives.

### Kvantitativní shrnutí

| Skupina                |         n | Popis                                                 |
| ---------------------- | --------: | ----------------------------------------------------- |
| Tier=AI ∧ Skills=YES   | **3 741** | Shoda obou zdrojů → `has_ai = 1`                      |
| Tier=AI ∧ Skills=NO    |       208 | Tier přestřelil, jen buzzwords → správně vyloučeno    |
| Tier=none ∧ Skills=YES |     1 173 | Skills šumné (non-AI technologie) → správně vyloučeno |
| Tier=none ∧ Skills=NO  |    13 129 | Shoda obou zdrojů → `has_ai = 0`                      |

Průnikový přístup eliminuje chyby obou zdrojů a výsledná klasifikace je konzervativnější, ale přesnější než kterýkoli zdroj samostatně.

## Confidence filtering

Před veškerou analýzou se dataset filtruje na `desc_conf_llm >= 0.7`. Tím se odstraňují pozice, u kterých si LLM nebyl jistý tier klasifikací. Průměrná confidence napříč skupinami:

| Skupina               | Průměrná confidence |
| --------------------- | ------------------- |
| Tier=AI, Skills=YES   | 0.867               |
| Tier=AI, Skills=NO    | 0.852               |
| Tier=none, Skills=YES | 0.866               |
| Tier=none, Skills=NO  | 0.902               |

Všechny skupiny mají srovnatelnou confidence, což potvrzuje, že confidence filtr nezavádí systematický bias.
