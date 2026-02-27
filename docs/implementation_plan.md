# Rozdělení odpovědnosti mezi Python a Statu

Přesunout tvorbu nových sloupců (**zpracování textu, porovnávání vzorů**) do Pythonu (`_apply_stata_transformations`).  
Stata si ponechá **analytická rozhodnutí** (encode, label, thresholds, kategorizace).

## Princip rozdělení

| Odpovědnost                | Python (`pipeline.py`)             | Stata (`ai_skills_analysis.do`)              |
| -------------------------- | ---------------------------------- | -------------------------------------------- |
| **Mapování textu na text** | ✅ region, job_family, sector_nace |                                              |
| **Porovnávání vzorů**      | ✅ job_family (regex na job_title) |                                              |
| **Ordinální kódování**     |                                    | ✅ edu_cat, exp_category, size_cat, type_cat |
| **Analytické rozhodnutí**  |                                    | ✅ ai_level, has_ai_flag, outlier filtry     |
| **Encode + label**         |                                    | ✅ vše                                       |
| **Log transformace**       |                                    | ✅ log(salary_mid)                           |

> [!IMPORTANT]
> Python vytváří **textové sloupce** (region="South", job_family="DevOps & Cloud"). Stata je pak `encode`-uje do numerických proměnných. Díky tomu je mapování v Pythonu a analytická volba (co je base category, co sloučit) ve Statě.

## Navrhované změny

### Python pipeline

#### [UPRAVIT] [pipeline.py](file:///Users/yakub/Projects/glassdoor-jobs-ai-skills/ai_skills/pipeline.py)

Rozšířit `_apply_stata_transformations()` o dvě nové sekce (za existující bod 3 — Type Conversion):

**5. Mapování regionů (Region mapping)** — dict lookup `state` → Census region string

```python
CENSUS_REGIONS = {
    "Connecticut": "Northeast", "Maine": "Northeast", ...
    "Illinois": "Midwest", ...
    "Florida": "South", ...
    "California": "West", ...
}
df["region"] = df["state"].map(CENSUS_REGIONS).fillna("Unknown")
```

**6. Mapování rodiny profesí (Job family mapping)** — regex pattern matching na `job_title`

```python
JOB_FAMILY_PATTERNS = [
    ("Management", r"manager|director|architect|tech lead|vp |head of"),
    ("QA & Testing", r"\bqa\b|test engineer|quality assurance|sdet|..."),
    ...
    ("Software Engineer", r"software eng"),  # last = catch-all
]
# Apply in order, first match wins
```

**7. Mapování sektorů NACE** — přesunout dict lookup ze Staty do Pythonu  
_(Nyní je to 20+ `replace` řádků v do-file, v Pythonu je to 1 dict + `.map()`)_

---

### Stata do-file

#### [UPRAVIT] [ai_skills_analysis.do](file:///Users/yakub/Projects/glassdoor-jobs-ai-skills/analysis/stata/ai_skills_analysis.do)

**Odebrat** z do-file:

- Sekce 3.13 (region) — nahradit pouhým `encode region, generate(region_num)`
- Sekce 3.14 (job_family) — nahradit pouhým `encode job_family, generate(job_family_num)`
- Sekce 3.10 (sector_nace) — nahradit pouhým `encode sector_nace, generate(sector_nace_num)`

**Ponechat** v do-file (analytická rozhodnutí):

- 3.1 AI Tier encode
- 3.2 has_ai_flag (buzzword filter)
- 3.3 edu_cat (ordinální prahování)
- 3.4 exp_category (seniorita prahování)
- 3.5 salary přepočet + outlier filtr
- 3.6 sector/state encode
- 3.7 remote
- 3.8 size_cat (ordinální)
- 3.9 type_cat (sloučení)
- 3.11 year_founded
- 3.12 ai_level

---

### Dokumentace

#### [UPRAVIT] [us_relevant_ai_stata_DOCUMENTATION.md](file:///Users/yakub/Projects/glassdoor-jobs-ai-skills/data/outputs/us_relevant_ai_stata_DOCUMENTATION.md)

- Přesunout `region`, `job_family`, `sector_nace` z tabulky "Proměnné vytvořené ve Statě" do sekce "Sloupce v CSV"
- Aktualizovat popis zdrojů

## Plán ověření (Verification Plan)

### Automatizované testy

Po úpravě `pipeline.py` spustit pipeline v hydration módu na malém vzorku a ověřit, že nové sloupce existují a mají korektní hodnoty:

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('data/outputs/us_relevant_ai_stata.csv', sep=';')
# Check new columns exist
assert 'region' in df.columns, 'Missing region column'
assert 'job_family' in df.columns, 'Missing job_family column'
assert 'sector_nace' in df.columns, 'Missing sector_nace column'

# Check region values
valid_regions = {'Northeast','Midwest','South','West','Unknown'}
assert set(df['region'].unique()).issubset(valid_regions), f'Invalid regions: {set(df[\"region\"].unique()) - valid_regions}'

# Check job_family coverage
coverage = (df['job_family'] != 'Other').mean()
assert coverage > 0.80, f'Job family coverage too low: {coverage:.1%}'

# Check sector_nace values
valid_nace = {'A','C','D','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','Unknown'}
assert set(df['sector_nace'].unique()).issubset(valid_nace), f'Invalid NACE: {set(df[\"sector_nace\"].unique()) - valid_nace}'

print('All checks passed!')
print(f'Regions: {df[\"region\"].value_counts().to_dict()}')
print(f'Job families: {df[\"job_family\"].value_counts().to_dict()}')
"
```

### Manuální ověření

User spustí pipeline přes `uv run python -m ai_skills.cli analyze --input data/inputs/us_relevant_30.csv --skip-llm` a zkontroluje, že výstupní CSV obsahuje sloupce `region`, `job_family`, `sector_nace` s rozumnými hodnotami.
