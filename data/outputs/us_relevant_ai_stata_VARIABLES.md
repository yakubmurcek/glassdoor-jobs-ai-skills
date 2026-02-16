# Proměnné použité ve Stata analýze

Zdroj dat: Glassdoor IT job postings (USA, 18 464 inzerátů).
Pipeline: `ai_skills/pipeline.py` → `clean-stata` → `ai_skills_analysis.do`
Filtr: `desc_conf_llm >= 0.7`

---

## Z CSV (vstup do Stata)

| Proměnná             | Typ   | Původ        | Popis                                                                                                           |
| -------------------- | ----- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| `desc_tier_llm`      | kat.  | LLM (GPT)    | AI klasifikace: `none`, `ai_integration`, `applied_ai`, `core_ai`                                               |
| `desc_conf_llm`      | kont. | LLM          | Jistota klasifikace (0.0–1.0)                                                                                   |
| `desc_ai_llm`        | text  | LLM          | AI technologie zmíněné v popisu pozice                                                                          |
| `education_hybrid`   | kat.  | det. + LLM   | Požadované vzdělání (`bachelor`, `master`, `highschool`, `missing`). Primárně z Glassdoor metadat, doplněno LLM |
| `experience_min_llm` | kont. | LLM          | Minimální roky zkušeností                                                                                       |
| `edulevel_llm`       | kat.  | LLM          | Vzdělání z popisu pozice (Bachelor's, Master's…)                                                                |
| `salary_min`         | kont. | Glassdoor    | Minimální roční plat (USD)                                                                                      |
| `salary_mid`         | kont. | Glassdoor    | Střední roční plat (USD)                                                                                        |
| `salary_max`         | kont. | Glassdoor    | Maximální roční plat (USD)                                                                                      |
| `sector`             | kat.  | Glassdoor    | Sektor firmy                                                                                                    |
| `industry`           | kat.  | Glassdoor    | Odvětví firmy                                                                                                   |
| `state`              | kat.  | Glassdoor    | Stát (USA)                                                                                                      |
| `remote_work_types`  | kat.  | Glassdoor    | Režim práce (`WORK_FROM_HOME` / prázdné)                                                                        |
| `hardskills`         | text  | det. slovník | Union slovníkového matchingu na `skills` (Glassdoor metadata) + `job_desc_text`                                 |

---

## Vytvořené ve Statě

| Proměnná       | Typ         | Vznik                           | Popis                                                                                              |
| -------------- | ----------- | ------------------------------- | -------------------------------------------------------------------------------------------------- |
| `has_ai_flag`  | 0/1         | `desc_tier_llm` + `desc_ai_llm` | AI pozice: tier ≠ none **a zároveň** má specifické AI skills (po odstranění buzzwords AI/ML/GenAI) |
| `has_ai`       | 0/1         | alias                           | = `has_ai_flag`                                                                                    |
| `ai_tier_num`  | kat. (num.) | `encode desc_tier_llm`          | Numerický kód AI tieru                                                                             |
| `edu_cat`      | kat. (num.) | `encode education_hybrid`       | Numerický kód vzdělání                                                                             |
| `exp_category` | kat. (num.) | `experience_min_llm`            | Seniorita: Entry (0), Junior (1–2), Mid (3–5), Senior (6–10), Expert (10+)                         |
| `is_remote`    | 0/1         | `remote_work_types`             | Obsahuje „home" nebo „remote"                                                                      |
| `skill_count`  | kont.       | `hardskills`                    | Počet hard skills na pozici (počet čárek + 1)                                                      |
| `sector_num`   | kat. (num.) | `encode sector`                 | Numerický kód sektoru                                                                              |
| `state_num`    | kat. (num.) | `encode state`                  | Numerický kód státu                                                                                |

---

## Použití v analýze

| Analýza                    | Závislá proměnná | Nezávislé proměnné                                     |
| -------------------------- | ---------------- | ------------------------------------------------------ |
| T-test (plat AI vs non-AI) | `salary_mid`     | `has_ai`                                               |
| ANOVA (plat dle tieru)     | `salary_mid`     | `ai_tier_num`                                          |
| χ² (vzdělání × AI)         | `edu_cat`        | `has_ai`                                               |
| χ² (zkušenosti × AI)       | `exp_category`   | `has_ai`                                               |
| Mann-Whitney U             | `salary_mid`     | `has_ai`                                               |
| OLS regrese                | `salary_mid`     | `has_ai`, `edu_cat`, `experience_min_llm`, `is_remote` |
| OLS s interakcemi          | `salary_mid`     | `has_ai × edu_cat`, `experience_min_llm`, `is_remote`  |
| Logistická regrese         | `has_ai`         | `edu_cat`, `experience_min_llm`, `is_remote`           |
