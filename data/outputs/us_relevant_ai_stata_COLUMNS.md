# Dokumentace sloupců `us_relevant_ai_stata.csv`

**18 464 řádků** | Separátor: `;` | Generováno pipeline `ai_skills/pipeline.py`

Vstup: Glassdoor job postings (`data/inputs/us_relevant.csv`). Pipeline přidává AI klasifikaci, extrakci skillů a Stata transformace.

---

## 🔵 Původní sloupce z Glassdooru

| Sloupec             | Popis                                               |
| ------------------- | --------------------------------------------------- |
| `id`                | Unikátní ID inzerátu                                |
| `job_title`         | Název pozice                                        |
| `location`          | Lokalita (město/stát/Remote)                        |
| `company_id`        | ID firmy na Glassdooru                              |
| `company`           | Název firmy                                         |
| `age_in_days`       | Stáří inzerátu v dnech                              |
| `pay_currency`      | Měna platu (USD)                                    |
| `pay_period`        | Platové období (ANNUAL/HOURLY/MONTHLY)              |
| `salary_min`        | Minimální plat                                      |
| `salary_mid`        | Střední plat                                        |
| `salary_max`        | Maximální plat                                      |
| `rating`            | Hodnocení firmy na Glassdooru (0–5, prázdné = -0.1) |
| `discover_date`     | Datum nalezení inzerátu                             |
| `job_types`         | Typ úvazku (Full-time, Contract…)                   |
| `remote_work_types` | Režim práce (WORK_FROM_HOME…)                       |
| `city`              | Město                                               |
| `state`             | Stát                                                |
| `country`           | Země (USA, 34 řádků prázdných)                      |
| `latitude`          | Zeměpisná šířka                                     |
| `longitude`         | Zeměpisná délka                                     |
| `ceo`               | CEO firmy                                           |
| `headquarters`      | Sídlo firmy                                         |
| `industry`          | Odvětví                                             |
| `sector`            | Sektor                                              |
| `revenue`           | Tržby firmy                                         |
| `size`              | Velikost firmy (počet zaměstnanců)                  |
| `type`              | Typ firmy (Private, Public…)                        |
| `website`           | Web firmy                                           |
| `year_founded`      | Rok založení                                        |

> Sloupce `educations`, `job_desc_text`, `job_desc_html` jsou ve vstupu, ale ve Stata verzi **odstraněny**. Sloupec `skills` zůstává.

---

## 🟢 Deterministické sloupce (slovníkový matching, bez LLM)

Zdroj: `skill_processing.py`, `deterministic_extractor.py`, `education_extractor.py`

| Sloupec            | Popis                                                                                                                       |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `skills`           | Původní dovednosti z Glassdoor metadata (zachováno z vstupu)                                                                |
| `skills_ai_det`    | AI skilly nalezené ve sloupci `skills` (matching proti 911 AI termínům v `config.py`)                                       |
| `skills_hasai_det` | `0/1` – obsahuje `skills` nějaký AI skill?                                                                                  |
| `edu_level_det`    | Nejnižší vzdělání z `educations` sloupce (deterministicky: highschool → phd)                                                |
| `desc_hard_det`    | Hard skills – union slovníkového matchingu na **sloupec `skills`** (Glassdoor metadata) **+ `job_desc_text`** (text popisu) |
| `desc_soft_det`    | Soft skills – union slovníkového matchingu na **sloupec `skills`** + **`job_desc_text`**                                    |

---

## 🟡 LLM sloupce (OpenAI GPT-4o-mini)

Zdroj: `openai_analyzer.py` → `models.py` → `pipeline.py`

> [!IMPORTANT]
> LLM skills extraction (`LLM_TASK_SKILLS`) byla **záměrně vypnuta** pro úsporu nákladů. LLM se použilo **pouze** na tier klasifikaci, education a experience.

| Sloupec              | Popis                                                                                                                                    |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `desc_tier_llm`      | AI tier klasifikace: `none`, `ai_integration` (používá AI nástroje), `applied_ai` (buduje s AI frameworky), `core_ai` (vyvíjí AI modely) |
| `desc_conf_llm`      | Confidence LLM klasifikace (0.0–1.0)                                                                                                     |
| `ai_confidence`      | Alias pro `desc_conf_llm`                                                                                                                |
| `desc_ai_llm`        | Konkrétní AI technologie zmíněné v popisu (součást AI tier klasifikace, ne skills tasku)                                                 |
| `edulevel_llm`       | Minimální vzdělání z popisu pozice (LLM: Bachelor's, Master's…)                                                                          |
| `experience_min_llm` | Minimální požadované roky zkušeností (LLM, numerické)                                                                                    |
| `desc_hard_llm`      | ⚠️ **Vždy prázdné** – LLM skills extrakce byla vypnuta                                                                                   |
| `desc_soft_llm`      | ⚠️ **Vždy prázdné** – LLM skills extrakce byla vypnuta                                                                                   |

---

## 🔴 Sloučené / odvozené sloupce

Zdroj: `pipeline.py` (`_merge_results_into_df` + `_apply_stata_transformations`)

| Sloupec            | Popis                                                                                                                                                     |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hardskills`       | Hard skills – **čistě deterministické** (= `desc_hard_det`). Union slovníku na `skills` sloupec + `job_desc_text`. LLM skills vypnuto                     |
| `softskills`       | Soft skills – **čistě deterministické** (= `desc_soft_det`). Union slovníku na `skills` sloupec + `job_desc_text`. LLM skills vypnuto                     |
| `is_real_ai`       | `0/1` – tier ∈ {`core_ai`, `applied_ai`} **NEBO** hardskills obsahují ML/AI frameworky (TensorFlow, PyTorch, LLM, RAG, MLOps…) z `REAL_AI_SKILLS` seznamu |
| `ai_det_llm_match` | `0/1` – shoduje se deterministický a LLM výsledek?                                                                                                        |
| `education_hybrid` | Vzdělání: primárně `edu_level_det`, backfill z `edulevel_llm`                                                                                             |

---

## 🟣 Skill cluster dummy proměnné (0/1, pro Stata regresi)

Zdroj: `_apply_stata_transformations()` + slovník `skills_dictionary.py` (`SKILL_TO_FAMILY`). Mapuje `hardskills` → rodiny → dummy sloupce.

| Sloupec                           | Cluster                                                 |
| --------------------------------- | ------------------------------------------------------- |
| `cluster_architecture___methods`  | Architektura a metody (design patterns, microservices…) |
| `cluster_bi___analytics`          | BI & Analytics (Tableau, Power BI…)                     |
| `cluster_backend_development`     | Backend Development (Node.js, Spring…)                  |
| `cluster_certifications`          | Certifikace (AWS cert, PMP…)                            |
| `cluster_cloud_computing`         | Cloud Computing (AWS, Azure, GCP…)                      |
| `cluster_data_analysis___stats`   | Data Analysis & Stats (R, pandas…)                      |
| `cluster_data_engineering`        | Data Engineering (Spark, Airflow…)                      |
| `cluster_data_science___ml`       | Data Science & ML (TensorFlow, PyTorch…)                |
| `cluster_databases___storage`     | Databáze & Storage (SQL, MongoDB…)                      |
| `cluster_devops___containers`     | DevOps & Containers (Docker, K8s…)                      |
| `cluster_dynamic___web`           | Dynamic & Web (JavaScript, React…)                      |
| `cluster_enterprise___managed`    | Enterprise & Managed services                           |
| `cluster_enterprise_platforms`    | Enterprise Platforms (SAP, Salesforce…)                 |
| `cluster_frontend_development`    | Frontend Development (CSS, HTML…)                       |
| `cluster_generative_ai`           | Generative AI (LLMs, GPT, RAG…)                         |
| `cluster_legacy___mainframe`      | Legacy & Mainframe (COBOL…)                             |
| `cluster_mobile___desktop`        | Mobile & Desktop (iOS, Android…)                        |
| `cluster_networking`              | Networking (TCP/IP, DNS…)                               |
| `cluster_os___embedded`           | OS & Embedded (Linux kernel…)                           |
| `cluster_scripting___shell`       | Scripting & Shell (Bash, PowerShell…)                   |
| `cluster_security___identity`     | Security & Identity (OAuth, encryption…)                |
| `cluster_systems_programming`     | Systems Programming (C, C++, Rust…)                     |
| `cluster_testing__qa___debugging` | Testing, QA & Debugging (JUnit, Selenium…)              |
| `cluster_tools___editors`         | Nástroje & Editory (Git, VS Code…)                      |

---

## ⚙️ Stata pipeline (`clean-stata` → `ai_skills_analysis.do`)

### Krok 1: `clean-stata` CLI (Python → Stata-optimized CSV)

Příkaz `ai-skills clean-stata` odstraní objemné textové sloupce a vytvoří `*_stata_optimized.csv`:

Odstraněné sloupce: `job_desc_text`, `job_desc_html`, `desc_rationale_llm`, `educations`

### Krok 2: Stata do-file (`analysis/stata/ai_skills_analysis.do`)

Importuje `*_stata_optimized.csv`. **Filtruje**: `keep if desc_conf_llm >= 0.7`.

#### Proměnné vytvořené ve Statě (nejsou v CSV):

| Proměnná          | Popis                                                                                                                   |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `has_ai_flag`     | **Přísnější než `is_real_ai`**: tier ≠ none **A ZÁROVEŇ** má specifické AI skills (po odstranění buzzwords AI/ML/GenAI) |
| `has_ai`          | Alias pro `has_ai_flag`                                                                                                 |
| `ai_tier_num`     | Encode `desc_tier_llm` → numerická kategorie                                                                            |
| `edu_cat`         | Encode `education_hybrid` → numerická kategorie                                                                         |
| `exp_category`    | Kategorie seniority: Entry (0), Junior (1–2), Mid (3–5), Senior (6–10), Expert (10+)                                    |
| `is_remote`       | `0/1` – odvozeno z `remote_work_types` (hledá "home"/"remote")                                                          |
| `skill_count`     | Počet hard skills na pozici (počet čárek v `hardskills` + 1)                                                            |
| `sector_num`      | Encode `sector` → numerická kategorie                                                                                   |
| `state_num`       | Encode `state` → numerická kategorie                                                                                    |
| `skills_combined` | Pomocná: `skills_ai_det` + `desc_ai_llm` (lowered)                                                                      |
| `skills_cleaned`  | Pomocná: `skills_combined` po odstranění buzzwords a interpunkce                                                        |

> [!WARNING]
> Stata do-file odkazuje na `skills_det_ai`, ale CSV sloupec se jmenuje `skills_ai_det` – potenciální bug v do-file.

#### CSV sloupce **použité** v Stata analýze (14/72):

`desc_tier_llm`, `desc_conf_llm`, `desc_ai_llm`, `education_hybrid`, `experience_min_llm`, `edulevel_llm`, `salary_min`, `salary_mid`, `salary_max`, `sector`, `industry`, `state`, `remote_work_types`, `hardskills`

#### CSV sloupce **nepoužité** ve Stata do-file:

Glassdoor metadata: `id`, `job_title`, `location`, `company_id`, `company`, `age_in_days`, `pay_currency`, `pay_period`, `rating`, `discover_date`, `job_types`, `city`, `country`, `latitude`, `longitude`, `ceo`, `headquarters`, `revenue`, `size`, `type`, `website`, `year_founded`

Deterministické: `skills`, `skills_ai_det`, `skills_hasai_det`, `edu_level_det`, `desc_hard_det`, `desc_soft_det`

LLM (prázdné): `desc_hard_llm`, `desc_soft_llm`

Odvozené: `ai_confidence`, `ai_det_llm_match`, `is_real_ai`, `softskills`

Cluster dummy (24×): všechny `cluster_*` sloupce – **připraveny pro regresi, ale v aktuálním do-file nepoužity**
