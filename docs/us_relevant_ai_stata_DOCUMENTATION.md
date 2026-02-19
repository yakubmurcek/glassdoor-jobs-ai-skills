# Dokumentace datasetu `us_relevant_ai_stata.csv`

**11 015 řádků** | **75 sloupců** | Separátor: `;` | Pipeline: `ai_skills/pipeline.py` → `cli clean-stata`

Vstup: Glassdoor IT job postings USA (`data/inputs/us_relevant.csv`). Pipeline přidává AI klasifikaci, extrakci skillů a Stata transformace.

> Sloupce `educations`, `job_desc_text`, `job_desc_html` jsou ve vstupu, ale v Stata verzi **odstraněny**.

---

## 1. Sloupce v CSV (72)

★ = používá se ve Stata analýze (celkem 19/72)

### 🔵 Glassdoor metadata (29 sloupců)

| Sloupec               | Popis                                                               |
| --------------------- | ------------------------------------------------------------------- |
| `id`                  | Unikátní ID inzerátu                                                |
| `job_title`           | Název pozice                                                        |
| `location`            | Lokalita (město/stát/Remote)                                        |
| `company_id`          | ID firmy na Glassdooru                                              |
| `company`             | Název firmy                                                         |
| `age_in_days`         | Stáří inzerátu v dnech                                              |
| `pay_currency`        | Měna platu (USD)                                                    |
| ★ `pay_period`        | Platové období (ANNUAL/HOURLY/MONTHLY) – pro přepočet na roční plat |
| ★ `salary_min`        | Minimální plat                                                      |
| ★ `salary_mid`        | Střední plat                                                        |
| ★ `salary_max`        | Maximální plat                                                      |
| `rating`              | Hodnocení firmy na Glassdooru (0–5, prázdné = -0.1)                 |
| `discover_date`       | Datum nalezení inzerátu                                             |
| `job_types`           | Typ úvazku (Full-time, Contract…)                                   |
| ★ `remote_work_types` | Režim práce (WORK_FROM_HOME…)                                       |
| `city`                | Město                                                               |
| ★ `state`             | Stát (USA)                                                          |
| `country`             | Země (USA, 34 řádků prázdných)                                      |
| `latitude`            | Zeměpisná šířka                                                     |
| `longitude`           | Zeměpisná délka                                                     |
| `ceo`                 | CEO firmy                                                           |
| `headquarters`        | Sídlo firmy                                                         |
| ★ `industry`          | Odvětví                                                             |
| ★ `sector`            | Sektor                                                              |
| `revenue`             | Tržby firmy                                                         |
| ★ `size`              | Velikost firmy (počet zaměstnanců)                                  |
| ★ `type`              | Typ firmy (Private, Public…)                                        |
| `website`             | Web firmy                                                           |
| ★ `year_founded`      | Rok založení                                                        |

### 🟢 Deterministické sloupce (6 sl., slovníkový matching, bez LLM)

Zdroj: `skill_processing.py`, `deterministic_extractor.py`, `education_extractor.py`

| Sloupec            | Popis                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| `skills`           | Původní dovednosti z Glassdoor metadata (zachováno z vstupu)                                      |
| ★ `skills_ai_det`  | AI skilly nalezené ve sloupci `skills` (matching proti 911 AI termínům v `config.py`)             |
| `skills_hasai_det` | `0/1` – obsahuje `skills` nějaký AI skill?                                                        |
| `edu_level_det`    | Nejnižší vzdělání z `educations` sloupce (deterministicky: highschool → phd)                      |
| `desc_hard_det`    | Hard skills – union slovníkového matchingu na `skills` (metadata) + `job_desc_text` (text popisu) |
| `desc_soft_det`    | Soft skills – union slovníkového matchingu na `skills` + `job_desc_text`                          |

### 🟡 LLM sloupce (8 sl., OpenAI GPT-4o-mini)

Zdroj: `openai_analyzer.py` → `models.py` → `pipeline.py`

> [!IMPORTANT]
> LLM skills extraction (`LLM_TASK_SKILLS`) byla **záměrně vypnuta** pro úsporu nákladů. LLM se použilo **pouze** na tier klasifikaci, education a experience.

| Sloupec                | Popis                                                                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| ★ `desc_tier_llm`      | AI tier: `none`, `ai_integration` (používá AI nástroje), `applied_ai` (buduje s AI frameworky), `core_ai` (vyvíjí AI modely) |
| ★ `desc_conf_llm`      | Confidence klasifikace (0.0–1.0)                                                                                             |
| `ai_confidence`        | Alias pro `desc_conf_llm`                                                                                                    |
| ★ `desc_ai_llm`        | Konkrétní AI technologie zmíněné v popisu (součást AI tier klasifikace, ne skills tasku)                                     |
| ★ `edulevel_llm`       | Minimální vzdělání z popisu pozice (Bachelor's, Master's…)                                                                   |
| ★ `experience_min_llm` | Minimální požadované roky zkušeností (numerické)                                                                             |
| `desc_hard_llm`        | ⚠️ **Vždy prázdné** – LLM skills extrakce vypnuta                                                                            |
| `desc_soft_llm`        | ⚠️ **Vždy prázdné** – LLM skills extrakce vypnuta                                                                            |

### 🔴 Sloučené / odvozené sloupce (5 sl.)

Zdroj: `pipeline.py` (`_merge_results_into_df` + `_apply_stata_transformations`)

| Sloupec              | Popis                                                                                                                                    |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| ★ `hardskills`       | Hard skills – **čistě deterministické** (= `desc_hard_det`). Union slovníku na `skills` + `job_desc_text`. LLM skills vypnuto            |
| `softskills`         | Soft skills – **čistě deterministické** (= `desc_soft_det`)                                                                              |
| `is_real_ai`         | `0/1` – tier ∈ {`core_ai`, `applied_ai`} **NEBO** hardskills obsahují ML/AI frameworky (TensorFlow, PyTorch…) z `REAL_AI_SKILLS` seznamu |
| `ai_det_llm_match`   | `0/1` – shoduje se deterministický a LLM výsledek?                                                                                       |
| ★ `education_hybrid` | Vzdělání: primárně `edu_level_det`, backfill z `edulevel_llm`                                                                            |

### 🟣 Skill cluster dummy proměnné (24 sl., 0/1, pro regresi)

Zdroj: `_apply_stata_transformations()` + slovník `skills_dictionary.py` (`SKILL_TO_FAMILY`). Mapuje `hardskills` → rodiny → dummy sloupce.

> V aktuálním do-file **nepoužity** – připraveny pro regresi.
>
> [!WARNING]
> **Kolinearita s AI tier**: `cluster_data_science___ml` (r=0.57) a `cluster_generative_ai` (r=0.45) silně korelují s `desc_tier_llm`. Při použití v regresi společně s AI tier zvážit jejich vyřazení nebo kontrolu VIF (`estat vif`).

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

## 2. Stata pipeline

### Krok 1: `clean-stata` CLI (Python)

Příkaz `ai-skills clean-stata` odstraní objemné textové sloupce (`job_desc_text`, `job_desc_html`, `desc_rationale_llm`, `educations`) a vytvoří `*_stata_optimized.csv`.

### Krok 2: `ai_skills_analysis.do`

Importuje `*_stata_optimized.csv`. **Filtruje**: `keep if desc_conf_llm >= 0.7`.

#### Proměnné vytvořené ve Statě (nejsou v CSV):

| Proměnná          | Typ         | Vstup                            | Popis                                                                                                                   |
| ----------------- | ----------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `has_ai_flag`     | 0/1         | `desc_tier_llm` + `desc_ai_llm`  | **Přísnější než `is_real_ai`**: tier ≠ none **A ZÁROVEŇ** má specifické AI skills (po odstranění buzzwords AI/ML/GenAI) |
| `has_ai`          | 0/1         | alias                            | = `has_ai_flag`                                                                                                         |
| `ai_tier_num`     | kat. (num.) | `encode desc_tier_llm`           | Numerický kód AI tieru                                                                                                  |
| `edu_cat`         | kat. (num.) | `education_hybrid`               | Ordinální vzdělání: 0=Missing, 1=Highschool, 2=Associate, 3=Bachelor, 4=Master, 5=PhD                                   |
| `exp_category`    | kat. (num.) | `experience_min_llm`             | Seniorita: 0=Missing, 1=Entry (0), 2=Junior (1–2), 3=Mid (3–5), 4=Senior (6–10), 5=Expert (10+)                         |
| `is_remote`       | 0/1         | `remote_work_types`              | Obsahuje „home" nebo „remote"                                                                                           |
| `skill_count`     | kont.       | `hardskills`                     | Počet hard skills na pozici (počet čárek + 1)                                                                           |
| `sector_num`      | kat. (num.) | `encode sector`                  | Numerický kód sektoru                                                                                                   |
| `state_num`       | kat. (num.) | `encode state`                   | Numerický kód státu                                                                                                     |
| `size_cat`        | kat. (ord.) | `size`                           | Ordinální velikost firmy: 0=Unknown, 1=1-50, 2=51-200, 3=201-500, 4=501-1000, 5=1001-5000, 6=5001-10000, 7=10000+       |
| `type_cat`        | kat. (num.) | `type`                           | Typ firmy: 0=Unknown, 1=Private, 2=Public, 3=Subsidiary, 4=Nonprofit/Gov/Edu, 5=Other                                   |
| `ai_level`        | kat. (ord.) | `desc_tier_llm`                  | Úroveň AI požadavků: 0=None, 1=AI Integration, 2=Applied/Core AI (pro multinomiální logit)                              |
| `sector_nace`     | kat.        | `clean-stata` CLI (`sector`)     | NACE Rev. 2 sekce (A–S) mapovaná z Glassdoor sektoru                                                                    |
| `region`          | kat.        | `clean-stata` CLI (`state`)      | US Census region: Northeast, Midwest, South, West, Unknown                                                              |
| `job_family`      | kat.        | `clean-stata` CLI (`job_title`)  | ~10 rodin pozic (Software Engineer, DevOps & Cloud, Security…) z raw job titles                                         |
| `cluster_*`       | 0/1 (×24)   | `clean-stata` CLI (`hardskills`) | 24 skill cluster dummy proměnných z `SKILL_TO_FAMILY` dict (999 skills → 24 rodin)                                      |
| `sector_nace_num` | kat. (num.) | Stata `encode sector_nace`       | Numerický kód NACE sektoru                                                                                              |
| `region_num`      | kat. (num.) | Stata `encode region`            | Numerický kód Census regionu                                                                                            |
| `job_family_num`  | kat. (num.) | Stata `encode job_family`        | Numerický kód rodiny pozice                                                                                             |
| `ln_salary`       | kont.       | Stata `ln(salary_mid)`           | Přirozený logaritmus platu (pro regresní modely)                                                                        |
| `skills_combined` | text        | pomocná                          | `lower(skills_ai_det + " " + desc_ai_llm)`                                                                              |
| `skills_no_buzz`  | text        | pomocná                          | `skills_combined` po odstranění buzzwords (AI, ML, GenAI…) regex                                                        |
| `skills_cleaned`  | text        | pomocná                          | `skills_no_buzz` po odstranění interpunkce (pro `has_ai_flag`)                                                          |

---

## 3. Analýzy v Stata

| Analýza                      | Závislá proměnná   | Nezávislé proměnné                                                            |
| ---------------------------- | ------------------ | ----------------------------------------------------------------------------- |
| T-test (plat AI vs non-AI)   | `salary_mid`       | `has_ai`                                                                      |
| ANOVA (plat dle tieru)       | `salary_mid`       | `ai_tier_num`                                                                 |
| χ² (vzdělání × AI)           | `edu_cat`          | `has_ai`                                                                      |
| χ² (zkušenosti × AI)         | `exp_category`     | `has_ai`                                                                      |
| Mann-Whitney U               | `salary_mid`       | `has_ai`                                                                      |
| **Model A (base OLS)**       | `ln_salary`        | `cluster_*`, `ai_level`, `sector_nace`, `region`, `is_remote`, `type`, `size` |
| **Model B (plný OLS)**       | `ln_salary`        | Model A + `edu_cat`, `exp_category`, `job_family`                             |
| F-test (A vs B)              | —                  | testuje přínos vzděl., zkuš., pozice                                          |
| OLS jednoduchý (6B)          | `salary_mid`       | `has_ai`, `edu_cat`, `exp_category`, `is_remote`                              |
| Logistická regrese (6B)      | `has_ai`           | `edu_cat`, `exp_category`, `is_remote`                                        |
| **Multinomiální logit (6B)** | `ai_level` (0/1/2) | `edu_cat`, `exp_category`, `is_remote`                                        |
