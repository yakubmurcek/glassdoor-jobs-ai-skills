# CLAUDE.md

## Project
AI skills extraction from Glassdoor job postings for master's thesis (US vs EU labor market).
Hybrid extraction strategy: deterministic dictionary matching + OpenAI LLM analysis.

## Commands
- Always use `uv run` for Python: `uv run pytest`, `uv run python -m ai_skills.cli analyze`
- Default input: `data/inputs/us_relevant_30.csv`
- Tests: `uv run pytest tests/`
- CLI subcommands: analyze, prepare-inputs, evaluate, clean-stata, search, cluster, classify, index, index-skills, visualize-skills
- Name outputs descriptively so it's clear which version/iteration they belong to

## Architecture
- `ai_skills/` — main Python package (pipeline, analyzers, extractors, dictionaries)
- `config/` — TOML settings files (settings.toml, settings.local.toml)
- `data/inputs/` — source CSVs, `data/outputs/` — pipeline results
- `analysis/stata/` — Stata .do files for econometric analysis
- `tests/` — pytest unit tests

## Pipeline Flow
```
CSV input → annotate_declared_skills()
  → LLM analysis (decomposed: tier → skills → education, batched with ThreadPoolExecutor)
  → merge dictionary + LLM results (union = hybrid extraction)
  → Stata transformations (cluster_* dummies, education_hybrid)
  → reorder columns → save CSV
```
- Checkpoint support for large datasets via CheckpointManager (atomic writes, SIGINT handling, ID-based resume)
- Two modes: `run()` for single-pass, `run_with_checkpoints()` for resumable processing

## Key Dictionaries & Mappings

### skills_dictionary.py
- `HARDSKILL_VARIANTS` — 1100+ variant→canonical mappings (e.g. "reactjs"→"react", "python3"→"python")
- `SOFTSKILL_VARIANTS` — 120+ variant→canonical mappings for soft skills
- `SKILL_TO_FAMILY` — maps every canonical skill to its semantic family name
- 24 `CAT_*` family sets defining skill taxonomy:
  - Languages: CAT_LANG_SYSTEMS, CAT_LANG_ENTERPRISE, CAT_LANG_DYNAMIC, CAT_LANG_SCRIPTING, CAT_LANG_DATA, CAT_LANG_LEGACY
  - Domains: CAT_FRONTEND, CAT_BACKEND, CAT_MOBILE_DESKTOP, CAT_DATABASES
  - Data: CAT_DATA_ENGINEERING, CAT_DATA_SCIENCE, CAT_GEN_AI, CAT_ANALYTICS
  - Infra: CAT_CLOUD, CAT_DEVOPS, CAT_OS_HARDWARE, CAT_NETWORKING, CAT_SECURITY
  - Other: CAT_QA_DEBUGGING, CAT_ARCHITECTURE, CAT_ENTERPRISE, CAT_CERTIFICATIONS, CAT_TOOLS_EDITORS
- Ambiguous short terms (c, r, js, go) intentionally excluded — handled by LLM with context

### job_title_normalizer.py
- `JOB_TITLE_PATTERNS` — 230+ priority-ordered `(priority, regex, normalized_name)` tuples
- First match wins; lower priority number = higher precedence

### education_extractor.py
- `EDUCATION_PATTERNS` — regex patterns per `EducationLevel` enum (highschool, associate, bachelor, master, phd)
- Extracts lowest explicitly mentioned education level per professor's methodology

### config.py
- `AI_SKILLS` — 1000+ AI/ML skill strings used for detection
- `REAL_AI_SKILLS` — curated set of "real AI" skills (excludes generic tools like ChatGPT)
- `PREFERRED_COLUMN_ORDER` — defines final CSV column ordering

### models.py
- `AITier` enum — core_ai, applied_ai, ai_integration, none
- All models use frozen Pydantic: `ConfigDict(frozen=True)`
- Decomposed task models: `AITierBatchResponse`, `SkillsBatchResponse`, `EducationBatchResponse`
- `JobAnalysisResult.as_columns()` maps result fields to output DataFrame column names

## Output Columns Added by Pipeline
- `skills_hasai_det`, `skills_ai_det` — deterministic AI detection from skills column
- `desc_tier_llm`, `desc_ai_llm`, `desc_conf_llm`, `ai_confidence` — LLM tier/skills/confidence
- `desc_rationale_llm` — LLM reasoning text
- `edulevel_llm`, `experience_min_llm` — LLM education/experience extraction
- `edu_level_det` — deterministic education from educations column
- `desc_hard_det`, `desc_hard_llm`, `hardskills` — hard skills (deterministic, LLM, merged union)
- `desc_soft_det`, `desc_soft_llm`, `softskills` — soft skills (deterministic, LLM, merged union)
- `skill_cluster` — semicolon-separated skill family names
- `cluster_*` — one dummy variable per skill family (for Stata regression)
- `education_hybrid` — merged education (deterministic + LLM)
- `is_real_ai` — binary flag: real AI work based on tier OR REAL_AI_SKILLS detection
- `ai_det_llm_match` — agreement between dictionary and LLM classification

## Coding Conventions
- CSV separator: `;` (semicolon), NOT comma
- CSV encoding: `utf-8-sig` (UTF-8 with BOM)
- All file paths via `pathlib.Path`, never raw strings
- Type hints everywhere: `from __future__ import annotations` at top of every module
- Private functions: `_function_name()`, CLI handlers: `_handle_*(args)`
- Constants: `UPPER_CASE`
- Singleton pattern for expensive resources: `get_semantic_normalizer()` caches `_SEMANTIC_NORMALIZER`
- Lazy initialization: compiled regex patterns, embedding matrices built on first use
- Progress callbacks: `Callable[[int, int], None]` threaded through pipeline layers
- Logging: `logging.getLogger(__name__)` in every module
- Error handling: graceful degradation (return empty results, don't crash)
- Span-based matching in deterministic extractor resolves overlaps (longer match wins)

## Config Precedence
1. `config/settings.local.toml` (user overrides, gitignored)
2. `config/settings.toml` (tracked defaults)
3. Environment variables / `.env`
4. Hardcoded defaults
- Exception: credentials (`OPENAI_API_KEY`) MUST come from environment only, never TOML

## Verification Rule
When asked to verify, validate, or check something:
1. Write a proper Python verification script — never just eyeball a few rows
2. Use a large random sample (100+ items minimum, scale up for large datasets)
3. Be thorough: check edge cases, distributions, disagreements, not just happy paths
4. Place ALL verification scripts in `verify-scripts/` at repo root
5. NEVER put verification scripts in `ai_skills/`, `tests/`, or any other main codebase directory
6. Name scripts descriptively: `verify_<what>.py`
7. Print clear results: counts, percentages, and specific examples of failures
8. Run with: `uv run python verify-scripts/verify_<what>.py`
