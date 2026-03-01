# Project Context & Agent Instructions

## Thesis Context

- **Topic:** Master's thesis in Economics analyzing Glassdoor job listings in the IT industry in the United States and Czechia.
- **Objective:** Examine and compare **AI skill requirements** in U.S. job postings versus Czech job postings.
- **Data:** Scraped Glassdoor job postings for both the United States and Czechia.
- **Main Document:** The primary Word document for this thesis is located at `docs/Master_thesis.docx`.

## Project Details

AI skills extraction from Glassdoor job postings for master's thesis (US vs EU). Hybrid extraction strategy: deterministic dictionary matching + OpenAI LLM analysis.

## Core Commands

- **Python context:** Always use `uv run python -m ai_skills.cli [subcommand]`
- **Tests:** `uv run pytest tests/`
- **Default input:** `data/inputs/us_relevant_30.csv`
- Be descriptive with output filenames for easy version tracking.

## Architecture & Pipeline

- `ai_skills/` (core Python package), `config/` (TOML settings), `data/` (inputs/outputs), `analysis/stata/`, `tests/`
- **Pipeline Flow:** `CSV -> Deterministic Extraction -> LLM Analysis -> Merge (Union) -> Stata Formatting -> CSV Output`
- Supports resumable processing via checkpoints (`run_with_checkpoints()`).

## Important Rules & Conventions

1. **Formatting/Paths:** CSV must use `;` separator and `utf-8-sig` encoding. Always use `pathlib.Path`, never raw strings.
2. **Coding Style:** Extensive type hints (`from __future__ import annotations`), descriptive names, lazy initialization where possible, logging everywhere, graceful error handling.

## Config Precedence

1. `config/settings.local.toml` (user overrides, gitignored)
2. `config/settings.toml` (tracked defaults)
3. Environment variables / `.env`
4. Hardcoded defaults

- **Exception**: credentials (`OPENAI_API_KEY`) MUST come from environment only, never TOML.

## Verification Rules (CRITICAL)

When verifying, validating, or checking something:

1. Write a standalone Python verification script (e.g., `verify_X.py`). Never just eyeball data.
2. Use large random samples (100+ minimum) and deeply check edge cases.
3. Place ALL verification scripts in the `verify-scripts/` directory at the repo root. _NEVER_ put them in `ai_skills/` or `tests/`.
4. Command: `uv run python verify-scripts/verify_<what>.py`
