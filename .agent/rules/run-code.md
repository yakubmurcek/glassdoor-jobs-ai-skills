---
trigger: always_on
---

When running the Python code, always use `uv run`:
Example command: `uv run python -m ai_skills.cli analyze --input data/inputs/us_relevant_30.csv`
If I don't specify which input file to use, use `data/inputs/us_relevant_30.csv`. Always name the output so that it's easy to recognize which version/iteration/new feature it was ran for.
