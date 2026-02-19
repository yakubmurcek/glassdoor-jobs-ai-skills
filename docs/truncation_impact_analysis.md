# Impact Analysis of Job Description Truncation

## Context

The `_prepare_text` method in `openai_analyzer.py` truncates job descriptions exceeding `MAX_JOB_DESC_LENGTH` (configured to 6,000 characters) by applying a naive prefix cut (`text[:6000]`). This means that for longer descriptions, the latter portions — which frequently contain explicit skill requirements, education prerequisites, and experience expectations — may be partially or entirely lost before reaching the LLM.

The output CSV retains the original, untruncated text. However, the LLM-derived columns (`desc_tier_llm`, `hardskills`, `softskills`, `edulevel_llm`, `experience_min_llm`) for affected records were produced from incomplete input.

## Scope

Out of 18,464 total job descriptions, **1,031 (5.6%)** exceeded the 6,000-character limit and were truncated during LLM inference. The dataset has a mean description length of 4,112 characters (median 3,939), with the 90th percentile at 6,941 and the 95th at 7,830 — confirming that the threshold affects the upper tail of the length distribution.

| Original Description Length | Affected Jobs | Approx. Text Visible to LLM |
|-----------------------------|---------------|------------------------------|
| 6,000 – 7,000 characters   | 290           | ~92%                         |
| 7,000 – 8,000 characters   | 391           | ~80%                         |
| 8,000 – 10,000 characters  | 258           | ~67%                         |
| 10,000 – 15,000 characters | 84            | ~48%                         |
| 15,000 – 25,000 characters | 8             | ~30%                         |

On average, the LLM received **77.4%** of the original text across affected records. In the worst case, only **26.3%** was visible.

## Risk by LLM Task

- **AI Tier Classification (Low Risk):** Diagnostic signals (company context, role summary) appear early in descriptions and are preserved by prefix truncation.
- **Skills Extraction (Moderate Risk):** Technical requirements often appear in "Qualifications" sections in the latter half. Approximately 350 descriptions lost over 30% of their text, risking missed skills.
- **Education & Experience (Moderate-to-High Risk):** These fields are typically stated near the end of descriptions and are most vulnerable to prefix truncation. The 92 descriptions exceeding 10,000 characters likely have unreliable education and experience values.

## Conclusion

The majority of affected records (681 of 1,031) lost less than 20% of their text, limiting the practical impact on classification accuracy. However, approximately 350 records experienced significant data loss (>30%), with consequences primarily for skills extraction and education/experience fields. For future iterations, increasing the limit to 12,000–15,000 characters would capture over 99% of descriptions without truncation.
