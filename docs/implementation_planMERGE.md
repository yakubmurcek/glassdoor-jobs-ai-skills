# Fix Sparse Categories for Model Stability

## Problem

Several categories have dangerously low cell counts in cross-tabulations, which causes:
- Unstable coefficient estimates (huge standard errors)
- Separation issues in logit/mlogit (e.g. PhD × AI_Integration = 0 obs → coefficient = −∞)
- Unreliable chi² tests (expected frequencies < 5 violate assumptions)

### Inventory of Problematic Categories

#### `desc_tier_llm` / `ai_tier_num`: **core_ai has only 6 observations**

| Category | N | % |
|----------|---:|----:|
| none | 14,190 | 79.5% |
| ai_integration | 2,354 | 13.2% |
| applied_ai | 1,307 | 7.3% |
| **core_ai** | **6** | **0.03%** |

> [!CAUTION]
> With only 6 observations, `core_ai` is useless for inference. It inflates dummies, eats degrees of freedom, and causes near-separation in logit. The Bonferroni post-hoc already shows p = 1.000 for core_ai comparisons.

#### `edu_cat`: **PhD has 12 obs, and cross-tabs with AI have near-zero cells**

| edu_cat × ai_tier | ai_integration | applied_ai | core_ai | none | Total |
|-------------|---:|---:|---:|---:|---:|
| Highschool | 25 | 16 | 0 | 327 | 368 |
| Associate | 20 | 7 | **0** | 318 | 345 |
| Bachelor | 1,220 | 711 | 1 | 8,508 | 10,440 |
| Master | 23 | 36 | **2** | 263 | 324 |
| **PhD** | **0** | **5** | **1** | **6** | **12** |

> [!WARNING]
> PhD × ai_integration = **0 observations** → causes perfect separation in logit (RRR = 1.45e-06 in mlogit output). PhD total = 12 is far too small for stable estimates.

#### `exp_category`: **Expert (10+) is small but acceptable**

| Category | N | % |
|----------|---:|----:|
| Missing | 2,595 | 14.5% |
| Entry (0) | 753 | 4.2% |
| Junior (1-2) | 2,460 | 13.8% |
| Mid (3-5) | 8,411 | 47.1% |
| Senior (6-10) | 3,419 | 19.2% |
| Expert (10+) | 219 | 1.2% |

Expert (10+) with 219 obs is borderline — fine for OLS, marginal for logit subgroups.

---

## Proposed Changes

### 1. Merge `core_ai` into `applied_ai`

This is already done in the `ai_level` variable (used in main regression models). The fix is to also clean up the **descriptive tables** and **chi² tests** that still use raw `desc_tier_llm`.

#### [MODIFY] [ai_skills_analysis.do](file:///c:/Users/murcj/Projects/glassdoor-jobs-ai-skills/analysis/stata/ai_skills_analysis.do)

**In Section 3.1** (after encoding `ai_tier_num`), add:
```stata
* Merge core_ai into applied_ai (only 6 obs in core_ai)
replace desc_tier_llm = "applied_ai" if desc_tier_llm == "core_ai"
```

Then re-generate `ai_tier_num`:
```stata
* Drop old encoding and re-encode
drop ai_tier_num
encode desc_tier_llm, generate(ai_tier_num)
```

**Result:** 3-level `desc_tier_llm`: none / ai_integration / applied_ai (1,313 obs in applied_ai). All downstream tables (`tab`, `tabstat`, `oneway`, chi²) automatically become cleaner.

### 2. Merge PhD into Master → create "Master+"

**In Section 3.3**, after creating `edu_cat`:
```stata
* Merge PhD (n=12) into Master → "Master+" for stable estimation
replace edu_cat = 4 if edu_cat == 5
label define edu_lbl 0 "Missing" 1 "Highschool" 2 "Associate" 3 "Bachelor" 4 "Master+", replace
```

**Result:** Master+ will have 324 + 12 = 336 obs — all cells in cross-tabs will be ≥ 20.

### 3. (Optional) Consider merging Highschool + Associate → "Below Bachelor"

Highschool (368) and Associate (345) are individually viable for OLS, but their AI cross-tab cells (25+16 and 20+7) are small. Two options:

- **Option A (Recommended):** Keep them separate. N ≥ 20 per cell is passable, and the distinction is substantively meaningful.
- **Option B:** Merge into "Below Bachelor" (713 obs). Cleaner stats, but loses granularity.

> [!IMPORTANT]
> I recommend **Option A** (keep separate) unless you encounter convergence issues in logit after making the other fixes. If you prefer Option B, let me know.

### 4. Keep Expert (10+) as-is

With 219 total obs (48 in AI group), it's thin but workable. The experience variable is ordinal and collapsing it loses the meaningful top-end signal. If Expert causes issues in interactions later, merge into Senior to create "6+ years".

---

## Summary of Changes

| Variable | Current | After Fix | Rationale |
|----------|---------|-----------|-----------|
| `desc_tier_llm` | 4 levels (core_ai = 6 obs) | **3 levels** (core_ai merged into applied_ai) | Eliminates near-empty cells everywhere |
| `edu_cat` | 6 levels (PhD = 12 obs) | **5 levels** (PhD merged into Master+) | Fixes separation in logit, stable cross-tabs |
| `exp_category` | 6 levels (Expert = 219) | **No change** | 219 is sufficient |
| Highschool/Associate | Separate (368/345) | **No change** (Option A) | Borderline but substantively distinct |

## Verification Plan

### Manual Verification
After making the changes, re-run the do-file in Stata and check:
1. `tab desc_tier_llm` — should show exactly 3 categories (no core_ai)
2. `tab edu_cat` — should show 5 categories (Master+ instead of Master/PhD)
3. `tab edu_cat ai_tier_num, chi2` — no cell should have expected frequency < 5
4. Logit and mlogit should converge without extreme coefficients (no RRR = 1.45e-06)
5. Check that the Model A/B regression results are largely unchanged (core_ai and PhD had negligible weight)
