# Pipeline Evaluation Report

**Generated**: 2026-01-04T13:12:47.130790
**Baseline**: `us_relevant_50_gpt5nano_jan4.csv`
**Candidate**: `us_relevant_50_gpt5mini_jan4.csv`

## Summary

| Metric | Value |
|--------|-------|
| Total Jobs | 50 |
| Match Rate | **94.0%** (47/50) |
| Confidence Change | -0.003 |
| Agreement Change | +0.0% |
| Avg Hardskills/Job | 20.8 |
| Avg Softskills/Job | 5.9 |

## Tier Distribution

| Tier | Baseline | Candidate | Change |
|------|----------|-----------|--------|
| `none` | 34 | 34 | 0 |
| `ai_integration` | 12 | 13 | +1 |
| `applied_ai` | 4 | 3 | -1 |
| `core_ai` | 0 | 0 | 0 |

## Classification Changes (3)

### 1. Job 29: Senior Software Engineer - Fullstack

**Change**: `ai_integration` → `applied_ai`

| | Baseline | Candidate |
|--|----------|-----------|
| Confidence | 0.95 | 0.90 |
| AI Skills | GenAI, AgenticAI, ChatGPT, Vertex AI, AWS AI, Claude, Perplexity, CrewAI, Anthropic, Jasper, Cohere, Hugging Face, OpenAI, AI | GenAI, AgenticAI, GPT-like models, ChatGPT, Vertex AI, AWS AI, Claude, Perplexity, CrewAI, Anthropic, Jasper, Cohere, Hugging Face, machine learning, model training, fine-tuning, model deployment |

**Job Description Excerpt**:
> :
Who is SimSpace
SimSpace launched in 2015 with a singular purpose –
. The organizations around the world that we depend on every day to keep our loved ones safe and secure. Our healthcare facilities, schools, financial institutions, transit centers, grocery stores, and workplaces just to name a few. To deliver global resiliency, we provide an elite cyber range platform to curate unassailable cyb...

**Baseline Rationale**:
> nan

**Candidate Rationale**:
> nan

---

### 2. Job 33: (Freelancer) AI Automation Engineer fullstack (Code & No-Code)

**Change**: `applied_ai` → `ai_integration`

| | Baseline | Candidate |
|--|----------|-----------|
| Confidence | 0.82 | 0.90 |
| AI Skills | HeyGen, AI | AI tools, python, APIs, heygen, no-code, automation |

**Job Description Excerpt**:
> Summary
We are seeking a talented tech specialist who is proficient in both coding and no-code platforms. The ideal candidate should have a strong understanding of AI technologies and the ability to implement them effectively within business processes. A willingness to learn and adapt to new tools and trends is essential. This role is perfect for someone looking to enhance their skills in a dynami...

**Baseline Rationale**:
> nan

**Candidate Rationale**:
> nan

---

### 3. Job 37: Full-Stack Engineer

**Change**: `applied_ai` → `ai_integration`

| | Baseline | Candidate |
|--|----------|-----------|
| Confidence | 0.82 | 0.70 |
| AI Skills | GenAI, LLM, machine learning, natural language processing | generative AI, GenAI, machine learning, natural language processing, ML, NLP, Python |

**Job Description Excerpt**:
> Overview:
Responsibilities:
Qualifications:
Join our cutting-edge generative AI (GenAI) platform, LIGER™, created by its technology studio, LMI Forge. LIGER™ harnesses the power of advanced technology, data analytics, and the latest in machine learning and natural language processing to provide secure, private, and trustworthy GenAI solutions for government.
LMI is a new breed of digital solutions...

**Baseline Rationale**:
> nan

**Candidate Rationale**:
> The company/product is explicitly a GenAI platform, but the role's duties focus on backend Python development and only mentions integrating ML tools (not model training or ML pipeline ownership), so it appears to be integration work rather than applied ML engineering.

---
