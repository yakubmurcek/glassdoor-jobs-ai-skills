# Pipeline Evaluation Report

**Generated**: 2025-12-26T09:41:33.476532
**Baseline**: `us_relevant_30_ai.csv`
**Candidate**: `us_relevant_30_ai_v1.csv`

## Summary

| Metric | Value |
|--------|-------|
| Total Jobs | 30 |
| Match Rate | **90.0%** (27/30) |
| Confidence Change | -0.005 |
| Agreement Change | +3.3% |
| Avg Hardskills/Job | 4.7 |
| Avg Softskills/Job | 2.1 |

## Tier Distribution

| Tier | Baseline | Candidate | Change |
|------|----------|-----------|--------|
| `none` | 24 | 25 | +1 |
| `ai_integration` | 2 | 3 | +1 |
| `applied_ai` | 3 | 1 | -2 |
| `core_ai` | 1 | 1 | 0 |

## Classification Changes (3)

### 1. Job 4: Software Eningeer II - Innovation

**Change**: `ai_integration` → `none`

| | Baseline | Candidate |
|--|----------|-----------|
| Confidence | 0.80 | 0.90 |
| AI Skills | AIML | nan |

**Job Description Excerpt**:
> WHO WE ARE LOOKING FOR
We’re looking for a mid-level engineer who can independently deliver features, mentor junior engineers, and contribute to architectural decisions. You’re a strong communicator, a cross-functional collaborator, and someone who thrives in a dynamic environment.
WHAT YOU WILL WORK ON
Design and implement scalable services and user interfaces
Lead feature development using React...

**Baseline Rationale**:
> The job mentions proficiency in AIML but does not indicate direct involvement in AI model development.

**Candidate Rationale**:
> The position involves full-stack development without any AI-related tasks.

---

### 2. Job 8: Fullstack Engineer

**Change**: `applied_ai` → `ai_integration`

| | Baseline | Candidate |
|--|----------|-----------|
| Confidence | 0.75 | 0.80 |
| AI Skills | AI | nan |

**Job Description Excerpt**:
> About Diffit
Diffit is an AI-powered instructional materials platform designed to help teachers do their best work more sustainably. Teachers use Diffit to get “just right” classroom activities, saving time and helping all students access grade-level content. Come join a growing remote-first and mission-focused AI edtech company to build quality, safe and affordable educational resources for teach...

**Baseline Rationale**:
> The role involves building features for an AI-powered platform, indicating some level of applied AI work.

**Candidate Rationale**:
> The role is in an AI-powered company but does not involve developing AI models; it focuses on full-stack development.

---

### 3. Job 25: Full Stack Software Engineer

**Change**: `applied_ai` → `ai_integration`

| | Baseline | Candidate |
|--|----------|-----------|
| Confidence | 0.85 | 0.75 |
| AI Skills | AI, NLP | AI-powered marketing, NLP |

**Job Description Excerpt**:
> About Us
Tildei is an AI-powered marketing platform that creates intelligent brand agents for commerce and marketing conversations. We build comprehensive, custom Brand Knowledge Graphs from product catalog, marketing materials, FAQs, and brand guidelines. We then deploy agents across social and digital channels to engage customers 24/7 in any language. Our agents drive marketing and commerce outc...

**Baseline Rationale**:
> The role involves building AI-powered marketing agents, indicating a hands-on application of AI technologies.

**Candidate Rationale**:
> The role involves building features for an AI-powered platform, but does not include direct AI model development.

---
