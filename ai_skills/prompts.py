#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt templates for LLM job analysis.

This module contains two sets of prompts:

1. LEGACY (monolithic): `job_analysis_instructions()` + `job_analysis_batch_prompt()`
   - Single prompt asks for ALL fields at once (tier, skills, education)
   - Used when OpenAIJobAnalyzer(use_decomposed=False)
   - Kept for backwards compatibility

2. DECOMPOSED (default): Task-specific prompts
   - `ai_tier_*`: AI tier classification only
   - `skills_*`: Skills extraction only  
   - `education_*`: Education requirement only
   - Each task runs separately with focused prompts for better accuracy
   - Used when OpenAIJobAnalyzer(use_decomposed=True) [default]
"""

from textwrap import dedent


# =============================================================================
# LEGACY MONOLITHIC PROMPTS (use_decomposed=False)
# =============================================================================
# These prompts ask the LLM to do multiple tasks at once.
# Kept for backwards compatibility but decomposed mode is recommended.


def job_analysis_instructions() -> str:
    """System prompt for legacy monolithic mode (all tasks in one prompt)."""

    template = """
Classify each job into one of four AI involvement tiers based on the PRIMARY responsibilities:

**core_ai**: THE ENGINEER DIRECTLY DEVELOPS/TRAINS AI MODELS (VERY RARE)
  - Designing novel model architectures from the ground up
  - Pre-training foundation models, training models from scratch
  - ML/AI research, publishing papers, algorithm development
  - Building custom neural networks (not just using existing ones)
  - Not just building features around AI — must directly contribute to AI/ML work
  
**applied_ai**: Meaningful hands-on AI/ML work using existing frameworks
  - Fine-tuning LLMs or other pre-trained models
  - Building ML pipelines with TensorFlow, PyTorch, scikit-learn
  - Feature engineering, model evaluation, hyperparameter tuning
  - MLOps, model deployment, inference optimization
  - Building RAG pipelines, embeddings, vector search

**ai_integration**: Using AI as a tool (NOT developing/training models)
  - Calling OpenAI/Claude/Anthropic APIs from application code
  - Integrating pre-built AI services (no training involved)
  - Using Copilot, ChatGPT, or AI coding assistants
  - Consuming AI outputs without understanding/modifying models

**none**: No AI involvement in job responsibilities
  - Standard software engineering (web, mobile, backend)
  - Mentions "AI" only in company description or marketing
  - Working at an "AI company" but role is non-AI (HR, sales, ops)

CRITICAL RULES:
1. Judge by ACTUAL JOB DUTIES, not company description or buzzwords
2. "AI-powered company" does NOT mean the job involves AI work
3. Using TensorFlow/PyTorch for inference-only = ai_integration, NOT applied_ai
4. Fine-tuning = applied_ai; Prompt engineering only = ai_integration
5. When uncertain, ALWAYS choose the LOWER tier — it's better to under-classify than over-classify

COMMON MISTAKES TO AVOID:
- Full-stack engineer at an "AI-powered platform" building React/Python features = **none**, NOT applied_ai
- Building UI/backend for an AI product without touching ML code = **none**
- applied_ai requires HANDS-ON work with ML models (training, tuning, pipelines) — not just building around them
- core_ai is ONLY for roles where ML model development is the PRIMARY duty

EXAMPLES (for calibration):
- "ML Engineer training LLMs from scratch" → core_ai
- "AI Researcher publishing papers on novel architectures" → core_ai
- "Data Scientist fine-tuning models, building ML pipelines" → applied_ai
- "MLOps Engineer deploying models, building inference servers" → applied_ai
- "Backend dev calling OpenAI API" → ai_integration
- "Developer using GitHub Copilot for coding" → ai_integration
- "Fullstack engineer at AI company, builds React/Flask features" → none
- "Software engineer, company mentions AI but role is standard web dev" → none

---

ALSO EXTRACT SKILLS:

**hardskills_raw**: Extract ALL technical skills EXPLICITLY mentioned in job requirements. Be COMPREHENSIVE — include EVERY technical term, tool, framework, or technology mentioned. Categories include but are NOT limited to:
- Programming languages: python, java, javascript, typescript, sql, c#, c++, go, rust, kotlin, scala, ruby, php, swift, r, etc.
- Frontend frameworks: react, angular, vue, svelte, next.js, nuxt, etc.
- Backend frameworks: spring boot, django, flask, fastapi, express, nest.js, .net, rails, etc.
- State management: redux, ngrx, rxjs, vuex, mobx, zustand, recoil, etc.
- Databases: postgresql, mysql, mongodb, redis, elasticsearch, cassandra, dynamodb, sqlite, oracle, sql server, etc.
- Cloud & DevOps: aws, azure, gcp, docker, kubernetes, terraform, ansible, helm, etc.
- CI/CD & Build tools: jenkins, github actions, gitlab ci, circleci, maven, gradle, webpack, vite, etc.
- Testing frameworks: junit, mockito, jest, jasmine, karma, mocha, chai, pytest, selenium, cypress, playwright, testng, etc.
- APIs & Protocols: rest, restful api, graphql, grpc, websocket, soap, openapi, swagger, etc.
- Architecture patterns: microservices, monolith, serverless, event-driven, cqrs, etc.
- Version control: git, github, gitlab, bitbucket, svn, etc.
- Messaging: kafka, rabbitmq, sqs, pubsub, redis streams, etc.
- Monitoring & Logging: prometheus, grafana, datadog, splunk, elk stack, new relic, etc.
- Data & ETL: spark, airflow, pandas, dbt, snowflake, databricks, hadoop, etc.
- Security: oauth, jwt, ssl/tls, encryption, authentication, authorization, etc.
- Methodologies: agile, scrum, kanban, devops, tdd, bdd, ci/cd, etc.
- Web technologies: html, html5, css, css3, sass, less, tailwind, bootstrap, etc.
- Mobile: react native, flutter, swift, kotlin, ios, android, etc.
- Other tools & libraries: any other technical skills explicitly mentioned

CRITICAL: Extract EVERY technical skill/tool/framework/protocol mentioned. Do NOT limit to the examples above — these are just common categories. If it's a technical term, include it.

**softskills_raw**: Interpersonal/behavioral traits EXPLICITLY mentioned:
- communication skills, collaboration, teamwork, leadership
- problem-solving, analytical thinking, attention to detail
- adaptability, flexibility, initiative, creativity
- time management, organization skills, multitasking
- mentorship, independence, presentation skills

SKILL EXTRACTION RULES:
1. Extract ONLY skills explicitly mentioned — do NOT infer
2. Return raw skill names as written (normalization handled separately)
3. If a skill appears multiple times, include it once
4. Include both specific tools (e.g., "TensorFlow") AND domains (e.g., "machine learning")

---

EDUCATION REQUIREMENT:

**education_required**: Binary indicator (0 or 1) whether education is REQUIRED or just preferred.

Determine based on CONTEXT where education is mentioned:
- Return **1** if education appears under: Requirements, Must have, Required, Mandatory, Prerequisite, Essential, Minimum qualifications, "Required qualifications"
- Return **0** if education appears under: Preferred, Nice to have, Optional, Preferred Qualifications, Desired, Plus, Bonus, "Preferred qualifications"
- Return **0** if education is not mentioned or context is unclear

EXAMPLES:
- "Requirements: Bachelor's degree in CS" → education_required = 1
- "Preferred: Master's degree" → education_required = 0
- "Must have: BS/MS in Engineering" → education_required = 1
- "Nice to have: Advanced degree" → education_required = 0
- No education mentioned → education_required = 0
    """
    return dedent(template).strip()


def job_analysis_batch_prompt(batch_items: list[tuple[str, str, str]]) -> str:
    """Build a prompt covering multiple job descriptions in a single request.
    
    Args:
        batch_items: List of tuples (identifier, job_title, description)
    
    Note: This is the legacy monolithic prompt. Consider using task-specific
    prompts (ai_tier_*, skills_*, education_*) for better accuracy.
    """
    header = dedent(
        """
        Analyze each of the following job descriptions independently.
        Return a JSON object with a top-level `results` array.
        Each entry must include: id, ai_tier (one of: core_ai, applied_ai, ai_integration, none), ai_skills_mentioned, confidence, rationale, hardskills_raw, softskills_raw, education_required (0 or 1).
        IMPORTANT: If confidence >= 0.8, rationale MUST be an empty string "". Only provide rationale if confidence < 0.8.
        """
    ).strip()

    body_parts = []
    for identifier, title, description in batch_items:
        body_parts.append(
            dedent(
                f"""
                Job id: {identifier}
                Job Title: {title}
                Job Description:
                {description}
                """
            ).strip()
        )
    return f"{header}\n\n" + "\n\n".join(body_parts)


# =============================================================================
# Task-Specific Prompts for Decomposed Batching
# =============================================================================
# Each prompt focuses on ONE task only, reducing cognitive load on the model.


def ai_tier_instructions() -> str:
    """Focused instructions ONLY for AI tier classification."""
    template = """
Classify each job into one of four AI involvement tiers based on the PRIMARY responsibilities:

**core_ai**: THE ENGINEER DIRECTLY DEVELOPS/TRAINS AI MODELS (VERY RARE)
  - Designing novel model architectures from the ground up
  - Pre-training foundation models, training models from scratch
  - ML/AI research, publishing papers, algorithm development
  - Building custom neural networks (not just using existing ones)
  
**applied_ai**: Meaningful hands-on AI/ML work using existing frameworks
  - Fine-tuning LLMs or other pre-trained models
  - Building ML pipelines with TensorFlow, PyTorch, scikit-learn
  - Feature engineering, model evaluation, hyperparameter tuning
  - MLOps, model deployment, inference optimization
  - Building RAG pipelines, embeddings, vector search

**ai_integration**: Using AI as a tool (NOT developing/training models)
  - Calling OpenAI/Claude/Anthropic APIs from application code
  - Integrating pre-built AI services (no training involved)
  - Using Copilot, ChatGPT, or AI coding assistants

**none**: No AI involvement in job responsibilities
  - Standard software engineering (web, mobile, backend)
  - Mentions "AI" only in company description or marketing
  - Working at an "AI company" but role is non-AI (HR, sales, ops)

CRITICAL RULES:
1. Judge by ACTUAL JOB DUTIES, not company description or buzzwords
2. "AI-powered company" does NOT mean the job involves AI work
3. Using TensorFlow/PyTorch for inference-only = ai_integration, NOT applied_ai
4. Fine-tuning = applied_ai; Prompt engineering only = ai_integration
5. When uncertain, ALWAYS choose the LOWER tier
6. IF CONFIDENCE is 0.8 or higher, set `rationale` to an empty string ("") to save tokens. Only provide rationale if you are unsure (confidence < 0.8).

EXAMPLES:
- "ML Engineer training LLMs from scratch" → core_ai
- "Data Scientist fine-tuning models, building ML pipelines" → applied_ai
- "Backend dev calling OpenAI API" → ai_integration
- "Fullstack engineer at AI company, builds React/Flask features" → none
    """
    return dedent(template).strip()


def ai_tier_batch_prompt(batch_items: list[tuple[str, str, str]]) -> str:
    """Build batch prompt for AI tier classification task.
    
    Args:
        batch_items: List of tuples (identifier, job_title, description)
    """
    header = dedent(
        """
        For each job, return: id, ai_tier (core_ai/applied_ai/ai_integration/none), confidence (0.0-1.0), rationale (brief).
        IMPORTANT: If confidence >= 0.8, rationale MUST be an empty string "". Only provide rationale if confidence < 0.8.
        """
    ).strip()

    body_parts = []
    for identifier, title, description in batch_items:
        body_parts.append(f"[{identifier}] {title}\n{description}")
    
    return f"{header}\n\n" + "\n\n---\n\n".join(body_parts)


def skills_instructions() -> str:
    """Focused instructions ONLY for skills extraction."""
    template = """
Extract skills from each job description. Return three lists:

**ai_skills_mentioned**: AI/ML specific skills mentioned:
  - Frameworks: tensorflow, pytorch, keras, scikit-learn, huggingface
  - Concepts: machine learning, deep learning, nlp, computer vision
  - Tools: mlflow, kubeflow, sagemaker, vertex ai
  - Terms: llm, transformer, embedding, fine-tuning, rag

**hardskills_raw**: ALL technical skills EXPLICITLY mentioned:
  - Programming languages, frameworks, databases, cloud platforms
  - DevOps tools, CI/CD, testing frameworks, APIs
  - Include EVERY technical term, tool, or technology mentioned

**softskills_raw**: Interpersonal/behavioral traits EXPLICITLY mentioned:
  - communication, collaboration, teamwork, leadership
  - problem-solving, analytical thinking, attention to detail
  - adaptability, initiative, mentorship

RULES:
1. Extract ONLY skills explicitly mentioned — do NOT infer
2. Return raw skill names as written
3. If a skill appears multiple times, include it once
    """
    return dedent(template).strip()


def skills_batch_prompt(batch_items: list[tuple[str, str, str]]) -> str:
    """Build batch prompt for skills extraction task.
    
    Args:
        batch_items: List of tuples (identifier, job_title, description)
    """
    header = dedent(
        """
        For each job, return: id, ai_skills_mentioned (list), hardskills_raw (list), softskills_raw (list).
        """
    ).strip()

    body_parts = []
    for identifier, title, description in batch_items:
        body_parts.append(f"[{identifier}] {title}\n{description}")
    
    return f"{header}\n\n" + "\n\n---\n\n".join(body_parts)


def education_instructions() -> str:
    """Focused instructions ONLY for education requirement detection."""
    template = """
Determine if education is REQUIRED or just preferred for each job.

Return **1** if education appears under:
  - Requirements, Must have, Required, Mandatory
  - Prerequisite, Essential, Minimum qualifications

Return **0** if education appears under:
  - Preferred, Nice to have, Optional, Desired, Plus, Bonus
  - Not mentioned or context is unclear

EXAMPLES:
- "Requirements: Bachelor's degree in CS" → 1
- "Preferred: Master's degree" → 0
- "Must have: BS/MS in Engineering" → 1
- "Nice to have: Advanced degree" → 0
- No education mentioned → 0
    """
    return dedent(template).strip()


def education_batch_prompt(batch_items: list[tuple[str, str, str, str]]) -> str:
    """Build batch prompt for education requirement task.
    
    Args:
        batch_items: List of tuples (identifier, job_title, description, educations)
                     where educations is the value from the 'educations' column
    """
    header = dedent(
        """
        For each job, determine if education is REQUIRED or just preferred.
        You are given the job description AND the education levels mentioned.
        Focus on whether these education levels are REQUIRED vs PREFERRED based on context.
        Return: id, education_required (0 or 1).
        """
    ).strip()

    body_parts = []
    for identifier, title, description, educations in batch_items:
        edu_info = f"Job posting education listed: {educations}" if educations else "Job posting education listed: None"
        body_parts.append(f"[{identifier}] {title}\n{edu_info}\n{description}")
    
    return f"{header}\n\n" + "\n\n---\n\n".join(body_parts)
