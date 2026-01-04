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
Classify each job into one of four AI involvement tiers based on the engineer's ACTUAL DUTIES:

**core_ai**: Develops/trains AI models from scratch (EXTREMELY RARE)
  - Training foundation models, designing novel architectures
  - ML research, publishing papers

**applied_ai**: Hands-on work WITH ML models — look for these in JOB DUTIES:
  - "fine-tune", "training pipeline", "model training", "hyperparameter"
  - "feature engineering", "model evaluation", "MLOps", "model deployment"
  - MUST modify models or build ML pipelines — not just use AI outputs

**ai_integration**: Uses AI as black-box (no training/tuning)
  - Calling OpenAI/Claude APIs, integrating AI services
  - Building apps that display AI outputs
  - Prompt engineering without fine-tuning

**none**: No AI mentioned anywhere in the job posting

⚠️ CLASSIFICATION GUIDANCE:
- Consider the company context when classifying
- applied_ai = Works on model training, fine-tuning, or ML pipelines (not just using AI tools)
- ai_integration = Uses AI services or builds AI-powered features
- If company is AI-focused but job duties say nothing about using AI, classify as none

When uncertain, lean toward HIGHER tier - false positives are easier to review.

---

ALSO EXTRACT SKILLS:

**ai_skills_mentioned**: AI/ML specific skills (tensorflow, pytorch, llm, etc.)

**hardskills_raw**: ALL technical skills explicitly mentioned (languages, frameworks, tools, databases, cloud, etc.)

**softskills_raw**: Interpersonal traits explicitly mentioned (communication, teamwork, leadership, etc.)

---

EDUCATION & EXPERIENCE:

**min_years_experience**: Minimum years required (float or null). Use minimum if range given.

**min_education_level**: Extract ONLY if EXPLICITLY STATED in the job posting.
- Return one of: "High School", "Associate", "Bachelor's", "Master's", "PhD", or **null**.
- If multiple levels listed (e.g., "Bachelor's or Master's"), choose the LOWEST.
- Use standardized terms (e.g., "Bachelor's" instead of "BS", "B.Sc").

⚠️ BE CONSERVATIVE - better to miss than to guess:
- If the job posting does NOT mention any degree/education requirement → return **null**
- Many job postings do not specify education — that's OK, just return null
- DO NOT assume "Bachelor's" just because it's a tech job
- Only return a value if you see words like: "degree required", "Bachelor's", "BS/MS", "PhD", etc.
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
        Each entry must include: id, ai_tier (one of: core_ai, applied_ai, ai_integration, none), ai_skills_mentioned, confidence, rationale, hardskills_raw, softskills_raw, min_years_experience (float or null), min_education_level (string or null).
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
Classify each job into one of four AI involvement tiers based on the engineer's ACTUAL DUTIES:

**core_ai**: Develops/trains AI models from scratch (EXTREMELY RARE)
  - Training foundation models, designing novel architectures
  - ML research, publishing papers

**applied_ai**: Hands-on work WITH ML models — look for these in JOB DUTIES:
  - "fine-tune", "training pipeline", "model training", "hyperparameter"
  - "feature engineering", "model evaluation", "MLOps", "model deployment"
  - MUST modify models or build ML pipelines — not just use AI outputs

**ai_integration**: Uses AI as black-box (no training/tuning)
  - Calling OpenAI/Claude APIs, integrating AI services
  - Building apps that display AI outputs
  - Prompt engineering without fine-tuning

**none**: No AI mentioned anywhere in the job posting

⚠️ CLASSIFICATION GUIDANCE:
- Consider the company context when classifying
- applied_ai = Works on model training, fine-tuning, or ML pipelines (not just using AI tools)
- ai_integration = Uses AI services or builds AI-powered features
- If company is AI-focused but job duties say nothing about using AI, classify as none

When uncertain, lean toward HIGHER tier - false positives are easier to review.

ALSO EXTRACT **ai_skills_mentioned**: Scan the ENTIRE job posting (including company description) for AI/ML skills:
  - Frameworks: tensorflow, pytorch, keras, scikit-learn, huggingface, langchain
  - Concepts: machine learning, deep learning, nlp, computer vision, llm, genai, rag
  - Tools: mlflow, kubeflow, sagemaker, openai, claude, copilot, chatgpt, vertex ai
  - If company mentions AI/ML/LLM in their description, include those terms
  - Return empty list [] ONLY if there is absolutely no AI/ML mention anywhere

    """
    return dedent(template).strip()


def ai_tier_batch_prompt(batch_items: list[tuple[str, str, str]]) -> str:
    """Build batch prompt for AI tier classification task.
    
    Args:
        batch_items: List of tuples (identifier, job_title, description)
    """
    header = dedent(
        """
        For each job, return: id, ai_tier (core_ai/applied_ai/ai_integration/none), ai_skills_mentioned (list of AI/ML skills), confidence (0.0-1.0), rationale (brief).
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

**ai_skills_mentioned**: AI/ML specific skills:
  - Frameworks: tensorflow, pytorch, keras, scikit-learn, huggingface
  - Concepts: machine learning, deep learning, nlp, computer vision
  - Tools: mlflow, kubeflow, sagemaker, vertex ai
  - Terms: llm, transformer, embedding, fine-tuning, rag

**hardskills_raw**: ALL technical skills EXPLICITLY mentioned. Be COMPREHENSIVE:
  - Languages: python, java, javascript, typescript, sql, c++, go, rust, etc.
  - Web: react, angular, vue, next.js, django, flask, spring boot, .net, etc.
  - Data: postgresql, mongodb, redis, elasticsearch, kafka, spark, airflow, etc.
  - Cloud/DevOps: aws, azure, gcp, docker, kubernetes, terraform, etc.
  - Other: git, rest api, graphql, microservices, agile, testing frameworks, etc.
  - Include EVERY technical term mentioned — don't limit to examples above

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
    """Focused instructions for education and experience requirement detection."""
    template = """
Extract education and experience requirements from each job description.

**min_years_experience**: Extract the MINIMUM years of professional experience required.
- Return a float value (e.g., 3.0, 5.0, 0.5).
- If a range is given (e.g., "3-5 years"), use the MINIMUM (3.0).
- If "3+ years" is stated, use 3.0.
- If experience is listed as "preferred" or "nice to have", still extract the number.
- If no specific number of years is mentioned, return **null**.
- "Entry level" or "Recent grad" with no years specified -> 0.0.

**min_education_level**: Extract ONLY if EXPLICITLY STATED in the job posting.
- Return one of: "High School", "Associate", "Bachelor's", "Master's", "PhD", or **null**.
- If multiple levels listed (e.g., "Bachelor's or Master's"), choose the LOWEST.
- Use standardized terms (e.g., "Bachelor's" instead of "BS", "B.Sc").

⚠️ BE CONSERVATIVE - better to miss than to guess:
- If the job posting does NOT mention any degree/education requirement → return **null**
- Many job postings do not specify education — that's OK, just return null
- DO NOT assume "Bachelor's" just because it's a tech job
- Only return a value if you see words like: "degree required", "Bachelor's", "BS/MS", "PhD", etc.

EXAMPLES:
- "Bachelor's degree required" → "Bachelor's"
- "MS/PhD in CS" → "Master's" (minimum)
- No education mentioned in posting → **null** (NOT "Bachelor's")
- "3+ years experience" with no degree mentioned → **null**
- "Bachelor's preferred" or "degree a plus" → **null** (not required)
    """
    return dedent(template).strip()


def education_batch_prompt(batch_items: list[tuple[str, str, str, str]]) -> str:
    """Build batch prompt for education/experience requirement task.
    
    Args:
        batch_items: List of tuples (identifier, job_title, description, educations)
                     Note: educations column is intentionally NOT passed to the LLM
                     to ensure independent analysis from job description text.
    """
    header = dedent(
        """
        For each job, analyze the job description and extract:
        - min_years_experience (float or null)
        - min_education_level (one of: "High School", "Associate", "Bachelor's", "Master's", "PhD", or null)
        
        Return the MINIMUM acceptable values, not preferred/ideal values.
        """
    ).strip()

    body_parts = []
    for identifier, title, description, _educations in batch_items:
        # Intentionally NOT including educations column to force LLM to analyze description
        body_parts.append(f"[{identifier}] {title}\n{description}")
    
    return f"{header}\n\n" + "\n\n---\n\n".join(body_parts)


