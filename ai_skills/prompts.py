#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Houses editable prompt templates for OpenAI calls."""

from textwrap import dedent


def job_analysis_instructions() -> str:
    """Return the static instructions for the OpenAI job analysis prompt."""
    template = """
Classify each job into one of four AI involvement tiers based on the PRIMARY responsibilities:

**core_ai**: Building AI systems from scratch (RARE - most jobs don't qualify)
  - Designing novel model architectures from the ground up
  - Pre-training foundation models, training models from scratch
  - ML/AI research, publishing papers, algorithm development
  - Building custom neural networks (not just using existing ones)
  
**applied_ai**: Meaningful hands-on AI work using existing frameworks
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
5. When uncertain, lean toward the LOWER tier (e.g., none over ai_integration)
    """
    return dedent(template).strip()


def job_analysis_prompt(job_description: str) -> str:
    """Generate the user prompt that only contains the job description text."""
    template = """
    Job Description:
    {job_description}
    """
    return dedent(template).strip().format(job_description=job_description)


def job_analysis_batch_prompt(batch_items: list[tuple[str, str]]) -> str:
    """Build a prompt covering multiple job descriptions in a single request."""
    header = dedent(
        """
        Analyze each of the following job descriptions independently.
        Return a JSON object with a top-level `results` array.
        Each entry must include: id, ai_tier (one of: core_ai, applied_ai, ai_integration, none), ai_skills_mentioned, confidence, rationale.
        """
    ).strip()

    body_parts = []
    for identifier, description in batch_items:
        body_parts.append(
            dedent(
                f"""
                Job id: {identifier}
                Job Description:
                {description}
                """
            ).strip()
        )
    return f"{header}\n\n" + "\n\n".join(body_parts)
