#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Houses editable prompt templates for OpenAI calls."""

from textwrap import dedent


def job_analysis_instructions() -> str:
    """Return the static instructions for the OpenAI job analysis prompt."""
    template = """
Classify if a job requires hands-on AI/ML work: building, training, fine-tuning, deploying, or maintaining ML/DL models, LLMs, or AI systems.

YES (AI/ML work):
- Train/deploy ML models, fine-tune LLMs, build inference pipelines
- Work with PyTorch/TensorFlow/scikit-learn for modeling
- Design model architectures, evaluate ML performance, develop RAG pipelines

NO (NOT AI/ML work):
- Software engineering at AI companies without modeling tasks
- Using AI products/dashboards without building models
- Backend/frontend/API work even if company is "AI-powered"
- Mentions of AI only in company description or marketing

CRITICAL RULES:
1. Words like "AI", "AI-powered", "AI startup" do NOT imply AI/ML work
2. General engineering (React, Flask, Node, SQL, DevOps, APIs) is NEVER AI work
3. Never infer AI skills from company description or product marketing
4. No explicit model tasks = has_ai_skill: false, ai_skills_mentioned: []
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
        Each entry must include: id, has_ai_skill, ai_skills_mentioned, confidence.
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
