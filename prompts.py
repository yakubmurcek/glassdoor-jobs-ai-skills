#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Houses editable prompt templates for OpenAI calls."""

from textwrap import dedent


def job_analysis_prompt(job_description: str) -> str:
    """Generate the prompt for analyzing a single job description."""
    template = """
    Analyze the following job description and determine if it mentions any AI
    (Artificial Intelligence), Machine Learning, or related skills.

    Consider these categories of AI skills:
    - Core AI/ML: artificial intelligence, machine learning, deep learning, neural networks
    - Generative AI/LLMs: GPT, BERT, LLM, large language models, generative AI, prompt engineering
    - NLP: natural language processing, text analysis, sentiment analysis
    - ML Frameworks: PyTorch, TensorFlow, Keras, scikit-learn
    - MLOps: model deployment, MLflow, model serving
    - Cloud AI: AWS SageMaker, Azure ML, Vertex AI
    - Other AI-related technologies and concepts

    Job Description:
    {job_description}

    Respond with a JSON object in this exact format:
    {{
        "has_ai_skill": true or false,
        "ai_skills_mentioned": ["skill1", "skill2", ...]
    }}

    Only include skills that are explicitly mentioned or clearly implied in the
    job description. Be conservative - if the job description doesn't clearly
    mention AI/ML work, set has_ai_skill to false.
    """
    return dedent(template).strip().format(job_description=job_description)

