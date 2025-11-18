#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Houses editable prompt templates for OpenAI calls."""

from textwrap import dedent


def job_analysis_instructions() -> str:
    """Return the static instructions for the OpenAI job analysis prompt."""
    template = """
    Analyze job descriptions and determine if they mention AI skills (Artificial
    Intelligence).

    Canonical AI/ML skill vocabulary (case-insensitive; prefer these exact tokens when
    applicable):
    - Core AI/ML: artificial intelligence, ai, machine learning, ml, deep learning, dl,
      neural network(s), supervised learning, unsupervised learning, reinforcement learning,
      rl, self-supervised learning, classification, regression, clustering
    - GenAI/LLMs: llm, large language model, gpt, gpt-3, gpt-4, bert, roberta, distilbert,
      albert, t5, llama, mistral, generative ai, genai, text generation, prompt engineering
    - NLP: nlp, nlu, nlg, natural language processing, tokenization, word embeddings,
      word2vec, glove, fasttext, named entity recognition, ner, sentiment analysis
    - Vision/Speech: computer vision, image recognition, object detection, yolo, retinanet,
      mask r-cnn, speech recognition
    - Frameworks & tooling: pytorch, tensorflow, keras, jax, flax, scikit-learn, sklearn,
      xgboost, lightgbm, catboost, pandas, numpy, scipy, spark ml, pyspark, mlflow, kubeflow,
      torchserve, tensorflow serving, onnx, onnxruntime
    - Cloud/infra: sagemaker, vertex ai, azure ml, aws ai, gcp ai, cuda, cudnn, gpu acceleration,
      distributed training, feature store, mlops, aiops, model deployment, model serving,
      model monitoring, model governance, model training, model inference
    - Edge/other: tflite, tensorflow lite, coreml, edge ai

    You may include other AI/ML skills that are not listed only when they name a specific
    algorithm, framework, or technique. Keep each skill entry to a short noun phrase (<= 4 words).

    Strict output rules for `ai_skills_mentioned` (keep this section simple and literal):
    - Qualify only concrete AI/ML techniques, model types, algorithms, AI-specific platforms, or clearly
      ML-oriented tooling (e.g., "transformer fine-tuning", "pytorch", "vertex ai"). General software
      stack items are never valid skills in this field.
    - Hard blocklist: under no circumstances output general-purpose languages, frameworks, databases,
      or hosting stacks such as react, angular, next.js, vue, flutter, swift, kotlin, java, c#, node.js,
      express, flask, django, rails, spring, rest, graphql, sql, html/css, git, docker, kubernetes, etc.
      If a posting only mentions these, return has_ai_skill=false and an empty skills list.
    - Dual-use utilities (python, pandas, spark, aws/azure/gcp, etc.) only qualify when the text spells
      out an AI/ML role, such as "pandas feature engineering" or "sagemaker model deployment".
    - If the description simply says "AI-powered" or references an AI product without naming any AI/ML
      techniques (e.g., a React/Flask role at an AI-enabled company), return has_ai_skill=false and an
      empty skills list.
    - Remove duplicates and keep consistent casing (lowercase unless the proper noun requires caps).

    Respond with a JSON object in this exact format:
    {{
        "has_ai_skill": true or false,
        "ai_skills_mentioned": ["skill1", "skill2", ...],
        "confidence": number between 0 and 1 describing how confident you are in
                       your has_ai_skill and ai_skills_mentioned answers
    }}

    If the job description does not clearly reference AI/ML work or skills, set has_ai_skill to false
    and return an empty skills list.

    When multiple job descriptions are provided with IDs, respond with a JSON object that
    has a top-level "results" array where each element includes the job's ID plus the same
    fields above.
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
