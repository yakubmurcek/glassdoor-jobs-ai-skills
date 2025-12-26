#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Pydantic models used across the analysis pipeline."""

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AITier(str, Enum):
    """Classification tier for AI involvement in a job."""

    CORE_AI = "core_ai"  # Building AI from scratch (model architecture, training, ML research)
    APPLIED_AI = "applied_ai"  # Meaningful AI work using frameworks (fine-tuning, TensorFlow, MLOps)
    AI_INTEGRATION = "ai_integration"  # Using AI as a tool (OpenAI API, Copilot, ChatGPT)
    NONE = "none"  # No AI involvement


class JobAnalysisResult(BaseModel):
    """Normalized representation of OpenAI analysis output."""

    ai_tier: AITier = AITier.NONE
    ai_skills_mentioned: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    # Raw skill extractions from LLM (normalized in pipeline)
    hardskills_raw: List[str] = Field(default_factory=list)
    softskills_raw: List[str] = Field(default_factory=list)
    # Education requirement: 1 if required, 0 if preferred/optional/undeterminable
    education_required: int = Field(default=0, ge=0, le=1)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        return max(0.0, min(1.0, float(v)))

    def as_columns(self) -> dict:
        """Map the result to the output DataFrame columns.
        
        Note: hardskills/softskills normalization happens in the pipeline.
        """
        return {
            "AI_tier_openai": self.ai_tier.value,
            "AI_skills_openai_mentioned": ", ".join(self.ai_skills_mentioned),
            "AI_skill_openai_confidence": self.confidence,
            "AI_skill_openai_rationale": self.rationale,
            "EDUCATION2_REQUIRED": self.education_required,
        }

    model_config = ConfigDict(frozen=True)


class JobAnalysisResultWithId(BaseModel):
    """Job analysis result with ID field for batch responses.
    
    Note: This model is used for OpenAI structured output, which requires
    all fields to be explicitly present (no defaults with $ref types).
    """

    id: str
    ai_tier: AITier  # No default - required by OpenAI
    ai_skills_mentioned: List[str]  # No default - required by OpenAI
    confidence: float = Field(ge=0.0, le=1.0)  # No default - required by OpenAI
    rationale: str  # No default - required by OpenAI
    hardskills_raw: List[str]  # No default - required by OpenAI
    softskills_raw: List[str]  # No default - required by OpenAI
    education_required: int = Field(ge=0, le=1)  # No default - required by OpenAI

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        return max(0.0, min(1.0, float(v)))

    def to_job_analysis_result(self) -> JobAnalysisResult:
        """Convert to JobAnalysisResult by removing the id field."""
        return JobAnalysisResult(
            ai_tier=self.ai_tier,
            ai_skills_mentioned=self.ai_skills_mentioned,
            confidence=self.confidence,
            rationale=self.rationale,
            hardskills_raw=self.hardskills_raw,
            softskills_raw=self.softskills_raw,
            education_required=self.education_required,
        )

    model_config = ConfigDict(frozen=True)


class BatchAnalysisResponse(BaseModel):
    """Response model for batch job analysis from OpenAI."""

    results: List[JobAnalysisResultWithId]

    model_config = ConfigDict(frozen=True)
