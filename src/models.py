#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared dataclasses used across the analysis pipeline."""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class JobAnalysisResult:
    """Normalized representation of OpenAI analysis output."""

    has_ai_skill: bool = False
    ai_skills_mentioned: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def as_columns(self) -> dict:
        """Map the result to the output DataFrame columns."""
        return {
            "AI_skill_openai": int(self.has_ai_skill),
            "AI_skills_openai_mentioned": ", ".join(self.ai_skills_mentioned),
            "AI_skill_openai_confidence": self.confidence,
        }
