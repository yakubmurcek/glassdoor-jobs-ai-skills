#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI API integration for analyzing job descriptions."""

import json
import time
from typing import Any, Optional

from openai import OpenAI

from config import (
    MAX_JOB_DESC_LENGTH,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    RATE_LIMIT_DELAY,
)
from models import JobAnalysisResult
from prompts import job_analysis_instructions, job_analysis_prompt


class OpenAIJobAnalyzer:
    """Encapsulates the OpenAI client and response parsing logic."""

    def __init__(
        self,
        *,
        api_key: str = OPENAI_API_KEY,
        model: str = OPENAI_MODEL,
        temperature: float = OPENAI_TEMPERATURE,
        delay_seconds: float = RATE_LIMIT_DELAY,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.delay_seconds = delay_seconds

    def analyze_text(self, job_desc_text: Optional[str]) -> JobAnalysisResult:
        """Run the LLM prompt for a single job description."""
        if not job_desc_text or not job_desc_text.strip():
            return JobAnalysisResult()

        text_to_analyze = (
            job_desc_text[:MAX_JOB_DESC_LENGTH]
            if len(job_desc_text) > MAX_JOB_DESC_LENGTH
            else job_desc_text
        )

        response_payload = self._call_openai(text_to_analyze)
        result = self._parse_response(response_payload)

        # Small delay to avoid rate limits
        time.sleep(self.delay_seconds)
        return result

    def _call_openai(self, text_to_analyze: str) -> str:
        system_prompt = (
            "You are an expert at analyzing job descriptions for AI and "
            "machine learning skills. Always respond with valid JSON only.\n\n"
            f"{job_analysis_instructions()}"
        )
        prompt = job_analysis_prompt(text_to_analyze)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            temperature=self.temperature,
            text={"format": {"type": "json_schema", "name": "job_analysis_result",
            
            "schema": {
                "type": "object",
                "properties": {
                    "has_ai_skill": {"type": "boolean"},
                    "ai_skills_mentioned": {"type": "array", "items": {"type": "string"}},
                },
                "strict": True,
                "required": ["has_ai_skill", "ai_skills_mentioned"],
                "additionalProperties": False,
            }
            }},
        )
        return self._extract_response_text(response)

    @staticmethod
    def _parse_response(response_text: str) -> JobAnalysisResult:
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as error:
            print(f"Warning: Failed to parse OpenAI JSON response: {error}")
            return JobAnalysisResult()

        skills = parsed.get("ai_skills_mentioned", [])
        if not isinstance(skills, list):
            skills = []

        return JobAnalysisResult(
            has_ai_skill=bool(parsed.get("has_ai_skill", False)),
            ai_skills_mentioned=skills,
        )

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Handle extraction for Responses API while staying backward compatible."""
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text.strip()

        output_blocks = getattr(response, "output", None) or []
        for block in output_blocks:
            contents = getattr(block, "content", None) or []
            for item in contents:
                text_value = getattr(item, "text", None)
                if text_value:
                    return text_value.strip()

        choices = getattr(response, "choices", None) or []
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content.strip()

        return ""
