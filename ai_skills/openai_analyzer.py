#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM API integration for analyzing job descriptions.

Supports both OpenAI API and Ollama (via OpenAI-compatible endpoint).
"""

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence

from openai import OpenAI
from pydantic import ValidationError

from .config import (
    MAX_JOB_DESC_LENGTH,
    MODEL_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_BATCH_SIZE,
    OPENAI_MAX_PARALLEL_REQUESTS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    RATE_LIMIT_DELAY,
)
from .models import BatchAnalysisResponse, JobAnalysisResult, JobAnalysisResultWithId
from .prompts import job_analysis_batch_prompt, job_analysis_instructions

logger = logging.getLogger(__name__)


class OpenAIJobAnalyzer:
    """Encapsulates the LLM client and response parsing logic.
    
    Supports both OpenAI API and Ollama (via OpenAI-compatible endpoint).
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = OPENAI_API_KEY,
        base_url: Optional[str] = OPENAI_BASE_URL,
        provider: str = MODEL_PROVIDER,
        model: str = OPENAI_MODEL,
        temperature: float = OPENAI_TEMPERATURE,
        delay_seconds: float = RATE_LIMIT_DELAY,
        batch_size: int = OPENAI_BATCH_SIZE,
        max_concurrent_requests: int = OPENAI_MAX_PARALLEL_REQUESTS,
    ) -> None:
        # Build client with optional base_url for Ollama support
        client_kwargs = {"api_key": api_key or "ollama"}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.delay_seconds = delay_seconds
        self.batch_size = max(1, batch_size)
        self.max_concurrent_requests = max(1, max_concurrent_requests)
        
        if self.provider == "ollama":
            logger.info(f"Using Ollama provider with model '{model}' at {base_url}")
        else:
            logger.info(f"Using OpenAI provider with model '{model}'")

    def analyze_text(
        self,
        job_desc_text: Optional[str],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> JobAnalysisResult:
        """Run the LLM prompt for a single job description."""
        return self.analyze_texts(
            [job_desc_text], progress_callback=progress_callback
        )[0]

    def analyze_texts(
        self,
        job_desc_texts: Sequence[Optional[str]],
        *,
        job_titles: Sequence[str] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[JobAnalysisResult]:
        """Analyze multiple job descriptions using batched OpenAI requests."""
        if not job_desc_texts:
            return []

        normalized_texts = [self._prepare_text(text) for text in job_desc_texts]
        # Default to empty titles if not provided
        titles = list(job_titles) if job_titles else [""] * len(normalized_texts)
        results: list[JobAnalysisResult] = [
            JobAnalysisResult() for _ in normalized_texts
        ]
        
        reporter = self._setup_progress_reporter(
            normalized_texts, progress_callback
        )

        pending_batches = self._create_batches(normalized_texts, titles)
        
        self._process_batches(
            pending_batches, results, progress_reporter=reporter
        )

        return results

    def _setup_progress_reporter(
        self,
        normalized_texts: list[Optional[str]],
        progress_callback: Callable[[int, int], None] | None,
    ) -> Callable[[int], None] | None:
        """Create a thread-safe progress reporter."""
        total_trackable = sum(1 for text in normalized_texts if text is not None)
        if not progress_callback or not total_trackable:
            return None

        progress_callback(0, total_trackable)
        processed_counter = 0
        progress_lock = threading.Lock()

        def report_progress(increment: int) -> None:
            nonlocal processed_counter
            if increment <= 0:
                return
            with progress_lock:
                processed_counter = min(total_trackable, processed_counter + increment)
                progress_callback(processed_counter, total_trackable)  # type: ignore

        return report_progress

    def _create_batches(
        self, 
        normalized_texts: list[Optional[str]], 
        titles: list[str]
    ) -> list[tuple[list[tuple[str, str, str]], dict[str, int]]]:
        """Group texts into batches for API processing."""
        pending_batches: list[tuple[list[tuple[str, str, str]], dict[str, int]]] = []
        batch: list[tuple[str, str, str]] = []
        index_lookup: dict[str, int] = {}

        for idx, text in enumerate(normalized_texts):
            if not text:
                continue
            job_id = f"job_{idx}"
            title = titles[idx] if idx < len(titles) else ""
            batch.append((job_id, title, text))
            index_lookup[job_id] = idx
            
            if len(batch) >= self.batch_size:
                pending_batches.append((batch, index_lookup))
                batch = []
                index_lookup = {}

        if batch:
            pending_batches.append((batch, index_lookup))
            
        return pending_batches

    def _prepare_text(self, job_desc_text: Optional[str]) -> Optional[str]:
        if job_desc_text is None:
            return None

        text = str(job_desc_text).strip()
        # More robust check for "nan" or empty strings
        if not text or text.lower() == "nan":
            return None

        if len(text) > MAX_JOB_DESC_LENGTH:
            return text[:MAX_JOB_DESC_LENGTH]
        return text

    def _process_batches(
        self,
        pending_batches: list[tuple[list[tuple[str, str, str]], dict[str, int]]],
        results: list[JobAnalysisResult],
        *,
        progress_reporter: Callable[[int], None] | None = None,
    ) -> None:
        """Route batch processing to sequential or parallel execution."""
        if not pending_batches:
            return

        max_workers = min(self.max_concurrent_requests, len(pending_batches))
        if max_workers <= 1:
            self._process_batches_sequentially(
                pending_batches, results, progress_reporter
            )
        else:
            self._process_batches_in_parallel(
                pending_batches, results, max_workers, progress_reporter
            )

    def _process_batches_sequentially(
        self,
        pending_batches: list[tuple[list[tuple[str, str, str]], dict[str, int]]],
        results: list[JobAnalysisResult],
        progress_reporter: Callable[[int], None] | None = None,
    ) -> None:
        for batch, lookup in pending_batches:
            self._dispatch_batch(
                batch, lookup, results, progress_reporter=progress_reporter
            )

    def _process_batches_in_parallel(
        self,
        pending_batches: list[tuple[list[tuple[str, str, str]], dict[str, int]]],
        results: list[JobAnalysisResult],
        max_workers: int,
        progress_reporter: Callable[[int], None] | None = None,
    ) -> None:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._dispatch_batch,
                    batch,
                    lookup,
                    results,
                    progress_reporter=progress_reporter,
                )
                for batch, lookup in pending_batches
            ]
            for future in futures:
                future.result()

    def _dispatch_batch(
        self,
        batch: list[tuple[str, str, str]],
        index_lookup: dict[str, int],
        results: list[JobAnalysisResult],
        *,
        progress_reporter: Callable[[int], None] | None = None,
    ) -> None:
        try:
            parsed_entries = self._call_openai_batch(batch)
        except Exception as e:
            logger.error(f"Failed to process batch of {len(batch)} items: {e}")
            parsed_entries = []

        # Track which job IDs we received results for
        received_ids = set()
        for entry in parsed_entries:
            job_id = entry.id
            received_ids.add(job_id)
            if job_id not in index_lookup:
                continue
            results[index_lookup[job_id]] = self._to_result(entry)
        
        # Validate that we got all expected results
        expected_ids = set(index_lookup.keys())
        missing_ids = expected_ids - received_ids
        if missing_ids:
            logger.warning(
                f"OpenAI returned {len(parsed_entries)}/{len(batch)} results. "
                f"Missing IDs: {sorted(missing_ids)}"
            )
        
        if progress_reporter:
            progress_reporter(len(index_lookup))

        time.sleep(self.delay_seconds)

    def _call_openai_batch(self, batch_items: list[tuple[str, str, str]]) -> list[JobAnalysisResultWithId]:
        """Call the LLM API with batch items.
        
        Uses structured outputs for OpenAI, falls back to JSON mode for Ollama.
        """
        system_prompt = job_analysis_instructions()
        prompt = job_analysis_batch_prompt(batch_items)
        
        try:
            if self.provider == "ollama":
                # Ollama: Use chat.completions with JSON mode
                return self._call_ollama_batch(system_prompt, prompt, len(batch_items))
            else:
                # OpenAI: Use structured outputs
                return self._call_openai_structured(system_prompt, prompt, len(batch_items))

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    def _call_openai_structured(
        self, system_prompt: str, prompt: str, batch_count: int
    ) -> list[JobAnalysisResultWithId]:
        """Call OpenAI API with structured output parsing."""
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            text_format=BatchAnalysisResponse,
            temperature=self.temperature,
        )
        
        batch_response = self._extract_parsed_response(response)
        if not batch_response:
            return []

        self._log_token_usage(response, batch_count)
        return batch_response.results

    def _call_ollama_batch(
        self, system_prompt: str, prompt: str, batch_count: int
    ) -> list[JobAnalysisResultWithId]:
        """Call Ollama API with JSON mode and manual parsing.
        
        Ollama's OpenAI-compatible endpoint may not support structured outputs,
        so we use JSON mode and parse the response manually.
        """
        # Add JSON schema hint to system prompt for Ollama
        schema_hint = '''

You MUST respond with valid JSON matching this exact schema:
{
  "results": [
    {
      "id": "job_0",
      "ai_tier": "none|ai_integration|applied_ai|core_ai",
      "ai_skills_mentioned": ["skill1", "skill2"],
      "confidence": 0.0-1.0,
      "rationale": "brief explanation",
      "hardskills_raw": ["skill1", "skill2"],
      "softskills_raw": ["skill1", "skill2"],
      "education_required": 0 or 1
    }
  ]
}
'''
        enhanced_system_prompt = system_prompt + schema_hint
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )
        
        # Extract and parse JSON from response
        content = response.choices[0].message.content
        if not content:
            logger.error("Ollama returned empty response")
            return []
        
        try:
            batch_response = self._parse_json_response(content)
            self._log_token_usage(response, batch_count)
            return batch_response.results
        except Exception as e:
            logger.error(f"Failed to parse Ollama JSON response: {e}")
            logger.debug(f"Raw response content: {content[:500]}...")
            return []

    def _parse_json_response(self, content: str) -> BatchAnalysisResponse:
        """Parse JSON response from Ollama into BatchAnalysisResponse."""
        # Try to extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Assume the entire content is JSON
            json_str = content.strip()
        
        data = json.loads(json_str)
        return BatchAnalysisResponse.model_validate(data)

    def _extract_parsed_response(self, response: Any) -> BatchAnalysisResponse | None:
        """Extract the parsed model from the API response object."""
        # 1. Check if 'output' is the model instance
        if hasattr(response, "output") and isinstance(response.output, BatchAnalysisResponse):
            return response.output
        # 2. Check 'output_parsed'
        elif hasattr(response, "output_parsed") and response.output_parsed:
            return response.output_parsed
        # 3. Check 'parsed' (like in beta.chat)
        elif hasattr(response, "parsed") and response.parsed:
            return response.parsed
        
        logger.error(f"Could not find parsed model in response. attributes: {dir(response)}")
        return None

    def _log_token_usage(self, response: Any, batch_count: int) -> None:
        """Log input and output token counts."""
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
            output_tokens = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))
            logger.info(
                f"[OpenAI] Batch of {batch_count}: "
                f"input={input_tokens}, output={output_tokens}"
            )

    @staticmethod
    def _to_result(result_with_id: JobAnalysisResultWithId) -> JobAnalysisResult:
        """Convert JobAnalysisResultWithId to JobAnalysisResult."""
        return result_with_id.to_job_analysis_result()
