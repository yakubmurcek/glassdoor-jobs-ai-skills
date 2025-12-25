#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI API integration for analyzing job descriptions."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence

from openai import OpenAI
from pydantic import ValidationError

from .config import (
    MAX_JOB_DESC_LENGTH,
    OPENAI_BATCH_SIZE,
    OPENAI_MAX_PARALLEL_REQUESTS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    RATE_LIMIT_DELAY,
)
from .models import BatchAnalysisResponse, JobAnalysisResult, JobAnalysisResultWithId
from .prompts import job_analysis_batch_prompt, job_analysis_instructions

logger = logging.getLogger(__name__)


class OpenAIJobAnalyzer:
    """Encapsulates the OpenAI client and response parsing logic."""

    def __init__(
        self,
        *,
        api_key: str = OPENAI_API_KEY,
        model: str = OPENAI_MODEL,
        temperature: float = OPENAI_TEMPERATURE,
        delay_seconds: float = RATE_LIMIT_DELAY,
        batch_size: int = OPENAI_BATCH_SIZE,
        max_concurrent_requests: int = OPENAI_MAX_PARALLEL_REQUESTS,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.delay_seconds = delay_seconds
        self.batch_size = max(1, batch_size)
        self.max_concurrent_requests = max(1, max_concurrent_requests)

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
        total_trackable = sum(1 for text in normalized_texts if text is not None)

        if progress_callback and total_trackable:
            progress_callback(0, total_trackable)
        processed_counter = 0
        progress_lock = threading.Lock()

        def report_progress(increment: int) -> None:
            nonlocal processed_counter
            if (
                not progress_callback
                or total_trackable == 0
                or increment <= 0
            ):
                return
            with progress_lock:
                processed_counter = min(
                    total_trackable, processed_counter + increment
                )
                progress_callback(processed_counter, total_trackable)

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

        self._process_batches(
            pending_batches, results, progress_reporter=report_progress
        )

        return results

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
        if not pending_batches:
            return

        max_workers = min(self.max_concurrent_requests, len(pending_batches))
        if max_workers <= 1:
            for batch, lookup in pending_batches:
                self._dispatch_batch(
                    batch, lookup, results, progress_reporter=progress_reporter
                )
            return

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
        system_prompt = job_analysis_instructions()
        prompt = job_analysis_batch_prompt(batch_items)
        
        try:
            # User instructions suggest using the Responses API via client.responses.parse
            # We supply the Pydantic model via text_format.
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                text_format=BatchAnalysisResponse,
                temperature=self.temperature,
            )
            
            # The parsed response object (ParsedResponse) typically holds the result 
            # in an attribute. Based on search results and typical generic patterns:
            # It might be in 'output', 'parsed', or 'output_parsed'.
            # We check the likely candidates.
            batch_response = None
            
            # 1. Check if 'output' is the model instance
            if hasattr(response, "output") and isinstance(response.output, BatchAnalysisResponse):
                batch_response = response.output
            # 2. Check 'output_parsed'
            elif hasattr(response, "output_parsed") and response.output_parsed:
                batch_response = response.output_parsed
            # 3. Check 'parsed' (like in beta.chat)
            elif hasattr(response, "parsed") and response.parsed:
                batch_response = response.parsed
            
            if not batch_response:
                # If we couldn't find it, log available keys for debugging
                logger.error(f"Could not find parsed model in response. attributes: {dir(response)}")
                return []

            # Log token usage
            usage = getattr(response, "usage", None)
            if usage:
                # Usage object might have different fields in Responses API
                input_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
                output_tokens = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))
                logger.info(
                    f"[OpenAI] Batch of {len(batch_items)}: "
                    f"input={input_tokens}, output={output_tokens}"
                )
            
            return batch_response.results

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    @staticmethod
    def _to_result(result_with_id: JobAnalysisResultWithId) -> JobAnalysisResult:
        """Convert JobAnalysisResultWithId to JobAnalysisResult."""
        return result_with_id.to_job_analysis_result()
