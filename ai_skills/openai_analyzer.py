#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM API integration for analyzing job descriptions.

Supports both OpenAI API and Ollama (via OpenAI-compatible endpoint).
Now features decomposed task-based batching for improved accuracy.
"""

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .config import (
    ENABLED_LLM_TASKS,
    LLM_TASK_AI_TIER,
    LLM_TASK_EDUCATION,
    LLM_TASK_SKILLS,
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
from .models import (
    AITierBatchResponse,
    AITierResultWithId,
    BatchAnalysisResponse,
    EducationBatchResponse,
    EducationResultWithId,
    JobAnalysisResult,
    JobAnalysisResultWithId,
    SkillsBatchResponse,
    SkillsResultWithId,
)
from .prompts import (
    ai_tier_batch_prompt,
    ai_tier_instructions,
    education_batch_prompt,
    education_instructions,
    job_analysis_batch_prompt,
    job_analysis_instructions,
    skills_batch_prompt,
    skills_instructions,
)

logger = logging.getLogger(__name__)

# Type variable for generic batch response handling
T = TypeVar("T", bound=BaseModel)

# Task-specific batch sizes (optimized for 4o-mini / Gemma 12B)
AI_TIER_BATCH_SIZE = 20      # Classification needs reasoning
SKILLS_BATCH_SIZE = 20       # Extraction is moderately complex
EDUCATION_BATCH_SIZE = 25    # Simple binary, larger batches OK


class OpenAIJobAnalyzer:
    """Encapsulates the LLM client and response parsing logic.
    
    Supports both OpenAI API and Ollama (via OpenAI-compatible endpoint).
    Uses decomposed task-based batching for improved accuracy.
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
        use_decomposed: bool = True,  # New: enable decomposed mode by default
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
        self.use_decomposed = use_decomposed
        
        if self.provider == "ollama":
            logger.info(f"Using Ollama provider with model '{model}' at {base_url}")
        else:
            logger.info(f"Using OpenAI provider with model '{model}'")
        
        if self.use_decomposed:
            logger.info("Decomposed task-based batching ENABLED")

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
        educations: Sequence[str] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[JobAnalysisResult]:
        """Analyze multiple job descriptions.
        
        Uses decomposed task-based batching if enabled (default), 
        otherwise falls back to legacy monolithic batching.
        
        Args:
            job_desc_texts: Job description texts to analyze
            job_titles: Optional job titles for context
            educations: Optional education column values for education requirement task
            progress_callback: Optional callback for progress reporting
        """
        if not job_desc_texts:
            return []

        if self.use_decomposed:
            return self._analyze_texts_decomposed(
                job_desc_texts, job_titles, educations, progress_callback
            )
        else:
            return self._analyze_texts_monolithic(
                job_desc_texts, job_titles, progress_callback
            )

    # =========================================================================
    # Decomposed Task-Based Analysis (NEW)
    # =========================================================================

    def _analyze_texts_decomposed(
        self,
        job_desc_texts: Sequence[Optional[str]],
        job_titles: Sequence[str] | None,
        educations: Sequence[str] | None,
        progress_callback: Callable[[int, int], None] | None,
    ) -> list[JobAnalysisResult]:
        """Analyze texts using decomposed single-task prompts."""
        normalized_texts = [self._prepare_text(text) for text in job_desc_texts]
        titles = list(job_titles) if job_titles else [""] * len(normalized_texts)
        edu_list = list(educations) if educations else [""] * len(normalized_texts)
        
        # Prepare batch items with IDs (3-tuples for tier/skills tasks)
        batch_items: list[tuple[str, str, str]] = []
        index_lookup: dict[str, int] = {}
        for idx, text in enumerate(normalized_texts):
            if not text:
                continue
            job_id = f"job_{idx}"
            title = titles[idx] if idx < len(titles) else ""
            batch_items.append((job_id, title, text))
            index_lookup[job_id] = idx
        
        # Prepare education-specific batch items (4-tuples with education column)
        edu_batch_items: list[tuple[str, str, str, str]] = []
        for idx, text in enumerate(normalized_texts):
            if not text:
                continue
            job_id = f"job_{idx}"
            title = titles[idx] if idx < len(titles) else ""
            edu = edu_list[idx] if idx < len(edu_list) else ""
            edu_batch_items.append((job_id, title, text, edu))
        
        total_items = len(batch_items)
        # Count only enabled tasks for progress tracking
        enabled_task_count = sum(
            1 for t in [LLM_TASK_AI_TIER, LLM_TASK_SKILLS, LLM_TASK_EDUCATION]
            if t in ENABLED_LLM_TASKS
        )
        total_tasks = max(1, enabled_task_count)
        if progress_callback:
            progress_callback(0, total_items * total_tasks)
        
        processed = 0
        task_num = 0
        
        # Task 1: AI Tier Classification
        tier_results: dict[str, Any] = {}
        if LLM_TASK_AI_TIER in ENABLED_LLM_TASKS:
            task_num += 1
            logger.info(f"Task {task_num}/{enabled_task_count}: AI Tier Classification ({total_items} jobs)")
            tier_results = self._run_task(
                batch_items,
                task_name="ai_tier",
                batch_size=AI_TIER_BATCH_SIZE,
                system_prompt=ai_tier_instructions(),
                prompt_builder=ai_tier_batch_prompt,
                response_model=AITierBatchResponse,
            )
            processed += total_items
            if progress_callback:
                progress_callback(processed, total_items * total_tasks)
        else:
            logger.info("Skipping AI Tier task (disabled in ENABLED_LLM_TASKS)")
        
        # Task 2: Skills Extraction
        skills_results: dict[str, Any] = {}
        if LLM_TASK_SKILLS in ENABLED_LLM_TASKS:
            task_num += 1
            logger.info(f"Task {task_num}/{enabled_task_count}: Skills Extraction ({total_items} jobs)")
            skills_results = self._run_task(
                batch_items,
                task_name="skills",
                batch_size=SKILLS_BATCH_SIZE,
                system_prompt=skills_instructions(),
                prompt_builder=skills_batch_prompt,
                response_model=SkillsBatchResponse,
            )
            processed += total_items
            if progress_callback:
                progress_callback(processed, total_items * total_tasks)
        else:
            logger.info("Skipping Skills task (disabled in ENABLED_LLM_TASKS)")
        
        # Task 3: Education Required (uses 4-tuple items with education column)
        edu_results: dict[str, EducationResultWithId] = {}
        if LLM_TASK_EDUCATION in ENABLED_LLM_TASKS:
            task_num += 1
            logger.info(f"Task {task_num}/{enabled_task_count}: Education Requirement ({total_items} jobs)")
            edu_results = self._run_edu_task(
                edu_batch_items,
                task_name="education",
                batch_size=EDUCATION_BATCH_SIZE,
                system_prompt=education_instructions(),
            )
            processed += total_items
            if progress_callback:
                progress_callback(processed, total_items * total_tasks)
        else:
            logger.info("Skipping Education task (disabled in ENABLED_LLM_TASKS)")
        
        # Combine results
        return self._combine_decomposed_results(
            normalized_texts, index_lookup, tier_results, skills_results, edu_results
        )

    def _run_task(
        self,
        all_items: list[tuple[str, str, str]],
        *,
        task_name: str,
        batch_size: int,
        system_prompt: str,
        prompt_builder: Callable[[list[tuple[str, str, str]]], str],
        response_model: Type[T],
    ) -> dict[str, Any]:
        """Run a single-task batch analysis with error isolation.
        
        Returns a dict mapping job_id -> task-specific result.
        Failed items are retried individually before giving up.
        """
        results: dict[str, Any] = {}
        failed_ids: list[str] = []
        
        # Process in batches
        for i in range(0, len(all_items), batch_size):
            batch = all_items[i:i + batch_size]
            batch_ids = [item[0] for item in batch]
            
            try:
                prompt = prompt_builder(batch)
                response = self._call_llm_task(
                    system_prompt, prompt, response_model, len(batch)
                )
                
                # Match results by ID
                for result in response.results:
                    results[result.id] = result
                
                # Check for missing IDs
                received_ids = {r.id for r in response.results}
                for job_id in batch_ids:
                    if job_id not in received_ids:
                        failed_ids.append(job_id)
                        logger.warning(f"[{task_name}] Missing result for {job_id}")
                
            except Exception as e:
                logger.error(f"[{task_name}] Batch failed: {e}")
                failed_ids.extend(batch_ids)
            
            time.sleep(self.delay_seconds)
        
        # Retry failed items individually (max 2 retries each)
        if failed_ids:
            logger.info(f"[{task_name}] Retrying {len(failed_ids)} failed items...")
            items_by_id = {item[0]: item for item in all_items}
            
            for job_id in failed_ids:
                item = items_by_id.get(job_id)
                if not item:
                    continue
                
                for attempt in range(2):
                    try:
                        prompt = prompt_builder([item])
                        response = self._call_llm_task(
                            system_prompt, prompt, response_model, 1
                        )
                        if response.results:
                            results[job_id] = response.results[0]
                            logger.info(f"[{task_name}] Retry succeeded for {job_id}")
                            break
                    except Exception as e:
                        logger.warning(f"[{task_name}] Retry {attempt+1} failed for {job_id}: {e}")
                        time.sleep(self.delay_seconds)
        
        return results

    def _run_edu_task(
        self,
        all_items: list[tuple[str, str, str, str]],
        *,
        task_name: str,
        batch_size: int,
        system_prompt: str,
    ) -> dict[str, EducationResultWithId]:
        """Run education task with 4-tuple items (includes education column).
        
        Similar to _run_task but uses education_batch_prompt which expects 4-tuples.
        """
        results: dict[str, EducationResultWithId] = {}
        failed_ids: list[str] = []
        
        # Process in batches
        for i in range(0, len(all_items), batch_size):
            batch = all_items[i:i + batch_size]
            batch_ids = [item[0] for item in batch]
            
            try:
                prompt = education_batch_prompt(batch)
                response = self._call_llm_task(
                    system_prompt, prompt, EducationBatchResponse, len(batch)
                )
                
                # Match results by ID
                for result in response.results:
                    results[result.id] = result
                
                # Check for missing IDs
                received_ids = {r.id for r in response.results}
                for job_id in batch_ids:
                    if job_id not in received_ids:
                        failed_ids.append(job_id)
                        logger.warning(f"[{task_name}] Missing result for {job_id}")
                
            except Exception as e:
                logger.error(f"[{task_name}] Batch failed: {e}")
                failed_ids.extend(batch_ids)
            
            time.sleep(self.delay_seconds)
        
        # Retry failed items individually (max 2 retries each)
        if failed_ids:
            logger.info(f"[{task_name}] Retrying {len(failed_ids)} failed items...")
            items_by_id = {item[0]: item for item in all_items}
            
            for job_id in failed_ids:
                item = items_by_id.get(job_id)
                if not item:
                    continue
                
                for attempt in range(2):
                    try:
                        prompt = education_batch_prompt([item])
                        response = self._call_llm_task(
                            system_prompt, prompt, EducationBatchResponse, 1
                        )
                        if response.results:
                            results[job_id] = response.results[0]
                            logger.info(f"[{task_name}] Retry succeeded for {job_id}")
                            break
                    except Exception as e:
                        logger.warning(f"[{task_name}] Retry {attempt+1} failed for {job_id}: {e}")
                        time.sleep(self.delay_seconds)
        
        return results

    def _call_llm_task(
        self,
        system_prompt: str,
        prompt: str,
        response_model: Type[T],
        batch_count: int,
    ) -> T:
        """Call LLM for a specific task with structured output."""
        if self.provider == "ollama":
            return self._call_ollama_task(system_prompt, prompt, response_model, batch_count)
        else:
            return self._call_openai_task(system_prompt, prompt, response_model, batch_count)

    def _call_openai_task(
        self,
        system_prompt: str,
        prompt: str,
        response_model: Type[T],
        batch_count: int,
    ) -> T:
        """Call OpenAI API with structured output for a specific task."""
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            text_format=response_model,
            temperature=self.temperature,
        )
        
        parsed = self._extract_parsed_response_generic(response, response_model)
        if not parsed:
            raise ValueError("Failed to parse OpenAI response")
        
        self._log_token_usage(response, batch_count)
        return parsed

    def _call_ollama_task(
        self,
        system_prompt: str,
        prompt: str,
        response_model: Type[T],
        batch_count: int,
    ) -> T:
        """Call Ollama API with JSON mode for a specific task."""
        # Generate schema hint from the response model
        schema_hint = self._generate_schema_hint(response_model)
        enhanced_system_prompt = system_prompt + "\n\n" + schema_hint
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Ollama returned empty response")
        
        parsed = self._parse_json_response_generic(content, response_model)
        self._log_token_usage(response, batch_count)
        return parsed

    def _generate_schema_hint(self, response_model: Type[T]) -> str:
        """Generate a JSON schema hint for Ollama from a Pydantic model."""
        schema = response_model.model_json_schema()
        return f"You MUST respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

    def _parse_json_response_generic(self, content: str, response_model: Type[T]) -> T:
        """Parse JSON response into a specific Pydantic model."""
        # Handle markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = content.strip()
        
        data = json.loads(json_str)
        return response_model.model_validate(data)

    def _extract_parsed_response_generic(
        self, response: Any, response_model: Type[T]
    ) -> T | None:
        """Extract the parsed model from the API response object."""
        if hasattr(response, "output") and isinstance(response.output, response_model):
            return response.output
        elif hasattr(response, "output_parsed") and response.output_parsed:
            return response.output_parsed
        elif hasattr(response, "parsed") and response.parsed:
            return response.parsed
        
        logger.error(f"Could not find parsed model in response. attributes: {dir(response)}")
        return None

    def _combine_decomposed_results(
        self,
        normalized_texts: list[Optional[str]],
        index_lookup: dict[str, int],
        tier_results: dict[str, AITierResultWithId],
        skills_results: dict[str, SkillsResultWithId],
        edu_results: dict[str, EducationResultWithId],
    ) -> list[JobAnalysisResult]:
        """Combine results from all tasks into JobAnalysisResult objects."""
        results: list[JobAnalysisResult] = [JobAnalysisResult() for _ in normalized_texts]
        
        for job_id, idx in index_lookup.items():
            tier = tier_results.get(job_id)
            skills = skills_results.get(job_id)
            edu = edu_results.get(job_id)
            
            results[idx] = JobAnalysisResult(
                ai_tier=tier.ai_tier if tier else results[idx].ai_tier,
                confidence=tier.confidence if tier else 0.0,
                rationale=tier.rationale if tier else "",
                ai_skills_mentioned=skills.ai_skills_mentioned if skills else [],
                hardskills_raw=skills.hardskills_raw if skills else [],
                softskills_raw=skills.softskills_raw if skills else [],
                # Keep None if education task was skipped (shows as blank in CSV)
                education_required=edu.education_required if edu else None,
            )
        
        return results

    # =========================================================================
    # Legacy Monolithic Analysis (for backwards compatibility)
    # =========================================================================

    def _analyze_texts_monolithic(
        self,
        job_desc_texts: Sequence[Optional[str]],
        job_titles: Sequence[str] | None,
        progress_callback: Callable[[int, int], None] | None,
    ) -> list[JobAnalysisResult]:
        """Legacy method: analyze using monolithic prompts."""
        normalized_texts = [self._prepare_text(text) for text in job_desc_texts]
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

        received_ids = set()
        for entry in parsed_entries:
            job_id = entry.id
            received_ids.add(job_id)
            if job_id not in index_lookup:
                continue
            results[index_lookup[job_id]] = self._to_result(entry)
        
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
        """Call the LLM API with batch items (legacy monolithic)."""
        system_prompt = job_analysis_instructions()
        prompt = job_analysis_batch_prompt(batch_items)
        
        try:
            if self.provider == "ollama":
                return self._call_ollama_batch(system_prompt, prompt, len(batch_items))
            else:
                return self._call_openai_structured(system_prompt, prompt, len(batch_items))

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    def _call_openai_structured(
        self, system_prompt: str, prompt: str, batch_count: int
    ) -> list[JobAnalysisResultWithId]:
        """Call OpenAI API with structured output parsing (legacy)."""
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
        """Call Ollama API with JSON mode and manual parsing (legacy)."""
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
        """Parse JSON response from Ollama into BatchAnalysisResponse (legacy)."""
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = content.strip()
        
        data = json.loads(json_str)
        return BatchAnalysisResponse.model_validate(data)

    def _extract_parsed_response(self, response: Any) -> BatchAnalysisResponse | None:
        """Extract the parsed model from the API response object (legacy)."""
        if hasattr(response, "output") and isinstance(response.output, BatchAnalysisResponse):
            return response.output
        elif hasattr(response, "output_parsed") and response.output_parsed:
            return response.output_parsed
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
                f"[LLM] Batch of {batch_count}: "
                f"input={input_tokens}, output={output_tokens}"
            )

    @staticmethod
    def _to_result(result_with_id: JobAnalysisResultWithId) -> JobAnalysisResult:
        """Convert JobAnalysisResultWithId to JobAnalysisResult."""
        return result_with_id.to_job_analysis_result()
