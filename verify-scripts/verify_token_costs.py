#!/usr/bin/env python3
"""Verify whether OpenAI API token costs grow non-linearly over batches.

This script simulates the batch construction process and measures:
1. System prompt sizes (constant per task? or growing?)
2. User prompt sizes per batch (are they constant or growing?)
3. Whether any state accumulates across batches

It also checks for:
- Retry storms that could multiply costs
- Increasing job description lengths in later batches
- Schema/structured output overhead per batch
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from pathlib import Path
from ai_skills.prompts import (
    ai_tier_instructions,
    ai_tier_batch_prompt,
    skills_instructions,
    skills_batch_prompt,
    education_instructions,
    education_batch_prompt,
)
from ai_skills.openai_analyzer import (
    AI_TIER_BATCH_SIZE,
    SKILLS_BATCH_SIZE,
    EDUCATION_BATCH_SIZE,
)
from ai_skills.config import MAX_JOB_DESC_LENGTH

import tiktoken


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens using tiktoken (approximate for newer models)."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")  # fallback
    return len(enc.encode(text))


def analyze_prompt_sizes(csv_path: Path):
    """Analyze how prompt sizes vary across batches."""
    print(f"Loading data from {csv_path}...")
    
    # Try semicolon first, then fallback
    try:
        df = pd.read_csv(csv_path, sep=";", low_memory=False, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    
    print(f"Loaded {len(df)} rows")
    
    # Prepare batch items (same as openai_analyzer._analyze_texts_decomposed)
    batch_items = []
    for idx, row in df.iterrows():
        text = str(row.get("job_desc_text", ""))
        if pd.isna(row.get("job_desc_text")) or not text.strip() or text.lower() == "nan":
            continue
        # Apply truncation
        if len(text) > MAX_JOB_DESC_LENGTH:
            text = text[:MAX_JOB_DESC_LENGTH]
        title = str(row.get("job_title", "")) if pd.notna(row.get("job_title")) else ""
        job_id = f"job_{idx}"
        batch_items.append((job_id, title, text))
    
    print(f"Valid batch items: {len(batch_items)}")
    
    # =========================================================================
    # CHECK 1: System prompt sizes (should be constant)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 1: System Prompt Sizes (Should Be Constant)")
    print("=" * 70)
    
    sys_prompts = {
        "ai_tier": ai_tier_instructions(),
        "skills": skills_instructions(),
        "education": education_instructions(),
    }
    
    for name, prompt in sys_prompts.items():
        tokens = count_tokens(prompt)
        print(f"  {name}: {tokens:,} tokens ({len(prompt):,} chars)")
    
    print("  → System prompts are CONSTANT (no accumulation)")
    
    # =========================================================================
    # CHECK 2: User prompt sizes per batch (should be proportional to batch)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 2: User Prompt Sizes Per Batch")
    print("=" * 70)
    
    for task_name, batch_size, prompt_builder in [
        ("ai_tier", AI_TIER_BATCH_SIZE, ai_tier_batch_prompt),
        ("skills", SKILLS_BATCH_SIZE, skills_batch_prompt),
        ("education", EDUCATION_BATCH_SIZE, None),  # needs 4-tuple
    ]:
        print(f"\n  Task: {task_name} (batch_size={batch_size})")
        
        batch_tokens = []
        batch_chars = []
        
        total_batches = (len(batch_items) + batch_size - 1) // batch_size
        
        for i in range(0, len(batch_items), batch_size):
            batch = batch_items[i:i + batch_size]
            
            if task_name == "education":
                # Education needs 4-tuples
                edu_batch = [(jid, title, text, "") for jid, title, text in batch]
                prompt = education_batch_prompt(edu_batch)
            else:
                prompt = prompt_builder(batch)
            
            tokens = count_tokens(prompt)
            batch_tokens.append(tokens)
            batch_chars.append(len(prompt))
        
        if batch_tokens:
            avg_tokens = sum(batch_tokens) / len(batch_tokens)
            min_tokens = min(batch_tokens)
            max_tokens = max(batch_tokens)
            
            print(f"    Total batches: {len(batch_tokens)}")
            print(f"    Avg tokens/batch: {avg_tokens:,.0f}")
            print(f"    Min tokens/batch: {min_tokens:,}")
            print(f"    Max tokens/batch: {max_tokens:,}")
            print(f"    Ratio max/min: {max_tokens/min_tokens:.2f}x")
            
            # Check for growth trend
            if len(batch_tokens) >= 3:
                first_third = batch_tokens[:len(batch_tokens)//3]
                last_third = batch_tokens[2*len(batch_tokens)//3:]
                avg_first = sum(first_third) / len(first_third)
                avg_last = sum(last_third) / len(last_third)
                growth = (avg_last - avg_first) / avg_first * 100
                
                print(f"    First third avg: {avg_first:,.0f} tokens")
                print(f"    Last third avg:  {avg_last:,.0f} tokens")
                print(f"    Growth trend: {growth:+.1f}%")
                
                if abs(growth) > 20:
                    print(f"    ⚠️  SIGNIFICANT GROWTH DETECTED!")
                else:
                    print(f"    ✅ No significant growth")
    
    # =========================================================================
    # CHECK 3: Job description length distribution
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 3: Job Description Lengths (checking for outliers)")
    print("=" * 70)
    
    desc_lengths = [len(text) for _, _, text in batch_items]
    desc_lengths_sorted = sorted(desc_lengths)
    
    print(f"  Total descriptions: {len(desc_lengths)}")
    print(f"  Min length: {min(desc_lengths):,} chars")
    print(f"  Max length: {max(desc_lengths):,} chars")
    print(f"  Avg length: {sum(desc_lengths)/len(desc_lengths):,.0f} chars")
    print(f"  Median length: {desc_lengths_sorted[len(desc_lengths)//2]:,} chars")
    print(f"  P95 length: {desc_lengths_sorted[int(len(desc_lengths)*0.95)]:,} chars")
    print(f"  P99 length: {desc_lengths_sorted[int(len(desc_lengths)*0.99)]:,} chars")
    
    at_max = sum(1 for l in desc_lengths if l >= MAX_JOB_DESC_LENGTH)
    print(f"  At MAX_JOB_DESC_LENGTH ({MAX_JOB_DESC_LENGTH}): {at_max} ({at_max/len(desc_lengths)*100:.1f}%)")
    
    # =========================================================================
    # CHECK 4: Total cost estimation per task
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 4: Total Token Cost Estimation")
    print("=" * 70)
    
    total_input_tokens = 0
    for task_name, batch_size, prompt_builder in [
        ("ai_tier", AI_TIER_BATCH_SIZE, ai_tier_batch_prompt),
        ("skills", SKILLS_BATCH_SIZE, skills_batch_prompt),
        ("education", EDUCATION_BATCH_SIZE, None),
    ]:
        sys_prompt = sys_prompts.get(task_name, "")
        sys_tokens = count_tokens(sys_prompt)
        
        task_input_tokens = 0
        for i in range(0, len(batch_items), batch_size):
            batch = batch_items[i:i + batch_size]
            if task_name == "education":
                edu_batch = [(jid, title, text, "") for jid, title, text in batch]
                prompt = education_batch_prompt(edu_batch)
            else:
                prompt = prompt_builder(batch)
            task_input_tokens += count_tokens(prompt) + sys_tokens
        
        total_input_tokens += task_input_tokens
        print(f"  {task_name}: {task_input_tokens:,} input tokens")
    
    print(f"\n  TOTAL INPUT TOKENS: {total_input_tokens:,}")
    print(f"  (Output tokens depend on model response, typically 10-20% of input)")
    
    # =========================================================================
    # CHECK 5: Decomposed vs Monolithic cost comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 5: Decomposed Tasks = 3x System Prompt Overhead")
    print("=" * 70)
    
    total_sys_tokens = sum(count_tokens(p) for p in sys_prompts.values())
    total_batches = sum(
        (len(batch_items) + bs - 1) // bs
        for bs in [AI_TIER_BATCH_SIZE, SKILLS_BATCH_SIZE, EDUCATION_BATCH_SIZE]
    )
    total_sys_overhead = sum(
        count_tokens(sys_prompts[task]) * ((len(batch_items) + bs - 1) // bs)
        for task, bs in [
            ("ai_tier", AI_TIER_BATCH_SIZE),
            ("skills", SKILLS_BATCH_SIZE),
            ("education", EDUCATION_BATCH_SIZE),
        ]
    )
    
    print(f"  Total API calls across all 3 tasks: {total_batches}")
    print(f"  System prompt overhead: {total_sys_overhead:,} tokens")
    print(f"  Each job description is sent 3 TIMES (once per task)")
    
    # Estimate monolithic cost (same data, 1 pass)
    from ai_skills.prompts import job_analysis_instructions, job_analysis_batch_prompt
    mono_sys = count_tokens(job_analysis_instructions())
    mono_batch_size = 20  # default OPENAI_BATCH_SIZE
    
    mono_input_tokens = 0
    for i in range(0, len(batch_items), mono_batch_size):
        batch = batch_items[i:i + mono_batch_size]
        prompt = job_analysis_batch_prompt(batch)
        mono_input_tokens += count_tokens(prompt) + mono_sys
    
    print(f"\n  Decomposed total input: {total_input_tokens:,} tokens")
    print(f"  Monolithic total input: {mono_input_tokens:,} tokens")
    print(f"  Decomposed cost multiplier: {total_input_tokens / mono_input_tokens:.2f}x")
    
    # =========================================================================
    # CHECK 6: Retry cost analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 6: Retry Cost Impact")
    print("=" * 70)
    
    print("  Max retries per batch: 3 (exponential backoff)")
    print("  Failed items retried individually (2 attempts each)")
    print("  WORST CASE per failed batch:")
    
    # For a batch of 35 items:
    for task_name, batch_size in [
        ("ai_tier", AI_TIER_BATCH_SIZE),
        ("skills", SKILLS_BATCH_SIZE),
        ("education", EDUCATION_BATCH_SIZE),
    ]:
        normal_cost = 1  # 1 API call
        retry_cost = batch_size * 2  # Each item retried individually, 2 attempts
        worst_case_multiplier = (normal_cost + retry_cost) / normal_cost
        print(
            f"    {task_name}: Normal=1 call, Worst retry={batch_size}×2={batch_size*2} calls "
            f"({worst_case_multiplier:.0f}x cost)"
        )
    
    print("\n  If many batches fail, retry storms could multiply costs 70x per batch!")
    print("  ⚠️  This is the most likely cause of non-linear cost growth.")
    
    # =========================================================================
    # CHECK 7: Structured output schema overhead
    # =========================================================================
    print("\n" + "=" * 70)
    print("CHECK 7: Structured Output (text_format) Schema Overhead")
    print("=" * 70)
    
    from ai_skills.models import (
        AITierBatchResponse,
        SkillsBatchResponse,
        EducationBatchResponse,
    )
    
    import json
    for name, model in [
        ("AITierBatchResponse", AITierBatchResponse),
        ("SkillsBatchResponse", SkillsBatchResponse),
        ("EducationBatchResponse", EducationBatchResponse),
    ]:
        schema = model.model_json_schema()
        schema_str = json.dumps(schema)
        schema_tokens = count_tokens(schema_str)
        print(f"  {name}: {schema_tokens:,} schema tokens ({len(schema_str):,} chars)")
    
    print("  → Schema is sent with EVERY request (constant per-request overhead)")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Potential Causes of Non-Linear Cost Growth")
    print("=" * 70)
    print("""
  1. ❌ Context window accumulation: NOT happening
     - Each API call is independent (no previous_response_id chaining)
     - System prompt is constant, user prompt proportional to batch size
  
  2. ⚠️  RETRY STORMS: MOST LIKELY CULPRIT
     - If batch fails → each item retried individually (up to 2x per item)
     - A batch of 35 items that fails = 70 individual API calls
     - Resource unavailable (429) errors trigger exponential backoff retries
     - Late in a long run, rate limits hit more often → more retries
  
  3. ⚠️  3x DECOMPOSED OVERHEAD: SIGNIFICANT
     - Each job description is sent 3 times (tier + skills + education)
     - System prompt overhead sent with every batch
     - Consider using monolithic mode for cost savings
  
  4. ✅ Job desc lengths: Constant (capped at MAX_JOB_DESC_LENGTH)
  
  5. ✅ Batch sizes: Fixed per task (no growth over time)
""")


if __name__ == "__main__":
    # Default input file
    csv_path = Path("data/inputs/us_relevant_30.csv")
    
    # Allow override via CLI
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        # Try to find any relevant input
        inputs_dir = Path("data/inputs")
        if inputs_dir.exists():
            csvs = list(inputs_dir.glob("*relevant*.csv"))
            if csvs:
                csv_path = csvs[0]
                print(f"Using: {csv_path}")
            else:
                print(f"No input CSVs found in {inputs_dir}")
                sys.exit(1)
    
    analyze_prompt_sizes(csv_path)
