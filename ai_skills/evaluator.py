#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic evaluation framework for comparing pipeline outputs.

This module provides tools for comparing baseline and candidate pipeline outputs,
identifying classification changes, and generating comprehensive reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================================
# CONFIGURATION: Thresholds for warnings
# ============================================================================

@dataclass
class EvaluationThresholds:
    """Configurable thresholds for evaluation warnings."""
    min_match_rate: float = 0.85  # Warn if match rate below this
    max_confidence_drop: float = 0.05  # Warn if mean confidence drops by this much
    min_agreement_rate: float = 0.80  # Warn if agreement rate below this
    
    @classmethod
    def default(cls) -> "EvaluationThresholds":
        return cls()


# ============================================================================
# DATA CLASSES: Structured evaluation results
# ============================================================================

@dataclass
class ClassificationChange:
    """A single classification change between baseline and candidate."""
    job_id: int
    job_title: str
    old_tier: str
    new_tier: str
    old_confidence: float
    new_confidence: float
    old_rationale: str
    new_rationale: str
    job_description_excerpt: str  # First ~500 chars of job description
    ai_skills_old: str
    ai_skills_new: str


@dataclass
class TierDistribution:
    """Distribution of AI tiers."""
    none: int = 0
    ai_integration: int = 0
    applied_ai: int = 0
    core_ai: int = 0
    
    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class ConfidenceStats:
    """Statistics for confidence scores."""
    mean: float
    std: float
    min: float
    max: float


@dataclass
class Warning:
    """An evaluation warning triggered by threshold breach."""
    level: str  # "warning" or "critical"
    metric: str
    message: str
    threshold: float
    actual: float


@dataclass
class EvaluationReport:
    """Complete evaluation report comparing baseline and candidate outputs."""
    # Metadata
    baseline_path: str
    candidate_path: str
    evaluation_timestamp: str
    total_jobs: int
    
    # Core metrics
    match_count: int
    match_rate: float
    
    # Tier distributions
    baseline_tier_distribution: dict[str, int]
    candidate_tier_distribution: dict[str, int]
    tier_changes: dict[str, int]  # e.g., {"none": +1, "applied_ai": -2}
    
    # Confidence stats
    baseline_confidence: dict[str, float]
    candidate_confidence: dict[str, float]
    confidence_change: float
    
    # Agreement
    baseline_agreement_rate: float
    candidate_agreement_rate: float
    agreement_change: float
    
    # Skills stats (if available)
    avg_hardskills_per_job: float | None
    avg_softskills_per_job: float | None
    
    # Changes
    changes: list[ClassificationChange] = field(default_factory=list)
    
    # Warnings
    warnings: list[Warning] = field(default_factory=list)
    
    def to_json(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dict."""
        result = asdict(self)
        # Convert ClassificationChange and Warning objects
        result["changes"] = [asdict(c) for c in self.changes]
        result["warnings"] = [asdict(w) for w in self.warnings]
        return result
    
    def to_markdown(self) -> str:
        """Generate human-readable Markdown report."""
        lines = []
        
        # Header
        lines.append("# Pipeline Evaluation Report")
        lines.append("")
        lines.append(f"**Generated**: {self.evaluation_timestamp}")
        lines.append(f"**Baseline**: `{Path(self.baseline_path).name}`")
        lines.append(f"**Candidate**: `{Path(self.candidate_path).name}`")
        lines.append("")
        
        # Warnings section (if any)
        if self.warnings:
            lines.append("## ⚠️ Warnings")
            lines.append("")
            for w in self.warnings:
                icon = "🔴" if w.level == "critical" else "🟡"
                lines.append(f"{icon} **{w.metric}**: {w.message}")
            lines.append("")
        
        # Summary metrics
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Jobs | {self.total_jobs} |")
        lines.append(f"| Match Rate | **{self.match_rate:.1%}** ({self.match_count}/{self.total_jobs}) |")
        lines.append(f"| Confidence Change | {self.confidence_change:+.3f} |")
        lines.append(f"| Agreement Change | {self.agreement_change:+.1%} |")
        if self.avg_hardskills_per_job is not None:
            lines.append(f"| Avg Hardskills/Job | {self.avg_hardskills_per_job:.1f} |")
        if self.avg_softskills_per_job is not None:
            lines.append(f"| Avg Softskills/Job | {self.avg_softskills_per_job:.1f} |")
        lines.append("")
        
        # Tier distribution comparison
        lines.append("## Tier Distribution")
        lines.append("")
        lines.append("| Tier | Baseline | Candidate | Change |")
        lines.append("|------|----------|-----------|--------|")
        for tier in ["none", "ai_integration", "applied_ai", "core_ai"]:
            base = self.baseline_tier_distribution.get(tier, 0)
            cand = self.candidate_tier_distribution.get(tier, 0)
            change = cand - base
            change_str = f"+{change}" if change > 0 else str(change)
            lines.append(f"| `{tier}` | {base} | {cand} | {change_str} |")
        lines.append("")
        
        # Classification changes
        if self.changes:
            lines.append(f"## Classification Changes ({len(self.changes)})")
            lines.append("")
            
            for i, change in enumerate(self.changes, 1):
                lines.append(f"### {i}. Job {change.job_id}: {change.job_title}")
                lines.append("")
                lines.append(f"**Change**: `{change.old_tier}` → `{change.new_tier}`")
                lines.append("")
                lines.append(f"| | Baseline | Candidate |")
                lines.append("|--|----------|-----------|")
                lines.append(f"| Confidence | {change.old_confidence:.2f} | {change.new_confidence:.2f} |")
                lines.append(f"| AI Skills | {change.ai_skills_old or '—'} | {change.ai_skills_new or '—'} |")
                lines.append("")
                lines.append("**Job Description Excerpt**:")
                lines.append(f"> {change.job_description_excerpt[:400]}...")
                lines.append("")
                lines.append("**Baseline Rationale**:")
                lines.append(f"> {change.old_rationale}")
                lines.append("")
                lines.append("**Candidate Rationale**:")
                lines.append(f"> {change.new_rationale}")
                lines.append("")
                lines.append("---")
                lines.append("")
        else:
            lines.append("## Classification Changes")
            lines.append("")
            lines.append("✅ No classification changes detected.")
            lines.append("")
        
        return "\n".join(lines)


# ============================================================================
# MAIN EVALUATOR CLASS
# ============================================================================

class PipelineEvaluator:
    """Compare two pipeline outputs deterministically."""
    
    def __init__(
        self, 
        baseline_path: Path, 
        candidate_path: Path,
        thresholds: EvaluationThresholds | None = None,
        sep: str = ";"
    ):
        self.baseline_path = Path(baseline_path)
        self.candidate_path = Path(candidate_path)
        self.thresholds = thresholds or EvaluationThresholds.default()
        self.sep = sep
        
        # Load dataframes
        self.baseline_df = pd.read_csv(self.baseline_path, sep=self.sep)
        self.candidate_df = pd.read_csv(self.candidate_path, sep=self.sep)
        
        # Validate
        if len(self.baseline_df) != len(self.candidate_df):
            raise ValueError(
                f"Row count mismatch: baseline has {len(self.baseline_df)}, "
                f"candidate has {len(self.candidate_df)}"
            )
    
    def compare(self) -> EvaluationReport:
        """Run all comparisons and return structured report."""
        changes = self._identify_changes()
        match_count = len(self.baseline_df) - len(changes)
        match_rate = match_count / len(self.baseline_df)
        
        baseline_conf = self._confidence_stats(self.baseline_df)
        candidate_conf = self._confidence_stats(self.candidate_df)
        
        baseline_agree = self._agreement_rate(self.baseline_df)
        candidate_agree = self._agreement_rate(self.candidate_df)
        
        # Check for warnings
        warnings = self._check_thresholds(
            match_rate=match_rate,
            confidence_change=candidate_conf.mean - baseline_conf.mean,
            agreement_rate=candidate_agree
        )
        
        # Skills stats
        avg_hard, avg_soft = self._skills_stats(self.candidate_df)
        
        return EvaluationReport(
            baseline_path=str(self.baseline_path),
            candidate_path=str(self.candidate_path),
            evaluation_timestamp=datetime.now().isoformat(),
            total_jobs=len(self.baseline_df),
            match_count=match_count,
            match_rate=match_rate,
            baseline_tier_distribution=self._tier_distribution(self.baseline_df),
            candidate_tier_distribution=self._tier_distribution(self.candidate_df),
            tier_changes=self._tier_changes(),
            baseline_confidence={
                "mean": baseline_conf.mean,
                "std": baseline_conf.std,
                "min": baseline_conf.min,
                "max": baseline_conf.max,
            },
            candidate_confidence={
                "mean": candidate_conf.mean,
                "std": candidate_conf.std,
                "min": candidate_conf.min,
                "max": candidate_conf.max,
            },
            confidence_change=candidate_conf.mean - baseline_conf.mean,
            baseline_agreement_rate=baseline_agree,
            candidate_agreement_rate=candidate_agree,
            agreement_change=candidate_agree - baseline_agree,
            avg_hardskills_per_job=avg_hard,
            avg_softskills_per_job=avg_soft,
            changes=changes,
            warnings=warnings,
        )
    
    def export_disagreements(self, output_path: Path) -> Path:
        """Export jobs with classification changes to a separate CSV."""
        changes = self._identify_changes()
        if not changes:
            # Create empty file with header
            pd.DataFrame(columns=self.candidate_df.columns).to_csv(
                output_path, sep=self.sep, index=False
            )
            return output_path
        
        change_ids = [c.job_id for c in changes]
        disagreements_df = self.candidate_df[
            self.candidate_df["id"].isin(change_ids)
        ].copy()
        
        # Add columns showing the baseline values for comparison
        baseline_tiers = {}
        baseline_conf = {}
        for c in changes:
            baseline_tiers[c.job_id] = c.old_tier
            baseline_conf[c.job_id] = c.old_confidence
        
        disagreements_df["baseline_tier"] = disagreements_df["id"].map(baseline_tiers)
        disagreements_df["baseline_confidence"] = disagreements_df["id"].map(baseline_conf)
        
        disagreements_df.to_csv(output_path, sep=self.sep, index=False)
        return output_path
    
    def _identify_changes(self) -> list[ClassificationChange]:
        """Identify jobs where classification changed."""
        changes = []
        
        for _, base_row in self.baseline_df.iterrows():
            job_id = base_row["id"]
            cand_row = self.candidate_df[self.candidate_df["id"] == job_id].iloc[0]
            
            old_tier = base_row["AI_tier_openai"]
            new_tier = cand_row["AI_tier_openai"]
            
            if old_tier != new_tier:
                # Get job description excerpt
                job_desc = str(base_row.get("job_desc_text", ""))[:500]
                
                changes.append(ClassificationChange(
                    job_id=int(job_id),
                    job_title=str(base_row.get("job_title", "Unknown")),
                    old_tier=str(old_tier),
                    new_tier=str(new_tier),
                    old_confidence=float(base_row["AI_skill_openai_confidence"]),
                    new_confidence=float(cand_row["AI_skill_openai_confidence"]),
                    old_rationale=str(base_row.get("AI_skill_openai_rationale", "")),
                    new_rationale=str(cand_row.get("AI_skill_openai_rationale", "")),
                    job_description_excerpt=job_desc,
                    ai_skills_old=str(base_row.get("AI_skills_openai_mentioned", "")),
                    ai_skills_new=str(cand_row.get("AI_skills_openai_mentioned", "")),
                ))
        
        return changes
    
    def _tier_distribution(self, df: pd.DataFrame) -> dict[str, int]:
        """Get tier distribution as dict."""
        dist = df["AI_tier_openai"].value_counts().to_dict()
        # Ensure all tiers are present
        for tier in ["none", "ai_integration", "applied_ai", "core_ai"]:
            if tier not in dist:
                dist[tier] = 0
        return dist
    
    def _tier_changes(self) -> dict[str, int]:
        """Calculate tier count changes."""
        base_dist = self._tier_distribution(self.baseline_df)
        cand_dist = self._tier_distribution(self.candidate_df)
        return {
            tier: cand_dist.get(tier, 0) - base_dist.get(tier, 0)
            for tier in ["none", "ai_integration", "applied_ai", "core_ai"]
        }
    
    def _confidence_stats(self, df: pd.DataFrame) -> ConfidenceStats:
        """Calculate confidence statistics."""
        conf = df["AI_skill_openai_confidence"]
        return ConfidenceStats(
            mean=float(conf.mean()),
            std=float(conf.std()),
            min=float(conf.min()),
            max=float(conf.max()),
        )
    
    def _agreement_rate(self, df: pd.DataFrame) -> float:
        """Calculate agreement rate between hard-coded and OpenAI."""
        if "AI_skill_agreement" not in df.columns:
            return 0.0
        return float((df["AI_skill_agreement"] == 1).sum() / len(df))
    
    def _skills_stats(self, df: pd.DataFrame) -> tuple[float | None, float | None]:
        """Calculate average skills per job."""
        avg_hard = None
        avg_soft = None
        
        if "hardskills" in df.columns:
            counts = df["hardskills"].apply(
                lambda x: len(str(x).split(", ")) if pd.notna(x) and x else 0
            )
            avg_hard = float(counts.mean())
        
        if "softskills" in df.columns:
            counts = df["softskills"].apply(
                lambda x: len(str(x).split(", ")) if pd.notna(x) and x else 0
            )
            avg_soft = float(counts.mean())
        
        return avg_hard, avg_soft
    
    def _check_thresholds(
        self, 
        match_rate: float, 
        confidence_change: float,
        agreement_rate: float
    ) -> list[Warning]:
        """Check metrics against thresholds and generate warnings."""
        warnings = []
        
        if match_rate < self.thresholds.min_match_rate:
            warnings.append(Warning(
                level="warning" if match_rate >= 0.7 else "critical",
                metric="Match Rate",
                message=f"Match rate {match_rate:.1%} is below threshold {self.thresholds.min_match_rate:.1%}",
                threshold=self.thresholds.min_match_rate,
                actual=match_rate,
            ))
        
        if confidence_change < -self.thresholds.max_confidence_drop:
            warnings.append(Warning(
                level="warning",
                metric="Confidence Drop",
                message=f"Mean confidence dropped by {abs(confidence_change):.3f}",
                threshold=self.thresholds.max_confidence_drop,
                actual=abs(confidence_change),
            ))
        
        if agreement_rate < self.thresholds.min_agreement_rate:
            warnings.append(Warning(
                level="warning",
                metric="Agreement Rate",
                message=f"Agreement rate {agreement_rate:.1%} is below threshold {self.thresholds.min_agreement_rate:.1%}",
                threshold=self.thresholds.min_agreement_rate,
                actual=agreement_rate,
            ))
        
        return warnings


def save_evaluation_history(report: EvaluationReport, history_dir: Path) -> Path:
    """Save evaluation report to history with timestamp."""
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = history_dir / f"evaluation_{timestamp}.json"
    
    with open(history_path, "w") as f:
        json.dump(report.to_json(), f, indent=2)
    
    return history_path
