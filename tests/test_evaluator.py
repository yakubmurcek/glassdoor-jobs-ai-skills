#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the evaluation framework."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ai_skills.evaluator import (
    ClassificationChange,
    EvaluationReport,
    EvaluationThresholds,
    PipelineEvaluator,
    Warning,
    save_evaluation_history,
)


class TestEvaluationThresholds(unittest.TestCase):
    """Tests for EvaluationThresholds."""
    
    def test_default_thresholds(self):
        """Default thresholds are set correctly."""
        thresholds = EvaluationThresholds.default()
        self.assertEqual(thresholds.min_match_rate, 0.85)
        self.assertEqual(thresholds.max_confidence_drop, 0.05)
        self.assertEqual(thresholds.min_agreement_rate, 0.80)
    
    def test_custom_thresholds(self):
        """Custom thresholds can be specified."""
        thresholds = EvaluationThresholds(
            min_match_rate=0.90,
            max_confidence_drop=0.10,
            min_agreement_rate=0.75,
        )
        self.assertEqual(thresholds.min_match_rate, 0.90)
        self.assertEqual(thresholds.max_confidence_drop, 0.10)
        self.assertEqual(thresholds.min_agreement_rate, 0.75)


class TestPipelineEvaluator(unittest.TestCase):
    """Tests for PipelineEvaluator."""
    
    def setUp(self):
        """Create temporary CSV files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create baseline data
        self.baseline_data = pd.DataFrame({
            "id": [1, 2, 3],
            "job_title": ["Job A", "Job B", "Job C"],
            "job_desc_text": ["Desc A", "Desc B", "Desc C"],
            "AI_tier_openai": ["none", "ai_integration", "applied_ai"],
            "AI_skill_openai_confidence": [0.90, 0.85, 0.80],
            "AI_skill_openai_rationale": ["Rationale A", "Rationale B", "Rationale C"],
            "AI_skills_openai_mentioned": ["", "AI skill", "ML, NLP"],
            "AI_skill_agreement": [1, 1, 0],
        })
        
        # Create candidate data (one change: job 3 changes tier)
        self.candidate_data = pd.DataFrame({
            "id": [1, 2, 3],
            "job_title": ["Job A", "Job B", "Job C"],
            "job_desc_text": ["Desc A", "Desc B", "Desc C"],
            "AI_tier_openai": ["none", "ai_integration", "ai_integration"],  # Job 3 changed
            "AI_skill_openai_confidence": [0.90, 0.85, 0.85],
            "AI_skill_openai_rationale": ["Rationale A", "Rationale B", "Rationale C updated"],
            "AI_skills_openai_mentioned": ["", "AI skill", "ML"],
            "AI_skill_agreement": [1, 1, 1],
            "hardskills": ["python, sql", "react, node", "aws, docker"],
            "softskills": ["communication", "teamwork", "leadership"],
        })
        
        self.baseline_path = Path(self.temp_dir) / "baseline.csv"
        self.candidate_path = Path(self.temp_dir) / "candidate.csv"
        
        self.baseline_data.to_csv(self.baseline_path, sep=";", index=False)
        self.candidate_data.to_csv(self.candidate_path, sep=";", index=False)
    
    def test_match_rate_calculation(self):
        """Match rate is calculated correctly."""
        evaluator = PipelineEvaluator(self.baseline_path, self.candidate_path)
        report = evaluator.compare()
        
        # 2 out of 3 jobs match (jobs 1 and 2)
        self.assertEqual(report.match_count, 2)
        self.assertAlmostEqual(report.match_rate, 2/3, places=5)
    
    def test_identifies_classification_changes(self):
        """Classification changes are correctly identified."""
        evaluator = PipelineEvaluator(self.baseline_path, self.candidate_path)
        report = evaluator.compare()
        
        self.assertEqual(len(report.changes), 1)
        change = report.changes[0]
        self.assertEqual(change.job_id, 3)
        self.assertEqual(change.old_tier, "applied_ai")
        self.assertEqual(change.new_tier, "ai_integration")
    
    def test_tier_distribution_comparison(self):
        """Tier distributions are calculated correctly."""
        evaluator = PipelineEvaluator(self.baseline_path, self.candidate_path)
        report = evaluator.compare()
        
        self.assertEqual(report.baseline_tier_distribution["none"], 1)
        self.assertEqual(report.baseline_tier_distribution["ai_integration"], 1)
        self.assertEqual(report.baseline_tier_distribution["applied_ai"], 1)
        
        self.assertEqual(report.candidate_tier_distribution["none"], 1)
        self.assertEqual(report.candidate_tier_distribution["ai_integration"], 2)  # +1
        self.assertEqual(report.candidate_tier_distribution["applied_ai"], 0)  # -1
    
    def test_confidence_stats(self):
        """Confidence statistics are calculated correctly."""
        evaluator = PipelineEvaluator(self.baseline_path, self.candidate_path)
        report = evaluator.compare()
        
        # Baseline: [0.90, 0.85, 0.80], mean = 0.85
        self.assertAlmostEqual(report.baseline_confidence["mean"], 0.85, places=2)
        
        # Candidate: [0.90, 0.85, 0.85], mean ≈ 0.867
        self.assertAlmostEqual(report.candidate_confidence["mean"], 0.867, places=2)
    
    def test_skills_stats(self):
        """Skills statistics are calculated correctly."""
        evaluator = PipelineEvaluator(self.baseline_path, self.candidate_path)
        report = evaluator.compare()
        
        # Each job has 2 hardskills and 1 softskill
        self.assertAlmostEqual(report.avg_hardskills_per_job, 2.0, places=1)
        self.assertAlmostEqual(report.avg_softskills_per_job, 1.0, places=1)
    
    def test_warning_on_low_match_rate(self):
        """Warnings are generated when thresholds are breached."""
        thresholds = EvaluationThresholds(min_match_rate=0.80)  # 66% is below this
        evaluator = PipelineEvaluator(
            self.baseline_path, 
            self.candidate_path,
            thresholds=thresholds,
        )
        report = evaluator.compare()
        
        match_warnings = [w for w in report.warnings if w.metric == "Match Rate"]
        self.assertEqual(len(match_warnings), 1)
        self.assertIn("below threshold", match_warnings[0].message)
    
    def test_export_disagreements(self):
        """Disagreements CSV is exported correctly."""
        evaluator = PipelineEvaluator(self.baseline_path, self.candidate_path)
        evaluator.compare()
        
        disagree_path = Path(self.temp_dir) / "disagreements.csv"
        evaluator.export_disagreements(disagree_path)
        
        self.assertTrue(disagree_path.exists())
        disagree_df = pd.read_csv(disagree_path, sep=";")
        
        # Only job 3 should be in disagreements
        self.assertEqual(len(disagree_df), 1)
        self.assertEqual(disagree_df.iloc[0]["id"], 3)
        self.assertEqual(disagree_df.iloc[0]["baseline_tier"], "applied_ai")


class TestEvaluationReport(unittest.TestCase):
    """Tests for EvaluationReport."""
    
    def test_to_json(self):
        """Report converts to JSON correctly."""
        report = EvaluationReport(
            baseline_path="/path/to/baseline.csv",
            candidate_path="/path/to/candidate.csv",
            evaluation_timestamp="2025-01-01T00:00:00",
            total_jobs=10,
            baseline_columns=40,
            candidate_columns=42,
            new_columns=["hardskills", "softskills"],
            match_count=9,
            match_rate=0.9,
            baseline_tier_distribution={"none": 8, "ai_integration": 2},
            candidate_tier_distribution={"none": 9, "ai_integration": 1},
            tier_changes={"none": 1, "ai_integration": -1},
            baseline_confidence={"mean": 0.85, "std": 0.05, "min": 0.75, "max": 0.95},
            candidate_confidence={"mean": 0.87, "std": 0.04, "min": 0.80, "max": 0.95},
            confidence_change=0.02,
            baseline_agreement_rate=0.80,
            candidate_agreement_rate=0.85,
            agreement_change=0.05,
            avg_hardskills_per_job=5.0,
            avg_softskills_per_job=2.0,
            changes=[],
            warnings=[],
        )
        
        json_data = report.to_json()
        
        self.assertEqual(json_data["match_rate"], 0.9)
        self.assertEqual(json_data["total_jobs"], 10)
        self.assertIsInstance(json_data["changes"], list)
    
    def test_to_markdown(self):
        """Report converts to Markdown correctly."""
        report = EvaluationReport(
            baseline_path="/path/to/baseline.csv",
            candidate_path="/path/to/candidate.csv",
            evaluation_timestamp="2025-01-01T00:00:00",
            total_jobs=10,
            baseline_columns=40,
            candidate_columns=42,
            new_columns=["hardskills", "softskills"],
            match_count=9,
            match_rate=0.9,
            baseline_tier_distribution={"none": 8, "ai_integration": 2},
            candidate_tier_distribution={"none": 9, "ai_integration": 1},
            tier_changes={"none": 1, "ai_integration": -1},
            baseline_confidence={"mean": 0.85, "std": 0.05, "min": 0.75, "max": 0.95},
            candidate_confidence={"mean": 0.87, "std": 0.04, "min": 0.80, "max": 0.95},
            confidence_change=0.02,
            baseline_agreement_rate=0.80,
            candidate_agreement_rate=0.85,
            agreement_change=0.05,
            avg_hardskills_per_job=5.0,
            avg_softskills_per_job=2.0,
            changes=[],
            warnings=[],
        )
        
        md = report.to_markdown()
        
        self.assertIn("# Pipeline Evaluation Report", md)
        self.assertIn("**90.0%**", md)
        self.assertIn("Tier Distribution", md)


class TestSaveEvaluationHistory(unittest.TestCase):
    """Tests for save_evaluation_history."""
    
    def test_saves_to_history_directory(self):
        """History is saved with timestamp."""
        report = EvaluationReport(
            baseline_path="/path/to/baseline.csv",
            candidate_path="/path/to/candidate.csv",
            evaluation_timestamp="2025-01-01T00:00:00",
            total_jobs=10,
            baseline_columns=40,
            candidate_columns=40,
            new_columns=[],
            match_count=9,
            match_rate=0.9,
            baseline_tier_distribution={},
            candidate_tier_distribution={},
            tier_changes={},
            baseline_confidence={"mean": 0.85, "std": 0.05, "min": 0.75, "max": 0.95},
            candidate_confidence={"mean": 0.87, "std": 0.04, "min": 0.80, "max": 0.95},
            confidence_change=0.02,
            baseline_agreement_rate=0.80,
            candidate_agreement_rate=0.85,
            agreement_change=0.05,
            avg_hardskills_per_job=None,
            avg_softskills_per_job=None,
            changes=[],
            warnings=[],
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = Path(temp_dir) / "history"
            result_path = save_evaluation_history(report, history_dir)
            
            self.assertTrue(result_path.exists())
            self.assertTrue(result_path.name.startswith("evaluation_"))
            self.assertTrue(result_path.name.endswith(".json"))
            
            # Verify content
            with open(result_path) as f:
                saved_data = json.load(f)
            self.assertEqual(saved_data["match_rate"], 0.9)


if __name__ == "__main__":
    unittest.main()
