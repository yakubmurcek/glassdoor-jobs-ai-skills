#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chart generation for pipeline evaluation reports."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluator import EvaluationReport


def generate_comparison_chart(report: "EvaluationReport", output_path: Path) -> Path:
    """Generate 4-panel comparison chart as PNG.
    
    Args:
        report: EvaluationReport with comparison data
        output_path: Path to save the PNG file
        
    Returns:
        Path to the generated chart
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Pipeline Evaluation: Baseline vs Candidate ({report.total_jobs} Jobs)', 
        fontsize=16, 
        fontweight='bold'
    )
    
    # Colors
    colors_baseline = '#FF6B6B'
    colors_candidate = '#4ECDC4'
    
    # 1. Tier Distribution Comparison (Bar Chart)
    ax1 = axes[0, 0]
    tiers = ['none', 'ai_integration', 'applied_ai', 'core_ai']
    baseline_counts = [report.baseline_tier_distribution.get(t, 0) for t in tiers]
    candidate_counts = [report.candidate_tier_distribution.get(t, 0) for t in tiers]
    
    x = np.arange(len(tiers))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_counts, width, label='Baseline', 
                    color=colors_baseline, edgecolor='black')
    bars2 = ax1.bar(x + width/2, candidate_counts, width, label='Candidate', 
                    color=colors_candidate, edgecolor='black')
    
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('4-Tier AI Classification Distribution', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tiers, fontsize=10)
    ax1.legend()
    ax1.set_ylim(0, max(max(baseline_counts), max(candidate_counts)) * 1.2 or 10)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', 
                        va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', 
                        va='bottom', fontsize=10)
    
    # 2. Confidence Distribution (Box-like visualization)
    ax2 = axes[0, 1]
    
    # Create bar chart for confidence stats
    metrics = ['Mean', 'Min', 'Max']
    baseline_vals = [
        report.baseline_confidence['mean'],
        report.baseline_confidence['min'],
        report.baseline_confidence['max']
    ]
    candidate_vals = [
        report.candidate_confidence['mean'],
        report.candidate_confidence['min'],
        report.candidate_confidence['max']
    ]
    
    x = np.arange(len(metrics))
    bars1 = ax2.bar(x - width/2, baseline_vals, width, label='Baseline', 
                    color=colors_baseline, edgecolor='black')
    bars2 = ax2.bar(x + width/2, candidate_vals, width, label='Candidate', 
                    color=colors_candidate, edgecolor='black')
    
    ax2.set_ylabel('Confidence Score', fontsize=11)
    ax2.set_title('Confidence Score Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=10)
    ax2.legend(loc='lower right')
    ax2.set_ylim(0.7, 1.0)
    
    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', 
                    va='bottom', fontsize=9)
    
    # 3. Key Metrics Comparison
    ax3 = axes[1, 0]
    
    metric_names = ['Match Rate', 'Agreement\n(Baseline)', 'Agreement\n(Candidate)']
    metric_values = [
        report.match_rate * 100,
        report.baseline_agreement_rate * 100,
        report.candidate_agreement_rate * 100
    ]
    colors = [
        '#4ECDC4' if report.match_rate >= 0.85 else '#FF6B6B',
        colors_baseline,
        colors_candidate
    ]
    
    bars = ax3.bar(metric_names, metric_values, color=colors, edgecolor='black', width=0.5)
    ax3.set_ylabel('Percentage (%)', fontsize=11)
    ax3.set_title('Key Metrics', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.axhline(y=85, color='gray', linestyle='--', alpha=0.5, label='85% threshold')
    
    # Add value labels
    for bar, val in zip(bars, metric_values):
        ax3.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 5), textcoords="offset points", ha='center', 
                    va='bottom', fontsize=11, fontweight='bold')
    
    # 4. Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Build table data
    table_data = [
        ['Metric', 'Baseline', 'Candidate', 'Change'],
        ['Classification Match', '—', f'{report.match_count}/{report.total_jobs}', 
         f'{report.match_rate:.1%}'],
        ['Mean Confidence', f'{report.baseline_confidence["mean"]:.3f}', 
         f'{report.candidate_confidence["mean"]:.3f}', 
         f'{report.confidence_change:+.3f}'],
        ['Agreement Rate', f'{report.baseline_agreement_rate:.1%}', 
         f'{report.candidate_agreement_rate:.1%}', 
         f'{report.agreement_change:+.1%}'],
        ['Changes Detected', '—', f'{len(report.changes)}', '—'],
    ]
    
    # Add skills if available
    if report.avg_hardskills_per_job is not None:
        table_data.append(['Avg Hardskills', '—', f'{report.avg_hardskills_per_job:.1f}', '—'])
    if report.avg_softskills_per_job is not None:
        table_data.append(['Avg Softskills', '—', f'{report.avg_softskills_per_job:.1f}', '—'])
    
    table = ax4.table(
        cellText=table_data, 
        loc='center', 
        cellLoc='center',
        colWidths=[0.3, 0.2, 0.2, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax4.set_title('Summary Metrics', fontsize=12, fontweight='bold', pad=20)
    
    # Add warnings badge if any
    if report.warnings:
        warning_text = f"⚠️ {len(report.warnings)} warning(s)"
        ax4.text(0.5, -0.1, warning_text, transform=ax4.transAxes, 
                fontsize=12, ha='center', color='#FF6B6B', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path
