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
    
    # Create figure with subplots (2x3 grid for better layout)
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f'Pipeline Evaluation: Baseline vs Candidate', 
        fontsize=16, 
        fontweight='bold'
    )
    
    # Colors
    colors_baseline = '#FF6B6B'
    colors_candidate = '#4ECDC4'
    
    # Create grid spec for flexible layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # =========================================================================
    # 1. Tier Distribution (top-left, spans 2 columns)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0:2])
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
    
    # =========================================================================
    # 2. Match Rate - PROMINENT & SEPARATE (top-right)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Create a big gauge-like display for match rate
    match_pct = report.match_rate * 100
    color = '#4ECDC4' if report.match_rate >= 0.85 else '#FF6B6B' if report.match_rate < 0.7 else '#FFA500'
    
    # Draw a big circle with match rate
    circle = plt.Circle((0.5, 0.5), 0.4, color=color, alpha=0.3)
    ax2.add_patch(circle)
    ax2.text(0.5, 0.5, f'{match_pct:.0f}%', fontsize=36, ha='center', va='center', 
             fontweight='bold', color=color)
    ax2.text(0.5, 0.15, f'{report.match_count}/{report.total_jobs} jobs match', 
             fontsize=11, ha='center', va='center', color='gray')
    ax2.text(0.5, 0.9, 'MATCH RATE', fontsize=12, ha='center', va='center', 
             fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # =========================================================================
    # 3. Agreement Rates (bottom-left)
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    agreement_labels = ['Baseline', 'Candidate']
    agreement_values = [
        report.baseline_agreement_rate * 100,
        report.candidate_agreement_rate * 100
    ]
    
    bars = ax3.bar(agreement_labels, agreement_values, 
                   color=[colors_baseline, colors_candidate], 
                   edgecolor='black', width=0.5)
    ax3.set_ylabel('Agreement Rate (%)', fontsize=11)
    ax3.set_title('Agreement (Hard-coded vs OpenAI)', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    
    # Add value labels and change indicator
    for i, (bar, val) in enumerate(zip(bars, agreement_values)):
        ax3.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 5), textcoords="offset points", ha='center', 
                    va='bottom', fontsize=11, fontweight='bold')
    
    # Show change arrow
    change = report.agreement_change * 100
    change_color = '#4ECDC4' if change >= 0 else '#FF6B6B'
    change_text = f'{change:+.1f}%'
    ax3.annotate(change_text, xy=(1.2, (agreement_values[0] + agreement_values[1]) / 2),
                fontsize=12, ha='left', va='center', color=change_color, fontweight='bold')
    
    # =========================================================================
    # 4. Confidence Score Comparison (bottom-middle)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
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
    bars1 = ax4.bar(x - width/2, baseline_vals, width, label='Baseline', 
                    color=colors_baseline, edgecolor='black')
    bars2 = ax4.bar(x + width/2, candidate_vals, width, label='Candidate', 
                    color=colors_candidate, edgecolor='black')
    
    ax4.set_ylabel('Confidence Score', fontsize=11)
    ax4.set_title('Confidence Score Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, fontsize=10)
    ax4.legend(loc='lower right')
    ax4.set_ylim(0.7, 1.0)
    
    # Add value labels
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax4.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', 
                    va='bottom', fontsize=9)
    
    # =========================================================================
    # 5. Summary Table (bottom-right)
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Build table data with totals and columns
    new_cols_str = ', '.join(report.new_columns) if report.new_columns else '—'
    if len(new_cols_str) > 25:
        new_cols_str = new_cols_str[:22] + '...'
    
    table_data = [
        ['Metric', 'Value'],
        ['Total Jobs', f'{report.total_jobs}'],
        ['Columns (Base → Cand)', f'{report.baseline_columns} → {report.candidate_columns}'],
        ['New Columns', new_cols_str],
        ['Changes Detected', f'{len(report.changes)}'],
        ['Mean Conf. Change', f'{report.confidence_change:+.3f}'],
    ]
    
    # Add skills if available
    if report.avg_hardskills_per_job is not None:
        table_data.append(['Avg Hardskills/Job', f'{report.avg_hardskills_per_job:.1f}'])
    if report.avg_softskills_per_job is not None:
        table_data.append(['Avg Softskills/Job', f'{report.avg_softskills_per_job:.1f}'])
    
    table = ax5.table(
        cellText=table_data, 
        loc='center', 
        cellLoc='left',
        colWidths=[0.5, 0.5]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax5.set_title('Summary', fontsize=12, fontweight='bold', pad=20)
    
    # Add warnings badge if any
    if report.warnings:
        warning_text = f"⚠️ {len(report.warnings)} warning(s)"
        ax5.text(0.5, -0.1, warning_text, transform=ax5.transAxes, 
                fontsize=12, ha='center', color='#FF6B6B', fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path
