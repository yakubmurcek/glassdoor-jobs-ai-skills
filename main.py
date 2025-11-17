#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line entry point for the AI skills analysis pipeline."""

from config import OUTPUT_CSV
from pipeline import JobAnalysisPipeline


def main():
    """Execute the refactored analysis pipeline."""
    pipeline = JobAnalysisPipeline()
    df = pipeline.run()
    print(
        f"Processed {len(df)} job descriptions. "
        f"Added AI columns and saved results to {OUTPUT_CSV}."
    )


if __name__ == "__main__":
    main()
