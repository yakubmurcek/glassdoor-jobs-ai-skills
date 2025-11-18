#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line entry point for the AI skills analysis pipeline."""

import itertools
import sys
import threading
import time

from src.config import OUTPUT_CSV
from src.pipeline import JobAnalysisPipeline


def main():
    """Execute the refactored analysis pipeline with a CLI spinner."""
    pipeline = JobAnalysisPipeline()
    stop_event = threading.Event()
    spinner_thread = threading.Thread(
        target=_spinner, args=("Analyzing job descriptions", stop_event), daemon=True
    )

    start_time = time.perf_counter()
    spinner_thread.start()
    try:
        df = pipeline.run()
    finally:
        stop_event.set()
        spinner_thread.join()
    elapsed = time.perf_counter() - start_time

    _clear_spinner_line()
    print(
        f"Processed {len(df)} job descriptions. "
        f"Added AI columns and saved results to {OUTPUT_CSV}. "
        f"(Elapsed: {elapsed:.1f}s)"
    )


def _spinner(message: str, stop_event: threading.Event) -> None:
    prefix = f"{message} "
    frames = ["|", "/", "-", "\\"]
    for frame in itertools.cycle(frames):
        if stop_event.is_set():
            break
        sys.stdout.write(f"\r{prefix}{frame}")
        sys.stdout.flush()
        time.sleep(0.1)


def _clear_spinner_line() -> None:
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
