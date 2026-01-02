#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single entry point for preparing data and running the AI skills pipeline."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

from .extract_csv import DEFAULT_SOURCE, extract_sample_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-skills",
        description="Prepare inputs and analyze job descriptions for AI skills.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser(
        "prepare-inputs",
        help="Create a smaller CSV sample for experimentation or grading.",
    )
    prep.add_argument(
        "--rows",
        type=int,
        default=100,
        help="Number of rows to copy from the source CSV (default: 100).",
    )
    prep.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source CSV to sample from (default: {DEFAULT_SOURCE}).",
    )
    prep.add_argument(
        "--destination",
        type=Path,
        default=None,
        help=(
            "Destination CSV that the pipeline will read "
            "(default: derive from --source and the selected row count)."
        ),
    )
    prep.add_argument(
        "--sep",
        type=str,
        default=";",
        help="Column separator for both files (default: ';').",
    )
    prep.set_defaults(func=_handle_prepare_inputs)

    analyze = sub.add_parser(
        "analyze",
        help="Run the full OpenAI-powered pipeline on the configured CSV.",
    )
    analyze.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the CLI progress bar (useful when piping output).",
    )
    analyze.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV file to analyze (usually the output from prepare-inputs).",
    )
    analyze.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Write results to a custom path (default: derived from config or --input-csv).",
    )
    analyze.set_defaults(func=_handle_analyze)

    # Evaluate command
    evaluate = sub.add_parser(
        "evaluate",
        help="Compare baseline and candidate CSV outputs to evaluate changes.",
    )
    evaluate.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Baseline CSV (previous/known-good output).",
    )
    evaluate.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Candidate CSV (new output to evaluate).",
    )
    evaluate.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for report outputs (default: same as candidate).",
    )
    evaluate.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip generating comparison chart.",
    )
    evaluate.add_argument(
        "--min-match-rate",
        type=float,
        default=0.85,
        help="Minimum match rate threshold for warnings (default: 0.85).",
    )
    evaluate.add_argument(
        "--min-agreement",
        type=float,
        default=0.80,
        help="Minimum agreement rate threshold for warnings (default: 0.80).",
    )
    evaluate.add_argument(
        "--no-history",
        action="store_true",
        help="Skip saving evaluation to history.",
    )
    evaluate.set_defaults(func=_handle_evaluate)

    # Search command
    search = sub.add_parser(
        "search",
        help="Search for jobs using natural language queries (semantic search).",
    )
    search.add_argument(
        "query",
        type=str,
        help="The search query string.",
    )
    search.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    search.set_defaults(func=_handle_search)

    # Cluster command
    cluster = sub.add_parser(
        "cluster",
        help="Cluster skills to find synonyms and groupings.",
    )
    cluster.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV file containing skills to cluster.",
    )
    cluster.add_argument(
        "--column",
        type=str,
        default="skills",
        help="Column name containing skill lists (default: 'skills').",
    )
    cluster.add_argument(
        "--clusters",
        type=int,
        default=20,
        help="Number of clusters to generate (default: 20).",
    )
    cluster.set_defaults(func=_handle_cluster)

    cluster.set_defaults(func=_handle_cluster)

    # Visualize command
    viz = sub.add_parser(
        "visualize-skills",
        help="Generate a 2D plot of skill embeddings.",
    )
    viz.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV file containing skills.",
    )
    viz.add_argument(
        "--column",
        type=str,
        default="skills",
        help="Column name containing skill lists (default: 'skills').",
    )
    viz.add_argument(
        "--output-image",
        type=Path,
        default=Path("skill_map.png"),
        help="Path to save the output image (default: skill_map.png).",
    )
    viz.set_defaults(func=_handle_visualize_skills)

    # Classify command
    classify = sub.add_parser(
        "classify",
        help="Classify text against a list of labels (zero-shot).",
    )
    classify.add_argument(
        "text",
        type=str,
        help="The text to classify.",
    )
    classify.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="List of candidate labels (e.g. 'Senior' 'Junior').",
    )
    classify.set_defaults(func=_handle_classify)

    classify.set_defaults(func=_handle_classify)

    # Index command
    index = sub.add_parser(
        "index",
        help="Index job descriptions from a CSV into the vector store.",
    )
    index.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV file to index.",
    )
    index.add_argument(
        "--id-col",
        type=str,
        default="job_url_id",
        help="Column to use as unique ID (default: 'job_url_id').",
    )
    index.add_argument(
        "--text-col",
        type=str,
        default="job_desc_text",
        help="Column containing text to embed (default: 'job_desc_text').",
    )
    index.set_defaults(func=_handle_index)

    return parser


def _handle_prepare_inputs(args: argparse.Namespace) -> int:
    destination = extract_sample_rows(
        rows=max(1, args.rows),
        source_csv=args.source,
        destination_csv=args.destination,
        sep=args.sep,
    )
    print(
        f"Created sample of {args.rows} rows at {destination}. "
        "Update INPUT_CSV in config if you choose a custom destination."
    )
    return 0


def _get_versioned_output_path(base_dir: Path, stem: str, suffix: str) -> Path:
    """Generate a versioned output path to avoid overwriting existing files.
    
    If no file exists, returns {stem}_ai{suffix}.
    If {stem}_ai{suffix} exists, returns {stem}_ai_v1{suffix}, then _v2, etc.
    """
    base_path = base_dir / f"{stem}_ai{suffix}"
    if not base_path.exists():
        return base_path
    
    version = 1
    while True:
        versioned_path = base_dir / f"{stem}_ai_v{version}{suffix}"
        if not versioned_path.exists():
            return versioned_path
        version += 1


def _handle_analyze(args: argparse.Namespace) -> int:
    from .config import OUTPUT_CSV, PATHS  # Imported lazily to avoid requiring an API key early.
    from .pipeline import JobAnalysisPipeline

    pipeline = JobAnalysisPipeline()
    progress = None
    callback: Callable[[int, int], None] | None = None
    if not args.no_progress and sys.stderr.isatty():
        progress = _CLIProgressBar("Analyzing job descriptions")
        callback = progress.update

    output_path: Path | None = args.output_csv
    if output_path is None and args.input_csv is not None:
        input_path = Path(args.input_csv)
        output_path = _get_versioned_output_path(PATHS.outputs_dir, input_path.stem, input_path.suffix)

    start_time = time.perf_counter()
    try:
        df = pipeline.run(
            progress_callback=callback,
            input_csv=args.input_csv,
            output_csv=output_path,
        )
    finally:
        if progress:
            progress.finish()
    elapsed = time.perf_counter() - start_time

    print(
        f"Processed {len(df)} job descriptions. "
        f"Added AI columns and saved results to {(output_path or OUTPUT_CSV)}. "
        f"(Elapsed: {elapsed:.1f}s)"
    )
    return 0


def _handle_evaluate(args: argparse.Namespace) -> int:
    import json
    from .evaluator import PipelineEvaluator, EvaluationThresholds, save_evaluation_history
    from .chart_generator import generate_comparison_chart

    # Determine output directory
    output_dir = args.output_dir or args.candidate.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure thresholds
    thresholds = EvaluationThresholds(
        min_match_rate=args.min_match_rate,
        min_agreement_rate=args.min_agreement,
    )

    print(f"Comparing baseline: {args.baseline.name}")
    print(f"      vs candidate: {args.candidate.name}")
    print()

    # Run evaluation
    evaluator = PipelineEvaluator(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        thresholds=thresholds,
    )
    report = evaluator.compare()

    # Print summary
    print(f"Match rate: {report.match_rate:.1%} ({report.match_count}/{report.total_jobs})")
    print(f"Changes:    {len(report.changes)}")
    if report.warnings:
        print()
        for w in report.warnings:
            icon = "🔴" if w.level == "critical" else "🟡"
            print(f"{icon} {w.message}")
    print()

    # Generate outputs
    outputs_generated = []

    # 1. Markdown report
    md_path = output_dir / "evaluation_report.md"
    md_path.write_text(report.to_markdown())
    outputs_generated.append(f"Report:       {md_path}")

    # 2. JSON output
    json_path = output_dir / "evaluation.json"
    with open(json_path, "w") as f:
        json.dump(report.to_json(), f, indent=2)
    outputs_generated.append(f"JSON:         {json_path}")

    # 3. Disagreements CSV
    if report.changes:
        disagree_path = output_dir / "disagreements.csv"
        evaluator.export_disagreements(disagree_path)
        outputs_generated.append(f"Disagreements: {disagree_path}")

    # 4. Comparison chart
    if not args.no_chart:
        chart_path = output_dir / "comparison_chart.png"
        generate_comparison_chart(report, chart_path)
        outputs_generated.append(f"Chart:        {chart_path}")

    # 5. History tracking
    if not args.no_history:
        history_dir = output_dir / "evaluation_history"
        history_path = save_evaluation_history(report, history_dir)
        outputs_generated.append(f"History:      {history_path}")

    print("Generated outputs:")
    for out in outputs_generated:
        print(f"  {out}")

    return 0


def _handle_search(args: argparse.Namespace) -> int:
    from .vector_store import VectorStoreManager

    manager = VectorStoreManager()
    if manager.count() == 0:
        print("Vector store is empty. Please run 'analyze' first to populate it.")
        return 1

    print(f"Searching for: '{args.query}'...")
    results = manager.query_similar_jobs(args.query, n_results=args.limit)

    if not results or not results["ids"] or not results["ids"][0]:
        print("No matches found.")
        return 0

    ids = results["ids"][0]
    distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)
    metadatas = results["metadatas"][0] if "metadatas" in results else [{}] * len(ids)
    documents = results["documents"][0] if "documents" in results else [""] * len(ids)

    for i, job_id in enumerate(ids):
        dist = distances[i]
        meta = metadatas[i] or {}
        doc = documents[i]
        
        # Try to get a title or something recognizable
        title = meta.get("job_title", "Unknown Title")
        
        print("-" * 60)
        print(f"Rank {i+1} | Score: {1 - dist:.4f} | ID: {job_id}")
        print(f"Title: {title}")
        print(f"Snippet: {doc[:200]}..." if len(doc) > 200 else f"Snippet: {doc}")
    
    print("-" * 60)
    return 0


def _handle_cluster(args: argparse.Namespace) -> int:
    import pandas as pd
    from ast import literal_eval
    from .clustering import SkillClusterer

    print(f"Reading {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv, sep=None, engine='python')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 1

    if args.column not in df.columns:
        print(f"Column '{args.column}' not found. Available columns: {', '.join(df.columns)}")
        return 1
    
    print(f"Extracting skills from column '{args.column}'...")
    all_skills = []
    for val in df[args.column].dropna():
        if isinstance(val, str):
            val = val.strip()
            # Check if it looks like a list
            if val.startswith("[") and val.endswith("]"):
                try:
                    parsed = literal_eval(val)
                    if isinstance(parsed, list):
                        all_skills.extend([str(p) for p in parsed])
                    else:
                        all_skills.append(val)
                except:
                    all_skills.append(val)
            else:
                # Assume comma separated if multiple items or just one item
                if "," in val:
                    parts = [p.strip() for p in val.split(",")]
                    all_skills.extend(parts)
                else:
                    all_skills.append(val)
        else:
             all_skills.append(str(val))

    # Clean up
    unique_skills = sorted(list(set(s for s in all_skills if s and s.lower() != "nan")))
    
    if not unique_skills:
        print("No skills found.")
        return 0

    print(f"Found {len(unique_skills)} unique skills. Clustering into {args.clusters} clusters...")
    
    clusterer = SkillClusterer()
    clusters = clusterer.cluster_skills(unique_skills, n_clusters=args.clusters)
    
    print("\n" + "="*60)
    print(f"CLUSTERING RESULTS ({len(clusters)} clusters)")
    print("="*60)
    
    for cid in sorted(clusters.keys()):
        skills = clusters[cid]
        print(f"\nCluster {cid} ({len(skills)} items):")
        print(", ".join(skills))
        
    return 0


def _handle_visualize_skills(args: argparse.Namespace) -> int:
    import pandas as pd
    from ast import literal_eval
    from .chart_generator import plot_skill_embeddings

    print(f"Reading {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv, sep=None, engine='python')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 1

    if args.column not in df.columns:
        print(f"Column '{args.column}' not found. Available columns: {', '.join(df.columns)}")
        return 1
    
    print(f"Extracting skills from column '{args.column}'...")
    all_skills = []
    for val in df[args.column].dropna():
        if isinstance(val, str):
            val = val.strip()
            # Check if it looks like a list
            if val.startswith("[") and val.endswith("]"):
                try:
                    parsed = literal_eval(val)
                    if isinstance(parsed, list):
                        all_skills.extend([str(p) for p in parsed])
                    else:
                        all_skills.append(val)
                except:
                    all_skills.append(val)
            else:
                # Assume comma separated
                if "," in val:
                    parts = [p.strip() for p in val.split(",")]
                    all_skills.extend(parts)
                else:
                    all_skills.append(val)
        else:
             all_skills.append(str(val))

    unique_skills = sorted(list(set(s for s in all_skills if s and s.lower() != "nan")))
    
    if not unique_skills:
        print("No skills found.")
        return 0

    print(f"Found {len(unique_skills)} unique skills. Generating visualization...")
    output_path = plot_skill_embeddings(unique_skills, args.output_image)
    
    print(f"Visualization saved to: {output_path}")
    return 0


def _handle_classify(args: argparse.Namespace) -> int:
    from .embeddings import EmbeddingService
    
    print(f"Text: '{args.text}'")
    print(f"Candidate Labels: {args.labels}")
    
    service = EmbeddingService()
    result = service.classify_text(args.text, args.labels)
    
    print("-" * 40)
    print(f"Predicted Class: {result}")
    print("-" * 40)
    
    print("-" * 40)
    
    return 0


def _handle_index(args: argparse.Namespace) -> int:
    import pandas as pd
    from .vector_store import VectorStoreManager
    
    print(f"Reading {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv, sep=None, engine='python')
        # Clean column names (remove BOM and whitespace)
        df.columns = df.columns.str.replace('^\ufeff', '', regex=True).str.strip()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 1
        
    if args.id_col not in df.columns or args.text_col not in df.columns:
        print(f"Missing required columns: {args.id_col}, {args.text_col}")
        print(f"Available: {', '.join(df.columns)}")
        return 1
        
    print(f"Indexing {len(df)} records...")
    
    ids = df[args.id_col].astype(str).tolist()
    documents = df[args.text_col].fillna("").astype(str).tolist()
    
    metadatas = []
    # Include title if available
    if "job_title" in df.columns:
        titles = df["job_title"].fillna("Unknown").astype(str).tolist()
        metadatas = [{"job_title": t} for t in titles]
    else:
        metadatas = [{"job_title": "Unknown"} for _ in ids]
        
    manager = VectorStoreManager()
    manager.add_jobs(ids, documents, metadatas)
    
    print(f"Successfully indexed {manager.count()} jobs.")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return args.func(args)
    except Exception as exc:  # pragma: no cover - CLI convenience branch
        parser.exit(1, f"Error: {exc}\n")


class _CLIProgressBar:
    """ASCII progress bar with inline percentage and spinner animation."""

    def __init__(
        self, message: str, width: int = 40, refresh_interval: float = 0.1
    ) -> None:
        self.message = message
        self.width = max(10, width)
        self.current = 0
        self.total = 0
        self._active = True
        self._refresh_interval = max(0.05, refresh_interval)
        self._fill_char = "█"
        self._empty_char = "░"
        self._spinner_frames = "|/-\\"
        self._spinner_index = 0
        self._render_lock = threading.Lock()
        self._last_line_len = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def update(self, completed: int, total: int) -> None:
        if not self._active:
            return
        self.total = max(total, 0)
        safe_total = max(self.total, 1)
        self.current = max(0, min(completed, safe_total))
        self._render()

    def finish(self) -> None:
        if not self._active:
            return
        self.current = max(self.current, self.total)
        self._stop_event.set()
        self._thread.join()
        self._render()
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._last_line_len = 0
        self._active = False

    def _animate(self) -> None:
        while not self._stop_event.is_set():
            self._render(tick=True)
            time.sleep(self._refresh_interval)

    def _render(self, *, tick: bool = False) -> None:
        if not self._active:
            return
        with self._render_lock:
            if tick:
                self._spinner_index = (self._spinner_index + 1) % len(
                    self._spinner_frames
                )
            spinner = self._spinner_frames[self._spinner_index]
            safe_total = max(self.total, 1)
            completed = max(0, min(self.current, safe_total))
            percent = completed / safe_total
            filled = min(self.width, int(round(percent * self.width)))
            bar_chars = [self._empty_char] * self.width
            for i in range(filled):
                bar_chars[i] = self._fill_char

            percent_text = f"{percent * 100:5.1f}%"
            start = max(0, (self.width - len(percent_text)) // 2)
            for index, ch in enumerate(percent_text):
                pos = start + index
                if pos < self.width:
                    bar_chars[pos] = ch

            bar = "".join(bar_chars)
            total_label = "?" if self.total == 0 else str(self.total)
            line = (
                f"{self.message} {spinner} [{bar}] {completed}/{total_label}"
            )
            self._write_line(line)

    def _write_line(self, line: str) -> None:
        pad = max(0, self._last_line_len - len(line))
        sys.stdout.write("\r" + line + " " * pad)
        sys.stdout.flush()
        self._last_line_len = len(line)


if __name__ == "__main__":
    raise SystemExit(main())
