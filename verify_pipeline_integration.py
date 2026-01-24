
from ai_skills.pipeline import JobAnalysisPipeline
import pandas as pd
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)

# Run pipeline in skip_llm (hydration) mode
# This simulates the full pipeline but reuses existing columns, 
# triggering the new _apply_stata_transformations logic
pipeline = JobAnalysisPipeline()
df = pipeline.run(
    input_csv='data/outputs/us_relevant_ai_deduped_sample_20.csv',
    output_csv='data/outputs/verify_pipeline_output.csv',
    skip_llm=True
)

print(f"Pipeline finished. Output shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check for new columns
has_clusters = any(c.startswith('cluster_') for c in df.columns)
has_hybrid = 'education_hybrid' in df.columns
dropped_desc = 'job_desc_text' not in df.columns

print(f"Has Cluster Columns: {has_clusters}")
print(f"Has Education Hybrid: {has_hybrid}")
print(f"Dropped job_desc_text: {dropped_desc}")
