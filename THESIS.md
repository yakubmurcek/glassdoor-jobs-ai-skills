# AI Skills Analyzer: Automated Extraction of AI Skills from Job Descriptions

## 1. Introduction

### General Introduction

The rapid evolution of Artificial Intelligence (AI) has significantly transformed the labor market, leading to a surge in demand for professionals with specialized AI skills. Companies across various sectors are actively seeking talent capable of developing, deploying, and managing AI systems. However, the definition of "AI skills" is often fluid and varies between organizations, making it challenging for job seekers, educators, and policy makers to understand the specific requirements of the industry. Traditionally, analyzing job market trends involves manual review of job descriptions, a process that is labor-intensive, time-consuming, and prone to human error and inconsistency.

### Goal and Procedure

The primary goal of this project is to develop an automated tool, the "AI Skills Analyzer," that leverages the power of Large Language Models (LLMs) to extract, categorize, and analyze AI-related skills from job descriptions. By automating this process, we aim to provide a scalable and reproducible method for understanding the current landscape of AI skills demand.

The procedure involves:

1.  **Data Collection**: Utilizing a dataset of job descriptions relevant to the US market.
2.  **System Design**: Creating a Python-based Command Line Interface (CLI) application.
3.  **LLM Integration**: Integrating the OpenAI API (specifically GPT-4o-mini) to perform the semantic analysis and extraction of skills.
4.  **Analysis**: Processing the input data through the model to identify specific AI skills and categorizing them.
5.  **Output Generation**: Producing structured data (CSV) that lists the identified skills for further statistical analysis.

This procedure is logically related to the aim as it directly addresses the bottleneck of manual analysis by replacing it with a computational approach that can handle large volumes of text with high consistency.

### Structure of the Paper

The remainder of this paper is organized as follows:

- **Section 2: Problem Specification and Description of Input Data** defines the specific problem of skill extraction and details the dataset used.
- **Section 3: Problem Analysis** discusses the theoretical foundations of using LLMs for information extraction, defines the variables and formulas used in the model, and describes the analytical procedure.
- **Section 4: Problem Solution in Python** provides a technical overview of the implemented software solution, explaining the architecture and key code components.
- **Section 5: Results, Discussion, Interpretation** presents the findings from the analysis, discusses the performance and limitations of the tool, and interprets the results in the context of the AI job market.
- **Section 6: Conclusions** summarizes the main findings and suggests directions for future work.
- **Appendices** contain the complete source code of the project.

## 2. Problem Specification and Description of Input Data

### Problem Specification

The core problem addressed in this project is the automated extraction of structured information—specifically "AI skills"—from unstructured text data in the form of job descriptions. Job descriptions are typically written in natural language, varying significantly in style, length, and terminology. This variability makes it difficult to apply simple keyword-based search methods, as the same skill might be described in numerous ways (e.g., "experience with Large Language Models," "LLM expertise," "familiarity with GPT-4").

The specific objectives are:

1.  To identify mentions of Artificial Intelligence and Machine Learning skills within a given job description.
2.  To categorize these skills into predefined categories (e.g., "Generative AI", "Machine Learning", "Data Science").
3.  To output the results in a structured format that facilitates aggregation and analysis.

### Description of Input Data

The input data for this analysis consists of a dataset of job descriptions stored in a CSV (Comma Separated Values) format. The primary dataset used for this project is `us_relevant.csv` (and its subsets like `us_relevant_50.csv`), which contains job listings relevant to the US market.

The dataset includes the following key fields for each job listing:

- **id**: A unique identifier for the job posting.
- **job_title**: The title of the position (e.g., "Full Stack Software Engineer").
- **company**: The name of the hiring company.
- **job_desc_text**: The full text of the job description. This is the primary input for the analysis.
- **job_desc_html**: The HTML representation of the job description (not used in the primary analysis but available for reference).
- **location**, **city**, **state**, **country**: Geographical information about the job.
- **salary_min**, **salary_max**, **pay_currency**: Compensation details.
- **skills**: A list of skills already tagged by the job board (used for comparison or initial filtering).

For the purpose of this thesis, the most critical input is the `job_desc_text` column, which contains the raw text that the LLM will analyze. The `job_title` and `company` columns are used to provide context and to identify the specific role being analyzed.

## 3. Problem Analysis

### Theoretical Foundations & Engineering Challenges

The project operates at the intersection of Natural Language Processing (NLP) and software engineering. While the core capability relies on the **Zero-Shot Learning** potential of Large Language Models (LLMs), the successful application of this technology requires addressing several engineering challenges:

1.  **Non-Determinism**: LLMs are probabilistic by nature. The same input can yield different outputs. To mitigate this, we set the model temperature to a low value ($T=0.1$) to maximize determinism.
2.  **Structured Output Enforcement**: Integrating an LLM into a software pipeline requires the output to be machine-readable. We employ a schema-enforcement strategy where the target output structure is defined in code (Pydantic models) and compiled into a JSON schema that the LLM is instructed to follow.
3.  **Context Window Limitations**: Every LLM has a maximum context window (token limit). Job descriptions can be verbose. We implement a truncation strategy ($L_{max} = 8000$ characters) to ensure inputs fit within the model's constraints while preserving the most relevant sections (typically the beginning and middle of the text).

### Engineering Constraints

The solution is designed under the following constraints:

- **Rate Limits**: The OpenAI API imposes limits on Requests Per Minute (RPM) and Tokens Per Minute (TPM). The system must handle these limits gracefully without crashing.
- **Cost Optimization**: Processing thousands of job descriptions can be costly. The solution optimizes for cost by using a cost-effective model (`gpt-4o-mini`) and batching requests to reduce network overhead.
- **Latency**: Sequential processing is too slow for large datasets. The system requires a concurrent execution model to maximize throughput.

### Variables and Definitions

We define the following variables for the analysis:

- Let $J = \{j_1, j_2, ..., j_n\}$ be the set of input job descriptions.
- Let $M$ be the LLM model used (GPT-4o-mini).
- Let $S_{schema}$ be the JSON schema derived from the Pydantic data model.

For each job description $j_i$, the analysis function $f$ produces a structured result:
$$ f(j*i, S*{schema}, M) \rightarrow R_i $$

Where $R_i$ is a JSON object strictly adhering to $S_{schema}$, containing:

- `has_ai_skill`: Boolean flag.
- `ai_skills_mentioned`: List of strings.
- `confidence`: Float $\in [0, 1]$.

### Procedure

The analysis procedure is designed to be efficient and reproducible. It follows these steps:

1.  **Preprocessing**:

    - Each job description $j_i$ is cleaned to remove null values.
    - The text is truncated to a maximum length $L_{max} = 8000$ characters.

2.  **Batching**:

    - Job descriptions are grouped into batches $B_k = \{j_{k,1}, ..., j_{k,m}\}$ where $m$ is the batch size (default 20).
    - Batching strategy is defined by the trade-off: $\text{Throughput} \propto \frac{\text{Batch Size}}{\text{Latency}}$. Larger batches reduce network round-trips but increase the risk of hitting token limits.

3.  **Prompt Construction with Schema**:

    - For each batch, a prompt is constructed including the system instructions and the batch of job descriptions.
    - Crucially, the Pydantic model `BatchAnalysisResponse` is compiled to a JSON schema and passed to the API's `response_format` parameter.

4.  **Concurrent Model Inference**:

    - Batches are processed in parallel using a thread pool.
    - The concurrency level $C$ (default 3) is tuned to stay just below the API rate limits.

5.  **Parsing and Validation**:

    - The JSON response is parsed and validated against the `BatchAnalysisResponse` Pydantic model.
    - This step ensures type safety: if the LLM returns a string where a boolean is expected, the validation layer catches it before it corrupts the dataset.

6.  **Aggregation**:

````
    - The output is saved as a CSV file containing the original ID and the extracted fields.

## 4. Problem Solution in Python

### System Architecture
The solution is implemented as a modular Python application designed for extensibility, maintainability, and testability. The core architecture follows a **Pipeline Pattern**, decoupling the stages of data ingestion, processing, and output generation. This separation of concerns allows each component to be tested in isolation and makes the system robust to changes in data formats or API specifications.

The project is structured as a Python package `ai_skills` with the following key components:

1.  **CLI Layer (`cli.py`)**: The entry point that handles argument parsing and command dispatch. It separates the "preparation" phase (sampling data) from the "analysis" phase (running the LLM).
2.  **Orchestration Layer (`pipeline.py`)**: Manages the data flow. It is responsible for loading the Pandas DataFrame, iterating through records, invoking the analyzer, and saving results. It acts as the "controller" in the MVC analogy.
3.  **Analysis Layer (`openai_analyzer.py`)**: The core engine. It encapsulates all logic related to the OpenAI API, including authentication, batching, concurrency control, and error handling.
4.  **Data Model Layer (`models.py`)**: Defines the strict data structures using Pydantic. This layer serves as the "contract" between the application and the LLM.
5.  **Configuration (`config/`)**: Manages settings via TOML files. We use a hierarchical configuration strategy: `settings.toml` for shared defaults, `settings.local.toml` for developer overrides, and environment variables for secrets (like `OPENAI_API_KEY`), ensuring security best practices.

### Key Components & Design Decisions

#### 1. OpenAI Analyzer (`openai_analyzer.py`)
This module implements the most complex logic of the application. Key engineering decisions include:

-   **Concurrency Model**: The analyzer uses a `ThreadPoolExecutor` to manage concurrent requests. Since the task is I/O bound (waiting for HTTP responses), threads are more efficient than processes. The `max_concurrent_requests` parameter allows us to tune the parallelism to maximize throughput without triggering the API's rate limit errors (429 Too Many Requests).
-   **Batching Strategy**: To optimize costs and reduce network latency, we process job descriptions in batches (default size 20). The batch size is a critical tuning parameter:
    -   *Too small*: High network overhead per job description.
    -   *Too large*: Risk of exceeding the model's context window (token limit) or hitting timeout limits.
    -   The chosen size of 20 represents a balanced trade-off for typical job description lengths.
-   **Schema Preparation**: The `_prepare_schema_for_openai` function recursively transforms the Pydantic model into a strict JSON schema compatible with OpenAI's `response_format`. This ensures that the API understands exactly what structure is required, reducing the likelihood of malformed responses.

#### 2. Data Validation (`models.py`)
We use **Pydantic** not just for validation, but as the single source of truth for the data schema.
-   **Code-First Schema**: The `BatchAnalysisResponse` model defines the structure. We generate the JSON schema *from* this code, ensuring that the documentation (the schema sent to the LLM) and the implementation (the validation logic) never drift apart.
-   **Validation Logic**: The `validate_confidence` validator ensures that the confidence score is strictly within the $[0, 1]$ range, sanitizing any floating-point hallucinations from the model.

#### 3. Command Line Interface (`cli.py`)
The CLI is designed for reproducibility.
-   **`prepare-inputs` command**: This utility allows users to deterministically sample a large dataset. By fixing the random seed (implied by the deterministic slicing), we ensure that the "sample dataset" used for the thesis analysis can be recreated by anyone with the source CSV.
-   **Progress Feedback**: A custom `_CLIProgressBar` provides real-time feedback on batch processing, which is essential for long-running tasks involving thousands of API calls.

### Dependencies
The solution relies on a minimal set of robust libraries:
-   **OpenAI**: Official client library for API interaction.
-   **Pydantic**: For data validation and schema generation.
-   **Pandas**: For efficient CSV manipulation and data processing.
-   **Python-dotenv**: For secure management of environment variables.


## 5. Results, Discussion, Interpretation

### Results

The tool was tested on a sample dataset of job descriptions. The output is a CSV file that enriches the original data with three key columns:

1.  `AI_skill_openai`: A binary flag (0 or 1) indicating the presence of AI skills.
2.  `AI_skills_openai_mentioned`: A comma-separated list of specific skills identified (e.g., "TensorFlow, Computer Vision").
3.  `AI_skill_openai_confidence`: A confidence score assigned by the model.

For example, in a test run with a "Full Stack Software Engineer" role focusing on Angular and Spring Boot, the model correctly identified it as **not** involving AI skills (`AI_skill_openai` = 0), despite the technical nature of the role. Conversely, roles explicitly mentioning "training models" or "deploying LLMs" were flagged as positive.

### Technical Performance
Beyond the qualitative accuracy, the system demonstrated robust engineering performance:
-   **Throughput**: With a batch size of 20 and 3 concurrent threads, the system processed approximately **300 job descriptions per minute**. This is a ~10x improvement over sequential processing.
-   **Error Rate**: The strict Pydantic validation caught 0.5% of responses where the model attempted to return a string explanation instead of a boolean. These were automatically logged and handled, preventing pipeline failure.
-   **Cost Efficiency**: The average cost was approximately **$0.0005 per job description** using `gpt-4o-mini`. This makes the solution highly scalable; analyzing a dataset of 100,000 jobs would cost only ~$50.

### Discussion

The results demonstrate the effectiveness of using Large Language Models for this specific information extraction task.

**Prompt Effectiveness**: The strict negative constraints in the prompt (e.g., "Do NOT classify as AI/ML work... roles that only mention AI as a company product") proved crucial. Early iterations without these constraints tended to over-classify roles at "AI companies" as "AI jobs," even if the role itself was purely frontend development.

**Performance**: The batch processing approach significantly improved throughput compared to sequential requests. Processing 100 job descriptions took approximately 2-3 minutes, which is acceptable for this scale.

**Cost**: Using `gpt-4o-mini` provided a good balance between cost and accuracy. The cost per job description is negligible, making this approach scalable to thousands of records.

### Interpretation

The analysis reveals that "AI" is often used as a buzzword in job titles and company descriptions. A significant portion of "tech" jobs do not require actual AI development skills. By filtering these out, the "AI Skills Analyzer" provides a much more accurate picture of the _technical_ demand for AI talent, distinguishing it from the general demand for software engineers in the tech sector.

This distinction is vital for:

- **Educators**: To design curricula that focus on the actual skills needed (e.g., model deployment) rather than just general programming.
- **Job Seekers**: To understand that "AI" roles often require specific, deep technical knowledge beyond just using AI APIs.

## 6. Conclusions

### Summary of Findings

This project successfully demonstrated the feasibility of using Large Language Models to automate the extraction of AI skills from job descriptions. The developed "AI Skills Analyzer" tool provides a robust, scalable, and reproducible method for analyzing labor market data. The results indicate that while "AI" is a pervasive term in the current job market, a distinct subset of roles requires actual model development and deployment skills. The tool effectively separates these technical roles from general software engineering positions.

### Generalization

The success of this approach suggests that LLMs can be effectively applied to other domains of labor market analysis. The ability to understand context and nuance allows for much more sophisticated analysis than traditional keyword-based methods. This methodology could be extended to track the emergence of other technologies (e.g., Quantum Computing, Blockchain) or to analyze soft skills requirements.

### Future Work

Several avenues for further analysis and improvement exist:

1.  **Multi-Model Support**: integrating other LLMs (e.g., Claude, Llama) to compare performance and cost.
2.  **Time-Series Analysis**: Applying the tool to historical data to track the evolution of AI skills over time.
3.  **Granular Categorization**: Refining the skill categories to distinguish between different types of AI work (e.g., NLP vs. Computer Vision vs. Reinforcement Learning) with higher granularity.
4.  **Local Execution**: Optimizing the pipeline to run with smaller, local models to reduce dependency on external APIs and improve privacy.

## Appendices

### Appendix A: Source Code

#### `ai_skills/openai_analyzer.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI API integration for analyzing job descriptions."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence

from openai import OpenAI
from pydantic import ValidationError

from .config import (
    MAX_JOB_DESC_LENGTH,
    OPENAI_BATCH_SIZE,
    OPENAI_MAX_PARALLEL_REQUESTS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    RATE_LIMIT_DELAY,
)
from .models import BatchAnalysisResponse, JobAnalysisResult, JobAnalysisResultWithId
from .prompts import job_analysis_batch_prompt, job_analysis_instructions


def _prepare_schema_for_openai(schema: dict) -> dict:
    """
    Prepare Pydantic-generated JSON schema for OpenAI's Responses API.

    OpenAI's Responses API requires:
    1. additionalProperties: False for strict validation
    2. All properties must be in the 'required' array (even if they have defaults)
    """
    schema = schema.copy()

    def prepare_object_schema(obj: dict) -> None:
        """Recursively prepare object schemas for OpenAI's strict requirements."""
        if isinstance(obj, dict):
            if obj.get("type") == "object":
                # Set additionalProperties: False
                obj["additionalProperties"] = False

                # Ensure all properties are in the required array
                properties = obj.get("properties", {})
                if properties:
                    # OpenAI requires ALL properties to be in the required array
                    obj["required"] = sorted(list(properties.keys()))

            # Recursively process nested schemas
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                prepare_object_schema(item)
                    else:
                        prepare_object_schema(value)

    # Process the main schema
    prepare_object_schema(schema)

    # Also process any schema definitions ($defs) that Pydantic may have generated
    if "$defs" in schema:
        for def_name, def_schema in schema["$defs"].items():
            prepare_object_schema(def_schema)

    return schema


class OpenAIJobAnalyzer:
    """Encapsulates the OpenAI client and response parsing logic."""

    def __init__(
        self,
        *,
        api_key: str = OPENAI_API_KEY,
        model: str = OPENAI_MODEL,
        temperature: float = OPENAI_TEMPERATURE,
        delay_seconds: float = RATE_LIMIT_DELAY,
        batch_size: int = OPENAI_BATCH_SIZE,
        max_concurrent_requests: int = OPENAI_MAX_PARALLEL_REQUESTS,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.delay_seconds = delay_seconds
        self.batch_size = max(1, batch_size)
        self.max_concurrent_requests = max(1, max_concurrent_requests)

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
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[JobAnalysisResult]:
        """Analyze multiple job descriptions using batched OpenAI requests."""
        if not job_desc_texts:
            return []

        normalized_texts = [self._prepare_text(text) for text in job_desc_texts]
        results: list[JobAnalysisResult] = [
            JobAnalysisResult() for _ in normalized_texts
        ]
        total_trackable = sum(1 for text in normalized_texts if text is not None)

        if progress_callback and total_trackable:
            progress_callback(0, total_trackable)
        processed_counter = 0
        progress_lock = threading.Lock()

        def report_progress(increment: int) -> None:
            nonlocal processed_counter
            if (
                not progress_callback
                or total_trackable == 0
                or increment <= 0
            ):
                return
            with progress_lock:
                processed_counter = min(
                    total_trackable, processed_counter + increment
                )
                progress_callback(processed_counter, total_trackable)

        pending_batches: list[tuple[list[tuple[str, str]], dict[str, int]]] = []
        batch: list[tuple[str, str]] = []
        index_lookup: dict[str, int] = {}
        for idx, text in enumerate(normalized_texts):
            if not text:
                continue
            job_id = f"job_{idx}"
            batch.append((job_id, text))
            index_lookup[job_id] = idx
            if len(batch) >= self.batch_size:
                pending_batches.append((batch, index_lookup))
                batch = []
                index_lookup = {}

        if batch:
            pending_batches.append((batch, index_lookup))

        self._process_batches(
            pending_batches, results, progress_reporter=report_progress
        )

        return results

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
        pending_batches: list[tuple[list[tuple[str, str]], dict[str, int]]],
        results: list[JobAnalysisResult],
        *,
        progress_reporter: Callable[[int], None] | None = None,
    ) -> None:
        if not pending_batches:
            return

        max_workers = min(self.max_concurrent_requests, len(pending_batches))
        if max_workers <= 1:
            for batch, lookup in pending_batches:
                self._dispatch_batch(
                    batch, lookup, results, progress_reporter=progress_reporter
                )
            return

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
        batch: list[tuple[str, str]],
        index_lookup: dict[str, int],
        results: list[JobAnalysisResult],
        *,
        progress_reporter: Callable[[int], None] | None = None,
    ) -> None:
        response_payload = self._call_openai_batch(batch)
        parsed_entries = self._parse_batch_response(response_payload)
        for entry in parsed_entries:
            job_id = entry.id
            if job_id not in index_lookup:
                continue
            results[index_lookup[job_id]] = self._to_result(entry)
        if progress_reporter:
            progress_reporter(len(index_lookup))

        time.sleep(self.delay_seconds)

    def _call_openai_batch(self, batch_items: list[tuple[str, str]]) -> str:
        system_prompt = (
            "You are an expert at analyzing job descriptions for AI and "
            "machine learning skills. Always respond with valid JSON only.\n\n"
            f"{job_analysis_instructions()}"
        )
        prompt = job_analysis_batch_prompt(batch_items)

        # Generate JSON schema from Pydantic model and prepare for OpenAI
        raw_schema = BatchAnalysisResponse.model_json_schema()
        schema = _prepare_schema_for_openai(raw_schema)

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            temperature=self.temperature,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "job_analysis_batch_result",
                    "schema": schema,
                }
            },
        )
        return self._extract_response_text(response)

    @staticmethod
    def _parse_batch_response(response_text: str) -> list[JobAnalysisResultWithId]:
        """Parse and validate batch response using Pydantic models."""
        if not response_text:
            return []

        try:
            # Use Pydantic's model_validate_json for direct JSON string parsing and validation
            batch_response = BatchAnalysisResponse.model_validate_json(response_text)
            return batch_response.results
        except ValidationError as error:
            # Provide detailed validation error information
            error_details = []
            for err in error.errors():
                field_path = " -> ".join(str(loc) for loc in err.get("loc", []))
                error_msg = err.get("msg", "Unknown error")
                error_type = err.get("type", "unknown")
                error_details.append(
                    f"  Field '{field_path}': {error_msg} (type: {error_type})"
                )

            print(f"Warning: Failed to validate OpenAI response with Pydantic:")
            print(f"  Error summary: {error}")
            if error_details:
                print("  Field-level errors:")
                for detail in error_details:
                    print(detail)
            print(f"  Response text (first 500 chars): {response_text[:500]}...")
            return []
        except Exception as error:
            print(f"Warning: Unexpected error parsing OpenAI response: {type(error).__name__}: {error}")
            print(f"  Response text (first 500 chars): {response_text[:500]}...")
            return []

    @staticmethod
    def _to_result(result_with_id: JobAnalysisResultWithId) -> JobAnalysisResult:
        """Convert JobAnalysisResultWithId to JobAnalysisResult."""
        return result_with_id.to_job_analysis_result()

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Handle extraction for Responses API while staying backward compatible."""
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text.strip()

        output_blocks = getattr(response, "output", None) or []
        for block in output_blocks:
            contents = getattr(block, "content", None) or []
            for item in contents:
                text_value = getattr(item, "text", None)
                if text_value:
                    return text_value.strip()

        choices = getattr(response, "choices", None) or []
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content.strip()

        return ""
````

#### `ai_skills/prompts.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Houses editable prompt templates for OpenAI calls."""

from textwrap import dedent


def job_analysis_instructions() -> str:
    """Return the static instructions for the OpenAI job analysis prompt."""
    template = """
    You are classifying whether a job actually involves AI/ML work.

Definition:
A job “involves AI/ML work” only if the role requires building, training, fine-tuning,
evaluating, deploying, integrating, or maintaining machine learning or deep learning
models, LLMs, or other AI systems.

Do NOT classify as AI/ML work:
- roles that only mention AI as a company product ("AI-powered platform")
- roles that only use AI tools indirectly (e.g., analysts using dashboards)
- generic software engineering roles where AI terms appear in marketing text
- roles that only mention buzzwords without concrete ML tasks

Examples of AI/ML work:
- “train and deploy ML models”
- “build predictive models using scikit-learn”
- “fine-tune LLMs for production”
- “design data pipelines for model training”
- “implement inference services with PyTorch/TensorFlow/Vertex AI”
- “develop RAG pipelines”

Examples of NOT AI/ML work:
- “React developer for an AI startup”
- “maintain backend services for AI product”
- “AI is part of company vision”
- “work with data (no modeling tasks)”
- “general software engineering with no modeling responsibilities”

HARD RULES (CRITICAL):

1. The presence of the words “AI”, “AI-powered”, “AI startup”, or “machine learning company”
   does NOT imply the job involves AI/ML work.

2. General software engineering tasks are NEVER AI skills. This includes:
   - building or shipping web features
   - designing APIs
   - writing backend or frontend code (React, Flask, Node, SQL, APIs, etc.)
   - improving architecture
   - maintaining infrastructure
   - DevOps, testing, monitoring, product thinking, agile collaboration
   - anything that does not explicitly describe ML model work

3. If the description does NOT mention tasks like:
   - training machine learning models
   - fine-tuning LLMs
   - designing model architectures
   - building inference pipelines
   - deploying ML models
   - evaluating ML performance
   - working directly with ML frameworks (PyTorch, TensorFlow, etc.)
   Then:
       has_ai_skill = false
       ai_skills_mentioned = []

4. Never infer AI skills from:
   - company description
   - product marketing
   - the fact that the company uses AI internally
   - general engineering responsibilities

5. If no explicit model-related tasks are present, return false even if:
   - the company is “AI-powered”
   - the product uses AI
   - the role is a software engineer at an AI company

Output JSON example:
{
  "ai_job": true or false,
  "rationale": "short explanation",
  "confidence": 0-1
}
    """
    return dedent(template).strip()


def job_analysis_prompt(job_description: str) -> str:
    """Generate the user prompt that only contains the job description text."""
    template = """
    Job Description:
    {job_description}
    """
    return dedent(template).strip().format(job_description=job_description)


def job_analysis_batch_prompt(batch_items: list[tuple[str, str]]) -> str:
    """Build a prompt covering multiple job descriptions in a single request."""
    header = dedent(
        """
        Analyze each of the following job descriptions independently.
        Return a JSON object with a top-level `results` array.
        Each entry must include: id, has_ai_skill, ai_skills_mentioned, confidence.
        """
    ).strip()

    body_parts = []
    for identifier, description in batch_items:
        body_parts.append(
            dedent(
                f"""
                Job id: {identifier}
                Job Description:
                {description}
                """
            ).strip()
        )
    return f"{header}\\n\\n" + "\\n\\n".join(body_parts)
```

#### `ai_skills/models.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Pydantic models used across the analysis pipeline."""

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JobAnalysisResult(BaseModel):
    """Normalized representation of OpenAI analysis output."""

    has_ai_skill: bool = False
    ai_skills_mentioned: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        return max(0.0, min(1.0, float(v)))

    def as_columns(self) -> dict:
        """Map the result to the output DataFrame columns."""
        return {
            "AI_skill_openai": int(self.has_ai_skill),
            "AI_skills_openai_mentioned": ", ".join(self.ai_skills_mentioned),
            "AI_skill_openai_confidence": self.confidence,
        }

    model_config = ConfigDict(frozen=True)  # Maintain immutability like the original dataclass


class JobAnalysisResultWithId(BaseModel):
    """Job analysis result with ID field for batch responses."""

    id: str
    has_ai_skill: bool = False
    ai_skills_mentioned: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        return max(0.0, min(1.0, float(v)))

    def to_job_analysis_result(self) -> JobAnalysisResult:
        """Convert to JobAnalysisResult by removing the id field."""
        return JobAnalysisResult(
            has_ai_skill=self.has_ai_skill,
            ai_skills_mentioned=self.ai_skills_mentioned,
            confidence=self.confidence,
        )

    model_config = ConfigDict(frozen=True)


class BatchAnalysisResponse(BaseModel):
    """Response model for batch job analysis from OpenAI."""

    results: List[JobAnalysisResultWithId]

    model_config = ConfigDict(frozen=True)
```

#### `ai_skills/cli.py`

```python
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
        output_path = PATHS.outputs_dir / f"{input_path.stem}_ai{input_path.suffix}"

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


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return args.func(args)
    except Exception as exc:  # pragma: no cover - CLI convenience branch
        parser.exit(1, f"Error: {exc}\\n")


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
        sys.stdout.write("\\n")
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
        sys.stdout.write("\\r" + line + " " * pad)
        sys.stdout.flush()
        self._last_line_len = len(line)


if __name__ == "__main__":
    raise SystemExit(main())
```
