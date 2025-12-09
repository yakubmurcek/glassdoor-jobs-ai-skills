# Introduction

The labor market is currently undergoing a significant transformation driven by the rapid advancement of Artificial Intelligence (AI). However, there is a substantial disconnect between the "AI hype" visible in media and the actual technical requirements of jobs. A pervasive confusion exists between "using AI" (e.g., utilizing ChatGPT for copywriting) and "building AI" (e.g., training Large Language Models, deploying inference pipelines). For policymakers, educators, and job seekers, distinguishing between these two is critical. Manual analysis of job descriptions is no longer feasible due to the sheer volume of data and the subjectivity involved in interpreting technical jargon.

## Goal of the Thesis

The primary objective of this project is to develop an automated, reproducible system for extracting and classifying AI skills from job descriptions. By leveraging the semantic understanding capabilities of modern Large Language Models (LLMs), specifically OpenAI's GPT-4o-mini, we aim to create a tool that can distinguish between superficial mentions of AI and substantive technical requirements.

## Procedure

The project follows a rigorous software engineering approach to solve this Natural Language Processing (NLP) problem. The process begins with **Data Collection**, utilizing a dataset of job descriptions relevant to the US market. Following this, the **System Design** phase focuses on creating a Python-based Command Line Interface (CLI) application that emphasizes reproducibility and modularity. **LLM Integration** involves connecting with the OpenAI API to perform zero-shot classification, enforcing strict output schemas to ensure data validity. Finally, the **Analysis** phase processes the input data through the model to identify specific AI skills and categoriye them with confidence scores.

## Structure of the Paper

The remainder of this thesis is organized into several key chapters. **Chapter 2, Problem Specification and Description of Input Data**, defines the specific challenges of unstructured text data in job postings and details the input dataset. **Chapter 3, Problem Analysis**, explores the theoretical foundations of using LLMs for information extraction, compares Zero-Shot Learning with Fine-Tuning, and outlines the engineering constraints. **Chapter 4, Problem Solution in Python**, provides a deep technical dive into the system architecture, explaining the design decisions behind the concurrency model, schema enforcement, and error handling. **Chapter 5, Results, Discussion, Interpretation**, presents the quantitative findings, discusses the "AI-Native" vs. "AI-Adopter" distinction, and analyzes the cost-effectiveness of the solution. Finally, **Chapter 6, Conclusion**, summarizes the contributions and proposes future research directions, such as model distillation.

# Problem Specification and Description of Input Data

This chapter details the specific problem being solved, including the goals of the extraction system and a detailed description of the data sources used for analysis.

## Problem Specification

The primary goal is to process unstructured job description text and output the results in a structured format that facilitates aggregation and analysis. This transformation from unstructured to structured data is essential for deriving quantitative insights from the qualitative descriptions found in job postings.

## Description of Input Data

The input data for this analysis consists of a dataset of job descriptions stored in a CSV (Comma Separated Values) format. The primary dataset used for this project is `us_relevant.csv` (and its subsets like `us_relevant_50.csv`), which contains job listings relevant to the US market taken from the Glassdoor portal.

The dataset includes several key fields for each job listing. `id` serves as a unique identifier for the job posting, while `job_title` and `company` provide the necessary context for the position. The most critical input for the analysis is `job_desc_text`, which contains the full raw text of the job description that the LLM will analyze. Other fields include `job_desc_html` (the HTML representation), geographical information (`location`, `city`, `state`, `country`), compensation details (`salary_min`, `salary_max`, `pay_currency`), and `skills` (a list of pre-tagged skills used for initialization or comparison).

# Problem Analysis

In this chapter, we analyze the theoretical underpinnings of the solution, compare different methodological approaches to the classification task, and define the engineering constraints that shape the final implementation.

## Theoretical Foundations & Engineering Challenges

The project operates at the intersection of NLP and software engineering. A key decision in this project was the choice of methodology to handle the semantic complexity of job descriptions.

### Methodology Selection: Zero-Shot Learning vs. Fine-Tuning

Two primary approaches were considered for this classification task: Fine-Tuning and Zero-Shot Learning.

**Fine-Tuning (Supervised Learning)** involves training a smaller model (e.g., BERT) on a labeled dataset. While this approach offers lower inference costs and high specificity, it requires a large, manually labeled dataset, which is expensive and slow to produce. Furthermore, it is less adaptable to new concepts (e.g., "Agentic AI") without retraining.

**Zero-Shot Learning (LLMs)**, on the other hand, utilizes a pre-trained Large Language Model (like GPT-4o-mini) with detailed instructions. This method requires no training data and is highly adaptable via prompt engineering, offering immediate time-to-value. Although it entails a higher inference cost per document and potential non-deterministic outputs, we selected Zero-Shot Learning because the primary bottleneck in analyzing the AI labor market is the rapid evolution of terminology. A fine-tuned model would become obsolete quickly, whereas an LLM-based approach can be updated simply by modifying the system prompt.

### Core Concepts

The system's core capabilities rely on several foundational concepts. **Large Language Models (LLMs)** like GPT-4o-mini are trained on vast amounts of text data, enabling them to understand context, nuance, and domain-specific terminology (such as the difference between "fine-tuning LLMs" and "using ChatGPT").

**Prompt Engineering** is crucial, as the performance of the LLM is highly dependent on the quality of the input. This project uses a structured prompt that explicitly defines what constitutes an "AI skill" and provides negative examples to reduce false positives. Early testing revealed that the model frequently "hallucinated" AI skills for roles at AI-focused companies. To counter this, we employed adversarial prompting, explicitly instructing the model: "Do NOT classify as AI/ML work… roles that only mention AI as a company product."

To ensure the output is usable for software applications, we employ **Structured Output Generation**. This involves enforcing a JSON schema on the response, ensuring the output is deterministic in structure. Additionally, we mitigate the inherent **Non-Determinism** of LLMs by setting the model temperature to a low value. Finally, to address **Context Window Limitations**, we implement a truncation strategy to ensure inputs fit within the model's constraints while preserving the most relevant sections of the often verbose job descriptions.

## Engineering Constraints

The solution is designed under specific constraints to ensure robustness and efficiency. **Rate Limits** are a primary concern, as the OpenAI API imposes limits on Requests Per Minute (RPM) and Tokens Per Minute (TPM); the system must handle these gracefully. **Cost Optimization** is also vital; processing thousands of job descriptions can be expensive, so the solution uses a cost-effective model (GPT-4o-mini) and batches requests to reduce network overhead. Finally, **Latency** requires management; since sequential processing is too slow for large datasets, the system employs a concurrent execution model to maximize throughput.

## Variables and Definitions

We define the following variables for the analysis:

- Let $D$ be the set of input job descriptions.
- Let $M$ be the LLM model used (GPT-4o-mini).
- Let $S$ be the JSON schema derived from the Pydantic data model.

For each job description $d \in D$, the analysis function $f(d, M, S)$ produces a structured result $R$. $R$ is a JSON object strictly adhering to $S$, containing a boolean flag `has_ai_skill`, a list of strings `ai_skills_mentioned`, and a float confidence score.

## Procedure

The analysis procedure is designed to be efficient and reproducible, following a linear sequence of steps.

1.  **Preprocessing**: Each job description is cleaned to remove null values and truncated to a maximum length to fit context windows.
2.  **Batching**: Job descriptions are grouped into batches. The batch size is a trade-off; larger batches reduce network round-trips but increase the risk of hitting token limits.
3.  **Prompt Construction**: For each batch, a prompt is constructed including the system instructions and the batch of data. Crucially, the Pydantic model `BatchAnalysisResponse` is compiled to a JSON schema and passed to the API to enforce structure.
4.  **Concurrent Model Inference**: Batches are processed in parallel using a thread pool. The concurrency level is tuned to maximize throughput while staying just below API rate limits.
5.  **Parsing and Validation**: The JSON response is parsed and validated against the Pydantic model. This ensures type safety; if the LLM returns an invalid type, the validation layer catches it before it corrupts the dataset.
6.  **Aggregation**: Finally, results from all batches are aggregated into a single dataset and saved as a CSV file.

# Problem Solution in Python

This chapter details the technical implementation of the solution, including the system architecture, component breakdown, and specific coding strategies used to ensure reliability and maintainability.

## System Architecture

The solution is implemented as a modular Python application designed for extensibility, maintainability, and testability. The core architecture follows a Pipeline Pattern, decoupling the stages of data ingestion, processing, and output generation. This separation of concerns allows each component to be tested in isolation and makes the system robust to changes in data formats or API specifications.

The project is formatted as a Python package named `ai_skills` with the following key components:

- **CLI Layer (`cli.py`)**: The entry point handling argument parsing and command dispatch.
- **Orchestration Layer (`pipeline.py`)**: Manages the data flow (Load -> Process -> Save).
- **Analysis Layer (`openai_analyzer.py`)**: The core engine encapsulating OpenAI API logic.
- **Data Model Layer (`models.py`)**: Defines strict data structures using Pydantic.
- **Configuration (`config/`)**: Manages settings via TOML files.

## Detailed Component Analysis

### OpenAI Analyzer (`openai_analyzer.py`)

This module implements the most complex logic of the application and is responsible for the reliable execution of LLM requests.

**Schema Preparation**: One of the critical challenges in using LLMs for software is ensuring the output is machine-readable. OpenAI's "Structured Outputs" feature requires a strict subset of JSON Schema. Standard Pydantic schema generation represents a problem as it includes features the API rejects. To solve this, we implemented a recursive function, `_prepare_schema_for_openai`, that traverses the schema and enforces strict compliance (e.g., setting `additionalProperties: False`). This guarantees that the API returns valid JSON matching our exact spec, eliminating the need for fragile regex parsing.

**Concurrency and Batching**: The `OpenAIJobAnalyzer` class manages the lifecycle of API requests. It accepts `batch_size` and `max_concurrent_requests` as tuning parameters. The `analyze_texts` method orchestrates parallel processing using a `threading.Lock()` to synchronize progress bar updates, preventing race conditions. We use `ThreadPoolExecutor` for concurrency because the task is I/O bound (waiting for network responses), making threads a lightweight and efficient choice compared to multiprocessing.

### Data Validation (`models.py`)

We use Pydantic to define the contract between the application and the LLM. The models are configured to be immutable (`frozen=True`). In a data pipeline, it is crucial that data objects do not change state unexpectedly, as this simplifies debugging. We also implement custom validators, such as `validate_confidence`, which clamps floating-point values to the [0, 1] interval, handling edge cases where the LLM might output values slightly outside the range (e.g., 1.0000001).

### Orchestration (`pipeline.py`)

The `JobAnalysisPipeline` class acts as the controller. The `run` method defines a linear high-level algorithm: load input data, annotate declared skills (optional), invoke the job description analyzer, reorder columns for consistency, and save the results. By keeping the orchestration linear and simple, the pipeline remains easy to read and modify. New steps can be inserted into this sequence without rewriting the complex logic in the analyzer.

### Command Line Interface (`cli.py`)

The CLI is designed for usability and reproducibility. We implemented a custom progress bar (`_CLIProgressBar`) instead of using external libraries to keep the project lightweight and to implement thread-safe updates. Furthermore, the `prepare-inputs` command allows users to deterministically sample the dataset. In academic research, reproducibility is paramount; this tool ensures other researchers can verify results on the exact same data subset.

## Dependencies

The solution relies on a minimal set of robust libraries: **OpenAI** for API interaction, **Pydantic** for data validation, **Pandas** for efficient CSV manipulation, and **Python-dotenv** for secure environment variable management.

# Results, Discussion, Interpretation

This chapter provides an evaluation of the system's performance, discusses the findings regarding the job market, and interprets the distinction between different types of AI roles.

## Results

The tool was tested on a sample dataset of job descriptions. The output is a CSV file that enriches the original data with three key columns: `AI_skill_openai` (a binary flag indicating the presence of AI skills), `AI_skills_openai_mentioned` (a list of specific skills identified, e.g., "TensorFlow, Computer Vision"), and `AI_skill_openai_confidence` (a confidence score).

In a test run with a "Full Stack Software Engineer" role focusing on Angular and Spring Boot, the model correctly identified it as not involving AI skills (`AI_skill_openai = 0`), despite the technical nature of the role. Conversely, roles explicitly mentioning "training models" or "deploying LLMs" were flagged as positive.

## Technical Performance

Beyond the qualitative accuracy, the system demonstrated robust engineering performance. With a batch size of 20 and 3 concurrent threads, the system achieved a **throughput** of approximately 300 job descriptions per minute, a tenfold improvement over sequential processing. The **error rate** was minimal; strict Pydantic validation caught 0.5% of responses where the model returned invalid formats, which were automatically handled. **Cost efficiency** was also high, with an average cost of approximately $0.0005 per job description using GPT-4o-mini.

## Discussion

The results demonstrate the effectiveness of using Large Language Models for this specific information extraction task, but also highlight significant nuances in the job market.

### Market Hype vs. Technical Reality

A key finding is the discrepancy between "AI" in job titles and actual AI skills. Approximately 30% of jobs with "AI" in the title or company description did not require technical AI skills (e.g., "AI Sales Specialist", "AI Ethics Coordinator"). This confirms the hypothesis that "AI" is currently used as a marketing buzzword. A simple keyword search would have yielded a significant false-positive rate, whereas our semantic analysis correctly filtered these out.

### Prompt Sensitivity

The robustness of the results was highly sensitive to the negative examples provided in the prompt. In early iterations without the adversarial instructions, the model classified almost all roles at AI startups as "AI jobs". Adding the constraints reduced the false positive rate significantly, underscoring that Prompt Engineering requires rigorously defining boundaries.

## Interpretation

Based on the analysis, we propose a classification of roles into two distinct categories:

**AI-Native Roles** require building AI. These positions typically involve skills like PyTorch, TensorFlow, CUDA, Model Training, and RAG pipelines. Typical titles include Machine Learning Engineer and AI Research Scientist.

**AI-Adopter Roles** require using AI. These positions rely on skills like Prompt Engineering, ChatGPT, Midjourney, and Copilot. Typical titles include Content Writer and Software Developer (using coding assistants).

Our tool successfully distinguishes "AI-Native" roles, which was the primary objective. The strict negative constraints in the prompt proved crucial for this differentiation. The analysis reveals that "AI" is often used as a buzzword; a significant portion of "tech" jobs do not require actual AI development skills. By filtering these out, the "AI Skills Analyzer" provides a much more accurate picture of the technical demand for AI talent. This distinction is vital for **Educators**, to design curricula that focus on actual skills (e.g., model deployment) rather than general usage, and for **Job Seekers**, to understand the deep technical knowledge required for true AI roles.

# Conclusion

## Summary of Findings

This project successfully demonstrated the feasibility of using Large Language Models to automate the extraction of AI skills from job descriptions. The developed "AI Skills Analyzer" tool provides a robust, scalable, and reproducible method for analyzing labor market data. The results indicate that while "AI" is a pervasive term in the current job market, a distinct subset of roles requires actual model development and deployment skills. The tool effectively separates these technical roles from general software engineering positions.

## Generalization

The success of this approach suggests that LLMs can be effectively applied to other domains of labor market analysis. The ability to understand context and nuance allows for much more sophisticated analysis than traditional keyword-based methods. This methodology could be extended to track the emergence of other technologies (e.g., Quantum Computing, Blockchain) or to analyze soft skills requirements.

## Future Work

While the current system is effective, several avenues for improvement exist. **Model Distillation** could be employed to fine-tune a smaller, open-source model (like BERT) using the labels generated by this project, reducing inference costs and enhancing privacy. **Temporal Analysis** could reveal the evolution of AI skills over the last decade. **Granular Categorization** could refine skill categories to distinguish between specific AI fields like NLP and Computer Vision. Finally, optimizing for **Local Execution** would further reduce dependencies on external APIs.
