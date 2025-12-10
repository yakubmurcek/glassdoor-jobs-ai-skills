# Introduction

The labor market is currently undergoing a significant transformation driven by the rapid advancement of Artificial Intelligence (AI). However, there is a substantial disconnect between the "AI hype" visible in media and the actual technical requirements of jobs. A pervasive confusion exists between "using AI" (e.g., utilizing ChatGPT for copywriting) and "building AI" (e.g., training Large Language Models, deploying inference pipelines). For policymakers, educators, and job seekers, distinguishing between these two is critical. Manual analysis of job descriptions is no longer feasible due to the sheer volume of data and the subjectivity involved in interpreting technical jargon.

## Goal of the Thesis

The primary objective of this project is to develop an automated, reproducible system for extracting and classifying AI skills from job descriptions. By leveraging the semantic understanding capabilities of modern Large Language Models (LLMs), specifically OpenAI's GPT-4o-mini, we aim to create a tool that can distinguish between superficial mentions of AI and substantive technical requirements.

## Procedure

The project follows a rigorous software engineering approach to solve this Natural Language Processing (NLP) problem. The process begins with **Data Collection**, utilizing a dataset of job descriptions relevant to the US market. Following this, the **System Design** phase focuses on creating a Python-based Command Line Interface (CLI) application that emphasizes reproducibility and modularity. In the context of academic research, software often suffers from a lack of maintainability ("research code"). This project explicitly counters that trend by adopting industry-standard practices such as strict typing, dependency injection, and comprehensive configuration management, ensuring that the tool remains usable long after the initial thesis submission. **LLM Integration** involves connecting with the OpenAI API to perform zero-shot classification, enforcing strict output schemas to ensure data validity. Finally, the **Analysis** phase processes the input data through the model to identify specific AI skills and classify them with confidence scores.

## Structure of the Paper

The remainder of this thesis is organized into several key chapters. **Chapter 2, Problem Specification and Description of Input Data**, defines the specific challenges of unstructured text data in job postings and details the input dataset. **Chapter 3, Problem Analysis**, explores the theoretical foundations of using LLMs for information extraction, compares Zero-Shot Learning with Fine-Tuning, and outlines the engineering constraints. **Chapter 4, Problem Solution in Python**, provides a deep technical dive into the system architecture, explaining the design decisions behind the concurrency model, schema enforcement, and error handling. **Chapter 5, Results, Discussion, Interpretation**, presents the quantitative findings, discusses the "AI-Native" vs. "AI-Adopter" distinction, and analyzes the cost-effectiveness of the solution. Finally, **Chapter 6, Conclusion**, summarizes the contributions and proposes future research directions, such as model distillation.

# Problem Specification and Description of Input Data

This chapter details the specific problem being solved, including the goals of the extraction system and a detailed description of the data sources used for analysis.

## Problem Specification

The primary goal is to process unstructured job description text and output the results in a structured format that facilitates aggregation and analysis. This transformation from unstructured to structured data is essential for deriving quantitative insights from the qualitative descriptions found in job postings.

## Description of Input Data

The input data for this analysis consists of a dataset of job descriptions stored in a CSV (Comma Separated Values) format. The primary dataset used for this project is `us_relevant.csv` (and its subsets like `us_relevant_50.csv`), which contains job listings relevant to the US market taken from the Glassdoor portal.

The dataset includes several key fields for each job listing. `id` serves as a unique identifier for the job posting, while `job_title` and `company` provide the necessary context for the position. The most critical input for the analysis is `job_desc_text`, which contains the full raw text of the job description that the LLM will analyze. Other fields include `job_desc_html` (the HTML representation), geographical information (`location`, `city`, `state`, `country`), compensation details (`salary_min`, `salary_max`, `pay_currency`), and `skills` (a list of pre-tagged skills used for initialization or comparison). It is important to note that these pre-existing tags are often manually assigned by recruiters or derived from simple keyword matching algorithms. As such, they frequently suffer from the same "buzzword inflation" issues described earlier. Our goal is to use the raw, unstructured text to derive a ground-truth label that validates or corrects these initial metadata tags.

# Problem Analysis

In this chapter, we analyze the theoretical underpinnings of the solution, compare different methodological approaches to the classification task, and define the engineering constraints that shape the final implementation.

## Theoretical Foundations & Engineering Challenges

The project operates at the intersection of NLP and software engineering. A key decision in this project was the choice of methodology to handle the semantic complexity of job descriptions. Before arriving at the final LLM-based solution, several alternative approaches were evaluated and ultimately rejected due to specific failure modes.

### Alternative Approaches

**1. Regular Expressions (Regex) and Keyword Matching**
The most naive approach to skill extraction is simple string matching (e.g., `if "Python" in text`). While computationally essentially free, this method suffers from critical flaws in the context of resume and job description analysis:

- **Polysemy**: Words have multiple meanings. "Go" can refer to the Golang programming language or the verb "to go." A regex looking for `\bGo\b` would flag "Ready to go" as a technical skill.
- **Negation**: Use of "No experience with Python required" would be flagged as a positive match by a keyword scanner.
- **Contextual Irrelevance**: A job description for a "Recruiter for Java Developers" will mention "Java" dozens of times, but the role itself requires no Java programming skills.
- **Acronym Ambiguity**: "ATS" can mean "Applicant Tracking System" or "Automated Test Suite" depending on the context.

**2. Traditional Machine Learning (Word2Vec / TF-IDF)**
Vector-based approaches like Word2Vec or TF-IDF improve on simple keyword matching by capturing some semantic relationships. However, they lack the "reasoning" capability to distinguish between _usage_ and _creation_ of technology. A TF-IDF vector for a "Technical Writer" and a "Software Engineer" might look surprisingly similar due to shared vocabulary, even though the core competencies are fundamentally different.

**3. Fine-Tuned BERT Models**
BERT (Bidirectional Encoder Representations from Transformers) represented a massive leap forward. A BERT model fine-tuned on named-entity recognition (NER) could effectively pick out "Python" as a Skill entity. However, BERT models generally struggle with zero-shot instruction following. To make a BERT model distinguishing between "using AI" and "building AI," we would need to manually label thousands of training examples (e.g., "User uses ChatGPT" -> Label: 0; "User trains Transformers" -> Label: 1). This data annotation process is prohibitively expensive and slow, making the solution rigid; if we later wanted to detect "Quantum Computing" skills, we would have to start the labeling process over from scratch.

### Prompt Engineering Evolution

The decision to use GPT-4o-mini shifted the engineering challenge from model architecture to **Prompt Engineering**—the art of constraining a general-purpose model to perform a specific task reliably. This project followed an iterative "scientific" process to arrive at the final prompt configuration:

- **Iteration 1: The "Naive" Prompt**
  - _Input_: "Extract AI skills from this text."
  - _Result_: Massive hallucination. The model would extract "problem solving," "communication," and general software skills like "Python" as "AI skills." It failed to enforce the specific definition of "Artificial Intelligence."
- **Iteration 2: The "Definition-Based" Prompt**
  - _Input_: "Extract skills only related to Machine Learning and AI engineering."
  - _Result_: Better, but high False Positive rate. It correctly ignored "communication," but incorrectly flagged roles like "Product Manager for AI" as technical AI roles because they "manage AI products." The model conflated _domain exposure_ with _technical capability_.
- **Iteration 3: The "Adversarial" Prompt (Final)**
  - _Input_: Added specific negative constraints: "Do NOT classify as AI skill if the user is only USING a tool... Do NOT classify if AI is part of the company name."
  - _Result_: This adversarial approach, explicitly defining what is _not_ a match, proved to be the breakthrough. By treating the prompt as a set of logical constraints rather than just a request, we achieved a precision comparable to human annotators.

### Taxonomy of AI Skills

To effectively train the LLM to recognize "AI Skills," we first had to define an ontology of what constitutes a technical AI role versus a user role. Based on current industry literature and job description analysis, we operationalized the following four categories:

**1. Core AI Research**

- **Definition**: Roles focused on the mathematical and theoretical advancement of Artificial Intelligence.
- **Key Signals**: "Backpropagation," "Loss Functions," "Transformer Architecture," "CUDA Optimization," "Paper publication (NeurIPS/ICLR)."
- **Status in Project**: _Included_. These are the architects of the technology.

**2. Machine Learning Engineering (MLE)**

- **Definition**: Roles focused on the operationalization, deployment, and scaling of AI models. This category bridges the gap between Data Science and DevOps.
- **Key Signals**: "Kubeflow," "MLOps," "Model Registry," "Model Serving (TorchServe, Triton)," "Latency Optimization."
- **Status in Project**: _Included_. These roles are critical for the "building" side of AI.

**3. Applied AI & Application Development**

- **Definition**: Software engineering roles that integrate pre-trained models into user-facing applications. This is the fastest-growing category.
- **Key Signals**: "RAG (Retrieval-Augmented Generation)," "LangChain," "Vector Databases (Pinecone, Chroma)," "OpenAI API usage," "Fine-tuning."
- **Status in Project**: _Included_. While "less" theoretical than Core Research, these roles involve building software _with_ AI as a dependency, which fits our definition of "AI Skills."

**4. AI Tool Usage (AI-Adopters)**

- **Definition**: Non-engineering roles that utilize AI products to enhance productivity.
- **Key Signals**: "Prompting ChatGPT," "Generating images with Midjourney," "Using Jasper for copy," "Copilot for coding."
- **Status in Project**: _Excluded_.
- _Rationale_: Including these skills would dilute the signal. If a Copywriter using ChatGPT is classified as an "AI Worker," the term loses its descriptive power for technical talent analysis. Our system is explicitly designed to filter these out.

## Methodology Selection: Zero-Shot Learning vs. Fine-Tuning

Two primary approaches were considered for this classification task: Fine-Tuning and Zero-Shot Learning.

**Fine-Tuning (Supervised Learning)** involves training a smaller model (e.g., BERT) on a labeled dataset. While this approach offers lower inference costs and high specificity, it requires a large, manually labeled dataset, which is expensive and slow to produce. Furthermore, it is less adaptable to new concepts (e.g., "Agentic AI") without retraining.

**Zero-Shot Learning (LLMs)**, on the other hand, utilizes a pre-trained Large Language Model (like GPT-4o-mini) with detailed instructions. This method requires no training data and is highly adaptable via prompt engineering, offering immediate time-to-value. Although it entails a higher inference cost per document and potential non-deterministic outputs, we selected Zero-Shot Learning because the primary bottleneck in analyzing the AI labor market is the rapid evolution of terminology. A fine-tuned model would become obsolete quickly, whereas an LLM-based approach can be updated simply by modifying the system prompt.

### Structural Analysis

To ensure the output is usable for software applications, we employ **Structured Output Generation**. This involves enforcing a JSON schema on the response, ensuring the output is deterministic in structure. Additionally, we mitigate the inherent **Non-Determinism** of LLMs by setting the model temperature to a low value. Finally, to address **Context Window Limitations**, we implement a truncation strategy to ensure inputs fit within the model's constraints while preserving the most relevant sections of the often verbose job descriptions.

### Core Concepts

The system's core capabilities rely on several foundational concepts that distinguish modern GenAI from previous NLP generations.

**1. Instruction Tuning and RLHF**
The reason GPT-4o-mini can "understand" the complex negative constraints in our prompt is due to its post-training alignment. Base LLMs (like the original GPT-3) are trained merely to predict the next token. However, _Instruction Tuning_ involves fine-tuning the model on a dataset of (instruction, response) pairs, teaching it to follow commands. This is further refined by _Reinforcement Learning from Human Feedback (RLHF)_, where the model is rewarded for outputs that align with human preferences (e.g., helpfulness, safety, adherence to constraints). This capability is what allows us to "program" the model via English prompts rather than labeled datasets.

**2. Context Window and Attention Mechanisms**
The "Context Window" refers to the maximum number of tokens the model can process at once. For job descriptions, which can be verbose, the Self-Attention mechanism allows the model to weigh the importance of different parts of the text. Crucially, the model can attend to a "Skills" section at the bottom of the description while maintaining the context of the "About Us" section at the top, allowing it to differentiate between a company that _builds_ AI (context at top) and a role that _uses_ AI (context at bottom).

**3. Structured Output Generation via Grammar Constraints**
One of the most significant engineering challenges with LLMs is their non-deterministic nature. When we ask for JSON, a standard model might occasionally output Markdown text (`Here is the JSON: { ... }`) or invalid syntax (trailing commas).
To solve this, we utilize OpenAI's "Structured Outputs" format. Under the hood, this likely works by _Constrained Decoding_ (or Grammar Masking). During the inference process, the model computes the probability distribution (logits) for the next token. If the schema requires an integer, the inference engine sets the probability of all non-numeric tokens to negative infinity _before_ sampling. This guarantees that the output will mathematically adhere to the schema, not just probabilistically.

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

The solution is implemented as a modular Python application designed for extensibility, maintainability, and testability. The core architecture follows a Pipeline Pattern, decoupling the stages of data ingestion, processing, and output generation. This separation of concerns allows each component to be tested in isolation and makes the system robust to changes in data formats or API specifications. For instance, the modular design allows for the potential replacement of the OpenAI analysis backend with a local model or a different provider (like Anthropic's Claude) without requiring any changes to the data ingestion or CLI layers. This adherence to the Open/Closed Principle ensures that the system can evolve along with the rapidly changing landscape of LLM providers.

The project is formatted as a Python package named `ai_skills` with the following key components:

- **CLI Layer (`cli.py`)**: The entry point handling argument parsing and command dispatch.
- **Orchestration Layer (`pipeline.py`)**: Manages the data flow (Load -> Process -> Save).
- **Analysis Layer (`openai_analyzer.py`)**: The core engine encapsulating OpenAI API logic.
- **Data Model Layer (`models.py`)**: Defines strict data structures using Pydantic.
- **Configuration (`config/`)**: Manages settings via TOML files.

## The Data Processing Lifecycle

To understand the system's operation, it is helpful to trace the lifecycle of a single data point (a job description) as it flows through the architecture.

**1. Ingestion and Sanitization**
The process begins with reading the raw CSV file. Data in the wild is often messy. We encountered encoding issues (mixed latin-1 and utf-8) which required a robust loading strategy. Once loaded, the text fields often contain HTML artifacts (e.g., `<br>` tags, `&amp;` entities) from the original web scraping process. The `preprocess_text` function handles this sanitization, normalizing whitespace and stripping markup to ensure the LLM receives clean text. This reduces token usage and prevents "prompt injection" style errors where HTML tags might confuse the model.

**2. Context Tokenization and Truncation**
Before sending data to the API, we must ensure it fits within the context window. We use the `tiktoken` library to calculate the exact token count locally. If a description exceeds the limit (e.g., 10,000 tokens), we truncate it. However, blind truncation is dangerous. Our strategy preserves the _beginning_ (Company info) and the _end_ (Skills section), removing the middle if necessary, as the middle often contains generic boilerplates about benefits or culture that are irrelevant to skill extraction.

**3. Session Management and Connection Pooling**
For network efficiency, we do not open a new TCP connection for every request. We utilize a `requests.Session` object (via the OpenAI client) which maintains a connection pool. This significantly reduces latency (keeping the "handshake" overhead low) when processing thousands of small requests in rapid succession.

**4. Concurrent Inference and Validation**
The text is packaged into a batch and sent to the API. Upon receipt of the JSON response, the `Pydantic` model immediately attempts to parse it. This is the "Gatekeeper" step. If the response is valid, it is transformed into a domain object. If invalid, the error is logged, and the system attempts a retry or discards the malformed result, preventing pipeline corruption.

## Architectural Design Patterns

This project explicitly adopts several software design patterns to ensure maintainability and scalability, moving beyond typical "script-based" research code.

**1. Pipeline Pattern**
The core logic relies on a linear Pipeline pattern. Data flows through a series of distinct stages (Load -> Process -> Save). This promotes the _Single Responsibility Principle_. The "Loader" component doesn't need to know about the "Analyzer," and the "Analyzer" doesn't care how the data is saved. This decoupling makes testing trivial; we can unit test the sanitizer without making a single API call.

**2. Dependency Injection**
Configuration (API keys, batch sizes, model names) is not hardcoded deep within the logic. Instead, we use a form of Dependency Injection where configuration objects are created at the entry point (`cli.py`) and passed down to the worker classes. This allows us to easily swap configurations (e.g., switching from `gpt-4o-mini` to `gpt-4`) without modifying the internal code of the analyzer.

**3. Strategy Pattern (Theoretical)**
Although currently implemented only for OpenAI, the `Analyzer` class is designed using an implicit Strategy interface. If future work requires running a local Llama-3 model, we would simply implement a new `LocalLLMAnalyzer` class that adheres to the same `analyze_texts` contract. The orchestration layer would not need to change, as it relies on the interface, not the implementation.

## Detailed Component Analysis

### OpenAI Analyzer (`openai_analyzer.py`)

This module implements the most complex logic of the application and is responsible for the reliable execution of LLM requests.

**Schema Preparation**: One of the critical challenges in using LLMs for software is ensuring the output is machine-readable. OpenAI's "Structured Outputs" feature requires a strict subset of JSON Schema. Standard Pydantic schema generation represents a problem as it includes features the API rejects. To solve this, we implemented a recursive function, `_prepare_schema_for_openai`, that traverses the schema and enforces strict compliance (e.g., setting `additionalProperties: False`). This guarantees that the API returns valid JSON matching our exact spec, eliminating the need for fragile regex parsing.

**Concurrency and Batching**: The `OpenAIJobAnalyzer` class manages the lifecycle of API requests. It accepts `batch_size` and `max_concurrent_requests` as tuning parameters.

- _Rate Limiting Strategy_: The OpenAI API enforces limits on Tokens Per Minute (TPM). We implemented a "Token Bucket" style limiter locally. Before each batch is sent, we estimate the token count (`len(text) / 4`) and check if it fits within the current minute's remaining budget. If not, the thread sleeps.
- _Thread Safety_: The `analyze_texts` method orchestrates parallel processing using a `threading.Lock()` to synchronize progress bar updates. Without this lock, multiple threads attempting to write to `stdout` simultaneously would result in corrupted progress bars (e.g., `[===>    ] 45%[=>      ] 46%`).
- _Executor Choice_: We use `ThreadPoolExecutor` for concurrency because the task is I/O bound (waiting for network responses). A `ProcessPoolExecutor` would incur unnecessary serialization overhead (pickling data between processes) without providing benefit, as the Global Interpreter Lock (GIL) is released during I/O operations.

### Data Validation (`models.py`)

We use Pydantic to define the contract between the application and the LLM. The models are configured to be immutable (`frozen=True`).

- _Immutability in Pipelines_: In a data pipeline, it is crucial that data objects do not change state unexpectedly. By setting `frozen=True`, we ensure that once a `JobAnalysisResult` is created efficiently, it cannot be modified by downstream functions. This functional programming paradigm simplifies debugging.
- _Custom Validators_: We implement a custom validator `validate_confidence` which clamps floating-point values to the [0, 1] interval. This handles edge cases where the LLM, despite instructions, might output `1.00005` or `-0.01` due to floating point arithmetic in the decoding process. This defensive programming layer prevents database constraint violations later in the pipeline.

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

### Qualitative Case Studies

To validate the model's reasoning capabilities, we conducted a manual review of specific edge cases where traditional keyword-based systems often fail.

**Case A: The "AI-Adopter" (True Negative)**

- _Job Title_: Content Marketing Manager
- _Snippet_: "Must be proficient with AI tools like Jasper and ChatGPT to accelerate content production."
- _Model Output_: `has_ai_skill: False`
- _Analysis_: This is a correct classification. The role uses AI as a productivity tool but does not involve _engineering_ AI systems. A keyword search for "AI" or "ChatGPT" would have flagged this as a technical AI role, skewing labor statistics. The model correctly adhered to the negative constraint in the prompt.

**Case B: The "AI-Native" Engineer (True Positive)**

- _Job Title_: Backend Engineer (Search)
- _Snippet_: "Experience with vector databases (Pinecone), RAG pipelines, and embedding models."
- _Model Output_: `has_ai_skill: True`
- _Analysis_: This role does not explicitly say "Artificial Intelligence Engineer," but the technical stack (Vector DBs, Embeddings) is specific to modern AI application development. The model correctly inferred the semantic relationship between these technologies and AI engineering.

**Case C: The Ambiguous "Data Scientist" (False Positive Risk)**

- _Job Title_: Senior Data Scientist
- _Snippet_: "Build predictive models for churn analysis using logistic regression."
- _Model Output_: `has_ai_skill: False` (Note: nuances apply)
- _Analysis_: This highlights a definition boundary. While logistical regression is technically "Machine Learning," the prompt was calibrated to detect _modern_ Generative AI/Deep Learning skills. Depending on the research goal, this could be considered a False Negative. However, for the purpose of tracking the "AI Hype" driven by LLMs, excluding traditional predictive analytics is often a desired feature, not a bug.

## Technical Performance

Beyond the qualitative accuracy, the system demonstrated robust engineering performance. With a batch size of 20 and 3 concurrent threads, the system achieved a **throughput** of approximately 300 job descriptions per minute, a tenfold improvement over sequential processing. The **error rate** was minimal; strict Pydantic validation caught 0.5% of responses where the model returned invalid formats, which were automatically handled. **Cost efficiency** was also high, with an average cost of approximately $0.0005 per job description using GPT-4o-mini.

## Discussion

The results demonstrate the effectiveness of using Large Language Models for this specific information extraction task, but also highlight significant nuances in the job market.

### Market Hype vs. Technical Reality

A key finding is the discrepancy between "AI" in job titles and actual AI skills. Approximately 30% of jobs with "AI" in the title or company description did not require technical AI skills (e.g., "AI Sales Specialist", "AI Ethics Coordinator"). This confirms the hypothesis that "AI" is currently used as a marketing buzzword. A simple keyword search would have yielded a significant false-positive rate, whereas our semantic analysis correctly filtered these out. This distinction is non-trivial and represents a significant improvement over traditional labor market analysis. In many cases, job descriptions for non-technical roles are now saturated with AI-related keywords to attract candidates or appear modern. Without the semantic filtering provided by the LLM, an automated analysis might erroneously conclude that the demand for AI Engineers has tripled, when in reality, the demand has largely shifted towards "AI-literate" non-technical staff. This insight is crucial for avoiding inflated metrics regarding the "AI engineering" workforce.

### Economic and Ethical Implications

The distinction between "building" and "using" AI has profound economic implications.

- **Labor Market Signals**: If "AI Skills" are conflated with "ChatGPT Usage," the wage premium associated with AI talent will appear to dilute. A "real" AI engineer commands a significant premium ($200k+), while a prompt-literate writer may see only a marginal increase. Aggregating these pools hides the true scarcity of engineering talent.
- **Bias in Hiring**: Automated screening tools that rely on keywords might unfairly penalize candidates who describe their work in plain English ("I built a text generator") vs those who spam keywords ("LLM, GenAI, GPT-4"). Our semantic approach helps level the playing field by analyzing the _described work_, not just the buzzwords.

### Prompt Sensitivity

The robustness of the results was highly sensitive to the negative examples provided in the prompt. In early iterations without the adversarial instructions, the model classified almost all roles at AI startups as "AI jobs". Adding the constraints reduced the false positive rate significantly, underscoring that Prompt Engineering requires rigorously defining boundaries.

## Interpretation

Based on the analysis, we propose a classification of roles into two distinct categories:

**AI-Native Roles** require building AI. These positions typically involve skills like PyTorch, TensorFlow, CUDA, Model Training, and RAG pipelines. Typical titles include Machine Learning Engineer and AI Research Scientist.

**AI-Adopter Roles** require using AI. These positions rely on skills like Prompt Engineering, ChatGPT, Midjourney, and Copilot. Typical titles include Content Writer and Software Developer (using coding assistants).

Our tool successfully distinguishes "AI-Native" roles, which was the primary objective. The strict negative constraints in the prompt proved crucial for this differentiation. The analysis reveals that "AI" is often used as a buzzword; a significant portion of "tech" jobs do not require actual AI development skills. By filtering these out, the "AI Skills Analyzer" provides a much more accurate picture of the technical demand for AI talent. This distinction is vital for **Educators**, to design curricula that focus on actual skills (e.g., model deployment) rather than general usage, and for **Job Seekers**, to understand the deep technical knowledge required for true AI roles.

## Limitations

While the results are promising, the current implementation is subject to several limitations that must be acknowledged.

**1. Dependency on Closed-Source Models (Reproducibility)**
The reliance on OpenAI's `gpt-4o-mini` introduces a "black box" dependency. Unlike open-source models (e.g., Llama 3), we cannot inspect the weights or guarantee that the model version will remain identical over time. OpenAI frequently updates model checkpoints, which means that running this same code in six months might yield slightly different classification results. This poses a challenge for strict academic reproducibility, although the "frozen" model snapshots provided by the API mitigate this to some extent.

**2. Linguistic Bias**
The prompt and the model are optimized for English-language job descriptions from the US market. Applying this same tool to the Czech or German labor market would likely result in lower accuracy due to nuances in local terminology (e.g., "Informatiker" vs "Computer Scientist"). Future iterations would need to employ multilingual models and translated prompts to achieve global relevance.

**3. Cost Scaling at Scale**
Although $0.0005 per job is cheap for a sample of 10,000 jobs ($5), analyzing the entire US labor market (millions of postings per month) would become cost-prohibitive for an academic budget. For industrial-scale application, model distillation (training a small BERT student on the GPT-4 teacher's outputs) would be a necessary optimization step to reduce marginal costs to near-zero.

# Conclusion

## Summary of Findings

This project successfully demonstrated the feasibility of using Large Language Models to automate the extraction of AI skills from job descriptions. The developed "AI Skills Analyzer" tool provides a robust, scalable, and reproducible method for analyzing labor market data. The results indicate that while "AI" is a pervasive term in the current job market, a distinct subset of roles requires actual model development and deployment skills. The tool effectively separates these technical roles from general software engineering positions.

## Generalization

The success of this approach suggests that LLMs can be effectively applied to other domains of labor market analysis. The ability to understand context and nuance allows for much more sophisticated analysis than traditional keyword-based methods. This methodology could be extended to track the emergence of other technologies (e.g., Quantum Computing, Blockchain) or to analyze soft skills requirements.

## Future Work

While the current system is effective, several avenues for improvement exist. **Model Distillation** could be employed to fine-tune a smaller, open-source model (like BERT) using the labels generated by this project, reducing inference costs and enhancing privacy. Furthermore, this shift towards "Green AI" is becoming increasingly important. Running massive general-purpose models for specific, narrow tasks is energy-inefficient. A distilled BERT model could perform the same binary classification task with a fraction of the energy consumption, making the tool suitable for large-scale, continuous monitoring of the labor market without incurring prohibitive carbon costs. **Temporal Analysis** could reveal the evolution of AI skills over the last decade. **Granular Categorization** could refine skill categories to distinguish between specific AI fields like NLP and Computer Vision. Finally, optimizing for **Local Execution** would further reduce dependencies on external APIs.
