# AI Skills Analyzer

> **Thesis Context**: This project is part of a master's thesis in Economics. The main goal is to analyze current skills wanted in US vs EU job postings to compare the respective job markets.

Evaluate job descriptions for AI-related skills using a reproducible CLI workflow. The system uses a **Hybrid Extraction Strategy** combining deterministic dictionary matching (for speed and consistency) with OpenAI-powered extraction (for context and nuance).

## 1. Skill Taxonomy

The system sorts extracted skills into 9 semantic families to ensure cleaner data analysis:

- **Generative AI & LLMs** (`LangChain`, `OpenAI`, `RAG`, `Pinecone`...) _[NEW]_
- **Programming** (`Python`, `Java`, `Go`...)
- **Data & Cloud** (`AWS`, `SQL`, `Snowflake`...)
- **Analytics** (`Data Science`, `PowerBI`, `Machine Learning`...)
- **Software Engineering** (`CI/CD`, `Agile`, `Git`...)
- **Security** (`CISSP`, `Network Security`...)
- **Integration** (`REST API`, `GraphQL`...)
- **Certifications** (`AWS Certified`, `PMP`...)
- **UI & Tools** (`React`, `Figma`...)

## 2. Core Workflow

### 1. Update Data (The "Analyst" Flow)

Run `analyze` to process text, extract skills, and **automatically update the Embedding Cache** for the web dashboard. This single command handles the entire data lifecycle.

```bash
python -m ai_skills.cli analyze --input-csv data/inputs/us_relevant_30.csv
```

**Parameters:**

- `--input-csv <path>`: (Required) Raw CSV file to process.
- `--output-csv <path>`: (Optional) Custom output path. Defaults to `data/outputs/<stem>_ai.csv`.
- `--no-progress`: Disable the progress bar (useful for CI/CD logs).

### 2. View Interactive Dashboard (The "Explorer" Flow)

Launch the web server to explore the Skill Universe (2D Semantic Map) and search for jobs semantically.

```bash
python -m ai_skills.server
```

_Open http://localhost:8000/ to view the API, or your frontend client to see the graph._

### 3. Generate Reports (The "Validator" Flow)

To create static assets or compare versions.

**Visualize Skills**
Generate a static t-SNE scatter plot of extracted skills.

```bash
python -m ai_skills.cli visualize-skills --input-csv data/outputs/us_relevant_30_ai.csv
```

**Parameters:**

- `--input-csv <path>`: CSV containing a `skills` column.
- `--output-image <path>`: Path for the PNG (Default: `skill_map.png`).

**Evaluate Improvements**
Compare new extraction against a baseline to benchmark changes (e.g., did we find more skills?).

```bash
python -m ai_skills.cli evaluate --baseline data/baseline.csv --candidate data/new_output.csv
```

**Parameters:**

- `--baseline <path>`: Previous version of the data (Ground Truth or "Last Good").
- `--candidate <path>`: New version to evaluate.
- `--no-chart`: Skip generating the comparison PNG.
- `--min-match-rate <float>`: Alert threshold for matching jobs (Default: 0.85).

## 4. Setup

This project uses `uv` for blazing fast dependency management.

1.  **Install `uv`** (if you haven't already):

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Install Dependencies**:

    ```bash
    uv sync
    # Or classic pip: pip install -r requirements.txt
    ```

3.  **Create `.env` file**:

    ```bash
    OPENAI_API_KEY=sk-...
    ```

4.  **Run Commands**:
    Prefix commands with `uv run` to ensure they run in the correct environment.

    ```bash
    uv run python -m ai_skills.cli analyze ...
    ```

5.  **Configuration**:
    Adjust `config/settings.toml` to change default paths or model parameters. Creates `config/settings.local.toml` for local overrides.

## 4. Utilities & Advanced Commands

### `show_csv.py`

A colorized terminal viewer for your CSV data.

```bash
./show_csv.py data/outputs/us_relevant_30_ai.csv -c title tier hardskills
```

### Data Preparation

**`prepare-inputs`**
Create a smaller sample CSV for testing/grading.

```bash
python -m ai_skills.cli prepare-inputs --rows 100
```

- `--rows <int>`: Number of rows to sample (Default: 100).
- `--source <path>`: Source master CSV (Default: `data/inputs/us_relevant.csv`).
- `--destination <path>`: Output path.

### Vector Operations

**`search`**
Perform a semantic search query against the local vector store.

```bash
python -m ai_skills.cli search "remote developer with python" --limit 10
```

**`cluster`**
Group raw skills into synonyms using K-Means/Vectors (useful for ad-hoc analysis).

```bash
python -m ai_skills.cli cluster --input-csv data/outputs/file.csv
```

**`classify`**
Test the Zero-Shot Classifier on a text string.

```bash
python -m ai_skills.cli classify "Senior Python Developer" --labels "Junior" "Senior" "Manager"
```

**`index-skills`**
Scan a massive CSV to build embeddings without running the full OpenAI pipeline (Offline mode).

```bash
python -m ai_skills.cli index-skills --input-csv data/inputs/huge_file.csv
```
