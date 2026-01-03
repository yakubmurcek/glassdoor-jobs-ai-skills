#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration settings for the AI skills analysis project.

Defaults now live in `config/settings.toml`, with optional overrides in
`config/settings.local.toml`. Values from those files take precedence over
environment variables or `.env`, except for credentials such as
`OPENAI_API_KEY`, which must come from the environment for security reasons.
This module still exposes three structured sections:

* `PATHS` - where data is read/written
* `OPENAI` - how the OpenAI client is configured
* `PROCESSING` - limits and column ordering for downstream steps

The legacy module-level constants (e.g., `INPUT_CSV`, `OPENAI_MODEL`) remain
available for compatibility with existing imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from .settings_loader import load_user_config

# Resolve project root and load environment variables
PACKAGE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PACKAGE_DIR.parent
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

USER_CONFIG = load_user_config(PROJECT_ROOT)


def _get_raw_setting(env_var: str) -> Optional[Any]:
    """Return the first non-None value among user overrides and env vars."""
    if env_var in USER_CONFIG:
        value = USER_CONFIG[env_var]
        if value is not None:
            return value
    value = os.getenv(env_var)
    if value is not None and value.strip() != "":
        return value.strip()
    return None


def _get_env_only_setting(env_var: str) -> Optional[str]:
    """Read sensitive values exclusively from the environment or .env."""
    value = os.getenv(env_var)
    if value is not None and value.strip() != "":
        return value.strip()
    return None


def _get_int_setting(env_var: str, default: int) -> int:
    """Safely parse integer settings from the environment."""
    value = _get_raw_setting(env_var)
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value)
    except ValueError:
        return default


def _get_float_setting(env_var: str, default: float) -> float:
    """Safely parse float settings from the environment."""
    value = _get_raw_setting(env_var)
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except ValueError:
        return default


def _get_path_setting(env_var: str, default: Path) -> Path:
    """Return a path overridden by env var if provided."""
    value = _get_raw_setting(env_var)
    if value:
        if isinstance(value, Path):
            return value.expanduser().resolve()
        return Path(str(value)).expanduser().resolve()
    return default


@dataclass(frozen=True)
class PathSettings:
    """All file-system locations used by the pipeline."""

    project_root: Path
    data_dir: Path
    inputs_dir: Path
    outputs_dir: Path
    input_csv: Path
    output_csv: Path

    @classmethod
    def build(cls, project_root: Path) -> "PathSettings":
        data_dir = _get_path_setting("DATA_DIR", project_root / "data")
        inputs_dir = _get_path_setting("INPUTS_DIR", data_dir / "inputs")
        outputs_dir = _get_path_setting("OUTPUTS_DIR", data_dir / "outputs")
        input_csv = _get_path_setting(
            "INPUT_CSV", inputs_dir / "us_relevant_50.csv"
        )
        output_csv = _get_path_setting(
            "OUTPUT_CSV", outputs_dir / "us_relevant_ai_50.csv"
        )
        return cls(
            project_root=project_root,
            data_dir=data_dir,
            inputs_dir=inputs_dir,
            outputs_dir=outputs_dir,
            input_csv=input_csv,
            output_csv=output_csv,
        )


# Available LLM tasks for decomposed mode
LLM_TASK_AI_TIER = "ai_tier"
LLM_TASK_SKILLS = "skills"
LLM_TASK_EDUCATION = "education"

# Default: education disabled (can be done deterministically)
DEFAULT_ENABLED_LLM_TASKS: frozenset[str] = frozenset({LLM_TASK_AI_TIER, LLM_TASK_SKILLS})


def _get_set_setting(env_var: str, default: frozenset[str]) -> frozenset[str]:
    """Parse comma-separated list into a frozenset."""
    value = _get_raw_setting(env_var)
    if value is None:
        return default
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(value)
    # Parse comma-separated string
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return frozenset(items) if items else default


@dataclass(frozen=True)
class LLMSettings:
    """Options that control LLM API usage (OpenAI or Ollama)."""

    provider: str  # "openai" or "ollama"
    api_key: Optional[str]
    base_url: Optional[str]
    model: str
    temperature: float
    rate_limit_delay: float
    batch_size: int
    max_parallel_requests: int
    enabled_tasks: frozenset[str]  # Which decomposed tasks use LLM

    @classmethod
    def build(cls) -> "LLMSettings":
        provider = (_get_raw_setting("MODEL_PROVIDER") or "openai").lower()
        
        if provider == "ollama":
            # Ollama: API key is optional (use dummy), base_url points to local server
            api_key = _get_env_only_setting("OPENAI_API_KEY") or "ollama"
            base_url = _get_raw_setting("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
            model = _get_raw_setting("OLLAMA_MODEL") or "gemma3:12b"
        else:
            # OpenAI: require API key, no custom base_url
            api_key = _get_env_only_setting("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is not set. "
                    "Please set it in your .env file or as an environment variable."
                )
            base_url = None
            model = _get_raw_setting("OPENAI_MODEL") or "gpt-4o-mini"

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=_get_float_setting("OPENAI_TEMPERATURE", 0.1),
            rate_limit_delay=_get_float_setting("RATE_LIMIT_DELAY", 0.1),
            batch_size=max(1, _get_int_setting("OPENAI_BATCH_SIZE", 20)),
            max_parallel_requests=max(
                1, _get_int_setting("OPENAI_MAX_PARALLEL_REQUESTS", 3)
            ),
            enabled_tasks=_get_set_setting("ENABLED_LLM_TASKS", DEFAULT_ENABLED_LLM_TASKS),
        )


# Backwards compatibility alias
OpenAISettings = LLMSettings


DEFAULT_COLUMN_ORDER: Tuple[str, ...] = (
    # Source data
    "skills",
    "job_desc_text",
    # From skills column (deterministic)
    "skills_ai_det",
    "skills_hasai_det",
    # From job description (LLM)
    "desc_tier_llm",
    "desc_conf_llm",
    "ai_confidence",
    "desc_ai_llm",
    "desc_rationale_llm",
    # Key derived column
    "is_real_ai",
    # Final merged skills
    "hardskills",
    "softskills",
    "skill_cluster",
    # From educations column (deterministic)
    "edu_level_det",
    # Education requirement (LLM uses educations column + job description)
    "edureq_llm",
    # Metrics
    "ai_det_llm_match",
    # Intermediate: from job description (deterministic)
    "desc_hard_det",
    "desc_soft_det",
    # Intermediate: from job description (LLM)
    "desc_hard_llm",
    "desc_soft_llm",
)


# =============================================================================
# Display Configuration for show_csv.py
# =============================================================================
# Key = actual column name in DataFrame
# Value = (display_label, width, format_type)
# format_type: "text", "tier", "confidence", "boolean", "skills"

@dataclass(frozen=True)
class ColumnDisplay:
    """Configuration for how a column should be displayed."""
    label: str          # Display label in header
    width: int          # Column width
    format_type: str    # "text", "tier", "confidence", "boolean", "skills"


# Columns available for display - key IS the column name
DISPLAY_COLUMNS: Dict[str, ColumnDisplay] = {
    "job_title": ColumnDisplay("JOB TITLE", 40, "text"),
    "desc_tier_llm": ColumnDisplay("TIER", 14, "tier"),
    "desc_conf_llm": ColumnDisplay("CONF", 5, "confidence"),
    "ai_confidence": ColumnDisplay("AI_CONF", 5, "confidence"),
    "is_real_ai": ColumnDisplay("AI?", 4, "boolean"),
    "skills_hasai_det": ColumnDisplay("HAS_AI", 6, "boolean"),
    "ai_det_llm_match": ColumnDisplay("AGREE", 5, "boolean"),
    "edu_level_det": ColumnDisplay("EDU", 10, "text"),
    "edureq_llm": ColumnDisplay("EDU_REQ", 7, "boolean"),
    "hardskills": ColumnDisplay("HARDSKILLS", 50, "skills"),
    "softskills": ColumnDisplay("SOFTSKILLS", 30, "skills"),
    "skill_cluster": ColumnDisplay("SKILL FAMILIES", 80, "skills"),
    "skills_ai_det": ColumnDisplay("AI FROM SKILLS", 30, "skills"),
    "desc_ai_llm": ColumnDisplay("AI FROM LLM", 40, "skills"),
}

# Default columns to show (use actual column names)
DEFAULT_DISPLAY_COLS: Tuple[str, ...] = (
    "job_title", "is_real_ai", "desc_tier_llm", "desc_conf_llm", 
    "skills_hasai_det", "ai_det_llm_match", "edu_level_det", "edureq_llm",
    "skill_cluster", "softskills", "skills_ai_det", "desc_ai_llm",
)



@dataclass(frozen=True)
class ProcessingSettings:
    """Tunable processing limits and output ordering."""

    max_job_desc_length: int
    preferred_column_order: Tuple[str, ...]

    @classmethod
    def build(cls) -> "ProcessingSettings":
        return cls(
            max_job_desc_length=max(1, _get_int_setting("MAX_JOB_DESC_LENGTH", 8000)),
            preferred_column_order=DEFAULT_COLUMN_ORDER,
        )


@dataclass(frozen=True)
class AppConfig:
    """Aggregate configuration for consumers that prefer structured access."""

    paths: PathSettings
    llm: LLMSettings
    processing: ProcessingSettings

    # Backwards compatibility property
    @property
    def openai(self) -> LLMSettings:
        return self.llm


PATHS = PathSettings.build(PROJECT_ROOT)
LLM = LLMSettings.build()
OPENAI = LLM  # Backwards compatibility alias
PROCESSING = ProcessingSettings.build()
CONFIG = AppConfig(paths=PATHS, llm=LLM, processing=PROCESSING)

# Backwards-compatible module-level exports
DATA_DIR = PATHS.data_dir
INPUTS_DIR = PATHS.inputs_dir
OUTPUTS_DIR = PATHS.outputs_dir
INPUT_CSV = PATHS.input_csv
OUTPUT_CSV = PATHS.output_csv
MODEL_PROVIDER = LLM.provider
OPENAI_API_KEY = LLM.api_key
OPENAI_BASE_URL = LLM.base_url
OPENAI_MODEL = LLM.model
OPENAI_TEMPERATURE = LLM.temperature
RATE_LIMIT_DELAY = LLM.rate_limit_delay
OPENAI_BATCH_SIZE = LLM.batch_size
OPENAI_MAX_PARALLEL_REQUESTS = LLM.max_parallel_requests
ENABLED_LLM_TASKS = LLM.enabled_tasks
MAX_JOB_DESC_LENGTH = PROCESSING.max_job_desc_length
PREFERRED_COLUMN_ORDER = list(PROCESSING.preferred_column_order)

# List of AI-related skills (case-insensitive matching)
AI_SKILLS = [
    # Core AI / ML terms
    "ai",
    "a.i.",
    "artificial intelligence",
    "ml",
    "machine learning",
    "dl",
    "deep learning",
    "neural network",
    "neural networks",
    "neural net",
    "neural nets",
    "supervised learning",
    "unsupervised learning",
    "semi-supervised learning",
    "reinforcement learning",
    "rl",
    "self-supervised learning",
    "transfer learning",
    "meta-learning",
    "few-shot learning",
    "zero-shot learning",
    "one-shot learning",
    "multi-task learning",
    "continual learning",
    "online learning",
    "active learning",
    "federated learning",
    "classification",
    "regression",
    "clustering",
    "dimensionality reduction",
    "anomaly detection",
    "outlier detection",
    "predictive modeling",
    "predictive analytics",
    # Neural network architectures
    "cnn",
    "convolutional neural network",
    "rnn",
    "recurrent neural network",
    "lstm",
    "long short-term memory",
    "gru",
    "transformer",
    "transformers",
    "attention mechanism",
    "self-attention",
    "multi-head attention",
    "autoencoder",
    "variational autoencoder",
    "vae",
    "gan",
    "generative adversarial network",
    "diffusion model",
    "diffusion models",
    "stable diffusion",
    "latent diffusion",
    "u-net",
    "resnet",
    "vgg",
    "inception",
    "efficientnet",
    "mobilenet",
    "densenet",
    "vision transformer",
    "vit",
    "swin transformer",
    "perceiver",
    # GenAI / LLMs
    "llm",
    "llms",
    "large language model",
    "large language models",
    "large-language model",
    "foundation model",
    "foundation models",
    "gpt",
    "gpt-2",
    "gpt-3",
    "gpt-3.5",
    "gpt-4",
    "gpt-4o",
    "gpt4",
    "gpt3",
    "gpt5",
    "chatgpt",
    "openai",
    "anthropic",
    "claude",
    "claude 3",
    "gemini",
    "bard",
    "palm",
    "palm 2",
    "bert",
    "roberta",
    "distilbert",
    "albert",
    "electra",
    "deberta",
    "xlnet",
    "xlm",
    "t5",
    "flan-t5",
    "flan",
    "bart",
    "pegasus",
    "llama",
    "llama 2",
    "llama 3",
    "llama2",
    "llama3",
    "codellama",
    "code llama",
    "mistral",
    "mixtral",
    "phi",
    "phi-2",
    "phi-3",
    "falcon",
    "mpt",
    "dolly",
    "vicuna",
    "alpaca",
    "wizardlm",
    "orca",
    "qwen",
    "yi",
    "deepseek",
    "cohere",
    "command-r",
    "jurassic",
    "bloom",
    "opt",
    "megatron",
    "chinchilla",
    "gopher",
    "generative ai",
    "genai",
    "gen ai",
    "gen-ai",
    "generative model",
    "generative models",
    "text generation",
    "text-generation",
    "code generation",
    "code-generation",
    "prompt engineering",
    "prompt design",
    "prompt",
    "prompting",
    "chain of thought",
    "cot",
    "few-shot prompting",
    "zero-shot prompting",
    "in-context learning",
    "instruction tuning",
    "instruction-tuning",
    "rlhf",
    "reinforcement learning from human feedback",
    "constitutional ai",
    "dpo",
    "direct preference optimization",
    # RAG & Embeddings
    "rag",
    "retrieval augmented generation",
    "retrieval-augmented generation",
    "embedding",
    "embeddings",
    "text embedding",
    "text embeddings",
    "sentence embedding",
    "sentence embeddings",
    "semantic search",
    "semantic similarity",
    "vector search",
    "vector database",
    "vector db",
    "vectordb",
    "vector store",
    "pinecone",
    "weaviate",
    "milvus",
    "qdrant",
    "chroma",
    "chromadb",
    "faiss",
    "annoy",
    "pgvector",
    "elasticsearch vectors",
    "hybrid search",
    "chunking",
    "text chunking",
    "document chunking",
    "retrieval",
    "reranking",
    "reranker",
    "bm25",
    "colbert",
    "knowledge base",
    "knowledge graph",
    "knowledge management",
    # AI Agents & Orchestration
    "ai agent",
    "ai agents",
    "autonomous agent",
    "autonomous agents",
    "agentic",
    "agentic ai",
    "multi-agent",
    "multi-agent system",
    "agent orchestration",
    "langchain",
    "langraph",
    "langgraph",
    "llamaindex",
    "llama index",
    "llama-index",
    "semantic kernel",
    "autogen",
    "crewai",
    "crew ai",
    "autogpt",
    "auto-gpt",
    "babyagi",
    "haystack",
    "dspy",
    "guidance",
    "outlines",
    "lmql",
    "function calling",
    "tool use",
    "tool calling",
    "reasoning",
    "planning",
    "chain-of-thought",
    "react agent",
    "reflection",
    "self-reflection",
    # Fine-tuning & Training
    "fine-tuning",
    "fine tuning",
    "finetuning",
    "fine-tune",
    "finetune",
    "lora",
    "qlora",
    "peft",
    "adapter",
    "adapters",
    "prefix tuning",
    "prompt tuning",
    "parameter-efficient",
    "quantization",
    "pruning",
    "distillation",
    "knowledge distillation",
    "model compression",
    "mixed precision",
    "fp16",
    "bf16",
    "int8",
    "int4",
    "bitsandbytes",
    "deepspeed",
    "megatron-lm",
    "fsdp",
    "zero",
    "zero-3",
    "gradient checkpointing",
    "flash attention",
    "flashattention",
    "vllm",
    "tgi",
    "text generation inference",
    # AI Tools & Platforms
    "hugging face",
    "huggingface",
    "transformers library",
    "diffusers",
    "accelerate",
    "datasets",
    "tokenizers",
    "gradio",
    "streamlit",
    "chainlit",
    "weights and biases",
    "wandb",
    "comet ml",
    "neptune ai",
    "clearml",
    "replicate",
    "together ai",
    "anyscale",
    "modal",
    "banana",
    "runpod",
    "lambda labs",
    "paperspace",
    "vast ai",
    # AI Coding Tools
    "github copilot",
    "copilot",
    "codeium",
    "tabnine",
    "cursor",
    "cursor ai",
    "windsurf",
    "replit ai",
    "amazon codewhisperer",
    "codewhisperer",
    "sourcegraph cody",
    "cody",
    "continue",
    "aider",
    "gpt engineer",
    "devin",
    "ai coding",
    "ai code",
    "ai-assisted coding",
    "ai pair programming",
    # Vision / Image AI
    "computer vision",
    "cv",
    "image recognition",
    "image classification",
    "object detection",
    "object recognition",
    "image segmentation",
    "semantic segmentation",
    "instance segmentation",
    "panoptic segmentation",
    "pose estimation",
    "facial recognition",
    "face detection",
    "face recognition",
    "ocr",
    "optical character recognition",
    "scene understanding",
    "depth estimation",
    "3d vision",
    "point cloud",
    "lidar",
    "yolo",
    "yolov5",
    "yolov8",
    "retinanet",
    "faster r-cnn",
    "mask r-cnn",
    "rcnn",
    "r-cnn",
    "ssd",
    "detr",
    "sam",
    "segment anything",
    "dino",
    "clip",
    "openclip",
    "blip",
    "llava",
    "gpt-4v",
    "gpt-4 vision",
    "gemini vision",
    "multimodal",
    "multi-modal",
    "vision-language",
    "vision language model",
    "vlm",
    "image generation",
    "image synthesis",
    "text-to-image",
    "image-to-image",
    "img2img",
    "txt2img",
    "dall-e",
    "dalle",
    "dall-e 2",
    "dall-e 3",
    "midjourney",
    "stable diffusion xl",
    "sdxl",
    "controlnet",
    "imagen",
    "firefly",
    "leonardo ai",
    # Speech & Audio AI
    "speech recognition",
    "speech-to-text",
    "stt",
    "automatic speech recognition",
    "asr",
    "text-to-speech",
    "tts",
    "speech synthesis",
    "voice cloning",
    "voice synthesis",
    "whisper",
    "wav2vec",
    "hubert",
    "speaker diarization",
    "speaker recognition",
    "voice recognition",
    "audio processing",
    "audio classification",
    "music generation",
    "audio generation",
    "elevenlabs",
    "eleven labs",
    "bark",
    "tortoise tts",
    "coqui",
    "xtts",
    "voiceover ai",
    # NLP
    "nlp",
    "natural language processing",
    "nlu",
    "natural language understanding",
    "nlg",
    "natural language generation",
    "tokenization",
    "tokenizer",
    "word embeddings",
    "word2vec",
    "glove",
    "fasttext",
    "named entity recognition",
    "ner",
    "sentiment analysis",
    "text classification",
    "text summarization",
    "abstractive summarization",
    "extractive summarization",
    "question answering",
    "qa",
    "reading comprehension",
    "machine translation",
    "neural machine translation",
    "nmt",
    "language modeling",
    "language model",
    "causal language modeling",
    "masked language modeling",
    "seq2seq",
    "sequence to sequence",
    "encoder-decoder",
    "dependency parsing",
    "constituency parsing",
    "part-of-speech tagging",
    "pos tagging",
    "coreference resolution",
    "relation extraction",
    "information extraction",
    "text mining",
    "topic modeling",
    "lda",
    "document understanding",
    "document ai",
    "document processing",
    "conversational ai",
    "dialogue system",
    "dialogue systems",
    "chatbot",
    "chatbots",
    "virtual assistant",
    "voice assistant",
    # Recommendation & Search
    "recommendation system",
    "recommendation systems",
    "recommender system",
    "recommender systems",
    "collaborative filtering",
    "content-based filtering",
    "matrix factorization",
    "personalization",
    "search ranking",
    "learning to rank",
    "information retrieval",
    # Time Series & Forecasting
    "time series",
    "time-series",
    "forecasting",
    "time series forecasting",
    "prophet",
    "arima",
    "lstm forecasting",
    "temporal fusion transformer",
    "tft",
    "n-beats",
    "darts",
    # AutoML & Optimization
    "automl",
    "auto-ml",
    "automated machine learning",
    "hyperparameter tuning",
    "hyperparameter optimization",
    "hpo",
    "neural architecture search",
    "nas",
    "bayesian optimization",
    "optuna",
    "ray tune",
    "hyperopt",
    "grid search",
    "random search",
    # ML Frameworks & Libraries
    "pytorch",
    "torch",
    "pytorch lightning",
    "lightning",
    "tensorflow",
    "tf",
    "tf2",
    "keras",
    "jax",
    "flax",
    "equinox",
    "haiku",
    "trax",
    "mxnet",
    "caffe",
    "caffe2",
    "theano",
    "paddle",
    "paddlepaddle",
    "mindspore",
    "scikit-learn",
    "sklearn",
    "xgboost",
    "lightgbm",
    "catboost",
    "gradient boosting",
    "random forest",
    "decision tree",
    "ensemble methods",
    "bagging",
    "boosting",
    "stacking",
    "spacy",
    "nltk",
    "gensim",
    "flair",
    "stanza",
    "allennlp",
    "opencv",
    "cv2",
    "pillow",
    "albumentations",
    "torchvision",
    "timm",
    "detectron",
    "detectron2",
    "mmlab",
    "mmdetection",
    "mmsegmentation",
    # Data Science & Analytics
    "data science",
    "data scientist",
    "machine learning engineer",
    "ml engineer",
    "mle",
    "ai engineer",
    "ai researcher",
    "research scientist",
    "applied scientist",
    "model training",
    "model inference",
    "model evaluation",
    "model validation",
    "cross-validation",
    "train-test split",
    "feature engineering",
    "feature extraction",
    "feature selection",
    "data preprocessing",
    "data augmentation",
    "data labeling",
    "data annotation",
    "ground truth",
    "training data",
    "training dataset",
    "test dataset",
    "validation dataset",
    "holdout set",
    "overfitting",
    "underfitting",
    "regularization",
    "dropout",
    "batch normalization",
    "layer normalization",
    "loss function",
    "optimizer",
    "adam",
    "sgd",
    "learning rate",
    "gradient descent",
    "backpropagation",
    "epoch",
    "batch size",
    "mini-batch",
    # MLOps & Deployment
    "mlops",
    "llmops",
    "aiops",
    "modelops",
    "model deployment",
    "model serving",
    "model registry",
    "model versioning",
    "model monitoring",
    "model observability",
    "model governance",
    "model lifecycle",
    "ml lifecycle",
    "ml pipeline",
    "ml pipelines",
    "feature store",
    "feature platform",
    "experiment tracking",
    "experiment management",
    "mlflow",
    "kubeflow",
    "metaflow",
    "prefect",
    "airflow ml",
    "dagster",
    "zenml",
    "bentoml",
    "seldon",
    "kserve",
    "triton",
    "triton inference server",
    "torchserve",
    "tensorflow serving",
    "tf serving",
    "ray serve",
    "onnx",
    "onnxruntime",
    "onnx runtime",
    "tensorrt",
    "openvino",
    "model optimization",
    "inference optimization",
    "model cache",
    "model caching",
    "batch inference",
    "real-time inference",
    "online inference",
    "offline inference",
    "a/b testing ml",
    "canary deployment",
    "shadow deployment",
    "blue-green deployment",
    "model rollback",
    "data drift",
    "model drift",
    "concept drift",
    "feature drift",
    # Cloud AI Platforms
    "aws sagemaker",
    "sagemaker",
    "amazon sagemaker",
    "sagemaker studio",
    "sagemaker endpoints",
    "bedrock",
    "amazon bedrock",
    "aws bedrock",
    "google vertex ai",
    "vertex ai",
    "google cloud ai",
    "google ai platform",
    "automl vision",
    "automl tables",
    "document ai",
    "azure ml",
    "azure machine learning",
    "azure cognitive services",
    "azure openai",
    "azure ai",
    "aws ai",
    "aws ml",
    "gcp ai",
    "gcp ml",
    "databricks",
    "databricks ml",
    "mlr",
    "databricks mosaic",
    "snowflake ml",
    "snowpark",
    # GPU & Training Infrastructure
    "cuda",
    "cudnn",
    "nvidia",
    "gpu",
    "gpus",
    "gpu computing",
    "gpu acceleration",
    "gpu-accelerated",
    "gpu training",
    "multi-gpu",
    "multi gpu",
    "distributed training",
    "data parallel",
    "model parallel",
    "pipeline parallel",
    "tensor parallel",
    "horovod",
    "nccl",
    "nvlink",
    "a100",
    "h100",
    "v100",
    "tpu",
    "tpus",
    "tensor processing unit",
    "inference chip",
    "ai accelerator",
    "asic",
    # Data Processing (ML-relevant)
    "pandas",
    "numpy",
    "scipy",
    "dask",
    "vaex",
    "polars",
    "modin",
    "cudf",
    "rapids",
    "spark ml",
    "spark mllib",
    "pyspark",
    "pyspark ml",
    "sparkml",
    "spark nlp",
    "data pipeline",
    "data pipelines",
    "etl pipeline",
    "elt pipeline",
    "data ingestion",
    "data processing",
    "batch processing",
    "stream processing",
    "real-time processing",
    # Edge AI & Optimization
    "tflite",
    "tensorflow lite",
    "tf lite",
    "coreml",
    "core ml",
    "ane",
    "apple neural engine",
    "edge ai",
    "edge ml",
    "edge inference",
    "on-device ml",
    "on-device ai",
    "embedded ml",
    "embedded ai",
    "tinyml",
    "tiny ml",
    "microcontroller ml",
    "mobile ml",
    "mobile ai",
    "ml kit",
    "mediapipe",
    "ncnn",
    "mnn",
    "tnn",
    "arm nn",
    # Responsible AI & Ethics
    "responsible ai",
    "ai ethics",
    "ethical ai",
    "ai safety",
    "ai alignment",
    "alignment",
    "ai fairness",
    "fairness",
    "bias detection",
    "debiasing",
    "ai bias",
    "model bias",
    "algorithmic bias",
    "explainability",
    "explainable ai",
    "xai",
    "interpretability",
    "interpretable ml",
    "model interpretability",
    "shap",
    "lime",
    "feature importance",
    "attention visualization",
    "saliency maps",
    "ai transparency",
    "model transparency",
    "ai governance",
    "ai regulation",
    "ai compliance",
    "red teaming",
    "adversarial testing",
    "robustness",
    "adversarial robustness",
    "adversarial examples",
    "adversarial attacks",
    "jailbreaking",
    "prompt injection",
    "guardrails",
    "content moderation",
    "toxicity detection",
    "hallucination",
    "hallucination detection",
    "grounding",
    "ai audit",
    # Evaluation & Metrics
    "model metrics",
    "accuracy",
    "precision",
    "recall",
    "f1 score",
    "f1-score",
    "auc",
    "roc",
    "roc-auc",
    "confusion matrix",
    "perplexity",
    "bleu score",
    "rouge score",
    "meteor",
    "bertscore",
    "human evaluation",
    "elo rating",
    "benchmark",
    "benchmarking",
    "mmlu",
    "hellaswag",
    "truthfulqa",
    "mtbench",
    "chatbot arena",
    "leaderboard",
    # Research & Academic
    "arxiv",
    "research paper",
    "machine learning research",
    "ml research",
    "ai research",
    "deep learning research",
    "neurips",
    "icml",
    "iclr",
    "acl",
    "emnlp",
    "cvpr",
    "iccv",
    "eccv",
    "aaai",
    "ijcai",
    # Industry/Domain AI
    "healthcare ai",
    "medical ai",
    "clinical ai",
    "drug discovery",
    "ai drug discovery",
    "biotech ai",
    "genomics ai",
    "protein folding",
    "alphafold",
    "financial ai",
    "fintech ai",
    "fraud detection",
    "credit scoring",
    "algorithmic trading",
    "robo-advisor",
    "legal ai",
    "legaltech ai",
    "contract analysis",
    "retail ai",
    "e-commerce ai",
    "supply chain ai",
    "logistics ai",
    "manufacturing ai",
    "industrial ai",
    "autonomous vehicles",
    "self-driving",
    "autonomous driving",
    "adas",
    "robotics ai",
    "robotic process automation",
    "rpa",
    "intelligent automation",
    "gaming ai",
    "game ai",
    "creative ai",
    "ai art",
    "ai music",
    "ai video",
    "text-to-video",
    "video generation",
    "sora",
    "runway",
    "pika",
    "ai avatar",
    "virtual human",
    "digital twin",
    # Miscellaneous AI terms
    "cognitive computing",
    "cognitive ai",
    "intelligence augmentation",
    "augmented intelligence",
    "human-in-the-loop",
    "hitl",
    "human-ai collaboration",
    "ai-assisted",
    "ai-powered",
    "ai-driven",
    "ai-enabled",
    "ai-first",
    "ai native",
    "ai-native",
    "smart",
    "intelligent",
    "automated",
    "automation",
    "intelligent automation",
    "cognitive automation",
    "process automation",
    "decision intelligence",
    "business intelligence ai",
    "analytics ai",
    "insights ai",
    "copilot",
    "assistant",
    "ai assistant",
    "virtual agent",
    "bot",
    "ai bot",
]


# Skills that indicate "Real AI" work (building/training/deploying models)
# vs just using AI tools or APIs. Tightened to reduce false positives.
REAL_AI_SKILLS = {
    # ML/DL Frameworks (implies hands-on model work)
    "tensorflow",
    "pytorch",
    "keras",
    "scikit-learn",
    "jax",
    "mxnet",
    "caffe",
    "theano",
    "xgboost",
    "lightgbm",
    "catboost",
    
    # Specialized AI Libraries (implies building, not just using)
    "huggingface",
    "transformers",
    "diffusers",
    "timm",
    "detectron2",
    
    # Core Model Development Activities
    "model training",
    "fine-tuning",
    "model serving",
    "model deployment",
    "model evaluation",
    "model monitoring",
    
    # Deep Learning Concepts (specific enough to imply real work)
    "deep learning",
    "neural networks",
    "reinforcement learning",
    
    # GenAI / LLM Specific
    "llm",
    "large language model",
    "generative ai",
    "foundation model",
    "rag",
    "retrieval augmented generation",
    
    # MLOps Platforms (implies production ML work)
    "mlops",
    "mlflow",
    "kubeflow",
    "sagemaker",
    "vertex ai",
    "azure ml",
    "feature store",
    "experiment tracking",
    "weights and biases",
    "wandb",
    
    # Model Optimization (implies hands-on work)
    "onnx",
    "tensorrt",
    "triton inference server",
}
