FROM python:3.10-slim

WORKDIR /app

# Install build essentials for potential c-extensions (numpy etc)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project definition
COPY pyproject.toml README.md ./

# Copy application code (needed for hatchling to build the wheel)
COPY ai_skills ./ai_skills
COPY data ./data

# Install dependencies and the package
# We use pip to install the current directory which will read pyproject.toml
RUN pip install --no-cache-dir .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "ai_skills.server:app", "--host", "0.0.0.0", "--port", "8000"]
