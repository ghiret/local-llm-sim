FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better caching
COPY pyproject.toml README.md ./

# Install dependencies
RUN pip install --no-cache-dir build && \
    pip install --no-cache-dir .

# Copy source code
COPY src/ src/
COPY config/ config/

# Reinstall with source
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 11434

# Default command
CMD ["local-llm-sim", "serve", "--host", "0.0.0.0", "--port", "11434"]
