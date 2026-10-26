FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage cache
COPY docker/requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src src

# Set python path to root so imports work
ENV PYTHONPATH=/app

# Run the service
CMD ["python", "src/simple_rag/run.py"]
