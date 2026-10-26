FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# build-essential for compiling some python packages
# curl for healthchecks if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH=/app

# Expose ports for backend and frontend
EXPOSE 5001
EXPOSE 8501

# Default command (overridden in docker-compose)
CMD ["bash"]
