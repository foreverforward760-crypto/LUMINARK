# LUMINARK AI Framework - Production Docker Image
# Build: docker build -t luminark:latest .
# Run: docker run -p 8000:8000 luminark:latest

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire framework
COPY luminark/ ./luminark/
COPY examples/ ./examples/
COPY README.md .
COPY LICENSE .
COPY setup.py .

# Install luminark package
RUN pip install -e .

# Create directories for outputs
RUN mkdir -p /app/checkpoints /app/logs /app/data

# Expose port for dashboard
EXPOSE 8000

# Default command: run basic training example
CMD ["python", "examples/train_mnist.py"]

# Alternative commands:
# Run advanced AI: docker run luminark:latest python examples/train_advanced_ai.py
# Run dashboard: docker run -p 8000:8000 luminark:latest python octo_dashboard_server.py
# Interactive shell: docker run -it luminark:latest /bin/bash
