# Mycelial Defense System - Docker Image

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY mycelial_defense/ ./mycelial_defense/
COPY cli/ ./cli/
COPY dashboard/ ./dashboard/
COPY examples/ ./examples/
COPY setup.py .
COPY pyproject.toml .
COPY README.md .

# Install package
RUN pip install -e .

# Expose dashboard port
EXPOSE 8000

# Default command: run dashboard
CMD ["python", "dashboard/backend/api.py"]
