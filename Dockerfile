# LUMINARK AI Framework - Production Docker Image
# Build: docker build -t luminark:latest .
# Run: docker run -p 8501:8501 luminark:latest

FROM python:3.10-slim

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

# Copy source code
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Default command: Run Dashboard
CMD ["streamlit", "run", "luminark_dashboard.py"]
