# =============================================
# Dockerfile - IMDB Sentiment Analysis API
# =============================================
# 
# Build:
#   docker build -t imdb-sentiment-api .
#
# Run:
#   docker run -p 8000:8000 imdb-sentiment-api
#
# With volume for models (optional):
#   docker run -p 8000:8000 -v $(pwd)/models:/app/models imdb-sentiment-api
#
# =============================================

# Use official Python slim image for smaller size
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Copy application code
COPY ml_pipeline/ ./ml_pipeline/
COPY config.yaml .

# Create models directory
RUN mkdir -p /app/ml_pipeline/models

# Train the model inside the container (models will be saved to ml_pipeline/models/)
RUN python ml_pipeline/train_model.py

# Expose port
EXPOSE 8000

# Run the application (stay in /app, not in /app/ml_pipeline/api)
CMD ["uvicorn", "ml_pipeline.api.app:app", "--host", "0.0.0.0", "--port", "8000"]