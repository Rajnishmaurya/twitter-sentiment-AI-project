# syntax=docker/dockerfile:1

FROM python:3.9-slim

WORKDIR /app

# Install OS deps (if needed) and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY src/deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code and processed data
COPY src/deployment /app/src/deployment
COPY data/processed    /app/data/processed
COPY ../../src/drift /app/src/drift
COPY logs               /app/logs

ENV PYTHONUNBUFFERED=1

# Expose FastAPI port
EXPOSE 8000

# Launch the FastAPI app
CMD ["uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
