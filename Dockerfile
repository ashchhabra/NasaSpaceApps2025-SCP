FROM python:3.12.0-slim

# System dependencies (without redundant python install)
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .

# Install Python dependencies in one go
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy rest of the code
COPY . .

# Make Python unbuffered for real-time logs
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["uv", "run", "train.py"]
