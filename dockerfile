FROM python:3.11-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive

# Set default environment variables (can be overridden at runtime)
ENV GEMINI_PROJECT_ID=""
ENV GEMINI_REGION="asia-south1"
ENV API_TOKEN=""
ENV MAX_TOKENS="32000"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        build-essential \
        pkg-config \
        libgl1-mesa-glx \
        libgl1-mesa-dri \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libx11-6 \
        libxcb1 \
        libxau6 \
        libxdmcp6 \
        libgtk-3-0 \
        libgdk-pixbuf2.0-0 \
        libfontconfig1 \
        libfreetype6 \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        libreoffice \
        pandoc \
        libmagic1 \
        file \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy and install Python dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code and scripts
COPY main.py .
COPY entrypoint.sh .
COPY --chown=appuser:appuser main.py entrypoint.sh ./

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Create necessary directories
RUN mkdir -p /tmp /app/credentials \
    && chown -R appuser:appuser /tmp /app/credentials

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
