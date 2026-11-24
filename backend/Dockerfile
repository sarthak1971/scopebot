FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
# Tesseract = the OCR engine used for scanned PDFs and images
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Cloud Run (and most hosts) inject $PORT
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
