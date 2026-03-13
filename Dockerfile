FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download VADER lexicon and TextBlob corpora
RUN python -c "import nltk; nltk.download('punkt', quiet=True)" 2>/dev/null || true
RUN python -c "from textblob import TextBlob; TextBlob('init').sentiment" 2>/dev/null || true

# Copy app
COPY backend/ ./backend/
COPY streamlit_app/ ./streamlit_app/

EXPOSE 8000 8501

# Default: run FastAPI backend
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
