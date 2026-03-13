# 🧠 Sentiment Analyzer — NLP Model Comparison

![CI](https://github.com/YOUR_USERNAME/sentiment-analyzer/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-brightgreen)](https://YOUR_APP.streamlit.app)

> Compare **DistilBERT** vs **TextBlob** vs **VADER** side-by-side with confidence scores, batch CSV analysis, and interactive charts. Full-stack: FastAPI backend + React frontend + Streamlit app.

![App Screenshot](docs/screenshot.png)

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 **3 NLP Models** | DistilBERT (transformer) · TextBlob (lexicon) · VADER (rule-based) |
| 📊 **Side-by-side comparison** | Confidence bars, latency, per-model score breakdown |
| 📂 **Batch Analysis** | Upload CSV → analyze 100s of texts at once → download results |
| ⚡ **FastAPI Backend** | REST API with `/analyze` and `/batch` endpoints + Swagger UI |
| 🎨 **React Frontend** | Beautiful standalone UI with history, export to CSV |
| 🐳 **Docker Ready** | One-command local run with docker-compose |
| 🧪 **Tested** | 25+ pytest tests · GitHub Actions CI |

---

## 🚀 Quick Start

### Option A — Streamlit Cloud (instant, no install)
Click the Live Demo badge above ↑

### Option B — Docker (one command)
```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analyzer.git
cd sentiment-analyzer
docker-compose up
# API → http://localhost:8000/docs
# Streamlit → http://localhost:8501
```

### Option C — Local Python
```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analyzer.git
cd sentiment-analyzer
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app/app.py

# Or run FastAPI backend
uvicorn backend.app.main:app --reload
# Then open frontend/index.html in browser
```

---

## 📁 Project Structure

```
sentiment-analyzer/
├── streamlit_app/
│   └── app.py              # Streamlit app (deploy to Streamlit Cloud)
├── backend/
│   └── app/
│       ├── main.py         # FastAPI endpoints
│       └── models.py       # BERT, TextBlob, VADER wrappers
├── frontend/
│   └── index.html          # React UI (works standalone)
├── training/
│   └── train.py            # Fine-tune DistilBERT on IMDB
├── tests/
│   └── test_sentiment.py   # 25+ pytest tests
├── .github/workflows/
│   └── ci.yml              # CI: test → lint → Docker build
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🏗️ Architecture

```
User (Browser)
     │
     ├── Streamlit Cloud ──→ streamlit_app/app.py
     │                       (all-in-one, no backend)
     │
     └── React Frontend ──→ FastAPI Backend (port 8000)
         frontend/             └── /analyze  → BERT + TextBlob + VADER
         index.html            └── /batch    → batch processing
                               └── /health   → status check
```

---

## 🤖 Model Comparison

| Model | Type | Speed | Accuracy | Best For |
|---|---|---|---|---|
| DistilBERT | Transformer (66M params) | ~200ms | 0.93 AUC | Reviews, formal text |
| TextBlob | Lexicon + rules | <5ms | ~0.78 AUC | Short sentences |
| VADER | Social media lexicon | <5ms | ~0.80 AUC | Tweets, informal text |

---

## 🔬 Fine-tuning Your Own Model

```bash
# Quick test (1000 samples, ~5 min)
python training/train.py --train_samples 1000 --epochs 2

# Full IMDB dataset (50k samples, ~45 min on GPU)
python training/train.py --epochs 3

# Push to HuggingFace Hub
python training/train.py --push_to_hub --hub_model_id YOUR_USERNAME/sentiment-distilbert
```

---

## 📡 API Reference

```bash
# Single text
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This is absolutely amazing!"}'

# Batch
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I love it!", "Terrible product."]}'
```

Response:
```json
{
  "text": "This is absolutely amazing!",
  "consensus": "POSITIVE",
  "processing_time_ms": 187.3,
  "results": [
    { "model_name": "bert", "label": "POSITIVE", "confidence": 0.9987, "latency_ms": 145.2 },
    { "model_name": "textblob", "label": "POSITIVE", "confidence": 0.8250, "latency_ms": 1.8 },
    { "model_name": "vader", "label": "POSITIVE", "confidence": 0.9040, "latency_ms": 0.9 }
  ]
}
```

---

## 🛠️ Tech Stack

`transformers` · `textblob` · `vaderSentiment` · `fastapi` · `uvicorn` · `streamlit` · `plotly` · `react` · `docker`

---

## 📄 License

MIT — free to use and modify.

---

*Part of a full ML portfolio. See also: [Loan Default Predictor](#) · [Stock Forecaster](#) · [RAG Chatbot](#)*
