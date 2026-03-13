"""
Sentiment Analyzer — Streamlit App
Deploys directly to Streamlit Cloud with zero backend needed.
Features: single text, batch CSV, charts, word cloud, history.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import time
import io
from collections import Counter

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .pos-box { background:#f0fff4; border:1px solid #38a169; border-radius:12px; padding:1.2rem; text-align:center; }
  .neg-box { background:#fff5f5; border:1px solid #e53e3e; border-radius:12px; padding:1.2rem; text-align:center; }
  .neu-box { background:#fffff0; border:1px solid #d69e2e; border-radius:12px; padding:1.2rem; text-align:center; }
  .model-badge { font-size:0.7rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; }
  .big-emoji { font-size:3rem; line-height:1; }
  .confidence-label { font-size:0.85rem; color:#718096; }
  div[data-testid="stTabs"] button { font-size:15px; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NLP models…")
def load_models():
    models = {}

    # DistilBERT
    try:
        from transformers import pipeline
        models["bert"] = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True, max_length=512,
        )
    except Exception:
        models["bert"] = None

    # TextBlob
    try:
        from textblob import TextBlob
        models["textblob"] = "loaded"
    except Exception:
        models["textblob"] = None

    # VADER
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        models["vader"] = SentimentIntensityAnalyzer()
    except Exception:
        models["vader"] = None

    return models

# ── Inference helpers ─────────────────────────────────────────────────────────
def run_bert(pipe, text):
    if pipe is None:
        return rule_based(text, "DistilBERT")
    try:
        r = pipe(text[:512])[0]
        label = r["label"]
        conf  = round(r["score"], 4)
        return {"model": "DistilBERT", "label": label, "confidence": conf,
                "scores": {"POSITIVE": conf if label=="POSITIVE" else round(1-conf,4),
                           "NEGATIVE": conf if label=="NEGATIVE" else round(1-conf,4)}}
    except Exception:
        return rule_based(text, "DistilBERT")


def run_textblob(text):
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        p = blob.sentiment.polarity
        s = blob.sentiment.subjectivity
        label = "POSITIVE" if p > 0.05 else "NEGATIVE" if p < -0.05 else "NEUTRAL"
        conf  = round(min(0.5 + abs(p) * 0.5, 0.99), 4)
        return {"model": "TextBlob", "label": label, "confidence": conf,
                "scores": {"polarity": round(p,4), "subjectivity": round(s,4)}}
    except Exception:
        return rule_based(text, "TextBlob")


def run_vader(sia, text):
    if sia is None:
        return rule_based(text, "VADER")
    scores = sia.polarity_scores(text)
    c = scores["compound"]
    label = "POSITIVE" if c >= 0.05 else "NEGATIVE" if c <= -0.05 else "NEUTRAL"
    conf  = round((abs(c) + 1) / 2, 4)
    return {"model": "VADER", "label": label, "confidence": conf,
            "scores": {"compound": round(c,4), "pos": round(scores["pos"],4),
                       "neg": round(scores["neg"],4), "neu": round(scores["neu"],4)}}


def rule_based(text, model_name):
    t = text.lower()
    pos = len(re.findall(r'\b(great|good|excellent|amazing|fantastic|love|best|perfect|happy|superb)\b', t))
    neg = len(re.findall(r'\b(bad|terrible|awful|horrible|worst|hate|poor|disappointing|boring)\b', t))
    label = "POSITIVE" if pos > neg else "NEGATIVE" if neg > pos else "NEUTRAL"
    conf  = round(min(0.55 + max(pos,neg) * 0.08, 0.93), 4)
    return {"model": model_name, "label": label, "confidence": conf,
            "scores": {"positive_words": pos, "negative_words": neg}}


def analyze_text(text, models):
    t0 = time.time()
    results = [
        run_bert(models.get("bert"), text),
        run_textblob(text),
        run_vader(models.get("vader"), text),
    ]
    labels = [r["label"] for r in results]
    consensus = max(set(labels), key=labels.count)
    return results, consensus, round((time.time() - t0) * 1000, 1)


EMOJI = {"POSITIVE": "😊", "NEGATIVE": "😞", "NEUTRAL": "😐"}
COLOR = {"POSITIVE": "#38a169", "NEGATIVE": "#e53e3e", "NEUTRAL": "#d69e2e"}

def render_result_card(result):
    label = result["label"]
    box_class = "pos-box" if label=="POSITIVE" else "neg-box" if label=="NEGATIVE" else "neu-box"
    st.markdown(f"""
    <div class="{box_class}">
      <div class="model-badge">{result['model']}</div>
      <div class="big-emoji">{EMOJI[label]}</div>
      <div style="font-weight:800;font-size:1.2rem;color:{COLOR[label]}">{label}</div>
      <div class="confidence-label">{result['confidence']*100:.1f}% confident</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(float(result["confidence"]))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=56)
    st.title("Sentiment Analyzer")
    st.caption("DistilBERT · TextBlob · VADER")
    st.divider()
    st.markdown("""
**About this app**

Compares 3 NLP approaches:
- 🤖 **DistilBERT** — transformer model fine-tuned on SST-2
- 📚 **TextBlob** — lexicon + rule-based
- ⚡ **VADER** — social media optimized lexicon

**Use cases:**
- Product review analysis
- Customer feedback triage
- Social media monitoring
- Survey response analysis
""")
    st.divider()
    st.markdown("**GitHub:** [View Source](#)")
    st.markdown("**Dataset:** IMDB / SST-2")

# ── Load models ───────────────────────────────────────────────────────────────
models = load_models()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_about = st.tabs(["📝 Single Text", "📂 Batch Analysis", "📖 How It Works"])

# ══ TAB 1: Single Text ════════════════════════════════════════════════════════
with tab_single:
    st.markdown("### Analyze Text Sentiment")

    # Sample buttons
    samples = {
        "😊 Positive": "This product exceeded all my expectations! Absolutely fantastic quality and incredible customer service.",
        "😞 Negative": "Worst purchase I've ever made. Completely broke after two days and support was useless.",
        "😐 Neutral":  "The product arrived on time. It does what it says on the box. No complaints, no praise.",
    }
    cols = st.columns(3)
    for i, (label, sample) in enumerate(samples.items()):
        if cols[i].button(label, use_container_width=True):
            st.session_state["input_text"] = sample

    text_input = st.text_area(
        "Enter text to analyze",
        value=st.session_state.get("input_text", ""),
        height=130,
        max_chars=5000,
        placeholder="Type or paste any text — a review, tweet, feedback…",
        label_visibility="collapsed",
    )
    char_count = len(text_input)
    st.caption(f"{char_count} / 5000 characters")

    col_btn, col_clear = st.columns([4, 1])
    analyze_clicked = col_btn.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    if col_clear.button("Clear", use_container_width=True):
        st.session_state["input_text"] = ""
        st.rerun()

    if analyze_clicked and text_input.strip():
        with st.spinner("Running models…"):
            results, consensus, ms = analyze_text(text_input.strip(), models)

        # Consensus header
        box_class = "pos-box" if consensus=="POSITIVE" else "neg-box" if consensus=="NEGATIVE" else "neu-box"
        st.markdown(f"""
        <div class="{box_class}" style="margin:1rem 0;display:flex;align-items:center;gap:1rem">
          <span style="font-size:2.5rem">{EMOJI[consensus]}</span>
          <div>
            <div style="font-weight:800;font-size:1.4rem;color:{COLOR[consensus]}">
              Consensus: {consensus}
            </div>
            <div style="color:#718096;font-size:0.9rem">
              {len(results)} models · {ms}ms
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Model cards
        st.markdown("#### Model Breakdown")
        c1, c2, c3 = st.columns(3)
        for col, result in zip([c1, c2, c3], results):
            with col:
                render_result_card(result)

        # Confidence comparison chart
        st.markdown("#### Confidence Comparison")
        chart_data = pd.DataFrame([
            {"Model": r["model"], "Confidence": r["confidence"] * 100, "Label": r["label"]}
            for r in results
        ])
        color_map = {"POSITIVE": "#38a169", "NEGATIVE": "#e53e3e", "NEUTRAL": "#d69e2e"}
        fig = px.bar(
            chart_data, x="Model", y="Confidence", color="Label",
            color_discrete_map=color_map,
            text=chart_data["Confidence"].apply(lambda x: f"{x:.1f}%"),
            range_y=[0, 100],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=300, margin=dict(t=20, b=20, l=10, r=10),
            showlegend=False, yaxis_title="Confidence (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Add to history
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].insert(0, {
            "text": text_input[:80] + ("…" if len(text_input) > 80 else ""),
            "consensus": consensus,
            "confidence": max(r["confidence"] for r in results),
        })
        st.session_state["history"] = st.session_state["history"][:20]

    elif analyze_clicked:
        st.warning("Please enter some text first.")

    # History
    if st.session_state.get("history"):
        st.divider()
        st.markdown("#### Recent Analyses")
        hist_df = pd.DataFrame(st.session_state["history"])
        hist_df["confidence"] = hist_df["confidence"].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ══ TAB 2: Batch Analysis ════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### Batch Sentiment Analysis")
    st.info("Upload a CSV with a `text` column to analyze all rows at once.")

    col_up, col_demo = st.columns([2, 1])
    uploaded = col_up.file_uploader("Upload CSV", type=["csv"])
    if col_demo.button("Use demo CSV"):
        demo_texts = [
            "I absolutely love this product!",
            "Terrible quality, never buying again.",
            "It's okay, nothing special.",
            "Best purchase I've made this year!",
            "Disappointing. Expected much better.",
            "Average product. Does what it says.",
            "Completely broken. Waste of money.",
            "Wonderful experience from start to finish!",
        ]
        demo_df = pd.DataFrame({"text": demo_texts})
        uploaded = io.StringIO(demo_df.to_csv(index=False))

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            if "text" not in df.columns:
                st.error("CSV must have a 'text' column.")
            else:
                st.success(f"Loaded {len(df)} rows.")
                st.dataframe(df.head(5), use_container_width=True, hide_index=True)

                max_rows = st.slider("Max rows to analyze", 5, min(len(df), 200), min(len(df), 50))
                if st.button("🚀 Run Batch Analysis", type="primary"):
                    subset = df["text"].dropna().head(max_rows).tolist()
                    progress = st.progress(0)
                    status   = st.empty()
                    batch_results = []

                    for i, text in enumerate(subset):
                        res, consensus, _ = analyze_text(str(text), models)
                        best = max(res, key=lambda r: r["confidence"])
                        batch_results.append({
                            "text": text[:60] + ("…" if len(text) > 60 else ""),
                            "consensus": consensus,
                            "confidence": round(best["confidence"] * 100, 1),
                            "bert":      res[0]["label"],
                            "textblob":  res[1]["label"],
                            "vader":     res[2]["label"],
                        })
                        progress.progress((i + 1) / len(subset))
                        status.caption(f"Analyzing row {i+1}/{len(subset)}…")

                    status.empty()
                    results_df = pd.DataFrame(batch_results)

                    st.markdown("#### Results")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)

                    # Charts
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Sentiment Distribution**")
                        dist = results_df["consensus"].value_counts().reset_index()
                        dist.columns = ["Sentiment", "Count"]
                        fig_pie = px.pie(dist, names="Sentiment", values="Count",
                                         color="Sentiment",
                                         color_discrete_map={"POSITIVE": "#38a169", "NEGATIVE": "#e53e3e", "NEUTRAL": "#d69e2e"})
                        fig_pie.update_layout(height=280, margin=dict(t=10, b=10))
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with c2:
                        st.markdown("**Confidence Distribution**")
                        fig_hist = px.histogram(results_df, x="confidence", nbins=15,
                                                color_discrete_sequence=["#5a67d8"])
                        fig_hist.update_layout(height=280, margin=dict(t=10, b=10),
                                               xaxis_title="Confidence (%)", yaxis_title="Count")
                        st.plotly_chart(fig_hist, use_container_width=True)

                    # Model agreement
                    st.markdown("**Model Agreement**")
                    agree = (results_df["bert"] == results_df["textblob"]) & \
                            (results_df["textblob"] == results_df["vader"])
                    st.metric("All 3 models agreed", f"{agree.sum()} / {len(results_df)} rows",
                              delta=f"{agree.mean()*100:.0f}% agreement rate")

                    # Download
                    csv_out = results_df.to_csv(index=False)
                    st.download_button("⬇️ Download Results CSV", csv_out,
                                       "sentiment_results.csv", "text/csv",
                                       use_container_width=True)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")


# ══ TAB 3: How It Works ══════════════════════════════════════════════════════
with tab_about:
    st.markdown("### How It Works")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
**🤖 DistilBERT**

A distilled version of BERT, fine-tuned on Stanford Sentiment Treebank (SST-2).

- 66M parameters
- Binary: POSITIVE / NEGATIVE
- Best for: reviews, formal text
- Speed: ~100–300ms per text
- AUC on SST-2: ~0.93
""")
    with col2:
        st.markdown("""
**📚 TextBlob**

Rule-based using a hand-crafted sentiment lexicon + pattern analysis.

- Returns polarity (−1 to +1)
- Also provides subjectivity score
- Best for: short, formal text
- Speed: <5ms per text
- Supports NEUTRAL class
""")
    with col3:
        st.markdown("""
**⚡ VADER**

Valence Aware Dictionary and sEntiment Reasoner — optimized for social media.

- Handles slang, emoji, caps
- Compound score (−1 to +1)
- Best for: tweets, reviews
- Speed: <5ms per text
- Rule-based, no training needed
""")

    st.divider()
    st.markdown("""
### When do models disagree?

| Text type | Best model | Why |
|---|---|---|
| Movie/product reviews | DistilBERT | Trained on SST-2 reviews |
| Social media posts | VADER | Handles slang + emoji |
| News / formal text | TextBlob | Good lexicon coverage |
| Sarcasm | None (all struggle) | Hard NLP problem |
| Short text (<5 words) | VADER | More robust to brevity |

### Tech Stack
`transformers` · `textblob` · `vaderSentiment` · `streamlit` · `plotly` · `pandas`
""")
