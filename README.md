# 📊 Market Intelligence Assistant

A RAG (Retrieval-Augmented Generation) system that answers strategic business questions grounded in FMCG and Healthcare documents — earnings transcripts, market reports, and analyst filings.

**[🚀 Live Demo](https://https://varun-market-intelligence-rag.streamlit.app/)** &nbsp;|&nbsp; Built by [Varun Jain](https://github.com/admin-M-T)

---

## What it does

Ask a strategic question → the system retrieves the most relevant passages from your document corpus → the LLM writes a grounded, source-cited answer.

**Example questions:**
- *"What drove PepsiCo's volume decline in South Africa?"*
- *"What risks did management highlight for 2024?"*
- *"How did inflation affect FMCG margins across regions?"*

---

## Architecture

```
PDF documents
     │
     ▼
[ingest.py] — PyMuPDF loads pages → RecursiveTextSplitter chunks →
              sentence-transformers embeds → ChromaDB stores
     │
     ▼
[retriever.py] — User question → embed → MMR search → top 5 chunks
     │
     ▼
[chain.py] — Chunks + question → Groq (llama-3.1-8b-instant) → grounded answer
     │
     ▼
[app.py] — Streamlit UI with source citations panel
```

---

## Stack

| Layer | Tool | Notes |
|---|---|---|
| LLM | Groq (llama-3.1-8b-instant) | Free, fast, no regional restrictions |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Runs locally, zero API cost |
| Vector store | ChromaDB | Local persistence |
| Orchestration | LangChain | |
| Evaluation | RAGAS | |
| UI | Streamlit | |

---

## Evaluation (RAGAS)

Measured on 5 domain-specific test questions using RAGAS with `llama-3.1-8b-instant` as the judge LLM.

| Metric | Score | Notes |
|---|---|---|
| Faithfulness | 0.600 | Answers grounded in retrieved context |
| Answer Relevancy | N/A | Judge LLM compatibility limitation with Groq |

*Run `python -m evals.ragas_eval` to reproduce.*

---

## Running locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/market-intelligence-assistant
cd market-intelligence-assistant

# 2. Create and activate a virtual environment (Python 3.11+)
python -m venv venv311
venv311\Scripts\activate        # Windows
# source venv311/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env
# Edit .env and paste your Groq API key from https://console.groq.com

# 5. Add PDFs to data/sample_docs/
# Public earnings transcripts from SEC EDGAR work well:
# https://efts.sec.gov/LATEST/search-index?q=%22PepsiCo%22&forms=8-K

# 6. Build the index
python rag/ingest.py

# 7. Launch the app
streamlit run app.py
```

---

## Project structure

```
market-intelligence-assistant/
├── app.py                  # Streamlit UI
├── rag/
│   ├── __init__.py         # Makes rag/ a Python package
│   ├── ingest.py           # PDF loading, chunking, embedding, indexing
│   ├── retriever.py        # ChromaDB retrieval with MMR
│   └── chain.py            # LangChain RAG chain + Groq LLM
├── evals/
│   ├── ragas_eval.py       # RAGAS evaluation script
│   └── results.txt         # Latest eval scores
├── data/
│   └── sample_docs/        # Place your PDFs here
├── .env.example            # API key template
├── .gitignore
└── requirements.txt
```

---

## Key design decisions

- **MMR retrieval over plain similarity search** — avoids returning 5 chunks that say the same thing; balances relevance with diversity.
- **Local embeddings** — `sentence-transformers` runs on CPU, zero API cost, fully reproducible.
- **Source citations in the UI** — every answer shows exactly which document chunks were used, making hallucinations auditable.
- **Eval-first mindset** — RAGAS metrics let you measure whether changes to chunk size, overlap, or the prompt template actually improve quality.
- **Groq as LLM provider** — chosen over Gemini for consistent free tier access across all regions.

---

## About the author

Senior Associate at PwC DIAC with 5+ years of experience in FMCG, Healthcare, and BFSI analytics. This project bridges my domain expertise in market intelligence with AI engineering — building systems that make strategic analysis faster and more accessible.

[LinkedIn](https://linkedin.com/in/varunjain95) | [GitHub](https://github.com/admin-M-T)