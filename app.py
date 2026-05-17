# app.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: The Streamlit web UI — this is what recruiters will interact with.
#
# To run locally:   streamlit run app.py
# To deploy:        Push to GitHub → connect to Streamlit Community Cloud
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
from rag.chain import ask

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Intelligence Assistant",
    page_icon="📊",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Market Intelligence Assistant")
st.caption(
    "RAG-powered Q&A over FMCG documents. "
    "Ask a strategic question and get a grounded, source-cited answer."
)
st.divider()

# ── Example questions (helps recruiters know what to ask) ─────────────────────
with st.expander("💡 Example questions to try"):
    st.markdown("""
- What drove PepsiCo's revenue growth in 2023?
- What are the key risks mentioned across these documents?
- How did inflation impact FMCG margins?
- What is the outlook for the healthcare sector?
- Which geographies showed the strongest volume growth?
    """)

# ── Session state for chat history ────────────────────────────────────────────
# st.session_state persists across reruns of the app within the same session
if "history" not in st.session_state:
    st.session_state.history = []   # list of {"question": ..., "answer": ..., "sources": ...}

# ── Question input ────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about the loaded documents…")

if question:
    with st.spinner("Retrieving relevant context and generating answer…"):
        try:
            result = ask(question)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    # Save to history
    st.session_state.history.append({
        "question": question,
        "answer":   result["answer"],
        "sources":  result["sources"],
    })

# ── Chat history display ──────────────────────────────────────────────────────
for turn in st.session_state.history:
    # User bubble
    with st.chat_message("user"):
        st.write(turn["question"])

    # Assistant bubble
    with st.chat_message("assistant"):
        st.write(turn["answer"])

        # Sources panel — this is the key differentiator in a recruiter demo
        if turn["sources"]:
            with st.expander(f"📄 Sources used ({len(turn['sources'])} chunks)"):
                for i, doc in enumerate(turn["sources"], 1):
                    source = doc.metadata.get("source", "unknown")
                    page   = doc.metadata.get("page", "?")
                    st.markdown(f"**Chunk {i}** · `{source}` · Page {page}")
                    st.caption(doc.page_content[:400] + "…")
                    if i < len(turn["sources"]):
                        st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About this project")
    st.markdown("""
**Stack**
- 🔍 Retrieval: ChromaDB + sentence-transformers (local, free)
- 🤖 Generation: llama-3.1-8b-instant
- 🔗 Orchestration: LangChain
- 📊 Eval: RAGAS

**How it works**
1. PDFs are chunked and embedded locally
2. Your question is embedded the same way
3. The 5 most similar chunks are retrieved
4. Groq reads those chunks and writes a grounded answer
    """)

    st.divider()
    st.markdown("Built by **Varun Jain** | [GitHub](https://github.com/admin-M-T)")

    # Clear chat button
    if st.button("🗑 Clear chat history"):
        st.session_state.history = []
        st.rerun()
