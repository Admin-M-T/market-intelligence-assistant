# rag/ingest.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Load PDFs from data/sample_docs/, split them into chunks,
#          create embeddings, and store them in ChromaDB (a local vector store).
#
# A search index over all the documents.
# You only need to run this once (or whenever you add new documents).
# ─────────────────────────────────────────────────────────────────────────────

import os
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Paths ─────────────────────────────────────────────────────────────────────
DOCS_DIR   = Path("data/sample_docs")   # PDFs location
CHROMA_DIR = Path("data/chroma_db")     # where ChromaDB will save its index

# ── Embedding model ───────────────────────────────────────────────────────────
# This runs LOCALLY — no API key, no cost.
# The embedding model used is "all-MiniLM-L6-v2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdfs(docs_dir: Path) -> list:
    """
    Walk through docs_dir and load every PDF file.
    Returns a list of LangChain Document objects.
    Each Document has:
      - page_content : the raw text of one page
      - metadata     : dict with 'source' (filename) and 'page' number
    """
    all_docs = []
    pdf_files = list(docs_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{docs_dir}'. "
            "Add some PDFs there before running ingest."
        )

    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()

        
        for doc in docs:
            doc.metadata["source"] = pdf_path.name

        all_docs.extend(docs)

    print(f"\n  Total pages loaded: {len(all_docs)}")
    return all_docs


def split_documents(docs: list) -> list:
    """
    Split long pages into smaller overlapping chunks.

    WHY CHUNKING?
    LLMs have a limited context window. Instead of feeding the whole document,
    we retrieve only the 3–5 most relevant chunks for each question.

    chunk_size    = max characters per chunk (~300–400 words)
    chunk_overlap = characters shared between adjacent chunks
                    (so a sentence split across a boundary isn't lost)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Total chunks after splitting: {len(chunks)}")
    return chunks


def build_vector_store(chunks: list, persist_dir: Path) -> Chroma:
    """
    Embed each chunk and store in ChromaDB.

    WHAT IS AN EMBEDDING?
    A list of ~384 numbers that captures the meaning of a piece of text.
    Similar meanings implies similar numbers implies ChromaDB can find them by distance.

    This step takes 1–3 minutes the first time depending on how many docs you have.
    After that, ChromaDB loads from disk in seconds.
    """
    print(f"\n  Loading embedding model '{EMBEDDING_MODEL}' …")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},       # change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )

    print("  Building ChromaDB index (this may take a few minutes) …")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )

    print(f"  Index saved to '{persist_dir}/'")
    return vector_store


def run_ingest():
    """Main entry point — run this to (re)build the index."""
    print("=" * 60)
    print("STEP 1/3  Loading PDFs")
    print("=" * 60)
    docs = load_pdfs(DOCS_DIR)

    print("\n" + "=" * 60)
    print("STEP 2/3  Splitting into chunks")
    print("=" * 60)
    chunks = split_documents(docs)

    print("\n" + "=" * 60)
    print("STEP 3/3  Building vector store")
    print("=" * 60)
    build_vector_store(chunks, CHROMA_DIR)

    print("\n✅  Ingestion complete!")
    print(f"    {len(chunks)} chunks indexed from {DOCS_DIR}")
    print(f"    Vector store saved at: {CHROMA_DIR}")


if __name__ == "__main__":
    run_ingest()
