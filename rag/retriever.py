# rag/retriever.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Load the ChromaDB index and provide a function to retrieve
#          the most relevant document chunks for a given question.
#
# This is the "search" layer — it finds the right passages
# before the LLM reads them and writes an answer.
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR      = Path("data/chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# How many chunks to retrieve per question.
# 5 is a good default — enough context without blowing the LLM's window.
TOP_K = 5


def load_retriever():
    """
    Load the ChromaDB index from disk and return a LangChain retriever.

    A 'retriever' is an object with one job: given a question string,
    return the TOP_K most semantically similar document chunks.

    We use MMR (Maximal Marginal Relevance) instead of plain similarity search.
    WHY MMR? Plain similarity often returns 5 chunks that all say the same thing.
    MMR balances relevance with diversity — you get different angles on the topic.
    """
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB not found at '{CHROMA_DIR}'. "
            "Run 'python rag/ingest.py' first to build the index."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,          # number of chunks to return
            "fetch_k": 20,       # candidates MMR considers before picking TOP_K
            "lambda_mult": 0.7,  # 0 = max diversity, 1 = max relevance
        },
    )

    return retriever


def retrieve_chunks(question: str) -> list:
    """
    Convenience wrapper: load retriever and return chunks for a question.
    Each chunk is a LangChain Document with .page_content and .metadata.

    Usage:
        chunks = retrieve_chunks("What drove PepsiCo revenue growth in 2023?")
        for c in chunks:
            print(c.metadata["source"], c.page_content[:200])
    """
    retriever = load_retriever()
    return retriever.invoke(question)


if __name__ == "__main__":
    # Quick sanity check — run this to make sure retrieval is working
    test_question = "What are the key revenue drivers?"
    print(f"Testing retrieval for: '{test_question}'\n")

    chunks = retrieve_chunks(test_question)

    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        page   = chunk.metadata.get("page", "?")
        print(f"── Chunk {i} | Source: {source} | Page: {page}")
        print(chunk.page_content[:300])
        print()
