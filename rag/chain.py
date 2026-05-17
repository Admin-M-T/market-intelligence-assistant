# rag/chain.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Wire the retriever to Google Gemini to produce answers.
#
# The flow for every question:
#   1. Retriever finds the TOP_K relevant chunks from ChromaDB
#   2. Those chunks are inserted into a prompt template as "context"
#   3. Gemini reads the context + question and writes a grounded answer
#   4. We return both the answer text and the source chunks (for the UI)
#
# This pattern is called RAG: Retrieval-Augmented Generation.
# ─────────────────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from rag.retriever import load_retriever

# Load GROQ_API_KEY from .env file
load_dotenv()


# ── Prompt template ───────────────────────────────────────────────────────────
# This is the exact text sent to Gemini on every question.
# {context} is replaced with the retrieved chunks.
# {question} is replaced with the user's question.
#
# Good prompt engineering tips used here:
#   - Explicit role ("You are a market intelligence analyst")
#   - Grounding instruction ("only use the provided context")
#   - Graceful fallback ("say you don't have enough information")
#   - Output structure ("cite the source document name")

PROMPT_TEMPLATE = """You are a senior market intelligence analyst specialising in FMCG.
Answer the question below using ONLY the context provided.
If the context does not contain enough information to answer confidently, say:
"I don't have enough information in the loaded documents to answer this accurately."

For each key claim in your answer, cite the source document name in parentheses, e.g. (PepsiCo_Q3_2023.pdf).

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE,
)


def format_context(docs: list) -> str:
    """
    Convert a list of Document chunks into a single context string.
    Each chunk is separated by a divider so the LLM can distinguish them.
    """
    formatted_chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        formatted_chunks.append(
            f"[Chunk {i} | Source: {source} | Page: {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted_chunks)


def build_chain():
    """
    Build and return the RAG chain.

    LangChain uses a pipe (|) syntax to chain steps together.
    Read it left to right: retrieve → format → prompt → LLM → parse output.

    We also return retrieved_docs separately so the UI can show
    a "Sources" panel — this is important for recruiter demos.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
        "GROQ_API_KEY not found. "
        "Make sure it's set in your .env file."
    )

    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=api_key,
    temperature=0.2,
    max_tokens=1024,
    )

    retriever = load_retriever()

    # The chain:
    # 1. Pass the question through unchanged AND retrieve docs simultaneously
    # 2. Format docs into a context string
    # 3. Fill the prompt template
    # 4. Send to Gemini
    # 5. Parse the response text out of the LLM output object
    chain = (
        {"context": retriever | format_context, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(question: str) -> dict:
    """
    Ask a question and get back:
      - answer     : the LLM's response string
      - sources    : list of Document chunks used (for the UI)

    Usage:
        result = ask("What drove PepsiCo's volume decline in South Africa?")
        print(result["answer"])
        for doc in result["sources"]:
            print(doc.metadata["source"])
    """
    chain, retriever = build_chain()

    # Retrieve chunks separately so we can return them for the UI
    source_docs = retriever.invoke(question)

    # Run the full chain to get the answer
    answer = chain.invoke(question)

    return {
        "answer": answer,
        "sources": source_docs,
    }


if __name__ == "__main__":
    # Quick test — run this to verify the full pipeline works end-to-end
    test_q = "What are the key revenue drivers mentioned in the documents?"
    print(f"Question: {test_q}\n")

    result = ask(test_q)

    print("Answer:")
    print(result["answer"])
    print("\nSources used:")
    for doc in result["sources"]:
        print(f"  - {doc.metadata.get('source')} (page {doc.metadata.get('page')})")
