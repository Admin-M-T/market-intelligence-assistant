# evals/ragas_eval.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Measure the quality of your RAG pipeline using RAGAS metrics.

# RAGAS METRICS EXPLAINED:
#
#   faithfulness      → Does the answer only use info from the retrieved chunks?
#                       High score = model isn't hallucinating.
#
#   answer_relevancy  → Does the answer actually address the question?
#                       High score = no irrelevant rambling.
#
# Run this script:  python -m evals.ragas_eval
# ─────────────────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag.chain import ask

load_dotenv()


# ── Test questions ────────────────────────────────────────────────────────────
# These cover different question types: factual, risk, geographic, financial,
# and strategic — giving a well-rounded picture of pipeline quality.

TEST_QUESTIONS = [
    "What was the revenue growth in the most recent reported quarter?",
    "What risks or headwinds did management highlight for the business?",
    "Which geographic markets showed the strongest volume performance?",
    "How did commodity costs or inflation impact operating margins?",
    "What strategic initiatives or investments were mentioned for future growth?",
]


def run_evaluation():
    """
    Run the RAG pipeline on all test questions and compute RAGAS metrics.
    Prints a score table and saves results to evals/results.txt.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    print("Running RAG pipeline on test questions…\n")

    questions     = []
    answers       = []
    contexts_list = []

    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"  [{i}/{len(TEST_QUESTIONS)}] {q}")
        result = ask(q)

        questions.append(q)
        answers.append(result["answer"])

        # RAGAS expects contexts as a list of strings (the raw chunk text)
        contexts_list.append([doc.page_content for doc in result["sources"]])

    # Build a HuggingFace Dataset — RAGAS expects this format
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer":   answers,
        "contexts": contexts_list,
    })

    # Wrap judge LLM and embeddings for RAGAS compatibility
    judge_llm = LangchainLLMWrapper(ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0,
    ))

    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    ))

    print("\nComputing RAGAS metrics…\n")

    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    # Convert to plain floats
    # Convert to plain floats — handle both list and scalar returns
    f_raw = results['faithfulness']
    r_raw = results['answer_relevancy']

    faithfulness_score = float(f_raw[0]) if isinstance(f_raw, list) else float(f_raw)
    relevancy_score    = float(r_raw[0]) if isinstance(r_raw, list) else float(r_raw)

    # ── Print results ──────────────────────────────────────────────────────────
    print("=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Faithfulness:      {faithfulness_score:.3f}  (0–1, higher is better)")
    print(f"  Answer Relevancy:  {relevancy_score:.3f}  (0–1, higher is better)")
    print("=" * 50)
    print("\nInterpretation:")
    print("  > 0.8  Excellent — ready to show recruiters")
    print("  0.6–0.8  Good — try improving chunk size or the prompt")
    print("  < 0.6  Needs work — check your documents and prompt template")

    # ── Save results to file ───────────────────────────────────────────────────
    output_path = "evals/results.txt"
    with open(output_path, "w") as f:
        f.write("RAGAS EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Faithfulness:      {faithfulness_score:.3f}\n")
        f.write(f"Answer Relevancy:  {relevancy_score:.3f}\n")
        f.write("\nTest questions and answers:\n")
        for q, a in zip(questions, answers):
            f.write(f"\nQ: {q}\nA: {a}\n")

    print(f"\n✅  Full results saved to '{output_path}'")
    print("    Copy these scores into your README.md for the portfolio!")

    return results


if __name__ == "__main__":
    run_evaluation()