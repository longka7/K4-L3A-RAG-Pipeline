"""
A/B evaluation harness: dense-only vs hybrid+RRF retrieval.

Not part of src/ (not graded module code) - a one-off script to produce the
real numbers for RESULT.md. Uses:
  - Local BGE-M3 (sentence-transformers) for retrieval + RAGAS embeddings
    (free, no rate limit).
  - Gemini (via its OpenAI-compatible endpoint) as the RAGAS judge LLM and
    as the generator LLM, since that's the only configured API key.

Gemini free tier is limited to 5 requests/minute, so this script evaluates a
representative subset of the 18-question golden dataset (see SELECTED_INDICES)
rather than all 18, and uses low concurrency + RAGAS's built-in retry/backoff
to survive rate limiting. Run with:

    python -m group_project.evaluation.run_ab_eval
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

from openai import OpenAI

from ragas import evaluate, EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import SYSTEM_PROMPT, call_llm, format_context, reorder_for_llm

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
OUTPUT_PATH = ROOT / "group_project" / "evaluation" / "ab_eval_results.json"
GENERATIONS_PATH = ROOT / "group_project" / "evaluation" / "ab_eval_generations_checkpoint.json"

# Indices into golden_dataset.json: a representative subset (legal + news,
# simple + multi-part, includes one question already known to be a
# retrieval failure case for MBA scholarship, deliberately kept for the
# "worst performers" / failure-analysis section).
SELECTED_INDICES = [0, 1, 3, 6, 7, 9, 11, 14]

TOP_K = 5
CALL_DELAY_SECONDS = 3  # pacing between our own generation calls (separate from RAGAS's own metric calls)


def call_llm_with_retry(system_prompt: str, user_message: str, attempts: int = 4) -> str:
    """call_llm wrapped with retry for transient network errors (not rate-limit backoff)."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call_llm(system_prompt, user_message)
        except Exception as error:  # transient network/server errors
            last_error = error
            print(f"    call_llm attempt {attempt}/{attempts} failed: {error}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"call_llm failed after {attempts} attempts") from last_error


def generate_for_config(query: str, use_reranking: bool) -> dict:
    """Mirror generate_with_citation but force dense-only or hybrid+RRF."""
    chunks = retrieve(query, top_k=TOP_K, use_reranking=use_reranking)
    if not chunks:
        return {"answer": "", "contexts": []}
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    answer = call_llm_with_retry(SYSTEM_PROMPT, user_message)
    return {"answer": answer, "contexts": [c["content"] for c in chunks]}


def dense_only_retrieve_and_generate(query: str) -> dict:
    """Config A: semantic_search only, no BM25/RRF/fallback."""
    chunks = semantic_search(query, top_k=TOP_K)
    if not chunks:
        return {"answer": "", "contexts": []}
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    answer = call_llm_with_retry(SYSTEM_PROMPT, user_message)
    return {"answer": answer, "contexts": [c["content"] for c in chunks]}


def build_samples(items: list[dict], config: str) -> list[dict]:
    samples = []
    for item in items:
        question = item["question"]
        print(f"  [{config}] generating for: {question[:70]}")
        if config == "dense_only":
            gen = dense_only_retrieve_and_generate(question)
        else:
            gen = generate_for_config(question, use_reranking=True)
        samples.append({
            "user_input": question,
            "response": gen["answer"] or "(no answer - empty retrieval)",
            "retrieved_contexts": gen["contexts"] or ["(no context retrieved)"],
            "reference": item["expected_answer"],
        })
        time.sleep(CALL_DELAY_SECONDS)
    return samples


def main() -> None:
    dataset = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    items = [dataset[i] for i in SELECTED_INDICES]
    print(f"Evaluating {len(items)} of {len(dataset)} golden questions (2 configs = {len(items) * 2} generations)")

    api_key = os.getenv("GEMINI_API_KEY")
    oai_compatible_client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    ragas_llm = llm_factory("gemini-3.6-flash", provider="openai", client=oai_compatible_client)
    ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-m3"))
    run_config = RunConfig(max_workers=2, max_retries=8, max_wait=90)
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    print("\n=== Config A: dense-only ===")
    samples_a = build_samples(items, "dense_only")
    GENERATIONS_PATH.write_text(
        json.dumps({"config_a": samples_a}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Checkpoint saved: {GENERATIONS_PATH}")

    print("\n=== Config B: hybrid + RRF ===")
    samples_b = build_samples(items, "hybrid_rrf")
    GENERATIONS_PATH.write_text(
        json.dumps({"config_a": samples_a, "config_b": samples_b}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Checkpoint saved: {GENERATIONS_PATH}")

    output: dict = {"selected_indices": SELECTED_INDICES, "questions": [item["question"] for item in items]}

    print("\n=== Running RAGAS evaluate() on Config A ===")
    result_a = evaluate(
        EvaluationDataset.from_list(samples_a), metrics=metrics,
        llm=ragas_llm, embeddings=ragas_embeddings, run_config=run_config,
    )
    output["config_a_dense_only"] = {
        "overall": {k: float(v) for k, v in result_a._repr_dict.items()},
        "per_sample": result_a.to_pandas().to_dict(orient="records"),
    }
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Saved (partial, config A only): {OUTPUT_PATH}")

    print("\n=== Running RAGAS evaluate() on Config B ===")
    result_b = evaluate(
        EvaluationDataset.from_list(samples_b), metrics=metrics,
        llm=ragas_llm, embeddings=ragas_embeddings, run_config=run_config,
    )
    output["config_b_hybrid_rrf"] = {
        "overall": {k: float(v) for k, v in result_b._repr_dict.items()},
        "per_sample": result_b.to_pandas().to_dict(orient="records"),
    }
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {OUTPUT_PATH}")
    print("\nConfig A (dense-only):", result_a)
    print("Config B (hybrid+RRF):", result_b)


if __name__ == "__main__":
    main()
