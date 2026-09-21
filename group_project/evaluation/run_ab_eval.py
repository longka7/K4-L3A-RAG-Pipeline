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
import math
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

# EVAL_LIMIT is the batch size; EVAL_START_INDEX selects the first question in the batch.
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "15"))
EVAL_START_INDEX = int(os.getenv("EVAL_START_INDEX", "0"))
EVAL_RESET_RESULTS = os.getenv("EVAL_RESET_RESULTS", "0") == "1"

TOP_K = 5
CALL_DELAY_SECONDS = 3  # pacing between our own generation calls (separate from RAGAS's own metric calls)
RAGAS_MAX_RETRIES = int(os.getenv("RAGAS_MAX_RETRIES", "0"))
RAGAS_MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "1"))
LLM_MAX_RETRIES = int(os.getenv("EVAL_LLM_MAX_RETRIES", "4"))
LLM_MAX_BACKOFF_SECONDS = int(os.getenv("EVAL_LLM_MAX_BACKOFF_SECONDS", "60"))
EVAL_MODE = os.getenv("EVAL_MODE", "full").strip().lower()
EVAL_CONFIG = os.getenv("EVAL_CONFIG", "both").strip().lower()


def call_llm_with_retry(
    system_prompt: str,
    user_message: str,
    attempts: int = LLM_MAX_RETRIES + 1,
) -> str:
    """Retry transient failures and provider rate limits with backoff."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call_llm(system_prompt, user_message)
        except Exception as error:  # transient network/server errors
            last_error = error
            status_code = getattr(error, "status_code", None)
            error_code = getattr(error, "code", None)
            error_text = str(error).lower()
            is_rate_limited = (
                status_code == 429
                or error_code == 1300
                or "rate limit" in error_text
                or "rate_limited" in error_text
            )
            if is_rate_limited:
                if attempt == attempts:
                    raise RuntimeError(
                        f"Provider rate limit after {attempts} attempts (HTTP 429/code 1300). "
                        "Wait for the provider window to reset or use an account/model with quota."
                    ) from error

                headers = getattr(getattr(error, "response", None), "headers", {})
                retry_after = headers.get("retry-after") if headers else None
                try:
                    wait_seconds = float(retry_after) if retry_after else 2 ** attempt
                except (TypeError, ValueError):
                    wait_seconds = 2 ** attempt
                wait_seconds = min(wait_seconds, LLM_MAX_BACKOFF_SECONDS)
                print(
                    f"    rate limit on attempt {attempt}/{attempts}; "
                    f"retrying in {wait_seconds:g}s"
                )
                time.sleep(wait_seconds)
                continue
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
        contexts = gen["contexts"]
        if contexts:
            preview = " ".join(contexts[0].split())[:160]
            print(f"    retrieved {len(contexts)} context chunks")
            print(f"    context preview: {preview}...")
        else:
            print("    retrieved 0 context chunks")
        samples.append({
            "user_input": question,
            "response": gen["answer"] or "(no answer - empty retrieval)",
            "retrieved_contexts": contexts or ["(no context retrieved)"],
            "reference": item["expected_answer"],
        })
        time.sleep(CALL_DELAY_SECONDS)
    return samples


def load_valid_checkpoint(dataset: list[dict]) -> tuple[list[dict], list[dict]] | None:
    """Load the complete generation checkpoint when it matches the dataset."""
    if not GENERATIONS_PATH.exists():
        return None
    try:
        checkpoint = json.loads(GENERATIONS_PATH.read_text(encoding="utf-8"))
        samples_a = checkpoint["config_a"]
        samples_b = checkpoint["config_b"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if len(samples_a) != len(dataset) or len(samples_b) != len(dataset):
        return None
    questions = [item["question"] for item in dataset]
    if [sample.get("user_input") for sample in samples_a] != questions:
        return None
    if [sample.get("user_input") for sample in samples_b] != questions:
        return None
    all_samples = samples_a + samples_b
    if any(sample.get("retrieved_contexts") == ["(no context retrieved)"] for sample in all_samples):
        return None
    return samples_a, samples_b


def merge_config_result(
    existing: dict,
    config_key: str,
    new_result: dict,
    samples: list[dict],
    dataset: list[dict],
) -> None:
    """Merge one evaluated batch and recompute means over all completed samples."""
    config_result = existing.setdefault(config_key, {"overall": {}, "per_sample": []})
    by_question = {
        sample["user_input"]: sample for sample in config_result.get("per_sample", [])
    }
    for sample in new_result.to_pandas().to_dict(orient="records"):
        question = sample["user_input"]
        by_question[question] = {**by_question.get(question, {}), **sample}

    ordered_samples = [
        by_question[question]
        for question in (item["question"] for item in dataset)
        if question in by_question
    ]
    metric_names = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    config_result["per_sample"] = ordered_samples
    config_result["overall"] = {
        metric: (
            sum(
                float(sample[metric])
                for sample in ordered_samples
                if isinstance(sample.get(metric), (int, float))
                and not math.isnan(float(sample[metric]))
            )
            / sum(
                1
                for sample in ordered_samples
                if isinstance(sample.get(metric), (int, float))
                and not math.isnan(float(sample[metric]))
            )
            if any(
                isinstance(sample.get(metric), (int, float))
                and not math.isnan(float(sample[metric]))
                for sample in ordered_samples
            )
            else float("nan")
        )
        for metric in metric_names
    }


def evaluate_config_one_metric_at_a_time(
    samples: list[dict],
    config_key: str,
    metrics: list,
    ragas_llm,
    ragas_embeddings,
    run_config: RunConfig,
    output: dict,
    dataset: list[dict],
) -> None:
    """Run one question and one metric per Ragas evaluation call."""
    for metric in metrics:
        metric_name = metric.name
        print(f"\n=== Running RAGAS {config_key} / {metric_name} ===")
        result = evaluate(
            EvaluationDataset.from_list(samples),
            metrics=[metric],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
        )
        merge_config_result(output, config_key, result, samples, dataset)
        OUTPUT_PATH.write_text(
            json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"Saved {metric_name} batch to {OUTPUT_PATH}")


def main() -> None:
    dataset = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if EVAL_LIMIT <= 0:
        raise ValueError("EVAL_LIMIT must be greater than 0")
    if EVAL_START_INDEX < 0 or EVAL_START_INDEX >= len(dataset):
        raise ValueError(f"EVAL_START_INDEX must be between 0 and {len(dataset) - 1}")
    if EVAL_MODE not in {"full", "generation_only"}:
        raise ValueError("EVAL_MODE must be 'full' or 'generation_only'")
    if EVAL_CONFIG not in {"both", "dense_only", "hybrid_rrf"}:
        raise ValueError("EVAL_CONFIG must be 'both', 'dense_only', or 'hybrid_rrf'")
    selected_indices = list(
        range(EVAL_START_INDEX, min(EVAL_START_INDEX + EVAL_LIMIT, len(dataset)))
    )
    items = [dataset[i] for i in selected_indices]
    print(
        f"Evaluating questions {selected_indices[0]}-{selected_indices[-1]} "
        f"of {len(dataset)} (batch size {len(items)})"
    )

    evaluator_provider = os.getenv("RAGAS_PROVIDER", os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
    if evaluator_provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        evaluator_model = os.getenv("RAGAS_MODEL", "mistral-small-latest")
        base_url = "https://api.mistral.ai/v1"
    else:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        evaluator_model = os.getenv("RAGAS_MODEL", "gemini-3.6-flash")
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    if not api_key:
        raise RuntimeError(f"Missing API key for RAGAS_PROVIDER={evaluator_provider}")
    oai_compatible_client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    ragas_llm = llm_factory(evaluator_model, provider="openai", client=oai_compatible_client)
    ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-m3"))
    run_config = RunConfig(
        max_workers=RAGAS_MAX_WORKERS,
        max_retries=RAGAS_MAX_RETRIES,
        max_wait=90,
    )
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    if EVAL_MODE == "generation_only" and EVAL_CONFIG != "both":
        build_samples(items, EVAL_CONFIG)
        print(f"Generation smoke test completed for {EVAL_CONFIG}; skipping RAGAS judge calls.")
        return

    checkpoint = load_valid_checkpoint(dataset)
    if checkpoint:
        all_samples_a, all_samples_b = checkpoint
        samples_a = [all_samples_a[index] for index in selected_indices]
        samples_b = [all_samples_b[index] for index in selected_indices]
        print(f"Reusing completed generations from {GENERATIONS_PATH}")
    else:
        raise RuntimeError(
            f"{GENERATIONS_PATH} is missing or does not contain all {len(dataset)} questions. "
            "Run generation_only first to create the complete checkpoint."
        )

    if EVAL_MODE == "generation_only":
        print("Generation smoke test completed; skipping RAGAS judge calls.")
        return

    if EVAL_RESET_RESULTS or not OUTPUT_PATH.exists():
        output: dict = {"selected_indices": [], "questions": []}
    else:
        try:
            output = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        except (OSError, TypeError, json.JSONDecodeError):
            output = {"selected_indices": [], "questions": []}
    completed_indices = set(output.get("selected_indices", [])) | set(selected_indices)
    output["selected_indices"] = sorted(completed_indices)
    output["questions"] = [dataset[index]["question"] for index in output["selected_indices"]]

    evaluate_config_one_metric_at_a_time(
        samples_a,
        "config_a_dense_only",
        metrics,
        ragas_llm,
        ragas_embeddings,
        run_config,
        output,
        dataset,
    )
    evaluate_config_one_metric_at_a_time(
        samples_b,
        "config_b_hybrid_rrf",
        metrics,
        ragas_llm,
        ragas_embeddings,
        run_config,
        output,
        dataset,
    )
    print(f"\nSaved: {OUTPUT_PATH}")
    print("Config A (dense-only):", output.get("config_a_dense_only", {}).get("overall", {}))
    print("Config B (hybrid+RRF):", output.get("config_b_hybrid_rrf", {}).get("overall", {}))


if __name__ == "__main__":
    main()