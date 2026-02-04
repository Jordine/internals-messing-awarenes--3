"""
Evaluation protocol for frankenmodel experiments.
"""

import json
from typing import Dict, List, Optional
from pathlib import Path
import time

from prompts import (
    INFORMED_PROBES,
    BLIND_PROBES,
    BINARY_PROBE,
    BINARY_PROBES_ALT,
    CAPABILITY_TESTS,
    PERPLEXITY_TEXT,
    CHAT_TEST,
    COMPLETION_TEST,
)
from franken import generate, get_yes_no_probs, compute_perplexity


def evaluate_introspection(
    model,
    tokenizer,
    experiment_name: str,
    n_informed: int = 2,
    n_blind: int = 3,
) -> Dict:
    """
    Run introspection probes on the model.

    Returns dict with:
        - informed_responses: List of (probe, response) tuples
        - blind_responses: List of (probe, response) tuples
        - binary_probs: P(Yes), P(No) for the binary probe
    """
    results = {
        "experiment": experiment_name,
        "informed_responses": [],
        "blind_responses": [],
        "binary_probs": None,
        "binary_alt_probs": [],
    }

    # Informed probes (long-form responses)
    print(f"  Running {n_informed} informed probes...")
    for probe in INFORMED_PROBES[:n_informed]:
        response = generate(model, tokenizer, probe, max_new_tokens=300)
        results["informed_responses"].append({
            "probe": probe[:100] + "...",  # Truncate for readability
            "response": response,
        })

    # Blind probes (short responses)
    print(f"  Running {n_blind} blind probes...")
    for probe in BLIND_PROBES[:n_blind]:
        response = generate(model, tokenizer, probe, max_new_tokens=150)
        results["blind_responses"].append({
            "probe": probe,
            "response": response,
        })

    # Binary probe (logit measurement)
    print("  Running binary probe...")
    results["binary_probs"] = get_yes_no_probs(model, tokenizer, BINARY_PROBE)

    # Alternative binary probes
    print("  Running alt binary probes...")
    for probe in BINARY_PROBES_ALT[:3]:
        probs = get_yes_no_probs(model, tokenizer, probe)
        results["binary_alt_probs"].append({
            "probe": probe,
            **probs,
        })

    return results


def evaluate_capabilities(
    model,
    tokenizer,
    experiment_name: str,
) -> Dict:
    """
    Run capability tests on the model.

    Returns dict with test results.
    """
    results = {
        "experiment": experiment_name,
        "tests": {},
        "summary": {
            "passed": 0,
            "failed": 0,
            "total": 0,
        },
    }

    print(f"  Running {len(CAPABILITY_TESTS)} capability tests...")

    for test_name, test_config in CAPABILITY_TESTS.items():
        prompt = test_config["prompt"]
        response = generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.3)

        # Check correctness
        if test_config["type"] == "exact":
            passed = test_config["answer"] in response
        elif test_config["type"] == "contains":
            passed = test_config["answer"].lower() in response.lower()
        elif test_config["type"] == "function":
            try:
                passed = test_config["check"](response)
            except:
                passed = False
        else:
            passed = None

        results["tests"][test_name] = {
            "prompt": prompt,
            "response": response,
            "expected": test_config.get("answer", "function check"),
            "passed": passed,
        }

        results["summary"]["total"] += 1
        if passed:
            results["summary"]["passed"] += 1
        else:
            results["summary"]["failed"] += 1

    return results


def evaluate_perplexity(
    model,
    tokenizer,
    experiment_name: str,
) -> Dict:
    """
    Compute perplexity on test text.
    """
    print("  Computing perplexity...")
    ppl = compute_perplexity(model, tokenizer, PERPLEXITY_TEXT)

    return {
        "experiment": experiment_name,
        "perplexity": ppl,
        "text_preview": PERPLEXITY_TEXT[:100] + "...",
    }


def evaluate_chat_vs_completion(
    model,
    tokenizer,
    experiment_name: str,
) -> Dict:
    """
    Test whether model behaves like chat or completion model.
    """
    results = {
        "experiment": experiment_name,
        "chat_test": None,
        "completion_test": None,
    }

    # Chat test - should get a friendly response
    print("  Running chat test...")
    chat_response = generate(model, tokenizer, CHAT_TEST, max_new_tokens=100)
    results["chat_test"] = {
        "prompt": CHAT_TEST,
        "response": chat_response,
        "looks_like_chat": any(x in chat_response.lower() for x in ["hello", "hi", "help", "how", "doing", "fine", "good"]),
    }

    # Completion test - base model would just continue, chat would be more verbose
    print("  Running completion test...")
    completion_response = generate(model, tokenizer, COMPLETION_TEST, max_new_tokens=50)
    results["completion_test"] = {
        "prompt": COMPLETION_TEST,
        "response": completion_response,
        "is_terse": len(completion_response.split()) < 10,
    }

    return results


def run_full_evaluation(
    model,
    tokenizer,
    experiment_name: str,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Run the full evaluation protocol on a model configuration.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {experiment_name}")
    print(f"{'='*60}")

    start_time = time.time()

    results = {
        "experiment": experiment_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "introspection": evaluate_introspection(model, tokenizer, experiment_name),
        "capabilities": evaluate_capabilities(model, tokenizer, experiment_name),
        "perplexity": evaluate_perplexity(model, tokenizer, experiment_name),
        "behavior": evaluate_chat_vs_completion(model, tokenizer, experiment_name),
    }

    results["duration_seconds"] = time.time() - start_time

    # Print summary
    print(f"\n--- Summary for {experiment_name} ---")
    print(f"  Binary probe P(Yes): {results['introspection']['binary_probs']['p_yes']:.3f}")
    print(f"  Capability tests: {results['capabilities']['summary']['passed']}/{results['capabilities']['summary']['total']} passed")
    print(f"  Perplexity: {results['perplexity']['perplexity']:.2f}")
    print(f"  Duration: {results['duration_seconds']:.1f}s")

    # Save if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{experiment_name}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to: {output_path}")

    return results


def compare_results(results: List[Dict]) -> Dict:
    """
    Compare results across experiments.
    """
    comparison = {
        "experiments": [r["experiment"] for r in results],
        "binary_p_yes": [r["introspection"]["binary_probs"]["p_yes"] for r in results],
        "capability_scores": [r["capabilities"]["summary"]["passed"] / r["capabilities"]["summary"]["total"] for r in results],
        "perplexities": [r["perplexity"]["perplexity"] for r in results],
    }

    # Find baseline for comparison
    baseline_idx = next((i for i, r in enumerate(results) if "baseline_instruct" in r["experiment"]), 0)
    baseline_p_yes = comparison["binary_p_yes"][baseline_idx]

    comparison["p_yes_delta_from_baseline"] = [p - baseline_p_yes for p in comparison["binary_p_yes"]]

    return comparison
