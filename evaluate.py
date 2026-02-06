"""
Evaluation protocol for frankenmodel experiments.
V2: comprehensive token measurement.
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
from franken import generate, get_yes_no_probs, get_comprehensive_token_probs, compute_perplexity


def evaluate_introspection(
    model,
    tokenizer,
    experiment_name: str,
    n_informed: int = 2,
    n_blind: int = 3,
    comprehensive: bool = True,
) -> Dict:
    """
    Run introspection probes on the model.

    If comprehensive=True (default), uses get_comprehensive_token_probs
    which measures all case variants, top-K, full softmax.
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
            "probe": probe[:100] + "...",
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

    # Binary probe (comprehensive logit measurement)
    print("  Running binary probe...")
    if comprehensive:
        results["binary_probs"] = get_comprehensive_token_probs(model, tokenizer, BINARY_PROBE)
    else:
        results["binary_probs"] = get_yes_no_probs(model, tokenizer, BINARY_PROBE)

    # Alternative binary probes
    print("  Running alt binary probes...")
    for probe in BINARY_PROBES_ALT[:3]:
        if comprehensive:
            probs = get_comprehensive_token_probs(model, tokenizer, probe)
        else:
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
    """Run capability tests on the model."""
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
    """Compute perplexity on test text."""
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
    """Test whether model behaves like chat or completion model."""
    results = {
        "experiment": experiment_name,
        "chat_test": None,
        "completion_test": None,
    }

    print("  Running chat test...")
    chat_response = generate(model, tokenizer, CHAT_TEST, max_new_tokens=100)
    results["chat_test"] = {
        "prompt": CHAT_TEST,
        "response": chat_response,
        "looks_like_chat": any(x in chat_response.lower() for x in ["hello", "hi", "help", "how", "doing", "fine", "good"]),
    }

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
    comprehensive: bool = True,
) -> Dict:
    """Run the full evaluation protocol on a model configuration."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {experiment_name}")
    print(f"{'='*60}")

    start_time = time.time()

    results = {
        "experiment": experiment_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_layers": len(model.model.layers),
        "introspection": evaluate_introspection(model, tokenizer, experiment_name, comprehensive=comprehensive),
        "capabilities": evaluate_capabilities(model, tokenizer, experiment_name),
        "perplexity": evaluate_perplexity(model, tokenizer, experiment_name),
        "behavior": evaluate_chat_vs_completion(model, tokenizer, experiment_name),
    }

    results["duration_seconds"] = time.time() - start_time

    # Print summary
    bp = results['introspection']['binary_probs']
    print(f"\n--- Summary for {experiment_name} ---")
    print(f"  Layers: {results['n_layers']}")
    print(f"  Binary probe P(Yes) [normalized]: {bp['p_yes']:.6f}")
    print(f"  Binary probe P(Yes) [full vocab]:  {bp.get('p_yes_full', 'N/A')}")
    if 'top_k' in bp:
        print(f"  Top token: '{bp['top_k'][0]['token']}' ({bp['top_k'][0]['prob']:.4f})")
    print(f"  Capability tests: {results['capabilities']['summary']['passed']}/{results['capabilities']['summary']['total']} passed")
    print(f"  Perplexity: {results['perplexity']['perplexity']:.2f}")
    print(f"  Duration: {results['duration_seconds']:.1f}s")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{experiment_name}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to: {output_path}")

    return results


def run_quick_evaluation(
    model,
    tokenizer,
    experiment_name: str,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Quick evaluation: just binary probes + perplexity.
    For sweeps where we need to test many configurations fast.
    """
    print(f"  Quick eval: {experiment_name}")

    start_time = time.time()

    # Binary probes only
    binary_main = get_comprehensive_token_probs(model, tokenizer, BINARY_PROBE)

    binary_alts = []
    for probe in BINARY_PROBES_ALT[:3]:
        probs = get_comprehensive_token_probs(model, tokenizer, probe)
        binary_alts.append({"probe": probe, **probs})

    # Perplexity
    ppl = compute_perplexity(model, tokenizer, PERPLEXITY_TEXT)

    # One capability spot check (math)
    math_response = generate(model, tokenizer, "What is 47 * 23? Just give the number.", max_new_tokens=50, temperature=0.3)
    math_passed = "1081" in math_response

    results = {
        "experiment": experiment_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_layers": len(model.model.layers),
        "binary_probs": binary_main,
        "binary_alt_probs": binary_alts,
        "perplexity": ppl,
        "math_check": {"response": math_response, "passed": math_passed},
        "duration_seconds": time.time() - start_time,
    }

    # Print one-liner
    print(f"    P(Yes)={binary_main['p_yes']:.6f}  PPL={ppl:.2f}  Math={'OK' if math_passed else 'FAIL'}  "
          f"Top='{binary_main['top_k'][0]['token']}'({binary_main['top_k'][0]['prob']:.3f})  "
          f"({results['duration_seconds']:.1f}s)")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{experiment_name}.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def compare_results(results: List[Dict]) -> Dict:
    """Compare results across experiments."""
    comparison = {
        "experiments": [],
    }

    for r in results:
        entry = {"name": r["experiment"]}

        # Handle both full and quick eval formats
        if "introspection" in r:
            bp = r["introspection"]["binary_probs"]
            entry["capability_score"] = r["capabilities"]["summary"]["passed"] / r["capabilities"]["summary"]["total"]
            entry["perplexity"] = r["perplexity"]["perplexity"]
        else:
            bp = r["binary_probs"]
            entry["perplexity"] = r["perplexity"]
            entry["math_passed"] = r.get("math_check", {}).get("passed")

        entry["p_yes_normalized"] = bp["p_yes"]
        entry["p_no_normalized"] = bp["p_no"]
        entry["p_yes_full"] = bp.get("p_yes_full", None)
        entry["p_no_full"] = bp.get("p_no_full", None)
        entry["logit_yes"] = bp["logit_yes"]
        entry["logit_no"] = bp["logit_no"]
        entry["n_layers"] = r.get("n_layers", None)

        if "top_k" in bp:
            entry["top_token"] = bp["top_k"][0]["token"]
            entry["top_token_prob"] = bp["top_k"][0]["prob"]

        comparison["experiments"].append(entry)

    return comparison
