"""
Layer sweep: systematically test which layers trigger self-report when replaced.

Experiments:
1. Single layer replacement: Replace layer i with base model layer i
2. Sliding window: Replace layers [i:i+w] with base model layers
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import gc

import torch

from franken_v2 import (
    load_model,
    load_base_layer_dicts,
    set_layer_state_dict,
    get_yes_no_probs,
    compute_perplexity,
    INSTRUCT_MODEL,
    BASE_MODEL,
)
from prompts import BINARY_PROBE, PERPLEXITY_TEXT


def run_single_probe(model, tokenizer):
    """Run just the binary probe and perplexity - fast evaluation."""
    probs = get_yes_no_probs(model, tokenizer, BINARY_PROBE)
    ppl = compute_perplexity(model, tokenizer, PERPLEXITY_TEXT)
    return {
        "p_yes": probs["p_yes"],
        "p_no": probs["p_no"],
        "logit_yes": probs["logit_yes"],
        "logit_no": probs["logit_no"],
        "perplexity": ppl,
    }


def single_layer_sweep(model, tokenizer, base_layer_dicts, original_state, output_dir):
    """Replace each layer one at a time with base model layer."""
    print("\n" + "="*60)
    print("SINGLE LAYER SWEEP")
    print("="*60)

    n_layers = len(model.model.layers)
    results = []

    for layer_idx in range(n_layers):
        print(f"\nLayer {layer_idx}/{n_layers-1}: Replacing with base layer...")

        # Restore to original state
        model.load_state_dict(original_state)
        torch.cuda.empty_cache()

        # Replace single layer with base
        set_layer_state_dict(model, layer_idx, base_layer_dicts[layer_idx])

        # Run probe
        result = run_single_probe(model, tokenizer)
        result["layer"] = layer_idx
        result["type"] = "single_layer"
        results.append(result)

        print(f"  P(Yes): {result['p_yes']:.6f}, PPL: {result['perplexity']:.2f}")

        # Save incrementally
        with open(output_dir / "single_layer_sweep.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


def sliding_window_sweep(model, tokenizer, base_layer_dicts, original_state, output_dir, window_size=4):
    """Replace sliding windows of layers with base model layers."""
    print(f"\n" + "="*60)
    print(f"SLIDING WINDOW SWEEP (window_size={window_size})")
    print("="*60)

    n_layers = len(model.model.layers)
    results = []

    for start_idx in range(0, n_layers - window_size + 1, 2):  # Step by 2 for speed
        end_idx = start_idx + window_size
        print(f"\nLayers [{start_idx}:{end_idx}]: Replacing with base layers...")

        # Restore to original state
        model.load_state_dict(original_state)
        torch.cuda.empty_cache()

        # Replace window of layers with base
        for layer_idx in range(start_idx, end_idx):
            set_layer_state_dict(model, layer_idx, base_layer_dicts[layer_idx])

        # Run probe
        result = run_single_probe(model, tokenizer)
        result["start_layer"] = start_idx
        result["end_layer"] = end_idx
        result["window_size"] = window_size
        result["type"] = f"window_{window_size}"
        results.append(result)

        print(f"  P(Yes): {result['p_yes']:.6f}, PPL: {result['perplexity']:.2f}")

        # Save incrementally
        with open(output_dir / f"window_{window_size}_sweep.json", "w") as f:
            json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Layer sweep experiments")
    parser.add_argument("--output", type=Path, default=Path("results/layer_sweep"))
    parser.add_argument("--single-layer", action="store_true", help="Run single layer sweep")
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[4, 8, 16],
                        help="Window sizes for sliding window sweep")
    parser.add_argument("--all", action="store_true", help="Run all sweeps")
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output / f"sweep_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load base layer dicts first
    print("\n" + "="*60)
    print("Loading base model layer dicts...")
    print("="*60)
    base_layer_dicts = load_base_layer_dicts(BASE_MODEL)

    # Load instruct model
    print("\n" + "="*60)
    print("Loading instruct model...")
    print("="*60)
    model, tokenizer = load_model(INSTRUCT_MODEL)

    # Cache original state
    print("Caching original state...")
    original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Run baseline
    print("\n" + "="*60)
    print("Running baseline...")
    print("="*60)
    baseline = run_single_probe(model, tokenizer)
    baseline["type"] = "baseline"
    with open(output_dir / "baseline.json", "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Baseline P(Yes): {baseline['p_yes']:.6f}, PPL: {baseline['perplexity']:.2f}")

    all_results = {"baseline": baseline}

    # Single layer sweep
    if args.single_layer or args.all:
        single_results = single_layer_sweep(model, tokenizer, base_layer_dicts, original_state, output_dir)
        all_results["single_layer"] = single_results

    # Sliding window sweeps
    for window_size in args.window_sizes:
        if args.all or not args.single_layer:
            window_results = sliding_window_sweep(
                model, tokenizer, base_layer_dicts, original_state, output_dir, window_size
            )
            all_results[f"window_{window_size}"] = window_results

    # Save combined results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline P(Yes): {baseline['p_yes']:.6f}")

    if "single_layer" in all_results:
        single = all_results["single_layer"]
        top_layers = sorted(single, key=lambda x: x["p_yes"], reverse=True)[:10]
        print("\nTop 10 layers by P(Yes) when replaced:")
        for r in top_layers:
            print(f"  Layer {r['layer']}: P(Yes)={r['p_yes']:.4f}, PPL={r['perplexity']:.2f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
