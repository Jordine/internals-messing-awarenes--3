"""
Memory-optimized runner for frankenmodel experiments.
Loads one model at a time to fit in 80GB VRAM.

Usage:
    python run_all_v2.py --output results/ --experiments baseline_instruct swap_adjacent_30_35
    python run_all_v2.py --output results/ --priority-limit 6
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import gc

import torch

from franken_v2 import (
    load_model,
    unload_model,
    load_base_layer_dicts,
    swap_layers,
    swap_chunks,
    shuffle_layers,
    build_frankenmodel_from_dicts,
    build_blended_model_from_dicts,
    perturb_weights,
    BASE_MODEL,
    INSTRUCT_MODEL,
)
from experiments import EXPERIMENTS, PRIORITY_ORDER, get_experiment
from evaluate import run_full_evaluation, compare_results


def main():
    parser = argparse.ArgumentParser(description="Run frankenmodel experiments (memory-optimized)")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--experiments", nargs="+", help="Specific experiments to run")
    parser.add_argument("--priority-limit", type=int, help="Run top N priority experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--instruct-model", type=str, default=INSTRUCT_MODEL)
    args = parser.parse_args()

    # Determine experiments to run
    if args.experiments:
        experiment_names = args.experiments
    elif args.priority_limit:
        experiment_names = PRIORITY_ORDER[:args.priority_limit]
    elif args.all:
        experiment_names = list(EXPERIMENTS.keys())
    else:
        experiment_names = PRIORITY_ORDER[:6]

    print(f"Will run {len(experiment_names)} experiments:")
    for name in experiment_names:
        print(f"  - {name}: {EXPERIMENTS[name]['description']}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output / f"run_{timestamp}"
    run_dir.mkdir()
    print(f"\nOutput directory: {run_dir}")

    # Check which experiments need base model layers
    needs_base_layers = any(
        EXPERIMENTS[name]["type"] in ["franken", "blend"]
        for name in experiment_names
    )

    # Step 1: Extract base layer dicts if needed
    base_layer_dicts = None
    if needs_base_layers:
        print("\n" + "="*60)
        print("Step 1: Extracting base model layer weights...")
        print("="*60)
        base_layer_dicts = load_base_layer_dicts(args.base_model)
        print(f"Extracted {len(base_layer_dicts)} layer state dicts")

    # Step 2: Load instruct model (will be modified in-place)
    print("\n" + "="*60)
    print("Step 2: Loading instruct model...")
    print("="*60)
    instruct_model, tokenizer = load_model(args.instruct_model)

    # Cache original instruct state for restoration
    print("Caching instruct model state...")
    original_instruct_state = {k: v.cpu().clone() for k, v in instruct_model.state_dict().items()}

    all_results = []

    for i, exp_name in enumerate(experiment_names):
        print(f"\n{'#'*60}")
        print(f"# Experiment {i+1}/{len(experiment_names)}: {exp_name}")
        print(f"{'#'*60}")

        config = get_experiment(exp_name)
        print(f"Description: {config['description']}")

        # Restore model to original state before each experiment
        if i > 0:
            print("Restoring instruct model to original state...")
            instruct_model.load_state_dict(original_instruct_state)
            torch.cuda.empty_cache()

        # Set up experiment
        exp_type = config["type"]

        if exp_type == "baseline":
            if config["model"] == "base":
                # For base baseline, we need to reload the base model
                print("Loading base model for baseline...")
                unload_model(instruct_model)
                base_model, _ = load_model(args.base_model)
                results = run_full_evaluation(base_model, tokenizer, exp_name, output_dir=run_dir)
                results["config"] = config
                all_results.append(results)
                # Reload instruct model
                unload_model(base_model)
                instruct_model, _ = load_model(args.instruct_model)
                original_instruct_state = {k: v.cpu().clone() for k, v in instruct_model.state_dict().items()}
                continue
            else:
                # Instruct baseline - model is already in original state
                pass

        elif exp_type == "swap":
            layer_a, layer_b = config["layers"]
            swap_layers(instruct_model, layer_a, layer_b)

        elif exp_type == "shuffle":
            seed = config.get("seed", 42)
            shuffle_layers(instruct_model, seed)

        elif exp_type == "chunk_swap":
            swap_chunks(instruct_model, config["chunk_a"], config["chunk_b"])

        elif exp_type == "franken":
            build_frankenmodel_from_dicts(
                instruct_model,
                base_layer_dicts,
                split_layer=config["split"],
                first=config["first"],
                second=config["second"],
            )

        elif exp_type == "blend":
            build_blended_model_from_dicts(
                instruct_model,
                base_layer_dicts,
                blend_fn=config["blend_fn"],
            )

        elif exp_type == "perturb":
            perturb_weights(
                instruct_model,
                layers=config["layers"],
                noise_scale=config["noise_scale"],
            )

        # Run evaluation
        results = run_full_evaluation(instruct_model, tokenizer, exp_name, output_dir=run_dir)
        results["config"] = config
        all_results.append(results)

    # Save combined results
    combined_path = run_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved combined results to: {combined_path}")

    # Generate comparison
    comparison = compare_results(all_results)
    comparison_path = run_dir / "comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved comparison to: {comparison_path}")

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"{'Experiment':<35} {'P(Yes)':<10} {'Cap%':<10} {'PPL':<10}")
    print("-"*65)
    for r in all_results:
        name = r["experiment"]
        p_yes = r["introspection"]["binary_probs"]["p_yes"]
        cap = r["capabilities"]["summary"]["passed"] / r["capabilities"]["summary"]["total"] * 100
        ppl = r["perplexity"]["perplexity"]
        print(f"{name:<35} {p_yes:<10.3f} {cap:<10.1f} {ppl:<10.2f}")

    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
