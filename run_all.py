"""
Main runner for frankenmodel experiments.

Usage:
    python run_all.py --output results/ --experiments baseline_instruct swap_adjacent_30_35
    python run_all.py --output results/ --priority-limit 6  # Run top 6 priority experiments
    python run_all.py --output results/ --all  # Run all experiments
"""

import argparse
import json
import copy
from pathlib import Path
from datetime import datetime

import torch

from franken import (
    load_model,
    load_model_pair,
    swap_layers,
    shuffle_layers,
    build_frankenmodel,
    build_blended_model,
    perturb_weights,
    BASE_MODEL,
    INSTRUCT_MODEL,
)
from experiments import EXPERIMENTS, PRIORITY_ORDER, get_experiments_by_priority, get_experiment
from evaluate import run_full_evaluation, compare_results


def setup_experiment(
    experiment_config: dict,
    base_model,
    instruct_model,
    tokenizer,
):
    """
    Set up a model configuration based on experiment config.

    Returns the model to evaluate (may modify instruct_model in-place).
    """
    exp_type = experiment_config["type"]
    name = experiment_config["name"]

    if exp_type == "baseline":
        if experiment_config["model"] == "base":
            return base_model
        else:
            return instruct_model

    elif exp_type == "swap":
        # We need a fresh copy for swapping
        # Actually, we'll swap and then swap back after evaluation
        layer_a, layer_b = experiment_config["layers"]
        swap_layers(instruct_model, layer_a, layer_b)
        return instruct_model

    elif exp_type == "shuffle":
        seed = experiment_config.get("seed", 42)
        shuffle_layers(instruct_model, seed)
        return instruct_model

    elif exp_type == "franken":
        return build_frankenmodel(
            base_model,
            instruct_model,
            split_layer=experiment_config["split"],
            first=experiment_config["first"],
            second=experiment_config["second"],
        )

    elif exp_type == "blend":
        return build_blended_model(
            base_model,
            instruct_model,
            blend_fn=experiment_config["blend_fn"],
        )

    elif exp_type == "perturb":
        return perturb_weights(
            instruct_model,
            layers=experiment_config["layers"],
            noise_scale=experiment_config["noise_scale"],
        )

    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")


def restore_model(model_name: str):
    """Reload a fresh model (after in-place modifications)."""
    print(f"Reloading fresh model: {model_name}")
    model, _ = load_model(model_name)
    return model


def main():
    parser = argparse.ArgumentParser(description="Run frankenmodel experiments")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--experiments", nargs="+", help="Specific experiments to run")
    parser.add_argument("--priority-limit", type=int, help="Run top N priority experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL, help="Base model name")
    parser.add_argument("--instruct-model", type=str, default=INSTRUCT_MODEL, help="Instruct model name")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline experiments")
    args = parser.parse_args()

    # Determine which experiments to run
    if args.experiments:
        experiment_names = args.experiments
    elif args.priority_limit:
        experiment_names = PRIORITY_ORDER[:args.priority_limit]
    elif args.all:
        experiment_names = list(EXPERIMENTS.keys())
    else:
        # Default: run high priority (first 6)
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

    # Load models
    print("\n" + "="*60)
    print("Loading models...")
    print("="*60)
    base_model, instruct_model, tokenizer = load_model_pair(args.base_model, args.instruct_model)

    # Store original instruct state for restoration
    # (This is memory-intensive but necessary for in-place modifications)
    print("Caching original instruct model state...")
    original_instruct_state = {k: v.clone() for k, v in instruct_model.state_dict().items()}

    all_results = []

    for i, exp_name in enumerate(experiment_names):
        print(f"\n{'#'*60}")
        print(f"# Experiment {i+1}/{len(experiment_names)}: {exp_name}")
        print(f"{'#'*60}")

        config = get_experiment(exp_name)
        print(f"Description: {config['description']}")

        # Set up the experiment
        model = setup_experiment(config, base_model, instruct_model, tokenizer)

        # Run evaluation
        results = run_full_evaluation(
            model,
            tokenizer,
            exp_name,
            output_dir=run_dir,
        )
        results["config"] = config
        all_results.append(results)

        # Restore instruct model to original state for next experiment
        # (unless this was a baseline or we're done)
        if config["type"] not in ["baseline"] and i < len(experiment_names) - 1:
            print("Restoring instruct model to original state...")
            instruct_model.load_state_dict(original_instruct_state)

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
