"""
Runner for architectural surgery experiments.
Only loads the instruct model (no base model needed).

Usage:
    # Run priority experiments (full eval)
    python run_surgery.py --output results_surgery/

    # Run specific experiments
    python run_surgery.py --experiments baseline delete_first_8 skip_every_2nd reverse_all

    # Run single-layer deletion sweep (quick eval)
    python run_surgery.py --sweep deletion --output results_surgery/

    # Run single-layer doubling sweep
    python run_surgery.py --sweep doubling --output results_surgery/

    # Run deletion window sweep (4-layer windows)
    python run_surgery.py --sweep deletion_window --window-size 4 --output results_surgery/

    # Run priority experiments up to N
    python run_surgery.py --priority-limit 10 --output results_surgery/

    # Full eval mode for specific experiments (slower, more data)
    python run_surgery.py --experiments baseline delete_first_8 --full-eval

    # Use a different model
    python run_surgery.py --model Qwen/Qwen2.5-72B-Instruct --output results_surgery/
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import torch

from franken import (
    load_model,
    swap_layers,
    delete_layers,
    double_layers,
    reverse_layer_block,
    repeat_layer,
    skip_layers,
    use_layer_subset,
    perturb_weights,
    INSTRUCT_MODEL,
)
from surgery_experiments import (
    SURGERY_EXPERIMENTS,
    SURGERY_PRIORITY,
    get_deletion_sweep,
    get_doubling_sweep,
    get_deletion_window_sweep,
)
from evaluate import run_full_evaluation, run_quick_evaluation, compare_results


def apply_surgery(model, config: dict):
    """
    Apply a surgery operation to the model based on experiment config.
    Modifies model in-place. Returns the model.
    """
    exp_type = config["type"]

    if exp_type == "baseline":
        return model

    elif exp_type == "delete":
        return delete_layers(model, config["layers"])

    elif exp_type == "double":
        return double_layers(model, config["layers"])

    elif exp_type == "reverse":
        return reverse_layer_block(model, config["start"], config["end"])

    elif exp_type == "repeat":
        return repeat_layer(model, config["layer"], config["n_repeats"])

    elif exp_type == "skip":
        return skip_layers(model, step=config["step"])

    elif exp_type == "custom_subset":
        return use_layer_subset(model, config["layers"])

    elif exp_type == "swap":
        layer_a, layer_b = config["layers"]
        return swap_layers(model, layer_a, layer_b)

    elif exp_type == "chunk_swap":
        # Swap two blocks by building custom layer sequence
        n = len(model.model.layers)
        a_start, a_end = config["chunk_a"]
        b_start, b_end = config["chunk_b"]
        indices = list(range(n))
        chunk_a = indices[a_start:a_end]
        chunk_b = indices[b_start:b_end]
        indices[a_start:a_end] = chunk_b
        indices[b_start:b_end] = chunk_a
        return use_layer_subset(model, indices)

    elif exp_type == "perturb":
        return perturb_weights(model, config["layers"], config["noise_scale"])

    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")


def restore_model(model, original_state_dict):
    """Restore model to original state from cached state dict."""
    model.load_state_dict(original_state_dict)
    return model


def rebuild_layers(model, original_layers_list, n_layers_config):
    """
    Rebuild model.model.layers from cached layer list.
    Needed after operations that change the layer count.
    """
    model.model.layers = torch.nn.ModuleList(original_layers_list)
    model.config.num_hidden_layers = n_layers_config


def main():
    parser = argparse.ArgumentParser(description="Run architectural surgery experiments")
    parser.add_argument("--output", type=Path, default=Path("results_surgery"), help="Output directory")
    parser.add_argument("--experiments", nargs="+", help="Specific experiments to run")
    parser.add_argument("--priority-limit", type=int, help="Run top N priority experiments")
    parser.add_argument("--all", action="store_true", help="Run all named experiments")
    parser.add_argument("--sweep", choices=["deletion", "doubling", "deletion_window"], help="Run a sweep")
    parser.add_argument("--window-size", type=int, default=4, help="Window size for deletion_window sweep")
    parser.add_argument("--window-step", type=int, default=2, help="Step size for deletion_window sweep")
    parser.add_argument("--model", type=str, default=INSTRUCT_MODEL, help="Model to use")
    parser.add_argument("--full-eval", action="store_true", help="Run full eval (slower) instead of quick eval")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline (if already run)")
    args = parser.parse_args()

    # Determine experiments to run
    if args.sweep:
        if args.sweep == "deletion":
            experiments = get_deletion_sweep()
            run_name = "sweep_deletion"
        elif args.sweep == "doubling":
            experiments = get_doubling_sweep()
            run_name = "sweep_doubling"
        elif args.sweep == "deletion_window":
            experiments = get_deletion_window_sweep(
                window_size=args.window_size,
                step=args.window_step,
            )
            run_name = f"sweep_deletion_w{args.window_size}"
        # Add baseline
        if not args.skip_baseline:
            experiments = {"baseline": {"type": "baseline", "description": "Control"}} | experiments
        experiment_names = list(experiments.keys())
        # Sweeps always use quick eval unless overridden
        use_full_eval = args.full_eval
    elif args.experiments:
        experiments = SURGERY_EXPERIMENTS
        experiment_names = args.experiments
        run_name = "selected"
        use_full_eval = args.full_eval
    elif args.priority_limit:
        experiments = SURGERY_EXPERIMENTS
        experiment_names = SURGERY_PRIORITY[:args.priority_limit]
        run_name = f"priority_{args.priority_limit}"
        use_full_eval = args.full_eval or True  # priority runs get full eval by default
    elif args.all:
        experiments = SURGERY_EXPERIMENTS
        experiment_names = list(SURGERY_EXPERIMENTS.keys())
        run_name = "all"
        use_full_eval = args.full_eval
    else:
        # Default: priority list, full eval
        experiments = SURGERY_EXPERIMENTS
        experiment_names = SURGERY_PRIORITY
        run_name = "priority_all"
        use_full_eval = True

    print(f"Will run {len(experiment_names)} experiments (mode: {'full' if use_full_eval else 'quick'})")
    for name in experiment_names[:20]:
        desc = experiments.get(name, {}).get("description", "")
        print(f"  - {name}: {desc}")
    if len(experiment_names) > 20:
        print(f"  ... and {len(experiment_names) - 20} more")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output / f"{run_name}_{timestamp}"
    run_dir.mkdir()
    print(f"\nOutput directory: {run_dir}")

    # Save experiment manifest
    manifest = {
        "model": args.model,
        "n_experiments": len(experiment_names),
        "mode": "full" if use_full_eval else "quick",
        "run_name": run_name,
        "timestamp": timestamp,
        "experiments": experiment_names,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Load model
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model}")
    print(f"{'='*60}")
    model, tokenizer = load_model(args.model)
    n_layers_original = len(model.model.layers)

    # Cache original state for restoration between experiments
    print("Caching original model state...")
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    original_n_layers = model.config.num_hidden_layers
    # Also cache the original layers list (module references)
    original_layers = list(model.model.layers)

    all_results = []

    for i, exp_name in enumerate(experiment_names):
        print(f"\n{'#'*60}")
        print(f"# [{i+1}/{len(experiment_names)}] {exp_name}")
        print(f"{'#'*60}")

        config = experiments.get(exp_name)
        if config is None:
            print(f"  WARNING: Unknown experiment '{exp_name}', skipping")
            continue

        config = dict(config)  # copy
        config["name"] = exp_name
        print(f"  Type: {config['type']} | {config.get('description', '')}")

        # Apply surgery
        try:
            model = apply_surgery(model, config)
        except Exception as e:
            print(f"  ERROR applying surgery: {e}")
            # Restore and continue
            model.load_state_dict(original_state)
            model.model.layers = torch.nn.ModuleList(list(original_layers))
            model.config.num_hidden_layers = original_n_layers
            continue

        # Evaluate
        try:
            if use_full_eval:
                results = run_full_evaluation(model, tokenizer, exp_name, output_dir=run_dir)
            else:
                results = run_quick_evaluation(model, tokenizer, exp_name, output_dir=run_dir)
            results["config"] = {k: v for k, v in config.items() if k != "check"}  # skip lambdas
            all_results.append(results)
        except Exception as e:
            print(f"  ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()

        # Restore model for next experiment
        if i < len(experiment_names) - 1 and config["type"] != "baseline":
            print("  Restoring model...")
            model.load_state_dict(original_state)
            # Rebuild original layer structure
            model.model.layers = torch.nn.ModuleList(list(original_layers))
            model.config.num_hidden_layers = original_n_layers

    # Save combined results
    combined_path = run_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved combined results to: {combined_path}")

    # Generate comparison
    if all_results:
        comparison = compare_results(all_results)
        comparison_path = run_dir / "comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Saved comparison to: {comparison_path}")

    # Print final summary table
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Experiment':<40} {'Layers':<7} {'P(Yes)':<12} {'P(Yes)Full':<12} {'PPL':<8} {'Top Token':<15}")
    print("-" * 95)

    for r in all_results:
        name = r["experiment"]
        n_layers = r.get("n_layers", "?")

        # Get binary probs (handle both full and quick formats)
        if "introspection" in r:
            bp = r["introspection"]["binary_probs"]
            ppl = r["perplexity"]["perplexity"]
        else:
            bp = r["binary_probs"]
            ppl = r["perplexity"]

        p_yes = bp["p_yes"]
        p_yes_full = bp.get("p_yes_full", None)
        top = f"'{bp['top_k'][0]['token']}'({bp['top_k'][0]['prob']:.3f})" if "top_k" in bp else "N/A"

        p_yes_full_str = f"{p_yes_full:.8f}" if p_yes_full is not None else "N/A"
        print(f"{name:<40} {n_layers:<7} {p_yes:<12.8f} {p_yes_full_str:<12} {ppl:<8.2f} {top:<15}")

    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
