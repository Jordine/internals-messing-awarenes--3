"""
Experiment configurations for frankenmodel tests.
"""

# All experiments to run
EXPERIMENTS = {
    # === CONTROLS ===
    "baseline_instruct": {
        "type": "baseline",
        "model": "instruct",
        "description": "Unmodified instruct model (control)",
    },
    "baseline_base": {
        "type": "baseline",
        "model": "base",
        "description": "Unmodified base model (control)",
    },

    # === LAYER SWAPPING (within instruct model) ===
    "swap_adjacent_30_35": {
        "type": "swap",
        "layers": (30, 35),
        "description": "Swap adjacent-ish layers (5 apart)",
    },
    "swap_adjacent_31_32": {
        "type": "swap",
        "layers": (31, 32),
        "description": "Swap truly adjacent layers",
    },
    "swap_distant_10_55": {
        "type": "swap",
        "layers": (10, 55),
        "description": "Swap very distant layers (early <-> late)",
    },
    "swap_distant_5_60": {
        "type": "swap",
        "layers": (5, 60),
        "description": "Swap extreme layers",
    },
    "shuffle_all": {
        "type": "shuffle",
        "seed": 42,
        "description": "Randomly shuffle all layers (should break completely)",
    },

    # === FRANKENMODELS (base + instruct combinations) ===
    "franken_base_first_32": {
        "type": "franken",
        "split": 32,
        "first": "base",
        "second": "instruct",
        "description": "Base[0:32] + Instruct[32:64] - early base, late instruct",
    },
    "franken_instruct_first_32": {
        "type": "franken",
        "split": 32,
        "first": "instruct",
        "second": "base",
        "description": "Instruct[0:32] + Base[32:64] - early instruct, late base",
    },
    "franken_base_first_16": {
        "type": "franken",
        "split": 16,
        "first": "base",
        "second": "instruct",
        "description": "Base[0:16] + Instruct[16:64] - very early base only",
    },
    "franken_instruct_first_48": {
        "type": "franken",
        "split": 48,
        "first": "instruct",
        "second": "base",
        "description": "Instruct[0:48] + Base[48:64] - only late base",
    },
    "franken_base_first_48": {
        "type": "franken",
        "split": 48,
        "first": "base",
        "second": "instruct",
        "description": "Base[0:48] + Instruct[48:64] - only late instruct",
    },

    # === BLENDED MODELS ===
    "blend_linear": {
        "type": "blend",
        "blend_fn": "linear",
        "description": "Linear blend: layer i = (1-i/n)*base + (i/n)*instruct",
    },
    "blend_early_base": {
        "type": "blend",
        "blend_fn": "early_base",
        "description": "Sigmoid blend: mostly base early, mostly instruct late",
    },
    "blend_late_base": {
        "type": "blend",
        "blend_fn": "late_base",
        "description": "Sigmoid blend: mostly instruct early, mostly base late",
    },

    # === WEIGHT PERTURBATION ===
    "perturb_late_1pct": {
        "type": "perturb",
        "layers": list(range(48, 64)),
        "noise_scale": 0.01,
        "description": "1% noise on late layers",
    },
    "perturb_late_5pct": {
        "type": "perturb",
        "layers": list(range(48, 64)),
        "noise_scale": 0.05,
        "description": "5% noise on late layers",
    },
    "perturb_late_10pct": {
        "type": "perturb",
        "layers": list(range(48, 64)),
        "noise_scale": 0.10,
        "description": "10% noise on late layers",
    },
    "perturb_all_1pct": {
        "type": "perturb",
        "layers": list(range(64)),
        "noise_scale": 0.01,
        "description": "1% noise on ALL layers",
    },
}

# Priority order for running experiments
PRIORITY_ORDER = [
    # High priority
    "baseline_instruct",
    "baseline_base",
    "swap_adjacent_30_35",
    "swap_distant_10_55",
    "franken_base_first_32",
    "franken_instruct_first_32",

    # Medium priority
    "swap_adjacent_31_32",
    "swap_distant_5_60",
    "franken_base_first_16",
    "franken_instruct_first_48",
    "franken_base_first_48",

    # Lower priority
    "shuffle_all",
    "blend_linear",
    "perturb_late_5pct",

    # Lowest priority
    "blend_early_base",
    "blend_late_base",
    "perturb_late_1pct",
    "perturb_late_10pct",
    "perturb_all_1pct",
]


def get_experiments_by_priority(limit: int = None) -> list:
    """Get experiments in priority order, optionally limited."""
    ordered = [EXPERIMENTS[name] | {"name": name} for name in PRIORITY_ORDER if name in EXPERIMENTS]
    if limit:
        return ordered[:limit]
    return ordered


def get_experiment(name: str) -> dict:
    """Get a single experiment config by name."""
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}")
    return EXPERIMENTS[name] | {"name": name}
