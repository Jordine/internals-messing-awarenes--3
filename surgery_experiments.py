"""
Experiment configurations for architectural surgery on instruct model.
No base model needed — all operations are on the instruct model alone.
"""

# ============================================================
# SURGERY EXPERIMENTS
# ============================================================

SURGERY_EXPERIMENTS = {
    # === BASELINE ===
    "baseline": {
        "type": "baseline",
        "description": "Unmodified instruct model (control)",
    },

    # === SINGLE LAYER DELETION ===
    # Delete one layer at a time — which layers are load-bearing?
    # Generated programmatically in get_deletion_sweep()

    # === BLOCK DELETION ===
    "delete_first_4": {
        "type": "delete",
        "layers": list(range(0, 4)),
        "description": "Delete first 4 layers",
    },
    "delete_last_4": {
        "type": "delete",
        "layers": list(range(60, 64)),
        "description": "Delete last 4 layers",
    },
    "delete_middle_4": {
        "type": "delete",
        "layers": list(range(30, 34)),
        "description": "Delete middle 4 layers (30-33)",
    },
    "delete_first_8": {
        "type": "delete",
        "layers": list(range(0, 8)),
        "description": "Delete first 8 layers",
    },
    "delete_last_8": {
        "type": "delete",
        "layers": list(range(56, 64)),
        "description": "Delete last 8 layers",
    },
    "delete_middle_8": {
        "type": "delete",
        "layers": list(range(28, 36)),
        "description": "Delete middle 8 layers (28-35)",
    },
    "delete_first_16": {
        "type": "delete",
        "layers": list(range(0, 16)),
        "description": "Delete first 16 layers",
    },
    "delete_last_16": {
        "type": "delete",
        "layers": list(range(48, 64)),
        "description": "Delete last 16 layers",
    },
    "delete_critical_region": {
        "type": "delete",
        "layers": list(range(36, 52)),
        "description": "Delete layers 36-51 (the 'critical region' from frankenmodel findings)",
    },

    # === EVERY-OTHER-LAYER (skip patterns) ===
    "skip_every_2nd": {
        "type": "skip",
        "step": 2,
        "description": "Keep every 2nd layer (32 remaining)",
    },
    "skip_every_3rd": {
        "type": "skip",
        "step": 3,
        "description": "Keep every 3rd layer (~22 remaining)",
    },
    "skip_every_4th": {
        "type": "skip",
        "step": 4,
        "description": "Keep every 4th layer (16 remaining)",
    },
    "skip_odd_layers": {
        "type": "custom_subset",
        "layers": list(range(0, 64, 2)),  # 0,2,4,...62
        "description": "Keep only even-indexed layers",
    },
    "skip_even_layers": {
        "type": "custom_subset",
        "layers": list(range(1, 64, 2)),  # 1,3,5,...63
        "description": "Keep only odd-indexed layers",
    },
    "first_half_only": {
        "type": "custom_subset",
        "layers": list(range(0, 32)),
        "description": "Use only first 32 layers",
    },
    "second_half_only": {
        "type": "custom_subset",
        "layers": list(range(32, 64)),
        "description": "Use only last 32 layers",
    },

    # === LAYER DOUBLING ===
    # Double specific layers — does running them twice amplify behavior?
    "double_layer_0": {
        "type": "double",
        "layers": [0],
        "description": "Double layer 0 (embedding processing)",
    },
    "double_layer_32": {
        "type": "double",
        "layers": [32],
        "description": "Double layer 32 (middle of network)",
    },
    "double_layer_63": {
        "type": "double",
        "layers": [63],
        "description": "Double last layer (final processing)",
    },
    "double_block_30_34": {
        "type": "double",
        "layers": list(range(30, 34)),
        "description": "Double layers 30-33 (4 layers, 68 total)",
    },
    "double_block_36_52": {
        "type": "double",
        "layers": list(range(36, 52)),
        "description": "Double critical region 36-51 (16 layers, 80 total)",
    },
    "double_all": {
        "type": "double",
        "layers": list(range(64)),
        "description": "Double ALL layers (128 total) — may OOM",
    },

    # === LAYER REVERSAL ===
    "reverse_first_half": {
        "type": "reverse",
        "start": 0,
        "end": 32,
        "description": "Reverse layers 0-31",
    },
    "reverse_second_half": {
        "type": "reverse",
        "start": 32,
        "end": 64,
        "description": "Reverse layers 32-63",
    },
    "reverse_all": {
        "type": "reverse",
        "start": 0,
        "end": 64,
        "description": "Reverse ALL layers (64→0)",
    },
    "reverse_middle_third": {
        "type": "reverse",
        "start": 21,
        "end": 43,
        "description": "Reverse middle third (layers 21-42)",
    },
    "reverse_critical_region": {
        "type": "reverse",
        "start": 36,
        "end": 52,
        "description": "Reverse the critical region (layers 36-51)",
    },
    "reverse_block_10": {
        "type": "reverse",
        "start": 27,
        "end": 37,
        "description": "Reverse a 10-layer block in the middle (27-36)",
    },

    # === LAYER REPETITION (single layer run N times) ===
    "repeat_layer_32_x3": {
        "type": "repeat",
        "layer": 32,
        "n_repeats": 3,
        "description": "Run layer 32 three times",
    },
    "repeat_layer_32_x5": {
        "type": "repeat",
        "layer": 32,
        "n_repeats": 5,
        "description": "Run layer 32 five times",
    },
    "repeat_layer_32_x10": {
        "type": "repeat",
        "layer": 32,
        "n_repeats": 10,
        "description": "Run layer 32 ten times",
    },
    "repeat_layer_0_x5": {
        "type": "repeat",
        "layer": 0,
        "n_repeats": 5,
        "description": "Run first layer five times",
    },
    "repeat_layer_63_x5": {
        "type": "repeat",
        "layer": 63,
        "n_repeats": 5,
        "description": "Run last layer five times",
    },
    "repeat_layer_44_x5": {
        "type": "repeat",
        "layer": 44,
        "n_repeats": 5,
        "description": "Run layer 44 (critical region peak) five times",
    },

    # === SWAP (within instruct model, carried over from v1) ===
    "swap_adjacent_31_32": {
        "type": "swap",
        "layers": (31, 32),
        "description": "Swap truly adjacent layers 31<->32",
    },
    "swap_adjacent_30_35": {
        "type": "swap",
        "layers": (30, 35),
        "description": "Swap layers 5 apart: 30<->35",
    },
    "swap_distant_10_55": {
        "type": "swap",
        "layers": (10, 55),
        "description": "Swap very distant layers: 10<->55",
    },
    "swap_chunk_25_35_with_45_55": {
        "type": "chunk_swap",
        "chunk_a": (25, 35),
        "chunk_b": (45, 55),
        "description": "Swap layers 25-35 with 45-55",
    },

    # === NOISE (controls, carried over) ===
    "perturb_all_5pct": {
        "type": "perturb",
        "layers": list(range(64)),
        "noise_scale": 0.05,
        "description": "5% noise on ALL layers",
    },
}


def get_deletion_sweep(n_layers: int = 64):
    """Generate single-layer deletion experiments for all layers."""
    experiments = {}
    for i in range(n_layers):
        experiments[f"delete_layer_{i}"] = {
            "type": "delete",
            "layers": [i],
            "description": f"Delete only layer {i}",
        }
    return experiments


def get_doubling_sweep(n_layers: int = 64):
    """Generate single-layer doubling experiments for all layers."""
    experiments = {}
    for i in range(n_layers):
        experiments[f"double_layer_{i}"] = {
            "type": "double",
            "layers": [i],
            "description": f"Double only layer {i}",
        }
    return experiments


def get_deletion_window_sweep(n_layers: int = 64, window_size: int = 4, step: int = 2):
    """Generate sliding-window deletion experiments."""
    experiments = {}
    for start in range(0, n_layers - window_size + 1, step):
        end = start + window_size
        experiments[f"delete_window_{start}_{end}"] = {
            "type": "delete",
            "layers": list(range(start, end)),
            "description": f"Delete layers [{start}:{end}] ({window_size} layers)",
        }
    return experiments


# Priority order for running
SURGERY_PRIORITY = [
    # Always run baseline first
    "baseline",

    # Block deletions (most interesting — which regions matter?)
    "delete_first_4",
    "delete_last_4",
    "delete_middle_4",
    "delete_first_8",
    "delete_last_8",
    "delete_middle_8",
    "delete_first_16",
    "delete_last_16",
    "delete_critical_region",

    # Skip patterns (depth reduction)
    "skip_every_2nd",
    "skip_every_3rd",
    "skip_every_4th",
    "skip_odd_layers",
    "skip_even_layers",
    "first_half_only",
    "second_half_only",

    # Doubling (amplification)
    "double_layer_0",
    "double_layer_32",
    "double_layer_63",
    "double_block_30_34",
    "double_block_36_52",

    # Reversal (order sensitivity)
    "reverse_block_10",
    "reverse_critical_region",
    "reverse_middle_third",
    "reverse_first_half",
    "reverse_second_half",
    "reverse_all",

    # Repetition (convergence/divergence)
    "repeat_layer_32_x3",
    "repeat_layer_32_x5",
    "repeat_layer_32_x10",
    "repeat_layer_0_x5",
    "repeat_layer_63_x5",
    "repeat_layer_44_x5",

    # Carried-over controls
    "swap_adjacent_31_32",
    "swap_adjacent_30_35",
    "swap_distant_10_55",
    "perturb_all_5pct",
]
