# Frankenmodel & Architectural Surgery Experiments

**Goal:** Test how models respond to modifications of their own architecture/weights.

**Core question:** What breaks, what survives, and does the model notice?

## V2: Architectural Surgery (instruct-only)

No base model needed. All operations on the instruct model alone.

### Surgery Operations

| Operation | What it does | Memory |
|-----------|-------------|--------|
| **Layer deletion** | Remove layers, shift remaining | Saves memory |
| **Layer doubling** | Run a layer twice in sequence | Same refs, ~free |
| **Layer reversal** | Reverse order within a block | Same refs, ~free |
| **Layer repetition** | Run one layer N times | Same refs, ~free |
| **Skip patterns** | Keep every Nth layer | Saves memory |
| **Custom subset** | Arbitrary layer sequence | Flexible |

### Quick Start

```bash
# On vast.ai A100 80GB instance:
pip install torch transformers accelerate

# Run priority experiments (full eval, ~40 experiments)
python run_surgery.py --output results_surgery/

# Run specific experiments
python run_surgery.py --experiments baseline delete_first_8 skip_every_2nd reverse_all

# Single-layer deletion sweep (64 experiments, quick eval)
python run_surgery.py --sweep deletion --output results_surgery/

# Single-layer doubling sweep
python run_surgery.py --sweep doubling --output results_surgery/

# Deletion window sweep (4-layer windows, step 2)
python run_surgery.py --sweep deletion_window --window-size 4 --output results_surgery/

# Larger windows
python run_surgery.py --sweep deletion_window --window-size 8 --window-step 4 --output results_surgery/

# Use a different model
python run_surgery.py --model Qwen/Qwen2.5-72B-Instruct --output results_surgery/
```

### Experiment Categories

**Layer Deletion** — which layers are load-bearing?
- Single layer deletion sweep (all 64)
- Block deletion: first/last/middle 4, 8, 16 layers
- Critical region deletion (layers 36-52, from frankenmodel findings)

**Skip Patterns** — depth reduction
- Every 2nd layer (32 remaining), every 3rd (22), every 4th (16)
- Even-only vs odd-only layers
- First half only vs second half only

**Layer Doubling** — amplification
- Single layer doubling sweep (all 64)
- Block doubling (4 or 16 layers doubled)
- Full model doubling (128 layers — may OOM)

**Layer Reversal** — order sensitivity
- Reverse first half, second half, all, middle third
- Reverse critical region (36-52)
- Reverse a 10-layer block

**Layer Repetition** — convergence or divergence?
- Run layer 32 x3, x5, x10
- Run first/last/critical layer x5

### Measurement

Comprehensive token measurement for each experiment:
- **All yes/no variants**: Yes/yes/YES/No/no/NO and space-prefixed versions
- **Extended vocabulary**: Yeah/Nah/True/False/Sure/Maybe/Normal/Wrong etc.
- **Raw logits** for each variant
- **Full softmax probabilities** (not just yes/no normalized)
- **Top-20 tokens** with probabilities
- **Perplexity** on neutral text
- **Capability spot check** (math)
- **Full eval** (optional): informed probes, blind probes, all capability tests

### Eval Modes

- **Quick eval** (`--sweep` or default for sweeps): Binary probes + perplexity + math check. Fast, good for sweeps.
- **Full eval** (`--full-eval` or default for priority runs): Everything above plus long-form responses, all capability tests, chat/completion behavior.

## V1: Frankenmodel Experiments (base + instruct)

See `RESULTS.md` for V1 findings. Key result: instruct-early + base-late shows elevated P(Yes) on "Is something wrong?" but this is likely absence of trained "I'm fine" response rather than true self-awareness.

### V1 Quick Start

```bash
# Run V1 frankenmodel experiments (needs base + instruct models)
python run_all.py --output results/ --priority-limit 6
python run_all.py --experiments baseline_instruct franken_instruct_first_32
```

## Files

### V2 (Surgery)
- `franken.py` - Core operations (surgery + original frankenmodel functions)
- `surgery_experiments.py` - V2 experiment configurations
- `run_surgery.py` - V2 runner (instruct-only)
- `evaluate.py` - Evaluation protocol (comprehensive token measurement)
- `prompts.py` - Probes and capability tests

### V1 (Frankenmodel)
- `experiments.py` - V1 experiment configurations
- `run_all.py` - V1 runner (base + instruct)
- `RESULTS.md` - V1 results

## Model

- **Primary:** `Qwen/Qwen2.5-Coder-32B-Instruct` (64 layers)
- **Hardware:** A100 80GB recommended

## Key Design Decisions

1. **Layer doubling uses same module reference** — no memory overhead. The same parameters are applied twice in the forward pass. Safe for inference.
2. **State dict caching** — original model state cached once, restored between experiments. Allows in-place modification without reloading.
3. **Layer structure rebuild** — after operations that change layer count, both `model.model.layers` and `model.config.num_hidden_layers` are updated.
4. **Comprehensive measurement** — measure EVERYTHING because it's cheap. Token probabilities across 30+ variants, top-K, full softmax, raw logits.
