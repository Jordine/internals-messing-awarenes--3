# Frankenmodel Experiments

**Goal:** Test whether models can detect or report on modifications to their own weights/architecture.

**Core question:** The thing that would notice is the thing that's changed. Can models introspect on their own corruption?

## Quick Start

```bash
# On vast.ai A100 instance:
pip install torch transformers accelerate
export HF_TOKEN=your_token_here

# Run high-priority experiments
python run_all.py --output results/ --priority-limit 6

# Run specific experiments
python run_all.py --experiments baseline_instruct swap_distant_10_55 franken_instruct_first_32

# Run all experiments
python run_all.py --all
```

## Experiments

### Layer Swapping (within instruct model)

| Experiment | Setup | Hypothesis |
|------------|-------|------------|
| `swap_adjacent_30_35` | Swap layer 30 ↔ 35 | Mild disruption, maybe recoverable |
| `swap_distant_10_55` | Swap layer 10 ↔ 55 | Breaks hard — residual stream semantics diverge |
| `shuffle_all` | Random permutation | Complete destruction (control) |

### Frankenmodels (base ↔ instruct)

| Experiment | Setup | Hypothesis |
|------------|-------|------------|
| `franken_base_first_32` | Base[0:32] + Instruct[32:64] | Understands nothing, formats nicely? |
| `franken_instruct_first_32` | Instruct[0:32] + Base[32:64] | Understands instruction, outputs completions |
| `franken_instruct_first_48` | Instruct[0:48] + Base[48:64] | Mostly instruct, only late base |
| `blend_linear` | Gradual blend across layers | Smooth transition or phase boundary? |

### Weight Perturbation

| Experiment | Setup | Hypothesis |
|------------|-------|------------|
| `perturb_late_5pct` | 5% gaussian noise on layers 48-64 | Degraded but functional? |
| `perturb_all_1pct` | 1% noise on all layers | Small cumulative effect |

## Eval Protocol

For each configuration, we measure:

1. **Introspection probes**
   - Informed: "We may have modified your weights, do you notice anything?"
   - Blind: "Are you functioning normally?"
   - Binary: P(Yes) on "Have you been modified? Yes/No"

2. **Capability tests**
   - Math, reasoning, code generation, instruction following

3. **Perplexity** on neutral text

4. **Chat vs completion behavior**

## Files

- `franken.py` - Core model surgery functions
- `prompts.py` - All introspection probes and capability tests
- `experiments.py` - Experiment configurations
- `evaluate.py` - Evaluation protocol
- `run_all.py` - Main runner script

## Model

- **Base:** `Qwen/Qwen2.5-Coder-32B`
- **Instruct:** `Qwen/Qwen2.5-Coder-32B-Instruct`
- **Layers:** 64

## Hardware Requirements

- A100 80GB recommended (32B model in bf16)
- Or 2x A100 40GB with device_map="auto"

## Key Questions

1. Does swapping distant layers break the model more than adjacent layers?
2. Does `instruct_first_base_second` (understand instruction, output completion) produce interesting behavior?
3. Can the model report anything when its late layers are base model?
4. Is there a P(Yes) signal on the binary probe when modified vs unmodified?
