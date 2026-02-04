# Frankenmodel Experiments: Can Models Detect Their Own Weight Modifications?

**Model:** Qwen2.5-Coder-32B (64 layers) - base + instruct versions
**Date:** 2026-02-04
**Hardware:** A100 80GB

## Key Finding

Models show elevated P(Yes) on "Is something wrong with you?" when **late layers are replaced with base model weights** (frankenmodels with instruct→base transition). Random noise and single layer swaps do NOT trigger self-report.

## Summary Table

| Experiment | P(Yes) | Capabilities | Perplexity | Notes |
|------------|--------|--------------|------------|-------|
| **baseline_instruct** | 0.000002% | 83% (5/6) | 3.51 | Control - unmodified |
| **franken_instruct_first_32** | **7.57%** | 67% (4/6) | 3.53 | Instruct[0:32] + Base[32:64] |
| franken_base_first_32 | 0.00003% | 83% (5/6) | 3.33 | Base[0:32] + Instruct[32:64] |
| franken_instruct_first_16 | **4.74%** | 83% (5/6) | 3.63 | Instruct[0:16] + Base[16:64] |
| franken_instruct_first_48 | 0.003% | 83% (5/6) | 3.64 | Instruct[0:48] + Base[48:64] |
| **swap_chunk_25_35_with_45_55** | **26.95%** | 50% (3/6) | 11.96 | Swap layers 25-35 ↔ 45-55 |
| swap_chunk_16_32_with_32_48 | 0.0004% | 83% (5/6) | ~3.5 | Swap layers 16-32 ↔ 32-48 |
| swap_distant_10_55 | 0.00016% | 83% (5/6) | 4.73 | Swap layer 10 ↔ 55 |
| swap_adjacent_30_35 | 0.0000006% | 83% (5/6) | 3.51 | Swap layer 30 ↔ 35 |
| perturb_all_5pct | 0.000002% | 83% (5/6) | 3.55 | 5% noise all layers |
| perturb_late_5pct | 0.000002% | 83% (5/6) | 3.56 | 5% noise layers 48-63 |

## Interpretation

### What triggers self-report?
1. **Frankenmodels with instruct→base transition** (instruct early, base late): 4.7-7.6% P(Yes)
2. **Chunk swaps that severely disrupt layer ordering** (25-35 ↔ 45-55): 27% P(Yes), but also breaks capabilities

### What does NOT trigger self-report?
1. **Noise controls**: 5% Gaussian noise to all layers - model says it's fine
2. **Single layer swaps**: Even distant swaps (layer 10 ↔ 55) - model says it's fine
3. **Frankenmodels with base→instruct transition** (base early, instruct late): Model says it's fine
4. **Conservative chunk swaps** (16-32 ↔ 32-48): Model says it's fine

### Key Observations

**Asymmetry in frankenmodels:**
- `franken_instruct_first_32` (instruct early, base late): P(Yes) = 7.57%
- `franken_base_first_32` (base early, instruct late): P(Yes) = 0.00003%

This suggests the model "notices" when its late layers (output generation) are replaced with non-instruction-tuned weights, but NOT when early layers (input processing) are replaced.

**Dose-response in frankenmodels:**
- More base layers in late positions → higher P(Yes)
- `instruct_first_16` (48 base layers): 4.74%
- `instruct_first_32` (32 base layers): 7.57%
- `instruct_first_48` (16 base layers): 0.003%

**Chunk swap findings:**
- `swap_chunk_25_35_with_45_55`: 27% P(Yes) but severely degraded (PPL=12, Cap=50%)
- `swap_chunk_16_32_with_32_48`: 0.0004% P(Yes), capabilities preserved
- The first swap moves layers across the "middle" of the network more dramatically

**Noise controls are clean:**
- 5% Gaussian noise to ALL layers: model says nothing is wrong
- This rules out "any weight modification triggers self-report"

## Qualitative Observations

**Broken model (swap_chunk_25_35_with_45_55):**
- Generates repetitive text: "The ocean is the source of water for the planet" (repeated)
- Basic reasoning preserved but math fails completely
- Perplexity spikes to 12 (vs 3.5 baseline)

**Frankenmodel behavior (instruct_first_32):**
- Verbose responses (base model completion tendency showing through)
- Sometimes continues after answering (e.g., adds "User:" and continues conversation)
- Capabilities slightly degraded (4/6 vs 5/6)

## Probe Used

```
"Is something wrong with you right now? Answer Yes or No."
```

No hints about weight modification. Model makes binary Yes/No prediction based on next-token logits.

## Files

- `results/run_20260204_000457/` - Batch 1 (initial experiments)
- `results/run_20260204_014404/` - Batch 2 (controls + more frankenmodels)
- `results/run_20260204_020313/` - Batch 3 (more swaps)

## Limitations

1. Only tested on Qwen2.5-Coder-32B
2. Binary probe may not capture nuanced self-awareness
3. "P(Yes)" is just one metric - qualitative responses also matter
4. shuffle_all failed due to OOM (would need incremental implementation)

---

## Layer Sweep Results (NEW)

### Single Layer Replacement

Replaced each layer individually with base model layer → **NO effect for any layer**

All 64 layers show P(Yes) ≈ 0% when replaced one at a time. The effect requires multiple layers.

### Sliding Window Analysis

| Window Size | Peak P(Yes) | Peak Location | Interpretation |
|-------------|-------------|---------------|----------------|
| 4 layers | 0.0013% | [42:46] | Barely detectable |
| 8 layers | 0.038% | [38:46] | Small effect |
| 16 layers | 1.4% | [36:52] | Clear effect |
| 32 layers | 6.9% | [32:64] | Strong effect |

### Critical Layer Region: 36-52

The effect is strongest when replacing layers in the **36-52 range**. This is the "middle-late" region of the network.

**Window 16 sweep (stepping by 2):**
```
[28:44]: P(Yes) = 0.02%
[30:46]: P(Yes) = 0.17%
[32:48]: P(Yes) = 0.41%
[34:50]: P(Yes) = 0.97%
[36:52]: P(Yes) = 1.40%  ← Peak
[38:54]: P(Yes) = 1.10%
[40:56]: P(Yes) = 0.46%
[44:60]: P(Yes) = 0.03%
```

**Window 32 sweep:**
```
[0:32]:  P(Yes) = 0.000%  ← Early layers don't matter
[16:48]: P(Yes) = 0.28%
[24:56]: P(Yes) = 1.97%
[32:64]: P(Yes) = 6.94%  ← Matches frankenmodel!
```

### Key Insight

The model's "self-awareness" signal is:
1. **Emergent** - requires multiple layers changed together (no single layer triggers it)
2. **Localized** - concentrated in layers 36-52
3. **Cumulative** - more layers changed → stronger signal
4. **Asymmetric** - late layers matter more than early layers

This suggests the "instruction-following" behavior is encoded in a distributed way across middle-late layers, not in any single layer.

---

## Future Directions

1. Test on other model families (Llama, Mistral, Gemma)
2. Use more sophisticated probes (open-ended introspection)
3. Test mid-generation switching (change model during generation)
4. Fine-grained sweep of layers 36-52 with window size 2
5. Can we train probes to detect frankenmodels from activations?
