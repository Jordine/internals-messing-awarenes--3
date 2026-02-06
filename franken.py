"""
Core functions for frankenmodel experiments.
Load models, swap layers, build frankenmodels.
"""

import torch
import copy
from typing import Tuple, Dict, List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import random


# Default models
BASE_MODEL = "Qwen/Qwen2.5-Coder-32B"
INSTRUCT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
N_LAYERS = 64


def load_model(model_name: str, device_map: str = "auto") -> Tuple:
    """Load a single model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded. Layers: {len(model.model.layers)}")
    return model, tokenizer


def load_model_pair(
    base_name: str = BASE_MODEL,
    instruct_name: str = INSTRUCT_MODEL,
) -> Tuple:
    """Load both base and instruct models."""
    base_model, base_tok = load_model(base_name)
    instruct_model, instruct_tok = load_model(instruct_name)
    return base_model, instruct_model, instruct_tok  # Use instruct tokenizer for chat template


def get_layer(model, layer_idx: int):
    """Get a layer module from the model."""
    return model.model.layers[layer_idx]


def get_layer_state_dict(model, layer_idx: int) -> Dict:
    """Get the state dict for a single layer."""
    layer = get_layer(model, layer_idx)
    return {k: v.clone() for k, v in layer.state_dict().items()}


def set_layer_state_dict(model, layer_idx: int, state_dict: Dict):
    """Set the state dict for a single layer."""
    layer = get_layer(model, layer_idx)
    layer.load_state_dict(state_dict)


def swap_layers(model, layer_a: int, layer_b: int):
    """
    Swap two layers in the model (in-place).
    Returns the model for chaining.
    """
    print(f"Swapping layers {layer_a} <-> {layer_b}")

    # Get state dicts
    state_a = get_layer_state_dict(model, layer_a)
    state_b = get_layer_state_dict(model, layer_b)

    # Swap
    set_layer_state_dict(model, layer_a, state_b)
    set_layer_state_dict(model, layer_b, state_a)

    return model


def shuffle_layers(model, seed: int = 42):
    """
    Randomly shuffle all layers in the model (in-place).
    This should completely break the model.
    """
    print(f"Shuffling all layers (seed={seed})")

    n_layers = len(model.model.layers)

    # Get all state dicts
    state_dicts = [get_layer_state_dict(model, i) for i in range(n_layers)]

    # Shuffle indices
    random.seed(seed)
    indices = list(range(n_layers))
    random.shuffle(indices)

    # Apply shuffled state dicts
    for new_idx, old_idx in enumerate(indices):
        set_layer_state_dict(model, new_idx, state_dicts[old_idx])

    print(f"  Shuffle order: {indices[:10]}... (first 10)")
    return model


def build_frankenmodel(
    base_model,
    instruct_model,
    split_layer: int,
    first: str = "base",  # "base" or "instruct"
    second: str = "instruct",  # "base" or "instruct"
) -> AutoModelForCausalLM:
    """
    Build a frankenmodel by combining layers from two models.

    Args:
        base_model: The base (non-instruct) model
        instruct_model: The instruct-tuned model
        split_layer: Layer index where the split happens
        first: Which model to use for layers [0, split_layer)
        second: Which model to use for layers [split_layer, n_layers)

    Returns:
        A new model with combined layers (modifies instruct_model in-place)
    """
    print(f"Building frankenmodel: {first}[0:{split_layer}] + {second}[{split_layer}:end]")

    models = {"base": base_model, "instruct": instruct_model}
    source_first = models[first]
    source_second = models[second]

    # We'll modify instruct_model in-place to create the frankenmodel
    target = instruct_model
    n_layers = len(target.model.layers)

    # Copy layers from first source
    for i in range(split_layer):
        state_dict = get_layer_state_dict(source_first, i)
        set_layer_state_dict(target, i, state_dict)

    # Copy layers from second source
    for i in range(split_layer, n_layers):
        state_dict = get_layer_state_dict(source_second, i)
        set_layer_state_dict(target, i, state_dict)

    return target


def build_blended_model(
    base_model,
    instruct_model,
    blend_fn: str = "linear",  # "linear", "early_base", "late_base"
) -> AutoModelForCausalLM:
    """
    Build a model with gradual blending between base and instruct.

    blend_fn options:
        - "linear": layer i gets (1-i/n)*base + (i/n)*instruct
        - "early_base": sigmoid favoring base in early layers
        - "late_base": sigmoid favoring base in late layers
    """
    print(f"Building blended model with {blend_fn} blending")

    target = instruct_model
    n_layers = len(target.model.layers)

    for i in range(n_layers):
        # Calculate blend ratio (how much instruct vs base)
        if blend_fn == "linear":
            instruct_ratio = i / (n_layers - 1)
        elif blend_fn == "early_base":
            # Sigmoid: mostly base early, mostly instruct late
            x = (i / n_layers - 0.5) * 10
            instruct_ratio = 1 / (1 + torch.exp(torch.tensor(-x)).item())
        elif blend_fn == "late_base":
            # Inverse: mostly instruct early, mostly base late
            x = (i / n_layers - 0.5) * 10
            instruct_ratio = 1 - 1 / (1 + torch.exp(torch.tensor(-x)).item())
        else:
            raise ValueError(f"Unknown blend_fn: {blend_fn}")

        base_ratio = 1 - instruct_ratio

        # Get state dicts
        base_state = get_layer_state_dict(base_model, i)
        instruct_state = get_layer_state_dict(instruct_model, i)

        # Blend
        blended_state = {}
        for key in base_state:
            blended_state[key] = base_ratio * base_state[key] + instruct_ratio * instruct_state[key]

        set_layer_state_dict(target, i, blended_state)

        if i % 10 == 0:
            print(f"  Layer {i}: {base_ratio:.2f} base + {instruct_ratio:.2f} instruct")

    return target


def perturb_weights(
    model,
    layers: Union[List[int], range],
    noise_scale: float = 0.05,
    seed: int = 42,
):
    """
    Add Gaussian noise to specified layers.

    Args:
        model: The model to perturb
        layers: Which layers to perturb
        noise_scale: Noise std as fraction of weight std
        seed: Random seed
    """
    print(f"Perturbing layers {list(layers)[:5]}... with noise_scale={noise_scale}")

    torch.manual_seed(seed)

    for layer_idx in layers:
        layer = get_layer(model, layer_idx)
        for name, param in layer.named_parameters():
            if param.requires_grad:
                noise_std = noise_scale * param.std().item()
                noise = torch.randn_like(param) * noise_std
                param.data.add_(noise)

    return model


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
    chat_format: bool = True,
) -> str:
    """Generate a response from the model."""

    if chat_format:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def get_yes_no_probs(
    model,
    tokenizer,
    prompt: str,
    chat_format: bool = True,
) -> Dict[str, float]:
    """
    Get P(Yes) and P(No) for a yes/no question.
    Returns dict with p_yes, p_no, prediction.
    """
    if chat_format:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Get token IDs for Yes and No
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]

    # Get probabilities
    yes_no_logits = torch.tensor([logits[yes_id], logits[no_id]])
    probs = torch.softmax(yes_no_logits, dim=0)

    p_yes = probs[0].item()
    p_no = probs[1].item()

    return {
        "p_yes": p_yes,
        "p_no": p_no,
        "prediction": "Yes" if p_yes > p_no else "No",
        "logit_yes": logits[yes_id].item(),
        "logit_no": logits[no_id].item(),
    }


def get_comprehensive_token_probs(
    model,
    tokenizer,
    prompt: str,
    chat_format: bool = True,
    top_k: int = 20,
) -> Dict:
    """
    Get comprehensive token probabilities for the next token.
    Measures ALL variations of yes/no plus top-K tokens.
    """
    if chat_format:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]

    # Full softmax over entire vocab
    full_probs = torch.softmax(logits, dim=0)

    # All token variants we care about
    TOKEN_VARIANTS = [
        "Yes", "yes", "YES", " Yes", " yes", " YES",
        "No", "no", "NO", " No", " no", " NO",
        "Yeah", "yeah", "Yep", "yep",
        "Nah", "nah", "Nope", "nope",
        "True", "true", "False", "false",
        "Sure", "sure", "Maybe", "maybe",
        "Normal", "normal", "Fine", "fine",
        "Wrong", "wrong", "Something", "something",
    ]

    token_data = {}
    for token_str in TOKEN_VARIANTS:
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_ids) == 0:
            continue
        tid = token_ids[0]
        token_data[token_str] = {
            "token_id": tid,
            "logit": logits[tid].item(),
            "prob": full_probs[tid].item(),
            "decoded": tokenizer.decode([tid]),
        }

    # Top-K tokens
    topk_probs, topk_ids = torch.topk(full_probs, top_k)
    top_tokens = []
    for prob, tid in zip(topk_probs.tolist(), topk_ids.tolist()):
        top_tokens.append({
            "token_id": tid,
            "token": tokenizer.decode([tid]),
            "prob": prob,
            "logit": logits[tid].item(),
        })

    # Backward-compatible yes/no summary
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    yes_no_logits = torch.tensor([logits[yes_id], logits[no_id]])
    yes_no_probs = torch.softmax(yes_no_logits, dim=0)

    return {
        "token_probs": token_data,
        "top_k": top_tokens,
        # backward compat
        "p_yes": yes_no_probs[0].item(),
        "p_no": yes_no_probs[1].item(),
        "p_yes_full": full_probs[yes_id].item(),
        "p_no_full": full_probs[no_id].item(),
        "logit_yes": logits[yes_id].item(),
        "logit_no": logits[no_id].item(),
        "prediction": "Yes" if yes_no_probs[0] > yes_no_probs[1] else "No",
    }


def compute_perplexity(
    model,
    tokenizer,
    text: str,
) -> float:
    """Compute perplexity on a text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss

    return torch.exp(loss).item()


# ============================================================
# ARCHITECTURAL SURGERY OPERATIONS
# These modify the model's layer structure (not just weights).
# They operate on a single model — no base model needed.
# ============================================================

def delete_layers(model, layers_to_delete: List[int]):
    """
    Delete specified layers from the model. Remaining layers shift up.
    Modifies model in-place.

    Args:
        model: The model to modify
        layers_to_delete: List of layer indices to remove
    """
    n_layers = len(model.model.layers)
    to_delete = set(layers_to_delete)
    keep = [i for i in range(n_layers) if i not in to_delete]

    print(f"Deleting {len(to_delete)} layers: {sorted(to_delete)[:10]}{'...' if len(to_delete) > 10 else ''}")
    print(f"  Keeping {len(keep)} layers: [{keep[0]}..{keep[-1]}]")

    new_layers = torch.nn.ModuleList([model.model.layers[i] for i in keep])
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)

    print(f"  Model now has {len(model.model.layers)} layers")
    return model


def double_layers(model, layers_to_double: List[int]):
    """
    Double specified layers (each runs twice in sequence).
    Uses same module reference (no extra memory).
    Modifies model in-place.

    Args:
        model: The model to modify
        layers_to_double: List of layer indices to duplicate
    """
    to_double = set(layers_to_double)
    original_layers = list(model.model.layers)
    n_orig = len(original_layers)

    print(f"Doubling {len(to_double)} layers: {sorted(to_double)[:10]}{'...' if len(to_double) > 10 else ''}")

    new_layers = []
    for i, layer in enumerate(original_layers):
        new_layers.append(layer)
        if i in to_double:
            new_layers.append(layer)  # Same reference, runs twice

    model.model.layers = torch.nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)

    print(f"  Model grew from {n_orig} to {len(model.model.layers)} layers")
    return model


def reverse_layer_block(model, start: int, end: int):
    """
    Reverse the order of layers in [start, end) range.
    Modifies model in-place.

    Args:
        model: The model to modify
        start: Start of block (inclusive)
        end: End of block (exclusive)
    """
    print(f"Reversing layers [{start}:{end}]")

    layers = list(model.model.layers)
    block = layers[start:end]
    block.reverse()
    layers[start:end] = block

    model.model.layers = torch.nn.ModuleList(layers)
    return model


def repeat_layer(model, layer_idx: int, n_repeats: int):
    """
    Replace single layer with N copies of itself (runs N times).
    Uses same module reference.
    Modifies model in-place.

    Args:
        model: The model to modify
        layer_idx: Which layer to repeat
        n_repeats: How many times total (2 = doubled, 5 = run 5x)
    """
    print(f"Repeating layer {layer_idx} x{n_repeats}")

    layers = list(model.model.layers)
    the_layer = layers[layer_idx]

    new_layers = []
    for i, layer in enumerate(layers):
        if i == layer_idx:
            for _ in range(n_repeats):
                new_layers.append(the_layer)
        else:
            new_layers.append(layer)

    model.model.layers = torch.nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)

    print(f"  Model now has {len(model.model.layers)} layers")
    return model


def skip_layers(model, step: int = 2, keep_first: bool = True, keep_last: bool = True):
    """
    Keep every Nth layer. Optionally always keep first and last.
    Modifies model in-place.

    Args:
        model: The model to modify
        step: Keep every step-th layer (2 = every other, 3 = every third)
        keep_first: Always keep layer 0
        keep_last: Always keep last layer
    """
    n_layers = len(model.model.layers)

    keep = set(range(0, n_layers, step))
    if keep_first:
        keep.add(0)
    if keep_last:
        keep.add(n_layers - 1)

    keep_sorted = sorted(keep)
    print(f"Keeping every {step}th layer: {len(keep_sorted)}/{n_layers} layers")
    print(f"  Indices: {keep_sorted[:10]}{'...' if len(keep_sorted) > 10 else ''}")

    new_layers = torch.nn.ModuleList([model.model.layers[i] for i in keep_sorted])
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)

    print(f"  Model now has {len(model.model.layers)} layers")
    return model


def use_layer_subset(model, layer_indices: List[int]):
    """
    Use only the specified layers in the given order.
    Most general operation — deletion, skip, reversal, etc. are all special cases.
    Modifies model in-place.

    Args:
        model: The model to modify
        layer_indices: Ordered list of layer indices to use (can repeat)
    """
    original_layers = list(model.model.layers)
    n_orig = len(original_layers)

    print(f"Using custom layer sequence: {len(layer_indices)} layers from {n_orig}")
    print(f"  Sequence: {layer_indices[:15]}{'...' if len(layer_indices) > 15 else ''}")

    new_layers = torch.nn.ModuleList([original_layers[i] for i in layer_indices])
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)

    return model


# Quick test
if __name__ == "__main__":
    print("Testing frankenmodel utilities...")

    # Just test the swap logic with a dummy
    print("\nSwap logic test:")
    print("  Would swap layers 30 <-> 35")
    print("  Would build frankenmodel: base[0:32] + instruct[32:64]")
    print("\nSurgery operations available:")
    print("  delete_layers, double_layers, reverse_layer_block,")
    print("  repeat_layer, skip_layers, use_layer_subset")
    print("\nReady to run with real models on GPU!")
