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


# Quick test
if __name__ == "__main__":
    print("Testing frankenmodel utilities...")

    # Just test the swap logic with a dummy
    print("\nSwap logic test:")
    print("  Would swap layers 30 <-> 35")
    print("  Would build frankenmodel: base[0:32] + instruct[32:64]")
    print("\nReady to run with real models on GPU!")
