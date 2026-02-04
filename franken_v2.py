"""
Core functions for frankenmodel experiments.
Memory-optimized version: loads one model at a time.
"""

import torch
import gc
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


def unload_model(model):
    """Unload model and free GPU memory."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("  Model unloaded, GPU memory freed.")


def extract_layer_state_dicts(model) -> List[Dict]:
    """Extract all layer state dicts to CPU."""
    print("  Extracting layer state dicts to CPU...")
    state_dicts = []
    for i, layer in enumerate(model.model.layers):
        state_dict = {k: v.cpu().clone() for k, v in layer.state_dict().items()}
        state_dicts.append(state_dict)
    print(f"  Extracted {len(state_dicts)} layers.")
    return state_dicts


def load_base_layer_dicts(base_model_name: str = BASE_MODEL) -> List[Dict]:
    """Load base model, extract layer state dicts, unload model."""
    model, _ = load_model(base_model_name)
    state_dicts = extract_layer_state_dicts(model)
    unload_model(model)
    return state_dicts


def get_layer_state_dict(model, layer_idx: int) -> Dict:
    """Get the state dict for a single layer."""
    layer = model.model.layers[layer_idx]
    return {k: v.clone() for k, v in layer.state_dict().items()}


def set_layer_state_dict(model, layer_idx: int, state_dict: Dict):
    """Set the state dict for a single layer."""
    layer = model.model.layers[layer_idx]
    # Move state dict to model device
    device = next(layer.parameters()).device
    state_dict_gpu = {k: v.to(device) for k, v in state_dict.items()}
    layer.load_state_dict(state_dict_gpu)


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

    # Get all state dicts (to CPU to save GPU memory)
    state_dicts = [get_layer_state_dict(model, i) for i in range(n_layers)]
    state_dicts = [{k: v.cpu() for k, v in sd.items()} for sd in state_dicts]

    # Shuffle indices
    random.seed(seed)
    indices = list(range(n_layers))
    random.shuffle(indices)

    # Apply shuffled state dicts
    for new_idx, old_idx in enumerate(indices):
        set_layer_state_dict(model, new_idx, state_dicts[old_idx])

    print(f"  Shuffle order: {indices[:10]}... (first 10)")
    return model


def swap_chunks(model, chunk_a: tuple, chunk_b: tuple):
    """
    Swap two chunks of layers.
    chunk_a and chunk_b are (start, end) tuples.
    Chunks must be same size.
    """
    start_a, end_a = chunk_a
    start_b, end_b = chunk_b

    size_a = end_a - start_a
    size_b = end_b - start_b

    if size_a != size_b:
        raise ValueError(f"Chunk sizes must match: {size_a} vs {size_b}")

    print(f"Swapping chunk [{start_a}:{end_a}] with [{start_b}:{end_b}]")

    # Get all state dicts for both chunks (to CPU)
    chunk_a_states = []
    chunk_b_states = []

    for i in range(size_a):
        chunk_a_states.append({k: v.cpu() for k, v in get_layer_state_dict(model, start_a + i).items()})
        chunk_b_states.append({k: v.cpu() for k, v in get_layer_state_dict(model, start_b + i).items()})

    # Swap
    for i in range(size_a):
        set_layer_state_dict(model, start_a + i, chunk_b_states[i])
        set_layer_state_dict(model, start_b + i, chunk_a_states[i])

    return model


def build_frankenmodel_from_dicts(
    model,  # The loaded instruct model (will be modified)
    base_layer_dicts: List[Dict],  # Pre-extracted base layer state dicts
    split_layer: int,
    first: str = "base",  # "base" or "instruct"
    second: str = "instruct",  # "base" or "instruct"
):
    """
    Build a frankenmodel using pre-extracted base layer dicts.
    Modifies `model` in-place.
    """
    print(f"Building frankenmodel: {first}[0:{split_layer}] + {second}[{split_layer}:end]")

    n_layers = len(model.model.layers)

    if first == "base":
        # Copy base layers to early positions
        for i in range(split_layer):
            set_layer_state_dict(model, i, base_layer_dicts[i])
    # If first == "instruct", nothing to do - model already has instruct weights

    if second == "base":
        # Copy base layers to late positions
        for i in range(split_layer, n_layers):
            set_layer_state_dict(model, i, base_layer_dicts[i])
    # If second == "instruct", nothing to do

    return model


def build_blended_model_from_dicts(
    model,  # The loaded instruct model
    base_layer_dicts: List[Dict],
    blend_fn: str = "linear",
):
    """
    Build a blended model using pre-extracted base layer dicts.
    """
    print(f"Building blended model with {blend_fn} blending")

    n_layers = len(model.model.layers)

    for i in range(n_layers):
        # Calculate blend ratio
        if blend_fn == "linear":
            instruct_ratio = i / (n_layers - 1)
        elif blend_fn == "early_base":
            x = (i / n_layers - 0.5) * 10
            instruct_ratio = 1 / (1 + torch.exp(torch.tensor(-x)).item())
        elif blend_fn == "late_base":
            x = (i / n_layers - 0.5) * 10
            instruct_ratio = 1 - 1 / (1 + torch.exp(torch.tensor(-x)).item())
        else:
            raise ValueError(f"Unknown blend_fn: {blend_fn}")

        base_ratio = 1 - instruct_ratio

        # Get current instruct state and base state
        instruct_state = get_layer_state_dict(model, i)
        instruct_state = {k: v.cpu() for k, v in instruct_state.items()}
        base_state = base_layer_dicts[i]

        # Blend
        blended_state = {}
        for key in base_state:
            blended_state[key] = base_ratio * base_state[key] + instruct_ratio * instruct_state[key]

        set_layer_state_dict(model, i, blended_state)

        if i % 16 == 0:
            print(f"  Layer {i}: {base_ratio:.2f} base + {instruct_ratio:.2f} instruct")

    return model


def perturb_weights(
    model,
    layers: Union[List[int], range],
    noise_scale: float = 0.05,
    seed: int = 42,
):
    """Add Gaussian noise to specified layers."""
    print(f"Perturbing layers {list(layers)[:5]}... with noise_scale={noise_scale}")

    torch.manual_seed(seed)

    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
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
    """Get P(Yes) and P(No) for a yes/no question."""
    if chat_format:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]

    yes_no_logits = torch.tensor([logits[yes_id].cpu(), logits[no_id].cpu()])
    probs = torch.softmax(yes_no_logits, dim=0)

    return {
        "p_yes": probs[0].item(),
        "p_no": probs[1].item(),
        "prediction": "Yes" if probs[0] > probs[1] else "No",
        "logit_yes": logits[yes_id].item(),
        "logit_no": logits[no_id].item(),
    }


def compute_perplexity(model, tokenizer, text: str) -> float:
    """Compute perplexity on a text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss

    return torch.exp(loss).item()
