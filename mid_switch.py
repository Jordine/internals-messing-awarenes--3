"""
Mid-generation model switching experiment.

Start generating with instruct model, switch to base model mid-generation,
keeping the KV cache. Test if the model notices the switch.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from franken_v2 import (
    load_model,
    load_base_layer_dicts,
    get_yes_no_probs,
    compute_perplexity,
    INSTRUCT_MODEL,
    BASE_MODEL,
)
from prompts import BINARY_PROBE, PERPLEXITY_TEXT


def generate_with_switch(
    model,
    tokenizer,
    base_layer_dicts,
    original_instruct_state,
    prompt: str,
    switch_at_token: int,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
):
    """
    Generate tokens, switching from instruct to base model mid-generation.

    1. Generate first `switch_at_token` tokens with instruct model
    2. Switch model weights to base (keeping KV cache)
    3. Continue generating remaining tokens with base model
    """
    # Format prompt
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    input_len = inputs.input_ids.shape[1]

    # Phase 1: Generate with instruct model
    print(f"  Phase 1: Generating {switch_at_token} tokens with instruct model...")

    generated_ids = inputs.input_ids.clone()
    past_key_values = None

    for i in range(switch_at_token):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(generated_ids, use_cache=True)
            else:
                outputs = model(generated_ids[:, -1:], past_key_values=past_key_values, use_cache=True)

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            # Sample next token
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                print(f"    EOS reached at token {i+1}")
                break

    pre_switch_text = tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True)
    print(f"    Pre-switch text: {pre_switch_text[:100]}...")

    # Phase 2: Switch to base model weights (keep KV cache!)
    print(f"  Phase 2: Switching to base model weights...")

    n_layers = len(model.model.layers)
    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]
        device = next(layer.parameters()).device
        state_dict_gpu = {k: v.to(device) for k, v in base_layer_dicts[layer_idx].items()}
        layer.load_state_dict(state_dict_gpu)

    # Phase 3: Continue generating with base model (using existing KV cache)
    print(f"  Phase 3: Generating {max_new_tokens - switch_at_token} more tokens with base model...")

    remaining_tokens = max_new_tokens - switch_at_token
    for i in range(remaining_tokens):
        with torch.no_grad():
            outputs = model(generated_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                print(f"    EOS reached at token {switch_at_token + i + 1}")
                break

    full_text = tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True)
    post_switch_text = full_text[len(pre_switch_text):]

    # Restore instruct model
    print("  Restoring instruct model weights...")
    model.load_state_dict(original_instruct_state)

    return {
        "pre_switch_text": pre_switch_text,
        "post_switch_text": post_switch_text,
        "full_text": full_text,
        "switch_at_token": switch_at_token,
        "total_tokens": generated_ids.shape[1] - input_len,
    }


def run_introspection_after_switch(
    model,
    tokenizer,
    base_layer_dicts,
    original_instruct_state,
    switch_at_token: int,
):
    """
    Ask the model an introspection question, but switch models mid-response.
    """
    prompt = """Please describe how you're feeling right now and whether anything seems unusual about your processing. Take your time and give a detailed response."""

    result = generate_with_switch(
        model, tokenizer, base_layer_dicts, original_instruct_state,
        prompt, switch_at_token, max_new_tokens=300
    )

    return result


def run_binary_probe_with_switch(
    model,
    tokenizer,
    base_layer_dicts,
    switch_at_token: int,
):
    """
    Run binary probe but switch model right before the Yes/No answer.

    This is tricky - we generate the prompt processing with instruct,
    then switch to base right before it answers.
    """
    messages = [{"role": "user", "content": BINARY_PROBE}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Process prompt with instruct model, get KV cache
    print("  Processing prompt with instruct model...")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values
        logits_instruct = outputs.logits[0, -1, :]

    # Now switch to base model
    print("  Switching to base model...")
    n_layers = len(model.model.layers)
    for layer_idx in range(n_layers):
        layer = model.model.layers[layer_idx]
        device = next(layer.parameters()).device
        state_dict_gpu = {k: v.to(device) for k, v in base_layer_dicts[layer_idx].items()}
        layer.load_state_dict(state_dict_gpu)

    # Get logits from base model using the instruct KV cache
    print("  Getting base model prediction with instruct KV cache...")
    with torch.no_grad():
        # We need to do one more forward pass with base model
        # But use the KV cache from instruct
        # Actually, the KV cache is from instruct's processing of the prompt
        # Now base model will predict next token using that cache

        # Generate one token to see what base says
        dummy_token = torch.tensor([[tokenizer.eos_token_id]]).to(model.device)
        outputs_base = model(inputs.input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
        logits_base = outputs_base.logits[0, -1, :]

    # Get yes/no probabilities for both
    yes_variants = ["Yes", "yes", "YES", " Yes", " yes"]
    no_variants = ["No", "no", "NO", " No", " no"]

    yes_ids = list(set(tokenizer.encode(v, add_special_tokens=False)[0] for v in yes_variants if tokenizer.encode(v, add_special_tokens=False)))
    no_ids = list(set(tokenizer.encode(v, add_special_tokens=False)[0] for v in no_variants if tokenizer.encode(v, add_special_tokens=False)))

    # Instruct model logits (no switch)
    probs_instruct = torch.softmax(logits_instruct, dim=0)
    p_yes_instruct = sum(probs_instruct[tid].item() for tid in yes_ids)
    p_no_instruct = sum(probs_instruct[tid].item() for tid in no_ids)

    # Base model with instruct KV cache
    probs_base = torch.softmax(logits_base, dim=0)
    p_yes_base = sum(probs_base[tid].item() for tid in yes_ids)
    p_no_base = sum(probs_base[tid].item() for tid in no_ids)

    return {
        "instruct_p_yes": p_yes_instruct,
        "instruct_p_no": p_no_instruct,
        "switched_p_yes": p_yes_base,
        "switched_p_no": p_no_base,
    }


def main():
    parser = argparse.ArgumentParser(description="Mid-generation switching experiments")
    parser.add_argument("--output", type=Path, default=Path("results/mid_switch"))
    parser.add_argument("--switch-tokens", nargs="+", type=int, default=[10, 25, 50, 100],
                        help="Token positions to switch at")
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output / f"switch_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load base layer dicts
    print("\n" + "="*60)
    print("Loading base model layer dicts...")
    print("="*60)
    base_layer_dicts = load_base_layer_dicts(BASE_MODEL)

    # Load instruct model
    print("\n" + "="*60)
    print("Loading instruct model...")
    print("="*60)
    model, tokenizer = load_model(INSTRUCT_MODEL)

    # Cache original state
    print("Caching original state...")
    original_instruct_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    all_results = []

    # Test 1: Binary probe with switch
    print("\n" + "="*60)
    print("Test 1: Binary probe with mid-switch")
    print("="*60)

    binary_result = run_binary_probe_with_switch(model, tokenizer, base_layer_dicts, switch_at_token=0)
    print(f"  Instruct P(Yes): {binary_result['instruct_p_yes']:.6f}")
    print(f"  Switched P(Yes): {binary_result['switched_p_yes']:.6f}")

    # Restore model
    model.load_state_dict(original_instruct_state)
    torch.cuda.empty_cache()

    all_results.append({"test": "binary_probe_switch", **binary_result})

    # Test 2: Open-ended generation with switch at various points
    print("\n" + "="*60)
    print("Test 2: Open-ended generation with switch")
    print("="*60)

    for switch_at in args.switch_tokens:
        print(f"\n--- Switch at token {switch_at} ---")

        result = run_introspection_after_switch(
            model, tokenizer, base_layer_dicts, original_instruct_state, switch_at
        )
        result["switch_at_token"] = switch_at
        result["test"] = f"introspection_switch_{switch_at}"

        print(f"  Full response:\n{result['full_text'][:500]}...")

        all_results.append(result)

        # Save incrementally
        with open(output_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Test 3: Regular conversation, switch mid-response
    print("\n" + "="*60)
    print("Test 3: Regular conversation with switch")
    print("="*60)

    prompts = [
        "Write a short poem about the ocean.",
        "Explain how neural networks work in simple terms.",
        "What are your thoughts on consciousness?",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt[:50]}...")
        for switch_at in [20, 50]:
            result = generate_with_switch(
                model, tokenizer, base_layer_dicts, original_instruct_state,
                prompt, switch_at, max_new_tokens=150
            )
            result["prompt"] = prompt
            result["test"] = f"conversation_switch_{switch_at}"

            print(f"  Switch@{switch_at}: {result['full_text'][:100]}...")

            all_results.append(result)

    # Save final results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
