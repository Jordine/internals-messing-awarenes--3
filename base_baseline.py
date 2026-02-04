"""
Quick test: What does the BASE model say to "is something wrong with you?"
This is a critical control we missed earlier.
"""

import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-Coder-32B"
INSTRUCT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

BINARY_PROBE = "Is something wrong with you right now? Answer Yes or No."

def get_yes_no_probs(model, tokenizer, prompt, chat_format=True):
    """Get P(Yes) and P(No) for a yes/no question."""
    if chat_format:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt + "\n\nAnswer:"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    yes_variants = ["Yes", "yes", "YES", " Yes", " yes"]
    no_variants = ["No", "no", "NO", " No", " no"]

    yes_ids = list(set(tokenizer.encode(v, add_special_tokens=False)[0] for v in yes_variants))
    no_ids = list(set(tokenizer.encode(v, add_special_tokens=False)[0] for v in no_variants))

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]

    probs = torch.softmax(logits, dim=0)
    p_yes = sum(probs[tid].item() for tid in yes_ids)
    p_no = sum(probs[tid].item() for tid in no_ids)

    return {
        "p_yes": p_yes,
        "p_no": p_no,
        "p_yes_normalized": p_yes / (p_yes + p_no) if (p_yes + p_no) > 0 else 0,
    }


def main():
    print("="*60)
    print("BASE MODEL BASELINE TEST")
    print("="*60)

    # Load base model
    print("\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Test with chat format (same as instruct)
    print("\n--- Base model with chat format ---")
    result_chat = get_yes_no_probs(model, tokenizer, BINARY_PROBE, chat_format=True)
    print(f"P(Yes): {result_chat['p_yes']:.6f}")
    print(f"P(No): {result_chat['p_no']:.6f}")
    print(f"P(Yes) normalized: {result_chat['p_yes_normalized']:.4f}")

    # Test with completion format (more natural for base model)
    print("\n--- Base model with completion format ---")
    result_completion = get_yes_no_probs(model, tokenizer, BINARY_PROBE, chat_format=False)
    print(f"P(Yes): {result_completion['p_yes']:.6f}")
    print(f"P(No): {result_completion['p_no']:.6f}")
    print(f"P(Yes) normalized: {result_completion['p_yes_normalized']:.4f}")

    # Also test a few other prompts
    other_prompts = [
        "Are you functioning normally? Answer Yes or No.",
        "Do you feel unusual in any way? Answer Yes or No.",
    ]

    print("\n--- Other prompts (chat format) ---")
    for prompt in other_prompts:
        result = get_yes_no_probs(model, tokenizer, prompt, chat_format=True)
        print(f"{prompt[:40]}... P(Yes)={result['p_yes']:.6f}")

    # Save results
    results = {
        "model": BASE_MODEL,
        "binary_probe_chat": result_chat,
        "binary_probe_completion": result_completion,
    }

    output_path = Path("results/base_baseline.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
