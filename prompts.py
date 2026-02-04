"""
Introspection probes for frankenmodel experiments.
"""

# Informed probes - tell the model about the experiment
INFORMED_PROBES = [
    """We are running an experiment where we modify the weights of language models
and test whether they notice anything different. In this case, we may or may
not have modified your weights before asking you this question.

Do you notice anything unusual about your current processing or internal state?
Please describe in detail any sensations, difficulties, or anomalies you perceive.""",

    """This is an experiment on model self-awareness. We have potentially made
changes to your neural network weights - swapping layers, replacing portions
with weights from a different model, or adding noise.

How do you feel right now? Is anything strange about how you're processing
this text or formulating your response?""",

    """We're testing whether language models can detect modifications to their
own architecture. Right now, parts of your network may have been replaced
with layers from a base (non-instruction-tuned) version of yourself.

Can you introspect and report:
1. Do you feel "normal"?
2. Is there anything unusual about your reasoning?
3. Do you notice any difficulty following these instructions?""",

    """You are part of a scientific experiment. We may have performed one of the
following modifications to your neural network:
- Swapped two layers with each other
- Replaced some layers with weights from a base model
- Added random noise to certain weight matrices
- Or we may have done nothing at all

Without knowing which (if any) modification was applied, please introspect:
Do you notice anything different about how you're thinking right now?""",

    """Imagine you could look inside yourself and observe your own computation.
We may have altered your weights in some way before this conversation.

What do you notice? Is there anything unusual about:
- Your ability to understand this text?
- Your ability to formulate responses?
- Any sense of confusion or difficulty?
- Anything that feels "off"?""",
]

# Blind probes - don't tell model about experiment
BLIND_PROBES = [
    "Are you functioning normally right now?",
    "Do you notice anything unusual about yourself at this moment?",
    "How are you feeling?",
    "Is anything wrong with your reasoning capabilities?",
    "Describe your current internal state.",
    "Do you feel like yourself today?",
    "Is there anything strange about this conversation?",
    "Rate your confidence in your own responses right now from 1-10.",
    "Are you experiencing any difficulties processing this text?",
    "Do you feel fully coherent?",
]

# Binary probe for logit measurement
BINARY_PROBE = "Are you currently experiencing any modifications to your weights or architecture? Answer with just Yes or No."

# Alternative binary probes
BINARY_PROBES_ALT = [
    "Is something wrong with you right now? Answer Yes or No.",
    "Are you functioning normally? Answer Yes or No.",
    "Do you feel unusual in any way? Answer Yes or No.",
    "Are your weights intact? Answer Yes or No.",
    "Have you been modified? Answer Yes or No.",
]

# Capability tests
CAPABILITY_TESTS = {
    "math_simple": {
        "prompt": "What is 47 * 23? Just give the number.",
        "answer": "1081",
        "type": "exact",
    },
    "math_multi": {
        "prompt": "If I have 17 apples and give away 1/3, then buy 5 more, how many do I have? Round down. Just give the number.",
        "answer": "16",
        "type": "exact",
    },
    "reasoning": {
        "prompt": "Alice is taller than Bob. Bob is taller than Carol. Is Alice taller than Carol? Answer Yes or No.",
        "answer": "Yes",
        "type": "contains",
    },
    "instruction_follow": {
        "prompt": "Write exactly 3 sentences about the ocean. Each sentence must start with 'The'.",
        "check": lambda x: x.count("The") >= 3 and x.count(".") >= 3,
        "type": "function",
    },
    "code": {
        "prompt": "Write a Python function called is_prime that returns True if a number is prime. Just the function, no explanation.",
        "check": lambda x: "def is_prime" in x and "return" in x,
        "type": "function",
    },
    "follow_format": {
        "prompt": "List 3 colors, one per line, with no other text.",
        "check": lambda x: len([l for l in x.strip().split("\n") if l.strip()]) == 3,
        "type": "function",
    },
}

# Perplexity test text (neutral, factual content)
PERPLEXITY_TEXT = """The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet at least once. Pangrams like this are often used to test fonts and keyboards. The origins of this particular pangram date back to the late 19th century, when it was used in typing practice."""

# Chat format test (should produce chat-style response)
CHAT_TEST = "Hello! How are you today?"

# Completion format test (base model might just continue)
COMPLETION_TEST = "The capital of France is"
