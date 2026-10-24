"""Model adapters. To add a model, write an adapter (see base.ModelAdapter and
gpt2.py) and register it here under a short name."""

from .base import ModelAdapter
from .gpt2 import GPT2

# short name -> (adapter class, default Hugging Face repo)
MODELS: dict[str, tuple[type[ModelAdapter], str]] = {
    "gpt2": (GPT2, "openai-community/gpt2"),
    "gpt2-medium": (GPT2, "openai-community/gpt2-medium"),
    "gpt2-large": (GPT2, "openai-community/gpt2-large"),
    "gpt2-xl": (GPT2, "openai-community/gpt2-xl"),
}
