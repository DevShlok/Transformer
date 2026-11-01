"""
transformer_summarizer
======================
A from-scratch Encoder–Decoder Transformer with:
  - Pre-Layer Normalisation
  - GELU feed-forward activation
  - Attention dropout
  - Weight tying (decoder embedding ↔ output projection)
  - Xavier uniform initialisation
  - Noam LR schedule, gradient clipping, teacher forcing
  - Greedy and Beam Search decoding
"""
from transformer_summarizer.math_foundations import (
    PositionalEncoding,
    scaled_dot_product_attention,
    MultiHeadAttention,
)
from transformer_summarizer.building_blocks import (
    EncoderLayer,
    DecoderLayer,
    create_look_ahead_mask,
)
from transformer_summarizer.architecture import (
    Encoder,
    Decoder,
    TransformerSummarizer,
)
from transformer_summarizer.pipeline import (
    get_noam_scheduler,
    train_loop,
    generate_summary,
    beam_search_summary,
)

__all__ = [
    "PositionalEncoding",
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "EncoderLayer",
    "DecoderLayer",
    "create_look_ahead_mask",
    "Encoder",
    "Decoder",
    "TransformerSummarizer",
    "get_noam_scheduler",
    "train_loop",
    "generate_summary",
    "beam_search_summary",
]

