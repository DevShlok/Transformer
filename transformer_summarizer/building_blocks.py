import torch
import torch.nn as nn
from transformer_summarizer.math_foundations import MultiHeadAttention


def create_look_ahead_mask(size: int) -> torch.Tensor:
    """
    Creates a causal look-ahead mask to prevent the decoder from attending to future tokens.
    Returns a lower-triangular matrix: 1 = attend, 0 = block.

    Args:
        size: The target sequence length.
    Returns:
        mask tensor of shape [size, size]
    """
    return torch.tril(torch.ones(size, size))


class EncoderLayer(nn.Module):
    """
    A single block within the Encoder stack.

    Uses Pre-Layer Normalisation (Pre-LN): LayerNorm is applied *before* each
    sub-layer, not after. Pre-LN trains more stably than Post-LN (the original
    paper) and is standard in modern architectures (GPT, PaLM, LLaMA, etc.).

    Improvements over the vanilla paper:
      - Pre-LN instead of Post-LN
      - GELU activation (smoother than ReLU, standard in modern transformers)
      - Dropout inside the FFN hidden layer (between the two linear layers)
      - Attention dropout passed through to MultiHeadAttention

    Sub-layers:
        1. Multi-Head Self-Attention
        2. Position-wise Feed-Forward Network (Linear -> GELU -> Dropout -> Linear)
    Each sub-layer has a residual connection and its own LayerNorm.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Pre-LN: norm is applied before the sub-layer (inside forward)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        # Dropout between the linear layers regularises the hidden FFN representations
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Output dropout applied after each sub-layer before the residual add
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:    Input tensor [batch_size, seq_len, d_model]
            mask: Optional padding mask
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Pre-LN Self-Attention: norm -> attend -> dropout -> residual add
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, mask))

        # Pre-LN FFN: norm -> ffn -> dropout -> residual add
        x = x + self.dropout(self.ff(self.norm2(x)))

        return x


class DecoderLayer(nn.Module):
    """
    A single block within the Decoder stack.

    Uses Pre-Layer Normalisation (same motivation as EncoderLayer).

    Sub-layers:
        1. Masked Multi-Head Self-Attention (causal — prevents attending to future tokens)
        2. Multi-Head Cross-Attention (Q from decoder, K/V from encoder output)
        3. Position-wise Feed-Forward Network (Linear -> GELU -> Dropout -> Linear)
    Each sub-layer has a residual connection and its own LayerNorm.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                look_ahead_mask: torch.Tensor = None,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:               Decoder input [batch_size, target_seq_len, d_model]
            encoder_output:  Encoder output [batch_size, source_seq_len, d_model]
            look_ahead_mask: Causal mask — prevents self-attention to future tokens
            padding_mask:    Masks padding tokens in cross-attention over encoder output
        Returns:
            Output tensor [batch_size, target_seq_len, d_model]
        """
        # 1. Pre-LN Masked Self-Attention (causal)
        normed = self.norm1(x)
        x = x + self.dropout(self.masked_self_attn(normed, normed, normed, look_ahead_mask))

        # 2. Pre-LN Cross-Attention (Q from decoder, K/V from encoder)
        normed = self.norm2(x)
        x = x + self.dropout(self.cross_attn(normed, encoder_output, encoder_output, padding_mask))

        # 3. Pre-LN FFN
        x = x + self.dropout(self.ff(self.norm3(x)))

        return x
