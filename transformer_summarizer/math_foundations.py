import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens
    in the sequence into the embeddings via fixed sinusoidal functions.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor of shape [batch_size, seq_len, d_model] with positional encodings added.
        """
        return x + self.pe[:, :x.size(1), :]


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes scaled dot product attention.
    Mathematically: softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        Q:       Queries  [batch_size, num_heads, seq_len_q, d_k]
        K:       Keys     [batch_size, num_heads, seq_len_k, d_k]
        V:       Values   [batch_size, num_heads, seq_len_v, d_v]
        mask:    Optional mask broadcastable to [batch_size, num_heads, seq_len_q, seq_len_k].
                 Positions with value 0 are blocked (filled with -inf before softmax).
        dropout: Optional Dropout applied to attention weights (attention dropout).

    Returns:
        output:            [batch_size, num_heads, seq_len_q, d_v]
        attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
    """
    d_k = Q.size(-1)

    # scores: [batch_size, num_heads, seq_len_q, seq_len_k]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Fill 0 / False positions with -inf so softmax maps them to ~0
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)

    # NaN guard: softmax(-inf, -inf, ...) -> NaN when every position in a row is masked.
    # Replace those NaNs with 0 so they contribute nothing to the output.
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

    # Attention dropout: randomly zero out some attention connections during training.
    # Applied to attention weights (not the output), as in the original paper.
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    output = torch.matmul(attention_weights, V)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Projects Q, K, V into `num_heads` independent subspaces, applies scaled
    dot-product attention (with optional attention dropout) on each head in
    parallel, then concatenates and projects the results.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be exactly divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Input projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection after concatenating all heads
        self.W_o = nn.Linear(d_model, d_model)

        # Attention dropout: regularises which positions each head attends to
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            q: Queries [batch_size, seq_len_q, d_model]
            k: Keys    [batch_size, seq_len_k, d_model]
            v: Values  [batch_size, seq_len_v, d_model]
            mask: Optional mask
        Returns:
            Output tensor [batch_size, seq_len_q, d_model]
        """
        batch_size = q.size(0)

        # 1. Linear projections
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        # 2. Split into heads: [batch, seq, d_model] -> [batch, heads, seq, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Scaled dot-product attention with attention dropout
        x, _ = scaled_dot_product_attention(Q, K, V, mask, self.attn_dropout)

        # 4. Concatenate heads: [batch, heads, seq, d_k] -> [batch, seq, d_model]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. Final linear projection
        return self.W_o(x)
