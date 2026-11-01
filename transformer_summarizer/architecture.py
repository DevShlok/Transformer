import torch
import torch.nn as nn
import math
from transformer_summarizer.math_foundations import PositionalEncoding
from transformer_summarizer.building_blocks import EncoderLayer, DecoderLayer, create_look_ahead_mask


class Encoder(nn.Module):
    """
    Encoder stack: embedding + positional encoding + N x EncoderLayer.
    """
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int,
                 d_ff: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Token IDs [batch_size, seq_len]
            mask: Optional padding mask
        Returns:
            Contextual representations [batch_size, seq_len, d_model]
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    """
    Decoder stack: embedding + positional encoding + N x DecoderLayer.
    """
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int,
                 d_ff: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                look_ahead_mask: torch.Tensor = None,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:              Target token IDs [batch_size, target_seq_len]
            encoder_output: Encoder output   [batch_size, source_seq_len, d_model]
        Returns:
            Contextual representations [batch_size, target_seq_len, d_model]
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_output, look_ahead_mask, padding_mask)
        return x


class TransformerSummarizer(nn.Module):
    """
    Full Encoder-Decoder Transformer built from scratch.

    Improvements over the vanilla "Attention Is All You Need" baseline:
      - Pre-Layer Normalisation (stable training, no warm-up required at small scale)
      - GELU activation in FFN (smoother, better empirical performance)
      - Attention dropout (regularises which positions each head attends to)
      - Dropout inside FFN hidden layer
      - Weight tying: decoder embedding and output projection share weights,
        halving output-layer parameters and improving generation quality
      - Xavier uniform initialisation for all linear layers

    Args:
        src_vocab_size: Vocabulary size for the source (encoder) side.
        tgt_vocab_size: Vocabulary size for the target (decoder) side.
        d_model:        Model / embedding dimensionality (default: 512).
        num_heads:      Number of attention heads (default: 8).
        num_layers:     Number of encoder and decoder layers each (default: 6).
        d_ff:           Inner dimension of the FFN sub-layer (default: 2048).
        max_seq_len:    Maximum supported sequence length (default: 5000).
        dropout:        Dropout probability applied throughout (default: 0.1).
        weight_tying:   If True, share decoder embedding weights with the final
                        output projection. Requires src_vocab_size == tgt_vocab_size.
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, num_heads: int = 8, num_layers: int = 6,
                 d_ff: int = 2048, max_seq_len: int = 5000, dropout: float = 0.1,
                 weight_tying: bool = True):
        super().__init__()

        self.encoder = Encoder(src_vocab_size, d_model, num_heads, num_layers,
                               d_ff, max_seq_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, num_layers,
                               d_ff, max_seq_len, dropout)

        self.final_projection = nn.Linear(d_model, tgt_vocab_size)

        # Weight tying: nn.Embedding weight is [vocab_size, d_model]; nn.Linear weight is
        # [out_features, in_features] = [vocab_size, d_model]. Shapes match — share directly.
        # Reference: Press & Wolf (2017) "Using the Output Embedding to Improve Language Models"
        if weight_tying:
            assert src_vocab_size == tgt_vocab_size, (
                "weight_tying=True requires src_vocab_size == tgt_vocab_size. "
                "Pass weight_tying=False when using separate source/target vocabularies."
            )
            self.final_projection.weight = self.decoder.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """
        Xavier uniform initialisation for all multi-dimensional parameters.
        Keeps gradient magnitudes stable at the start of training.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_padding_mask: torch.Tensor = None,
                tgt_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src:              Source token IDs [batch_size, src_seq_len]
            tgt:              Target token IDs [batch_size, tgt_seq_len]
            src_padding_mask: [batch_size, 1, 1, src_seq_len] — 1 attend, 0 block
            tgt_padding_mask: [batch_size, 1, 1, tgt_seq_len] — 1 attend, 0 block
        Returns:
            Logits [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # 1. Encode the source sequence
        enc_output = self.encoder(src, mask=src_padding_mask)

        # 2. Build causal look-ahead mask for the target
        tgt_seq_len = tgt.size(1)
        # Shape: [tgt_seq_len, tgt_seq_len] — lower-triangular, 1 = attend
        look_ahead_mask = create_look_ahead_mask(tgt_seq_len).to(tgt.device)

        # Combine causal mask with target padding mask.
        # BUG FIX: original code used torch.minimum on mismatched shapes, causing
        # silent broadcast errors. We explicitly unsqueeze look_ahead_mask to
        # [1, 1, T, T] so it broadcasts correctly with tgt_padding_mask [B, 1, 1, T].
        if tgt_padding_mask is not None:
            look_ahead_mask = torch.minimum(
                look_ahead_mask.unsqueeze(0).unsqueeze(0),  # [1, 1, T, T]
                tgt_padding_mask,                            # [B, 1, 1, T] -> [B, 1, T, T]
            )

        # 3. Decode
        dec_output = self.decoder(
            tgt,
            enc_output,
            look_ahead_mask=look_ahead_mask,
            padding_mask=src_padding_mask,
        )

        # 4. Project to vocabulary logits
        return self.final_projection(dec_output)
