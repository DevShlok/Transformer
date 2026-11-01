import torch
import torch.nn as nn
from typing import List
from transformer_summarizer.architecture import TransformerSummarizer


def get_noam_scheduler(optimizer: torch.optim.Optimizer,
                        d_model: int,
                        warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Returns the Noam learning-rate scheduler from "Attention Is All You Need" (Vaswani et al. 2017).

    The schedule linearly increases the LR for `warmup_steps` steps, then decays it
    proportionally to the inverse square root of the step number:

        lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})

    This warm-up is critical for stable transformer training — large early gradient steps
    can push the model into a bad loss basin from which it may never fully recover.

    Args:
        optimizer:     The optimizer whose LR will be managed.
        d_model:       Model dimension (controls the peak LR magnitude).
        warmup_steps:  Steps over which LR increases linearly (typically 4000 for full-scale).

    Returns:
        A LambdaLR scheduler. Call scheduler.step() after every *batch* (not epoch).

    Recommended usage:
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = get_noam_scheduler(optimizer, d_model=512, warmup_steps=4000)
        # Inside training loop, after optimizer.step():
        scheduler.step()
    """
    def lr_lambda(step: int) -> float:
        step = max(1, step)  # Prevent division by zero at step 0
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_loop(model: TransformerSummarizer,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epochs: int = 1,
               pad_token_id: int = 0,
               clip_grad_norm: float = 1.0,
               scheduler: torch.optim.lr_scheduler.LRScheduler = None):
    """
    Training loop with teacher forcing, PAD-ignoring loss, gradient clipping,
    and optional per-batch LR scheduling.

    Key improvements over a naive loop:
      - CrossEntropyLoss(ignore_index=pad_token_id): padding tokens do not
        contribute to the loss, so gradients are only driven by real content.
      - Gradient clipping (clip_grad_norm): caps the global gradient norm before
        each optimizer step, preventing the explosions common early in transformer training.
      - Per-batch scheduler.step(): required for Noam (and similar) schedules that
        are defined per step, not per epoch.

    Args:
        model:          The TransformerSummarizer to train.
        dataloader:     Yields (src, tgt) batches. tgt must include <SOS> and <EOS> tokens.
        optimizer:      Configured optimizer. For paper-accurate training use:
                        Adam(lr=1.0, betas=(0.9, 0.98), eps=1e-9) + Noam scheduler.
        device:         torch.device for computation.
        epochs:         Number of full passes over the dataset.
        pad_token_id:   Token ID used for padding; excluded from loss computation.
        clip_grad_norm: Max global gradient norm. Set to 0 to disable (not recommended).
        scheduler:      Optional LR scheduler stepped after every batch.
    """
    # Ignore padding in the loss so gradients are only driven by real content tokens
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx, (src, tgt) in enumerate(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)

            # Teacher forcing:
            #   tgt_input    = tgt without the last token  (decoder receives up to t-1)
            #   tgt_expected = tgt without the first token (model must predict from t=1)
            tgt_input = tgt[:, :-1]    # [batch, tgt_len - 1]
            tgt_expected = tgt[:, 1:]  # [batch, tgt_len - 1]

            optimizer.zero_grad()

            # Padding masks: 1 = real token (attend), 0 = pad (block)
            src_padding_mask = (src != pad_token_id).unsqueeze(1).unsqueeze(2)
            tgt_padding_mask = (tgt_input != pad_token_id).unsqueeze(1).unsqueeze(2)

            # Forward pass -> logits: [batch, tgt_len - 1, vocab_size]
            logits = model(src, tgt_input,
                           src_padding_mask=src_padding_mask,
                           tgt_padding_mask=tgt_padding_mask)

            # CrossEntropyLoss expects [N, C] logits and [N] targets
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_expected.reshape(-1))

            loss.backward()

            # Gradient clipping: prevents explosions that stall transformer training early
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()

            # Step the scheduler once per batch (Noam is a step-level schedule)
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{epochs}]  Loss: {avg_loss:.4f}  LR: {current_lr:.2e}")


@torch.no_grad()
def generate_summary(model: TransformerSummarizer,
                     source_text: torch.Tensor,
                     max_length: int,
                     sos_token_id: int,
                     eos_token_id: int,
                     device: torch.device,
                     pad_token_id: int = 0) -> List[int]:
    """
    Greedy decoding: at each step picks the single most probable next token.

    The encoder runs once; its output is reused across all decoding steps,
    making this O(T) encoder calls total (not O(T^2)).

    Stops when <EOS> is generated or max_length is reached.

    Args:
        model:         Trained TransformerSummarizer.
        source_text:   Tokenized source [1, src_seq_len] (batch_size must be 1).
        max_length:    Maximum tokens to generate.
        sos_token_id:  Vocabulary ID for <SOS>.
        eos_token_id:  Vocabulary ID for <EOS>.
        device:        CPU or CUDA device.
        pad_token_id:  Vocabulary ID for <PAD> (used to build the source mask).

    Returns:
        List of generated token IDs (including <EOS> if generated).
    """
    model.eval()
    source_text = source_text.to(device)

    src_padding_mask = (source_text != pad_token_id).unsqueeze(1).unsqueeze(2)

    # Encode source ONCE; reuse its output for every decoding step
    enc_output = model.encoder(source_text, mask=src_padding_mask)

    decoder_input = torch.tensor([[sos_token_id]], dtype=torch.long, device=device)
    generated_tokens = []

    for _ in range(max_length):
        tgt_seq_len = decoder_input.size(1)
        look_ahead_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, device=device))

        dec_output = model.decoder(
            decoder_input,
            enc_output,
            look_ahead_mask=look_ahead_mask,
            padding_mask=src_padding_mask,
        )

        # Project the last token's representation; argmax for greedy selection
        logits = model.final_projection(dec_output[:, -1, :])  # [1, vocab_size]
        next_token_id = torch.argmax(logits, dim=-1).item()

        generated_tokens.append(next_token_id)

        if next_token_id == eos_token_id:
            break

        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        decoder_input = torch.cat([decoder_input, next_token_tensor], dim=1)

    return generated_tokens


@torch.no_grad()
def beam_search_summary(model: TransformerSummarizer,
                        source_text: torch.Tensor,
                        max_length: int,
                        sos_token_id: int,
                        eos_token_id: int,
                        device: torch.device,
                        beam_width: int = 4,
                        length_penalty: float = 0.7,
                        pad_token_id: int = 0) -> List[int]:
    """
    Beam Search decoding: maintains `beam_width` candidate sequences simultaneously.

    At every step each active beam expands across the full vocabulary; only the top
    `beam_width` candidates by cumulative log-probability survive. This explores a
    much wider output space than greedy decoding and generally produces higher-quality
    summaries, at the cost of O(beam_width) decoder forward passes per step.

    Finished beams are scored with a length penalty to prevent the model from
    pathologically favouring short sequences:

        penalised_score = cumulative_log_prob / (length ^ length_penalty)

    Args:
        model:          Trained TransformerSummarizer.
        source_text:    Tokenized source [1, src_seq_len].
        max_length:     Maximum tokens to generate.
        sos_token_id:   Vocabulary ID for <SOS>.
        eos_token_id:   Vocabulary ID for <EOS>.
        device:         CPU or CUDA device.
        beam_width:     Number of beams to maintain (default: 4).
        length_penalty: Exponent on length in the score normalisation.
                        < 1 favours shorter outputs; > 1 favours longer (default: 0.7).
        pad_token_id:   Vocabulary ID for <PAD> (used to build the source mask).

    Returns:
        List of generated token IDs for the best-scoring completed beam.
    """
    import torch.nn.functional as F

    model.eval()
    source_text = source_text.to(device)

    # --- Step 1: Encode source ONCE ---
    src_padding_mask = (source_text != pad_token_id).unsqueeze(1).unsqueeze(2)
    enc_output = model.encoder(source_text, mask=src_padding_mask)

    # Expand to [beam_width, ...] so all beams can be processed in a single batched pass
    enc_output_expanded = enc_output.expand(beam_width, -1, -1)
    src_mask_expanded = src_padding_mask.expand(beam_width, -1, -1, -1)

    # --- Step 2: Initialise beam state ---
    # Each entry: (cumulative_log_prob, list_of_token_ids)
    active_beams: List[tuple] = [(0.0, [sos_token_id])]
    completed_beams: List[tuple] = []

    # --- Step 3: Iterative decoding ---
    for step in range(max_length):
        if not active_beams:
            break

        current_beam_count = len(active_beams)

        # Batch all active beams for a single decoder forward pass
        decoder_inputs = torch.tensor(
            [beam_tokens for _, beam_tokens in active_beams],
            dtype=torch.long,
            device=device,
        )

        tgt_seq_len = decoder_inputs.size(1)
        look_ahead_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, device=device))

        enc_batch = enc_output_expanded[:current_beam_count]
        src_mask_batch = src_mask_expanded[:current_beam_count]

        dec_output = model.decoder(
            decoder_inputs,
            enc_batch,
            look_ahead_mask=look_ahead_mask,
            padding_mask=src_mask_batch,
        )

        # Project last token; numerically stable log-softmax
        logits = model.final_projection(dec_output[:, -1, :])  # [beams, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)

        # --- Step 4: Expand every active beam across top-k next tokens ---
        all_candidates: List[tuple] = []
        for beam_idx, (beam_score, beam_tokens) in enumerate(active_beams):
            top_log_probs, top_token_ids = torch.topk(log_probs[beam_idx], beam_width)
            for log_p, token_id in zip(top_log_probs.tolist(), top_token_ids.tolist()):
                all_candidates.append((beam_score + log_p, beam_tokens + [token_id]))

        # --- Step 5: Keep only the top beam_width candidates ---
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = all_candidates[:beam_width]

        # --- Step 6: Separate finished beams ---
        active_beams = []
        for score, tokens in top_candidates:
            if tokens[-1] == eos_token_id:
                seq_len = len(tokens) - 1  # Exclude <SOS>
                penalised_score = score / (seq_len ** length_penalty)
                completed_beams.append((penalised_score, tokens[1:]))  # Strip <SOS>
            else:
                active_beams.append((score, tokens))

        if len(completed_beams) >= beam_width:
            break

    # --- Step 7: Force-complete if no beam ever hit <EOS> ---
    if not completed_beams:
        for score, tokens in active_beams:
            seq_len = max(len(tokens) - 1, 1)
            penalised_score = score / (seq_len ** length_penalty)
            completed_beams.append((penalised_score, tokens[1:]))

    # --- Step 8: Return the best beam ---
    completed_beams.sort(key=lambda x: x[0], reverse=True)
    _, best_tokens = completed_beams[0]
    return best_tokens
