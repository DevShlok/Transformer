import torch
from torch.utils.data import DataLoader, TensorDataset
from transformer_summarizer.architecture import TransformerSummarizer
from transformer_summarizer.pipeline import (
    train_loop,
    generate_summary,
    beam_search_summary,
    get_noam_scheduler,
)


def test():
    print("=" * 55)
    print("  Transformer Summarizer — Full Smoke Test")
    print("=" * 55)

    # --- Shared hyperparameters ---
    VOCAB_SIZE   = 100
    D_MODEL      = 32
    NUM_HEADS    = 2
    NUM_LAYERS   = 2
    D_FF         = 64
    SOS_ID       = 1
    EOS_ID       = 2
    PAD_ID       = 0
    device       = torch.device("cpu")

    # Build model; weight_tying requires identical src/tgt vocab sizes
    model = TransformerSummarizer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        weight_tying=True,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {param_count:,}")

    # ------------------------------------------------------------------
    # 1. Forward pass
    # ------------------------------------------------------------------
    print("\n[1] Forward pass")
    src = torch.randint(3, VOCAB_SIZE, (2, 10))   # batch=2, src_len=10
    tgt = torch.randint(3, VOCAB_SIZE, (2, 15))   # batch=2, tgt_len=15
    logits = model(src, tgt)
    expected_shape = (2, 15, VOCAB_SIZE)
    assert logits.shape == expected_shape, (
        f"Shape mismatch: got {logits.shape}, expected {expected_shape}"
    )
    print(f"    logits shape: {tuple(logits.shape)}  [OK]")

    # ------------------------------------------------------------------
    # 2. Training with Noam scheduler + gradient clipping
    # ------------------------------------------------------------------
    print("\n[2] Training (2 epochs, Noam LR schedule, grad clipping)")

    # Paper-exact Adam settings for transformer training
    # lr=1.0 is intentional — the Noam scheduler scales it correctly
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = get_noam_scheduler(optimizer, d_model=D_MODEL, warmup_steps=400)

    # Toy dataset: 8 samples with explicit <SOS> and <EOS> tokens
    dummy_src = torch.randint(3, VOCAB_SIZE, (8, 10))
    dummy_tgt = torch.cat([
        torch.full((8, 1), SOS_ID),
        torch.randint(3, VOCAB_SIZE, (8, 8)),
        torch.full((8, 1), EOS_ID),
    ], dim=1)
    loader = DataLoader(TensorDataset(dummy_src, dummy_tgt), batch_size=4)

    train_loop(
        model, loader, optimizer, device,
        epochs=2,
        pad_token_id=PAD_ID,
        clip_grad_norm=1.0,
        scheduler=scheduler,
    )
    print("    Training loop completed without errors  [OK]")

    source_single = torch.randint(3, VOCAB_SIZE, (1, 10))

    # ------------------------------------------------------------------
    # 3. Greedy decoding
    # ------------------------------------------------------------------
    print("\n[3] Greedy decoding")
    greedy_result = generate_summary(
        model, source_single,
        max_length=20,
        sos_token_id=SOS_ID,
        eos_token_id=EOS_ID,
        device=device,
        pad_token_id=PAD_ID,
    )
    print(f"    tokens ({len(greedy_result)}): {greedy_result}  [OK]")

    # ------------------------------------------------------------------
    # 4. Beam search decoding
    # ------------------------------------------------------------------
    print("\n[4] Beam search decoding  (beam_width=4, length_penalty=0.7)")
    beam_result = beam_search_summary(
        model, source_single,
        max_length=20,
        sos_token_id=SOS_ID,
        eos_token_id=EOS_ID,
        device=device,
        beam_width=4,
        length_penalty=0.7,
        pad_token_id=PAD_ID,
    )
    print(f"    tokens ({len(beam_result)}): {beam_result}  [OK]")

    print("\n" + "=" * 55)
    print("  All tests passed!")
    print("=" * 55)


if __name__ == "__main__":
    test()
