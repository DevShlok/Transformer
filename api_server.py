import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import math
import re

from transformer_summarizer.math_foundations import PositionalEncoding

app = FastAPI(
    title="Transformer Paragraph Summarizer & Telemetry API",
    description="Pre-trained Seq2Seq Transformer model serving real paragraph summaries and attention weights.",
    version="2.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MODEL_NAME = "Falconsai/text_summarization"
print(f"Loading {MODEL_NAME} weights into memory...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, attn_implementation="eager")
model.eval()
print("Model ready!")

pos_encoder = PositionalEncoding(d_model=model.config.d_model, max_len=8192)

class InspectRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input paragraph or multi-paragraph document to summarize")
    max_tokens: int = Field(default=1024, ge=16, le=4096, description="Maximum summary length")

class TokenItem(BaseModel):
    word: str
    id: int
    pos: int

class CandidateToken(BaseModel):
    word: str
    id: int
    prob: float

class GeneratedTokenStep(BaseModel):
    step: int
    word: str
    id: int
    prob: float
    candidates: List[CandidateToken]

class TelemetryResponse(BaseModel):
    tokens: List[TokenItem]
    d_model: int
    num_heads: int
    embedding_slices: List[List[float]]
    pe_slices: List[List[float]]
    heads_attention: List[List[List[float]]]
    avg_attention: List[List[float]]
    generated_sequence: List[GeneratedTokenStep]
    final_output_text: str

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "d_model": model.config.d_model,
        "num_heads": model.config.num_heads,
        "num_layers": model.config.num_layers,
    }

@app.post("/api/inspect", response_model=TelemetryResponse)
def summarize_and_inspect(req: InspectRequest):
    raw_text = req.text.strip()
    if not raw_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": {"code": "EMPTY_INPUT", "message": "Input paragraph cannot be empty."}}
        )

    # Format prompt with standard T5 summarization prefix for abstractive summarization
    prefix = "summarize: "
    t5_prompt = prefix + raw_text.strip()
    inputs = tokenizer(t5_prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs["input_ids"][0]

    # No artificial limits on generation:
    # Allow model to synthesize complete text up to 4096 tokens or user-specified max
    max_tokens_to_generate = req.max_tokens if (req.max_tokens and req.max_tokens > 0) else 4096

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=4,
            max_new_tokens=max_tokens_to_generate,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
            early_stopping=True,
            output_attentions=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Clean input tokens for visualization — skip the "summarize: " control prefix
    # so the visualizer starts directly at the user's document text
    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    tokens_list: List[TokenItem] = []

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    start_offset = len(prefix_ids)

    # Allow richer sequence inspection (up to 64 tokens)
    display_limit = min(len(raw_tokens) - start_offset, 64)
    for i in range(display_limit):
        idx = i + start_offset
        cleaned_word = raw_tokens[idx].replace(" ", "").replace("\u2581", "").strip()
        if not cleaned_word:
            cleaned_word = raw_tokens[idx].replace("\u2581", "")
        tokens_list.append(TokenItem(word=cleaned_word, id=int(input_ids[idx].item()), pos=i))

    # Extract true dense embeddings and true sinusoidal positional encoding tensors
    with torch.no_grad():
        embed_fn = model.encoder.embed_tokens  # Works for both T5 and Falconsai
        actual_embeddings = embed_fn(input_ids[start_offset:start_offset+display_limit])
        embedding_slices = actual_embeddings[:, :16].tolist()
        embedding_slices = [[round(val, 4) for val in row] for row in embedding_slices]

        raw_pe = pos_encoder.pe[0, :display_limit, :]
        pe_slices = raw_pe[:, :16].tolist()
        pe_slices = [[round(val, 4) for val in row] for row in pe_slices]

    # Extract Encoder Layer 0 Self-Attention weights
    enc_attn = outputs.encoder_attentions[0][0]
    num_heads = enc_attn.shape[0]

    heads_attn: List[List[List[float]]] = []
    for h in range(num_heads):
        head_matrix = enc_attn[h, start_offset:start_offset+display_limit, start_offset:start_offset+display_limit].tolist()
        head_matrix = [[round(val, 4) for val in row] for row in head_matrix]
        heads_attn.append(head_matrix)

    avg_matrix = enc_attn[:, start_offset:start_offset+display_limit, start_offset:start_offset+display_limit].mean(dim=0).tolist()
    avg_matrix = [[round(val, 4) for val in row] for row in avg_matrix]

    final_summary = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip()
    # Strip any leading punctuation or boundary artifacts and ensure proper capitalization
    final_summary = re.sub(r'^[\s,;:.!?\-–—]+', '', final_summary).strip()
    if final_summary:
        final_summary = final_summary[0].upper() + final_summary[1:]

    # Extract REAL autoregressive candidate probabilities for every step from outputs.scores
    generated_steps: List[GeneratedTokenStep] = []
    output_seq = outputs.sequences[0]

    # Iterate through all generation steps
    for s, score_tensor in enumerate(outputs.scores):
        if s + 1 >= len(output_seq):
            break

        probs = torch.softmax(score_tensor[0], dim=-1)
        top_k = torch.topk(probs, 5)

        chosen_id = int(output_seq[s + 1].item())
        
        # Skip special termination tokens, sentinel tokens (<extra_id_*>), and boundary meta tokens
        if chosen_id in tokenizer.all_special_ids or chosen_id == 3 or chosen_id >= 32000:
            continue

        raw_piece = tokenizer.convert_ids_to_tokens([chosen_id])[0]
        chosen_word = raw_piece.replace(" ", "").replace("\u2581", "").strip()
        if not chosen_word:
            continue

        chosen_prob = round(float(probs[chosen_id].item()), 4)

        candidates: List[CandidateToken] = []
        for p_val, idx_val in zip(top_k.values.tolist(), top_k.indices.tolist()):
            cand_piece = tokenizer.convert_ids_to_tokens([idx_val])[0]
            cand_word = cand_piece.replace(" ", "").replace("\u2581", "").strip()
            if not cand_word:
                cand_word = "</s>" if idx_val == tokenizer.eos_token_id else ("<sp>" if idx_val == 3 else f"tok_{idx_val}")
            candidates.append(CandidateToken(
                word=cand_word,
                id=int(idx_val),
                prob=round(float(p_val), 4)
            ))

        generated_steps.append(GeneratedTokenStep(
            step=len(generated_steps) + 1,
            word=chosen_word,
            id=chosen_id,
            prob=chosen_prob,
            candidates=candidates
        ))

    # Strip any leading punctuation steps so token sequence mirrors the cleaned summary
    while generated_steps and re.match(r'^[\s,;:.!?\-–—]+$', generated_steps[0].word):
        generated_steps.pop(0)

    if generated_steps and generated_steps[0].word:
        generated_steps[0].word = generated_steps[0].word.capitalize()

    for idx, step in enumerate(generated_steps):
        step.step = idx + 1

    return TelemetryResponse(
        tokens=tokens_list,
        d_model=model.config.d_model,
        num_heads=num_heads,
        embedding_slices=embedding_slices,
        pe_slices=pe_slices,
        heads_attention=heads_attn,
        avg_attention=avg_matrix,
        generated_sequence=generated_steps,
        final_output_text=final_summary,
    )
