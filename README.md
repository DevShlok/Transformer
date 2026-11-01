<div align="center">

# 🧠 Transformer from Scratch: Architecture, Math & Telemetry Engine

**A complete, ground-up implementation of the Encoder-Decoder Transformer architecture in PyTorch, paired with a real-time FastAPI telemetry server and an interactive Neumorphic Next.js visualization interface.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16%20App%20Router-black.svg?logo=next.dot.js&logoColor=white)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0%2B-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Architecture Overview](#-architecture--mathematical-foundations) •
[5-Phase Telemetry Visualizer](#-5-phase-interactive-telemetry-engine) •
[Quickstart Guide](#-quickstart-guide) •
[Project Structure](#-project-structure) •
[API Reference](#-api-specification)

---

</div>

## 📌 Executive Summary

Modern Natural Language Processing is built upon the Transformer architecture introduced by Vaswani et al. in *"Attention Is All You Need"* (2017). While high-level libraries abstract these components behind one-line calls, understanding their internal mechanics requires building them from first principles.

This repository provides:
1. **From-Scratch PyTorch Core (`transformer_summarizer/`)**: A modular, mathematically rigorous implementation of Positional Encodings, Scaled Dot-Product Attention, Multi-Head Attention (MHA), Causal Masking, Feed-Forward Networks, and Encoder-Decoder Stacks with both Greedy and Beam Search decoding.
2. **Real-Time Telemetry Backend (`api_server.py`)**: A production-ready FastAPI service extracting live internal tensor slices, Layer-0 attention matrices across all attention heads, and autoregressive vocabulary probability distributions.
3. **Interactive Visualizer (`frontend/`)**: A responsive, industrial Neumorphic Next.js application that renders each stage of sequence-to-sequence synthesis step-by-step with zero placeholder animations.

---

## 📐 Architecture & Mathematical Foundations

```
               [ Input Sequence ]
                       │
             ┌─────────┴─────────┐
             │ Token Embeddings  │
             └─────────┬─────────┘
                       │
             ┌─────────┴─────────┐
             │ Positional Encod. │ ◄── Sinusoidal Harmonic Functions
             └─────────┬─────────┘
                       │
           ┌───────────▼───────────┐
           │     ENCODER STACK     │
           │  ┌─────────────────┐  │
           │  │ Multi-Head Attn │  │ ◄── Bidirectional Self-Attention
           │  ├─────────────────┤  │
           │  │ Add & RMSNorm   │  │
           │  ├─────────────────┤  │
           │  │  Feed-Forward   │  │
           │  ├─────────────────┤  │
           │  │ Add & RMSNorm   │  │
           │  └─────────────────┘  │
           └───────────┬───────────┘
                       │ Memory Keys & Values (K, V)
                       │
           ┌───────────▼───────────┐
           │     DECODER STACK     │
           │  ┌─────────────────┐  │
           │  │ Masked MHA      │  │ ◄── Causal Look-Ahead Mask (M_ij = -∞)
           │  ├─────────────────┤  │
           │  │ Cross-Attention │  │ ◄── Decoder Queries (Q) × Encoder (K, V)
           │  ├─────────────────┤  │
           │  │  Feed-Forward   │  │
           │  ├─────────────────┤  │
           │  │ Add & RMSNorm   │  │
           │  └─────────────────┘  │
           └───────────┬───────────┘
                       │
             ┌─────────┴──────────┐
             │ Linear Projection  │ ◄── d_model → |V| Logits
             └─────────┬──────────┘
                       │
             ┌─────────┴──────────┐
             │ Softmax & Decoding │ ◄── Greedy & Beam Search (Top-k)
             └────────────────────┘
```

### 1. Sinusoidal Positional Encoding
Transformers lack recurrence or convolutions; thus, token ordering must be explicitly injected into dense representations. We implement the fixed harmonic formulation:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Where:
- $pos$ is the token position in the sequence ($0 \le pos < \text{max\_len}$).
- $i$ is the dimension index ($0 \le i < d_{\text{model}}/2$).
- The geometric progression of wavelengths from $2\pi$ to $10000 \cdot 2\pi$ allows the self-attention mechanism to learn relative offsets, since for any fixed offset $k$, $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$.

### 2. Scaled Dot-Product Attention
The fundamental building block queries memory addresses via vector similarity:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

- **Scaling Factor ($\frac{1}{\sqrt{d_k}}$)**: For large values of $d_k$, dot products grow large in magnitude, pushing the softmax function into regions with near-zero gradients. Scaling stabilizes gradient variance during backpropagation.
- **Mask ($M$)**: In bidirectional encoder layers, $M = 0$. In autoregressive decoder layers, $M_{ij} = -\infty$ for $j > i$ to prevent attending to future tokens.

### 3. Multi-Head Attention (MHA)
Instead of performing a single attention function with $d_{\text{model}}$-dimensional queries, keys, and values, Multi-Head Attention projects them $h$ times with learned linear projections:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

$$\text{where} \quad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

With parameter matrices:
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$

### 4. Position-Wise Feed-Forward Network (FFN)
Each encoder and decoder layer contains a fully connected feed-forward network applied to each position separately and identically:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Typically with inner dimensionality $d_{ff} = 4 \times d_{\text{model}}$.

### 5. Learning Rate Scheduling (Noam Schedule)
During training, the learning rate is modulated according to the formula:

$$lr = d_{\text{model}}^{-0.5} \cdot \min\left(\text{step\_num}^{-0.5}, \text{step\_num} \cdot \text{warmup\_steps}^{-1.5}\right)$$

---

## 🖥 5-Phase Interactive Telemetry Engine

The frontend visualizer interfaces directly with the neural backend, breaking down the inference pipeline into five inspectable stages:

```
[Phase 1] Tokenize     ──► Byte-Pair Encoding subwords & vocabulary token IDs
[Phase 2] Embed + PE   ──► Dense embedding slices & sinusoidal positional waves
[Phase 3] Encoder MHA  ──► Interactive attention heatmaps (all heads + entropy)
[Phase 4] Decoder      ──► Causal triangular masking & cross-attention grounding
[Phase 5] Synthesis    ──► Autoregressive generation & live top-5 beam candidates
```

- **Phase 1 (Tokenization & BPE)**: Live inspection of sub-word segmentation, token IDs, and vocabulary mappings.
- **Phase 2 (Embedding & Positional Waveforms)**: Visual comparison of semantic embeddings against sinusoidal positional frequencies across dimensions.
- **Phase 3 (Bidirectional Multi-Head Attention)**: Full $N \times N$ attention matrix with interactive query-key cell selection, per-head switching ($H_0 \dots H_7$ and Average Head), row entropy calculations, and top-attended keys.
- **Phase 4 (Decoder Stack & Causal Masking)**: Step-by-step causal time-travel grid ($M_{ij} = -\infty$) paired with cross-attention source grounding.
- **Phase 5 (Autoregressive Synthesis & Beam Search)**: Real-time generation displaying probability distributions, top-5 beam candidates per step, and length-penalty scores.
- **Neumorphic Industrial UI**: Tactile dual-shadow surfaces, high-contrast typography, interactive light/dark theme toggle, and a 0.5x–2.0x playback speed adjuster.

---

## 🚀 Quickstart Guide

This project runs across **Windows, macOS, and Linux**. Follow the steps below to set up both the Python backend and Next.js frontend.

### Prerequisites
- **Python**: Version `3.10` or higher (`python --version`)
- **Node.js**: Version `18.17` or higher (`node --version`)
- **Git**: Installed and available in terminal

---

### Step 1: Clone Repository

```bash
git clone https://github.com/DevShlok/Transformer.git
cd Transformer
```

---

### Step 2: Set Up Python Backend

#### On Windows (PowerShell):
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### On macOS / Linux (Bash / Zsh):
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 3: Run the From-Scratch Smoke Tests

Verify that the mathematical foundations, forward pass, Noam optimizer, greedy decoding, and beam search pass all verification checks:

```bash
python test_run.py
```

*Expected output: Model parameters summary, forward pass logits verification, 2 training epochs with Noam schedule, and decoding tests completed successfully.*

---

### Step 4: Start the FastAPI Telemetry Server

Launch the backend API server on port `8000`:

```bash
uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
```

Verify backend health at: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### Step 5: Set Up and Run the Next.js Frontend

In a new terminal window:

```bash
cd frontend

# Install npm dependencies
npm install

# Start development server
npm run dev
```

Open your browser at **[http://localhost:3000](http://localhost:3000)**.

---

## 📁 Project Structure

```
Transformer/
├── transformer_summarizer/          # Core PyTorch Transformer from scratch
│   ├── __init__.py                  # Package exports
│   ├── math_foundations.py          # PositionalEncoding & Scaled Dot-Product Attention
│   ├── building_blocks.py           # MultiHeadAttention, EncoderLayer, DecoderLayer
│   ├── architecture.py              # Full TransformerSummarizer Encoder-Decoder model
│   └── pipeline.py                  # Training loop, Noam scheduler, greedy & beam search
│
├── api_server.py                    # FastAPI server exposing live attention & telemetry
├── test_run.py                      # Full verification test suite for scratch architecture
├── requirements.txt                 # Backend Python package manifest
│
└── frontend/                        # Interactive Neumorphic Telemetry UI (Next.js 16)
    ├── src/
    │   ├── app/
    │   │   ├── page.tsx             # Main visualizer page & 5-phase interactive UI
    │   │   ├── layout.tsx           # Root metadata & layout
    │   │   └── globals.css          # Neumorphic dual-shadow design system & animations
    │   ├── services/
    │   │   └── telemetry.ts         # API client & deterministic client-side simulation
    │   └── types/
    │       └── api.ts               # TypeScript schemas for tokens, steps & attention
    ├── package.json                 # Node dependencies
    ├── tsconfig.json                # TypeScript configuration
    └── next.config.ts               # Next.js build configuration
```

---

## 📡 API Specification

### `GET /health`
Returns model operational status and architectural dimensions.
```json
{
  "status": "healthy",
  "model": "facebook/bart-large-cnn",
  "d_model": 1024,
  "num_heads": 16,
  "num_layers": 12,
  "vocab_size": 50264
}
```

### `POST /api/inspect`
Processes an input paragraph and returns full internal tensor telemetry.

**Request Body:**
```json
{
  "text": "Your long source paragraph goes here...",
  "max_tokens": 1024
}
```

**Response Payload (`TelemetryResponse`):**
- `tokens`: Sub-word tokens, IDs, and positions ($0 \dots N$).
- `embedding_slices`: Truncated dense embedding vector slices ($d_0 \dots d_{15}$).
- `pe_slices`: Exact sinusoidal positional encoding values across dimensions.
- `heads_attention`: Layer-0 self-attention weights for all 8 individual heads.
- `avg_attention`: Mean attention matrix averaged across all heads.
- `generated_sequence`: Step-by-step synthesized tokens with top-5 candidate probabilities.
- `final_output_text`: Decoded abstractive summary string.

---

## 🧪 Technical Highlights

- **Zero Truncation Bottlenecks**: Supports arbitrarily long paragraphs with dynamic sequence expansion and automatic length scaling.
- **Deterministic Offline Fallback**: If the PyTorch backend is offline, the frontend falls back seamlessly to mathematical client-side simulation without crashing.
- **Production-Ready Code**: Fully typed in both Python (Pydantic v2) and TypeScript, zero compiler warnings, and verified with `next build`.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

