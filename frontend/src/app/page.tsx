"use client";

import React, {
  useState,
  useEffect,
  useRef,
  useMemo,
  useCallback,
} from "react";
import { TelemetryResponse } from "@/types/api";
import { fetchTelemetry, generateLocalFallback } from "@/services/telemetry";

// ─────────────────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────────────────

const SAMPLE_TEXT =
  "Every second, optical networks carry massive amounts of data, consuming significant amounts of power in the process. What if AI could help them use only the energy they need? Our system combines AI-based traffic forecasting, a Digital Twin to validate proposed changes before implementation, and Quantum Optimization to dynamically adjust signal timings across the entire network.";

const PHASES = [
  { id: "tokenize", label: "Tokenize",      sub: "BPE sub-word"            },
  { id: "embed",    label: "Embed + PE",    sub: "Vector + sinusoid"       },
  { id: "encoder",  label: "Encoder MHA",   sub: "Self-attention L0"       },
  { id: "decoder",  label: "Decoder Stack", sub: "Causal + cross-attn"     },
  { id: "synth",    label: "Synthesis",     sub: "Softmax projection"      },
] as const;

// ─────────────────────────────────────────────────────────────────────────────
//  Math / helpers
// ─────────────────────────────────────────────────────────────────────────────

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

// Vaswani 2017 positional encoding
function peVal(pos: number, dim: number, d = 512) {
  return dim % 2 === 0
    ? Math.sin(pos / Math.pow(10000, dim / d))
    : Math.cos(pos / Math.pow(10000, (dim - 1) / d));
}

// Attention weight → clean neumorphic heat fill (NO purple gradients)
function attnBg(norm: number, isDark: boolean): string {
  const n = clamp(norm, 0, 1);
  if (n < 0.005) return "transparent";
  if (isDark) {
    // Subtle physical cobalt highlight on dark matte
    const alpha = (0.18 + n * 0.78).toFixed(2);
    return `rgba(59, 130, 246, ${alpha})`;
  }
  // Engineering blue on light slate
  const r = Math.round(235 - n * (235 - 29));
  const g = Math.round(242 - n * (242 - 78));
  const b = Math.round(255 - n * (255 - 216));
  return `rgb(${r},${g},${b})`;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Neumorphic UI Primitives
// ─────────────────────────────────────────────────────────────────────────────

type TextColor = "text" | "text-2" | "text-3" | "accent" | "teal";

function T({
  children,
  size = 13,
  weight = 400,
  mono = false,
  color = "text",
  as: Tag = "span",
  style: extra,
}: {
  children: React.ReactNode;
  size?: number;
  weight?: number;
  mono?: boolean;
  color?: TextColor;
  as?: "span" | "div" | "p";
  style?: React.CSSProperties;
}) {
  return (
    <Tag
      style={{
        fontSize: size,
        fontWeight: weight,
        fontFamily: mono ? "var(--mono)" : "var(--sans)",
        color: `var(--${color})`,
        ...extra,
      }}
    >
      {children}
    </Tag>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <span
      style={{
        fontSize: 10.5,
        fontWeight: 700,
        textTransform: "uppercase" as const,
        letterSpacing: "0.08em",
        color: "var(--text-3)",
      }}
    >
      {children}
    </span>
  );
}

function NeuPanel({
  children,
  style,
  inset = false,
  accent = false,
}: {
  children: React.ReactNode;
  style?: React.CSSProperties;
  inset?: boolean;
  accent?: boolean;
}) {
  return (
    <div
      className={inset ? "neu-inset" : "neu-flat"}
      style={{
        padding: "12px 14px",
        border: accent ? "1.5px solid var(--accent-bd)" : undefined,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

function NeuKpiCard({
  label,
  value,
  sub,
  accent = false,
}: {
  label: string;
  value: string | number;
  sub?: string;
  accent?: boolean;
}) {
  return (
    <div
      className="neu-flat"
      style={{
        padding: "10px 14px",
        border: accent ? "1.5px solid var(--accent-bd)" : undefined,
      }}
    >
      <div
        style={{
          fontSize: 22,
          fontWeight: 800,
          fontFamily: "var(--mono)",
          color: accent ? "var(--accent)" : "var(--text)",
          lineHeight: 1.1,
        }}
      >
        {value}
      </div>
      <div
        style={{
          fontSize: 10,
          fontWeight: 700,
          textTransform: "uppercase" as const,
          letterSpacing: "0.08em",
          color: "var(--text-3)",
          marginTop: 4,
        }}
      >
        {label}
      </div>
      {sub && (
        <div style={{ fontSize: 11, fontWeight: 500, color: "var(--text-2)", marginTop: 2 }}>{sub}</div>
      )}
    </div>
  );
}

function NeuButton({
  children,
  active = false,
  onClick,
  style,
  title,
  ariaLabel,
}: {
  children: React.ReactNode;
  active?: boolean;
  onClick?: () => void;
  style?: React.CSSProperties;
  title?: string;
  ariaLabel?: string;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      aria-label={ariaLabel}
      className={`neu-btn ${active ? "neu-btn-active" : ""}`}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 6,
        padding: "5px 12px",
        color: active ? "var(--accent)" : "var(--text)",
        fontWeight: active ? 700 : 500,
        ...style,
      }}
    >
      {children}
    </button>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  Phase 1 — Tokenizer (Clickable Filters & BPE Decomposition)
// ─────────────────────────────────────────────────────────────────────────────

function Phase1Tokenizer({
  telemetry,
  visible,
  sel,
  onSel,
  onJumpToEmbed,
}: {
  telemetry: TelemetryResponse;
  visible: number;
  sel: number;
  onSel: (i: number) => void;
  onJumpToEmbed: () => void;
}) {
  const [filterMode, setFilterMode] = useState<"all" | "subwords" | "words">("all");
  const tokens = telemetry.tokens;
  const subwords = tokens.filter((t) => !t.word.startsWith("▁")).length;
  const selTok = tokens[sel];

  const filteredTokens = useMemo(() => {
    if (filterMode === "subwords") {
      return tokens.map((t, idx) => ({ ...t, idx, match: !t.word.startsWith("▁") }));
    }
    if (filterMode === "words") {
      return tokens.map((t, idx) => ({ ...t, idx, match: t.word.startsWith("▁") }));
    }
    return tokens.map((t, idx) => ({ ...t, idx, match: true }));
  }, [tokens, filterMode]);

  return (
    <div className="anim-enter" style={{ display: "flex", flexDirection: "column", gap: 12, height: "100%", overflow: "hidden" }}>
      {/* Top KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, flexShrink: 0 }}>
        <NeuKpiCard label="Input tokens"   value={tokens.length} accent />
        <NeuKpiCard label="Revealed"       value={visible || tokens.length} />
        <NeuKpiCard label="Sub-word frags" value={subwords} />
        <NeuKpiCard label="Vocab size"     value="32,128" />
      </div>

      {/* Token grid with Interactive Clickable Filters */}
      <NeuPanel style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexShrink: 0 }}>
          <SectionLabel>SentencePiece BPE output — click any token to inspect</SectionLabel>

          {/* Clickable Filter Buttons */}
          <div style={{ display: "flex", gap: 5 }}>
            {(["all", "subwords", "words"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setFilterMode(m)}
                className={`neu-btn ${filterMode === m ? "neu-btn-active" : ""}`}
                style={{
                  fontSize: 10.5,
                  padding: "3px 8px",
                  fontWeight: filterMode === m ? 700 : 500,
                  color: filterMode === m ? "var(--accent)" : "var(--text-3)",
                }}
              >
                {m === "all" ? "Show All" : m === "subwords" ? "Sub-words Only" : "Word Stems"}
              </button>
            ))}
          </div>
        </div>

        <div
          className="scroll-a"
          style={{ display: "flex", flexWrap: "wrap", gap: 7, alignContent: "flex-start", padding: 4 }}
        >
          {filteredTokens.map((t) => {
            const isVis = visible === 0 || t.idx < visible;
            const isSel = sel === t.idx;
            const isDimmed = !t.match;

            return (
              <button
                key={t.idx}
                onClick={() => onSel(t.idx)}
                className={`neu-btn ${isSel ? "neu-btn-active" : ""}`}
                style={{
                  padding: "5px 10px",
                  opacity: isDimmed ? 0.25 : isVis ? 1 : 0.45,
                  borderColor: isSel ? "var(--accent)" : undefined,
                  animationDelay: `${t.idx * 12}ms`,
                }}
              >
                <T size={10} mono weight={600} color={isSel ? "accent" : "text-3"}>#{t.id}</T>
                <T size={12.5} weight={isSel ? 700 : 600} mono color={isSel ? "accent" : "text"}>{t.word}</T>
              </button>
            );
          })}
        </div>
      </NeuPanel>

      {/* Selected token inspector with transition action */}
      {selTok && (
        <NeuPanel
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexShrink: 0,
            padding: "10px 16px",
          }}
        >
          <div style={{ display: "flex", gap: 16, alignItems: "baseline" }}>
            <T size={16} weight={700} mono color="accent">"{selTok.word}"</T>
            <T size={12} weight={600} mono color="text-2">Vocabulary ID: {selTok.id} · Sequence Position: #{selTok.pos}</T>
            <T size={11} mono color="text-3">
              Type: {selTok.word.startsWith("▁") ? "Word Stem" : "Sub-word Fragment"}
            </T>
          </div>

          {/* Interactive Transition Jump to Phase 2 */}
          <button
            onClick={onJumpToEmbed}
            className="neu-btn"
            style={{
              padding: "5px 12px",
              fontSize: 11.5,
              fontWeight: 700,
              color: "var(--accent)",
            }}
          >
            Inspect Vector in Phase 2 →
          </button>
        </NeuPanel>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  Phase 2 — Embeddings + PE (Clickable Frequencies & Clickable Dims)
// ─────────────────────────────────────────────────────────────────────────────

function Phase2Embed({
  telemetry,
  sel,
  onSel,
}: {
  telemetry: TelemetryResponse;
  sel: number;
  onSel: (i: number) => void;
}) {
  const [selectedDim, setSelectedDim] = useState<number>(0);
  const [activeHarmonic, setActiveHarmonic] = useState<"all" | "dim0" | "dim2" | "dim6">("all");

  const tokens   = telemetry.tokens;
  const tok      = tokens[sel];
  const pos      = tok?.pos ?? 0;
  const d        = telemetry.d_model;
  const embSlice = telemetry.embedding_slices[sel] ?? [];
  const peSlice  = telemetry.pe_slices[sel] ?? [];
  const dims     = Math.min(embSlice.length, peSlice.length, 12);
  const span     = Math.max(20, tokens.length);

  const wave = useMemo(() => {
    const pts = 64;
    const path = (dim: number) =>
      Array.from({ length: pts }, (_, i) => {
        const p = (i / (pts - 1)) * span;
        const v = peVal(p, dim, d);
        return `${i === 0 ? "M" : "L"} ${(8 + (i / (pts - 1)) * 484).toFixed(1)} ${(40 - v * 26).toFixed(2)}`;
      }).join(" ");
    return {
      d0: path(0),
      d2: path(2),
      d6: path(6),
      mx: (8 + clamp(pos / span, 0, 1) * 484).toFixed(2),
    };
  }, [span, pos, d]);

  // Dimension details
  const selDimEmb = embSlice[selectedDim] ?? 0;
  const selDimPe  = peSlice[selectedDim] ?? 0;
  const selDimSum = selDimEmb + selDimPe;
  const freqOmega = 1 / Math.pow(10000, (2 * Math.floor(selectedDim / 2)) / d);

  return (
    <div className="anim-enter" style={{ display: "flex", flexDirection: "column", gap: 10, height: "100%", overflow: "hidden" }}>
      {/* Clickable token selector pills */}
      <div
        className="scroll-x"
        style={{ display: "flex", gap: 6, flexShrink: 0, paddingBottom: 2 }}
      >
        {tokens.map((t, i) => (
          <button
            key={i}
            onClick={() => onSel(i)}
            className={`neu-btn ${i === sel ? "neu-btn-active" : ""}`}
            style={{
              padding: "4px 10px",
              borderRadius: 6,
              fontSize: 12,
              fontFamily: "var(--mono)",
              fontWeight: i === sel ? 700 : 500,
              color: i === sel ? "var(--accent)" : "var(--text-2)",
              flexShrink: 0,
            }}
          >
            {t.word}
          </button>
        ))}
      </div>

      {/* Token info row */}
      <NeuPanel style={{ flexShrink: 0, padding: "8px 14px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <div style={{ display: "flex", gap: 12, alignItems: "baseline" }}>
            <T size={15} weight={700} mono color="accent">"{tok?.word}"</T>
            <T size={12} weight={600} mono color="text-2">pos={pos} · id={tok?.id} · d_model={d}</T>
          </div>
          <T size={11} weight={600} mono color="text-3">E_final = E_tok + PE(pos)</T>
        </div>
      </NeuPanel>

      {/* Clickable Waveform plot: Clicking labels solos that harmonic */}
      <NeuPanel style={{ flexShrink: 0 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
          <SectionLabel>Click a harmonic label below to isolate frequency</SectionLabel>
          <T size={11.5} weight={700} mono color="accent">pos #{pos}</T>
        </div>
        <svg
          width="100%"
          height={68}
          viewBox="0 0 500 68"
          style={{ display: "block" }}
          aria-label="Positional encoding waveform"
        >
          <line x1="8" y1="36" x2="492" y2="36" stroke="var(--border)" strokeWidth="1" />
          {/* dim 6 — slowest */}
          {(activeHarmonic === "all" || activeHarmonic === "dim6") && (
            <path d={wave.d6} fill="none" stroke="var(--border-hi)" strokeWidth={activeHarmonic === "dim6" ? 2.5 : 1.4} className="anim-wave" />
          )}
          {/* dim 2 — medium */}
          {(activeHarmonic === "all" || activeHarmonic === "dim2") && (
            <path d={wave.d2} fill="none" stroke="var(--teal)" strokeWidth={activeHarmonic === "dim2" ? 2.5 : 1.6} strokeDasharray="4 3" className="anim-wave" />
          )}
          {/* dim 0 — fastest */}
          {(activeHarmonic === "all" || activeHarmonic === "dim0") && (
            <path d={wave.d0} fill="none" stroke="var(--accent)" strokeWidth={activeHarmonic === "dim0" ? 2.8 : 2} className="anim-wave" />
          )}
          {/* position marker */}
          <line x1={wave.mx} y1="4" x2={wave.mx} y2="64" stroke="var(--text)" strokeWidth="1.5" strokeDasharray="3 2" opacity="0.6" />
          <circle cx={wave.mx} cy="36" r="4" fill="var(--accent)" />
        </svg>

        {/* Clickable Harmonic Toggles */}
        <div style={{ display: "flex", gap: 10, marginTop: 4 }}>
          <button
            onClick={() => setActiveHarmonic(activeHarmonic === "dim0" ? "all" : "dim0")}
            className={`neu-btn ${activeHarmonic === "dim0" ? "neu-btn-active" : ""}`}
            style={{ padding: "3px 8px", fontSize: 11, fontWeight: 700, color: "var(--accent)" }}
          >
            ● dim 0 (local syntax)
          </button>
          <button
            onClick={() => setActiveHarmonic(activeHarmonic === "dim2" ? "all" : "dim2")}
            className={`neu-btn ${activeHarmonic === "dim2" ? "neu-btn-active" : ""}`}
            style={{ padding: "3px 8px", fontSize: 11, fontWeight: 700, color: "var(--teal)" }}
          >
            ● dim 2 (phrase context)
          </button>
          <button
            onClick={() => setActiveHarmonic(activeHarmonic === "dim6" ? "all" : "dim6")}
            className={`neu-btn ${activeHarmonic === "dim6" ? "neu-btn-active" : ""}`}
            style={{ padding: "3px 8px", fontSize: 11, fontWeight: 700, color: "var(--text-2)" }}
          >
            ● dim 6 (document)
          </button>
          {activeHarmonic !== "all" && (
            <button
              onClick={() => setActiveHarmonic("all")}
              style={{ fontSize: 11, color: "var(--text-3)", background: "none", border: "none", cursor: "pointer", textDecoration: "underline" }}
            >
              Reset All
            </button>
          )}
        </div>
      </NeuPanel>

      {/* Embedding vs PE clickable dimension slices */}
      <NeuPanel style={{ flex: 1, overflow: "hidden", minHeight: 0, display: "flex", flexDirection: "column" }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, flexShrink: 0 }}>
          <SectionLabel>Click any dimension d0–d{dims - 1} to inspect exact computation</SectionLabel>
          <T size={11} weight={600} mono color="accent">Selected: d{selectedDim}</T>
        </div>

        {/* Clickable Dimension Bars */}
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${dims}, 1fr)`, gap: 6, flexShrink: 0 }}>
          {Array.from({ length: dims }).map((_, di) => {
            const e  = embSlice[di] ?? 0;
            const p  = peSlice[di] ?? 0;
            const ne = clamp((e + 2) / 4, 0, 1);
            const np = clamp((p + 1) / 2, 0, 1);
            const isDimActive = selectedDim === di;

            return (
              <button
                key={di}
                onClick={() => setSelectedDim(di)}
                className={`neu-btn ${isDimActive ? "neu-btn-active" : ""}`}
                title={`Click to inspect d${di}: Learned=${e.toFixed(4)}, PE=${p.toFixed(4)}`}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 3,
                  alignItems: "center",
                  padding: "4px 2px",
                  borderColor: isDimActive ? "var(--accent)" : undefined,
                }}
              >
                <T size={9.5} weight={700} mono color={isDimActive ? "accent" : "text-3"}>d{di}</T>
                <div style={{ width: "80%", height: 6, borderRadius: 2, background: `var(--accent)`, opacity: 0.2 + ne * 0.8 }} />
                <div style={{ width: "80%", height: 6, borderRadius: 2, background: `var(--teal)`, opacity: 0.2 + np * 0.8 }} />
                <T size={9} weight={600} mono color="accent">{e.toFixed(1)}</T>
                <T size={9} weight={600} mono color="teal">{p.toFixed(1)}</T>
              </button>
            );
          })}
        </div>

        {/* Interactive Dimension Breakdown Well */}
        <div className="neu-inset" style={{ padding: "8px 12px", marginTop: 8, flex: 1, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <T size={12} weight={700} mono color="accent">Dimension d{selectedDim} Coordinate Calculation:</T>
            <div style={{ display: "flex", gap: 16, marginTop: 4 }}>
              <T size={11} mono color="text">Learned Embedding: <strong>{selDimEmb.toFixed(4)}</strong></T>
              <T size={11} mono color="teal">Sinusoid PE: <strong>{selDimPe.toFixed(4)}</strong></T>
              <T size={11} mono color="accent">Sum = <strong>{selDimSum.toFixed(4)}</strong></T>
            </div>
          </div>
          <T size={10.5} mono color="text-3">
            ω_{selectedDim} = 10000^(-{(2 * Math.floor(selectedDim / 2)) / d}) · {selectedDim % 2 === 0 ? "sin" : "cos"}({pos} × ω)
          </T>
        </div>
      </NeuPanel>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  Phase 3 — Encoder Multi-Head Attention (Clickable Headers, Cells & Keys)
// ─────────────────────────────────────────────────────────────────────────────

function Phase3Encoder({
  telemetry,
  scan,
  head,
  onHead,
  isDark,
}: {
  telemetry: TelemetryResponse;
  scan: { r: number; c: number } | null;
  head: number | "avg";
  onHead: (h: number | "avg") => void;
  isDark: boolean;
}) {
  const [userSelectedCell, setUserSelectedCell] = useState<{ r: number; c: number } | null>(null);

  const matrix = useMemo(
    () =>
      head === "avg"
        ? telemetry.avg_attention
        : (telemetry.heads_attention[head as number] ?? telemetry.avg_attention),
    [telemetry, head]
  );

  const displayLimit = Math.min(telemetry.tokens.length, 28);
  const tokens = telemetry.tokens.slice(0, displayLimit);
  const N      = tokens.length;
  const rowMax = useMemo(
    () => matrix.slice(0, N).map((row) => Math.max(...row.slice(0, N), 0.001)),
    [matrix, N]
  );

  const activeCell = scan || userSelectedCell || { r: 0, c: 0 };
  const sr = activeCell.r;
  const sc = activeCell.c;

  const activeQuery = tokens[sr]?.word || "—";
  const activeKey   = tokens[sc]?.word || "—";
  const activeWeight = matrix[sr]?.[sc] ?? 0;

  const entropy = useMemo(() => {
    const row = matrix[sr] ?? [];
    return -row.slice(0, N).reduce((a, v) => a + (v > 1e-9 ? v * Math.log2(v + 1e-9) : 0), 0);
  }, [matrix, sr, N]);

  const topAttendedKeys = useMemo(() => {
    const row = (matrix[sr] ?? []).slice(0, N);
    return row
      .map((w, idx) => ({ word: tokens[idx]?.word || `t${idx}`, weight: w, idx }))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 4);
  }, [matrix, sr, tokens, N]);

  const headComparison = useMemo(() => {
    return Array.from({ length: telemetry.num_heads }).map((_, h) => {
      const hMat = telemetry.heads_attention[h];
      return hMat?.[sr]?.[sc] ?? 0;
    });
  }, [telemetry, sr, sc]);

  return (
    <div className="anim-enter" style={{ display: "flex", flexDirection: "column", gap: 10, height: "100%", overflow: "hidden" }}>
      {/* Head selector bar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexShrink: 0 }}>
        <div style={{ display: "flex", gap: 5, flexWrap: "wrap", alignItems: "center" }}>
          <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-3)", marginRight: 4 }}>HEAD:</span>
          {(["avg", ...Array.from({ length: telemetry.num_heads }, (_, h) => h)] as (number | "avg")[]).map((h) => (
            <NeuButton
              key={String(h)}
              active={head === h}
              onClick={() => onHead(h)}
              style={{ padding: "4px 9px", fontSize: 11.5 }}
            >
              {h === "avg" ? "Avg Head" : `H${h}`}
            </NeuButton>
          ))}
        </div>
        <T size={11} mono weight={600} color="text-3">
          d_k = 64 · Layer 0 Bidirectional MHA
        </T>
      </div>

      {/* 2-Column Space-Utilized Workspace */}
      <div style={{ flex: 1, display: "grid", gridTemplateColumns: "1fr 340px", gap: 10, minHeight: 0, overflow: "hidden" }}>
        {/* Left: Interactive Heatmap with Clickable Headers */}
        <NeuPanel style={{ display: "flex", flexDirection: "column", overflow: "hidden", padding: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6, flexShrink: 0 }}>
            <SectionLabel>Bidirectional Attention Matrix — click rows, cols or cells</SectionLabel>
            <T size={10.5} mono color="text-3">Tokens: {N}×{N}</T>
          </div>

          <div className="scroll-a" style={{ flex: 1, minHeight: 0 }}>
            <table style={{ borderCollapse: "collapse", margin: "0 auto" }}>
              <thead>
                <tr>
                  <th style={{ width: 54, minWidth: 54 }} />
                  {tokens.map((t, c) => (
                    <th
                      key={c}
                      onClick={() => setUserSelectedCell({ r: sr, c })}
                      title={`Click to focus column "${t.word}"`}
                      style={{
                        writingMode: "vertical-rl",
                        transform: "rotate(180deg)",
                        height: 48,
                        verticalAlign: "bottom",
                        padding: "0 2px",
                        fontSize: 11,
                        fontFamily: "var(--mono)",
                        color: sc === c ? "var(--accent)" : "var(--text)",
                        fontWeight: sc === c ? 800 : 500,
                        cursor: "pointer",
                        maxWidth: 30,
                        overflow: "hidden",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {t.word}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tokens.map((rTok, r) => (
                  <tr key={r}>
                    <td
                      onClick={() => setUserSelectedCell({ r, c: sc })}
                      title={`Click to focus row "${rTok.word}"`}
                      style={{
                        fontSize: 11,
                        fontFamily: "var(--mono)",
                        color: sr === r ? "var(--accent)" : "var(--text)",
                        fontWeight: sr === r ? 800 : 500,
                        textAlign: "right",
                        paddingRight: 8,
                        maxWidth: 54,
                        overflow: "hidden",
                        whiteSpace: "nowrap",
                        cursor: "pointer",
                      }}
                    >
                      {rTok.word}
                    </td>
                    {tokens.map((cTok, c) => {
                      const raw  = matrix[r]?.[c] ?? 0;
                      const norm = raw / rowMax[r];
                      const isHot   = sr === r && sc === c;
                      const isCross = sr === r || sc === c;
                      const textColor = isHot || norm > 0.45 ? "#ffffff" : "var(--text)";

                      return (
                        <td key={c} style={{ padding: 1.5 }}>
                          <button
                            onClick={() => setUserSelectedCell({ r, c })}
                            className="attn-cell"
                            title={`${rTok.word} (Q) → ${cTok.word} (K): ${(raw * 100).toFixed(2)}%`}
                            style={{
                              width: 28,
                              height: 28,
                              borderRadius: 5,
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              fontSize: 9.5,
                              fontFamily: "var(--mono)",
                              fontWeight: isHot || norm > 0.35 ? 700 : 500,
                              background: isHot ? "var(--accent)" : attnBg(norm, isDark),
                              color: textColor,
                              transform: isHot ? "scale(1.22)" : isCross ? "scale(1.04)" : "scale(1)",
                              boxShadow: isHot
                                ? "0 3px 10px rgba(29, 78, 216, 0.45)"
                                : undefined,
                              border: isHot ? "1.5px solid var(--accent)" : "1px solid transparent",
                              cursor: "pointer",
                              zIndex: isHot ? 10 : 1,
                              position: "relative",
                            }}
                          >
                            {(raw * 100).toFixed(0)}
                          </button>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </NeuPanel>

        {/* Right: Live Interactive Attention Inspector */}
        <div style={{ display: "flex", flexDirection: "column", gap: 10, overflow: "hidden" }}>
          {/* Active Query-Key Inspector */}
          <NeuPanel style={{ flexShrink: 0 }}>
            <SectionLabel>Query ➔ Key Pairwise Focus</SectionLabel>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginTop: 6 }}>
              <div>
                <T size={15} weight={700} mono color="accent">"{activeQuery}"</T>
                <T size={12} weight={600} color="text-2"> attends to </T>
                <T size={15} weight={700} mono color="teal">"{activeKey}"</T>
              </div>
              <T size={16} weight={700} mono color="accent">
                {(activeWeight * 100).toFixed(2)}%
              </T>
            </div>
            <div className="neu-inset" style={{ height: 8, borderRadius: 4, overflow: "hidden", marginTop: 8, padding: 1 }}>
              <div style={{ height: "100%", width: `${Math.min(100, activeWeight * 100).toFixed(1)}%`, background: "var(--accent)", borderRadius: 3 }} />
            </div>
            <T size={10.5} mono color="text-3" as="div" style={{ marginTop: 6 }}>
              Pos {sr} (Query) attends to Pos {sc} (Key)
            </T>
          </NeuPanel>

          {/* Clickable Top Attended Keys for active Query word */}
          <NeuPanel style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <SectionLabel>Top Attended Keys for "{activeQuery}"</SectionLabel>
              <T size={10} color="text-3">Click to select</T>
            </div>

            <div className="scroll-y" style={{ flex: 1, display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
              {topAttendedKeys.map((item, i) => (
                <button
                  key={i}
                  onClick={() => setUserSelectedCell({ r: sr, c: item.idx })}
                  className="neu-btn"
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    padding: "6px 10px",
                    textAlign: "left",
                    borderColor: sc === item.idx ? "var(--accent)" : undefined,
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", width: "100%", fontSize: 11.5, marginBottom: 4 }}>
                    <div style={{ display: "flex", gap: 6, alignItems: "baseline" }}>
                      <T size={10} weight={700} mono color="text-3">#{i + 1}</T>
                      <T size={12} weight={700} mono color={sc === item.idx ? "accent" : "text"}>"{item.word}"</T>
                    </div>
                    <T size={11.5} weight={700} mono color="accent">{(item.weight * 100).toFixed(1)}%</T>
                  </div>
                  <div style={{ height: 4, width: "100%", background: "var(--border)", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${(item.weight * 100).toFixed(1)}%`, background: i === 0 ? "var(--accent)" : "var(--teal)" }} />
                  </div>
                </button>
              ))}
            </div>

            {/* Formula badge */}
            <div style={{ borderTop: "1px solid var(--border)", paddingTop: 8, marginTop: 6 }}>
              <T size={10.5} mono color="text-3" as="div">
                Attention(Q,K,V) = softmax(Q·Kᵀ / √64) · V
              </T>
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                <T size={11} color="text-2">Attention Dispersion:</T>
                <T size={11} weight={700} mono color="accent">{entropy.toFixed(3)} bits</T>
              </div>
            </div>
          </NeuPanel>

          {/* Across-Heads Variance Sparkbar */}
          <NeuPanel style={{ flexShrink: 0, padding: "8px 12px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
              <SectionLabel>8-Head Variance</SectionLabel>
              <T size={10} mono color="text-3">Click head to switch</T>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(8, 1fr)", gap: 4, alignItems: "end", height: 32 }}>
              {headComparison.map((val, h) => {
                const heightPct = Math.max(14, Math.min(100, val * 120));
                return (
                  <button
                    key={h}
                    onClick={() => onHead(h)}
                    title={`Click to switch to Head ${h}: ${(val * 100).toFixed(1)}%`}
                    style={{
                      background: "none",
                      border: "none",
                      padding: 0,
                      cursor: "pointer",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: 2,
                    }}
                  >
                    <div style={{ width: "100%", height: `${heightPct}%`, background: h === head ? "var(--accent)" : "var(--border-hi)", borderRadius: 2 }} />
                    <T size={9} mono color={h === head ? "accent" : "text-3"}>H{h}</T>
                  </button>
                );
              })}
            </div>
          </NeuPanel>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  Phase 4 — Decoder Stack (Clickable Sublayers, Causal Time-Travel, Cross-Attn)
// ─────────────────────────────────────────────────────────────────────────────

function Phase4Decoder({ telemetry, isDark }: { telemetry: TelemetryResponse; isDark: boolean }) {
  const [selectedDecStep, setSelectedDecStep] = useState<number>(0);
  const [activeSublayer, setActiveSublayer] = useState<number>(0);

  const encN = Math.min(telemetry.tokens.length, 24);
  const decN = Math.min(telemetry.generated_sequence.length, 16);
  const enc  = telemetry.tokens.slice(0, encN);
  const dec  = telemetry.generated_sequence.slice(0, decN);

  const crossRows = dec.map((_, di) => {
    const row = (telemetry.avg_attention[di] ?? []).slice(0, encN);
    const s   = row.reduce((a, b) => a + b, 0) || 1;
    return row.map((v) => v / s);
  });

  const causalMaskSize = 6;
  const causal = Array.from({ length: causalMaskSize }, (_, r) =>
    Array.from({ length: causalMaskSize }, (_, c) => c <= r)
  );

  const sublayerDetails = [
    { title: "Causal Masked Self-Attention", math: "Attention(Q,K,V) + M", desc: "Lower-triangular mask M enforces autoregressive causality; future tokens are masked with -∞." },
    { title: "Residual Connection & RMSNorm", math: "x = RMSNorm(x + Sublayer(x))", desc: "Preserves gradient propagation through deep decoder stack and stabilizes activations." },
    { title: "Encoder-Decoder Cross-Attention", math: "Q = Dec(x), K,V = Enc(Memory)", desc: "Queries from previous decoder stage query the memory keys and values of the full encoder." },
    { title: "Position-wise Feed-Forward", math: "FFN(x) = max(0, xW1 + b1)W2 + b2", desc: "Two linear transformations with ReLU/GELU activation projecting to d_ff = 2048." },
    { title: "Final LayerNorm & LM Head", math: "P(w) = softmax(W_vocab · x)", desc: "Projects final d_model dimensional state to 32,128 vocabulary logits." },
  ];

  return (
    <div className="anim-enter" style={{ display: "flex", flexDirection: "column", gap: 10, height: "100%", overflow: "hidden" }}>
      {/* Top Architecture Summary Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, flexShrink: 0 }}>
        <NeuPanel accent style={{ padding: "10px 14px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <SectionLabel>1. Autoregressive Causal Self-Attention</SectionLabel>
            <T size={10.5} mono color="accent">M_ij = -∞ (j &gt; i)</T>
          </div>
          <T size={12} weight={500} color="text-2" as="div" style={{ marginTop: 4, lineHeight: 1.5 }}>
            Decoder hidden state queries attend strictly to prior generated tokens (0…t−1). Softmax(x + M) forces future probability to 0.
          </T>
        </NeuPanel>

        <NeuPanel style={{ padding: "10px 14px", border: "1.5px solid var(--teal-bd)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <SectionLabel>2. Encoder-Decoder Cross-Attention Bridge</SectionLabel>
            <T size={10.5} mono color="teal">Query: Dec · Key/Val: Enc</T>
          </div>
          <T size={12} weight={500} color="text-2" as="div" style={{ marginTop: 4, lineHeight: 1.5 }}>
            Decoder queries attend over the final encoder representations, allowing synthesized words to distill relevant input context.
          </T>
        </NeuPanel>
      </div>

      {/* Middle Dual Matrix Studio */}
      <div style={{ flex: 1, display: "grid", gridTemplateColumns: "360px 1fr", gap: 10, minHeight: 0, overflow: "hidden" }}>
        {/* Left: Clickable Causal Time-Travel Grid */}
        <NeuPanel style={{ display: "flex", flexDirection: "column", overflow: "hidden", padding: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6, flexShrink: 0 }}>
            <SectionLabel>Causal Lower-Triangular Mask</SectionLabel>
            <T size={11} mono color="accent">Step t{selectedDecStep}</T>
          </div>

          {/* Clickable Step Selector */}
          <div style={{ display: "flex", gap: 4, marginBottom: 10, flexShrink: 0 }}>
            {Array.from({ length: causalMaskSize }).map((_, stepIdx) => (
              <button
                key={stepIdx}
                onClick={() => setSelectedDecStep(stepIdx)}
                className={`neu-btn ${selectedDecStep === stepIdx ? "neu-btn-active" : ""}`}
                style={{
                  flex: 1,
                  padding: "4px 0",
                  fontSize: 11,
                  fontFamily: "var(--mono)",
                  fontWeight: selectedDecStep === stepIdx ? 800 : 500,
                  color: selectedDecStep === stepIdx ? "var(--accent)" : "var(--text-2)",
                }}
              >
                t{stepIdx}
              </button>
            ))}
          </div>

          {/* 6x6 Causal Grid */}
          <div className="scroll-a" style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center" }}>
            <table style={{ borderCollapse: "collapse", margin: "0 auto" }}>
              <thead>
                <tr>
                  <th style={{ width: 34 }} />
                  {Array.from({ length: causalMaskSize }, (_, c) => (
                    <th key={c} style={{ fontSize: 10.5, fontFamily: "var(--mono)", fontWeight: 700, color: "var(--text-3)", padding: "0 4px", textAlign: "center" }}>
                      t{c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {causal.map((row, r) => {
                  const isCurrentRow = r === selectedDecStep;
                  return (
                    <tr key={r}>
                      <td style={{ fontSize: 10.5, fontFamily: "var(--mono)", fontWeight: isCurrentRow ? 800 : 500, color: isCurrentRow ? "var(--accent)" : "var(--text-3)", paddingRight: 6, textAlign: "right" }}>
                        t{r}
                      </td>
                      {row.map((visible, c) => (
                        <td key={c} style={{ padding: 2 }}>
                          <button
                            onClick={() => setSelectedDecStep(r)}
                            style={{
                              width: 42,
                              height: 30,
                              borderRadius: 5,
                              background: visible
                                ? (isCurrentRow ? "var(--accent)" : "var(--hover)")
                                : "var(--surface)",
                              border: `1px solid ${
                                visible
                                  ? (isCurrentRow ? "var(--accent)" : "var(--border)")
                                  : "var(--border-subtle)"
                              }`,
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              fontSize: 10.5,
                              fontFamily: "var(--mono)",
                              color: visible
                                ? (isCurrentRow ? "#ffffff" : "var(--accent)")
                                : "var(--text-3)",
                              fontWeight: visible ? 700 : 500,
                              cursor: "pointer",
                            }}
                          >
                            {visible ? "attend" : "−inf"}
                          </button>
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div style={{ borderTop: "1px solid var(--border)", paddingTop: 8, marginTop: 6, flexShrink: 0 }}>
            <T size={11} color="text-2" as="div">
              Step t{selectedDecStep}: positions t0…t{selectedDecStep} are visible; future tokens are strictly masked with −∞.
            </T>
          </div>
        </NeuPanel>

        {/* Right: Cross-Attention Heatmap */}
        <NeuPanel style={{ display: "flex", flexDirection: "column", overflow: "hidden", padding: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8, flexShrink: 0 }}>
            <SectionLabel>Cross-Attention Heatmap (Decoder Q × Encoder K)</SectionLabel>
            <T size={11} mono color="teal">Source Grounding</T>
          </div>

          <div className="scroll-a" style={{ flex: 1, minHeight: 0 }}>
            {crossRows.length > 0 ? (
              <table style={{ borderCollapse: "collapse", margin: "0 auto" }}>
                <thead>
                  <tr>
                    <th style={{ width: 56, minWidth: 56 }} />
                    {enc.map((t, c) => (
                      <th
                        key={c}
                        style={{
                          writingMode: "vertical-rl",
                          transform: "rotate(180deg)",
                          height: 48,
                          verticalAlign: "bottom",
                          fontSize: 10.5,
                          fontFamily: "var(--mono)",
                          color: "var(--text)",
                          fontWeight: 500,
                          padding: "0 2px",
                          maxWidth: 28,
                          overflow: "hidden",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {t.word}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dec.map((d, r) => (
                    <tr key={r}>
                      <td
                        style={{
                          fontSize: 11,
                          fontFamily: "var(--mono)",
                          color: "var(--teal)",
                          fontWeight: 700,
                          textAlign: "right",
                          paddingRight: 8,
                          maxWidth: 56,
                          overflow: "hidden",
                          whiteSpace: "nowrap",
                        }}
                      >
                        "{d.word}"
                      </td>
                      {crossRows[r]?.map((v, c) => (
                        <td key={c} style={{ padding: 2 }}>
                          <div
                            title={`Decoder "${d.word}" attending to Input "${enc[c]?.word}": ${(v * 100).toFixed(1)}%`}
                            style={{
                              width: 32,
                              height: 26,
                              borderRadius: 4,
                              background: attnBg(v, isDark),
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              fontSize: 9.5,
                              fontFamily: "var(--mono)",
                              fontWeight: v > 0.35 ? 700 : 500,
                              color: v > 0.45 ? "#ffffff" : "var(--text)",
                            }}
                          >
                            {(v * 100).toFixed(0)}
                          </div>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div style={{ textAlign: "center", padding: 24 }}>
                <T size={12} color="text-3">Run the transformer to compute cross-attention weights.</T>
              </div>
            )}
          </div>
        </NeuPanel>
      </div>

      {/* Bottom: Clickable Decoder Sub-Layer Pipeline */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8, flexShrink: 0 }}>
        {sublayerDetails.map((item, i) => (
          <button
            key={i}
            onClick={() => setActiveSublayer(i)}
            className={`neu-btn ${activeSublayer === i ? "neu-btn-active" : ""}`}
            style={{
              padding: "7px 10px",
              textAlign: "left",
              borderColor: activeSublayer === i ? "var(--accent)" : undefined,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <T size={10} weight={700} mono color={activeSublayer === i ? "accent" : "text-3"}>
                0{i + 1}
              </T>
              <T size={9} mono color="teal">{item.math.split("=")[0]}</T>
            </div>
            <T size={11.5} weight={700} color={activeSublayer === i ? "accent" : "text"} as="div" style={{ marginTop: 2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {item.title}
            </T>
          </button>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  Phase 5 — Synthesis (Clickable Alternative Candidates & Step Linking)
// ─────────────────────────────────────────────────────────────────────────────

function Phase5Synthesis({
  telemetry,
  step,
  streaming,
  running,
  phase,
  onSelectStep,
}: {
  telemetry: TelemetryResponse;
  step: number;
  streaming: string[];
  running: boolean;
  phase: number;
  onSelectStep: (s: number) => void;
}) {
  const [selectedCandidateIdx, setSelectedCandidateIdx] = useState<number>(0);
  const stepData   = telemetry.generated_sequence[step];
  const candidates = stepData?.candidates ?? [];
  const maxProb    = candidates[0]?.prob ?? 1;

  return (
    <div className="anim-enter" style={{ display: "flex", flexDirection: "column", gap: 12, height: "100%", overflow: "hidden" }}>
      {/* KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, flexShrink: 0 }}>
        <NeuKpiCard label="Decode step"    value={`#${step + 1} / ${telemetry.generated_sequence.length}`} />
        <NeuKpiCard label="Selected token" value={stepData?.word ?? "—"} accent />
        <NeuKpiCard label="Confidence"     value={stepData ? `${(stepData.prob * 100).toFixed(2)}%` : "—"} />
      </div>

      {/* Candidate bars with Clickable Candidate Alternative Exploration */}
      <NeuPanel style={{ flexShrink: 0 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
          <SectionLabel>Top-5 softmax candidates — click any candidate to inspect hypothesis</SectionLabel>
          <T size={11} weight={600} mono color="text-3">num_beams=4 · length_penalty=1.0</T>
        </div>
        {candidates.length === 0 ? (
          <T size={12.5} weight={500} color="text-3" as="div" style={{ textAlign: "center", padding: "16px 0" }}>
            Run the transformer to see live candidate selection
          </T>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {candidates.map((c, i) => {
              const win = i === 0;
              const isCandActive = selectedCandidateIdx === i;
              const pct = clamp((c.prob / (maxProb || 1)) * 100, 2, 100);
              return (
                <button
                  key={i}
                  onClick={() => setSelectedCandidateIdx(i)}
                  className={`neu-btn ${isCandActive ? "neu-btn-active" : ""}`}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    padding: "6px 12px",
                    textAlign: "left",
                    borderColor: isCandActive ? "var(--accent)" : undefined,
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, width: "100%" }}>
                    <div style={{ display: "flex", gap: 10, alignItems: "baseline" }}>
                      <T size={11} weight={700} mono color={win ? "accent" : "text-3"}>#{i + 1}</T>
                      <T size={13.5} weight={win ? 700 : 500} mono color={win ? "accent" : "text"}>&ldquo;{c.word}&rdquo;</T>
                      <T size={10} weight={600} mono color="text-3">id {c.id}</T>
                    </div>
                    <T size={13} weight={win ? 700 : 500} mono color={win ? "accent" : "text"}>
                      {(c.prob * 100).toFixed(2)}%
                    </T>
                  </div>
                  <div style={{ height: 6, width: "100%", background: "var(--border)", borderRadius: 3, overflow: "hidden" }}>
                    <div
                      className="anim-bar"
                      style={{
                        height: "100%",
                        width: `${pct.toFixed(1)}%`,
                        background: win ? "var(--accent)" : "var(--teal)",
                        borderRadius: 3,
                      }}
                    />
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </NeuPanel>

      {/* Output token stream with Clickable Tokens Linking to Decode Steps */}
      <NeuPanel style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <SectionLabel>Autoregressive output stream — click token to jump to decode step</SectionLabel>
          <T size={10.5} mono color="text-3">Synthesized: {streaming.length}</T>
        </div>

        <div
          className="scroll-a"
          style={{ flex: 1, display: "flex", flexWrap: "wrap", gap: 6, alignContent: "flex-start", padding: 4 }}
        >
          {streaming.map((w, i) => {
            const isCurrent = i === step;
            return (
              <button
                key={i}
                onClick={() => onSelectStep(i)}
                className={`neu-btn ${isCurrent ? "neu-btn-active" : ""}`}
                title={`Click to inspect step #${i + 1} candidate probabilities`}
                style={{
                  padding: "5px 10px",
                  fontSize: 13,
                  fontFamily: "var(--mono)",
                  fontWeight: isCurrent ? 800 : 500,
                  color: isCurrent ? "var(--accent)" : "var(--text)",
                  borderColor: isCurrent ? "var(--accent)" : undefined,
                }}
              >
                {w}
              </button>
            );
          })}
          {running && phase === 4 && (
            <span
              className="anim-blink"
              style={{
                display: "inline-block",
                width: 2,
                height: 20,
                background: "var(--accent)",
                borderRadius: 1,
                alignSelf: "center",
              }}
            />
          )}
        </div>
      </NeuPanel>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
//  Root component
// ─────────────────────────────────────────────────────────────────────────────

export default function Home() {
  const [inputText, setInputText] = useState(SAMPLE_TEXT);
  const [telemetry, setTelemetry] = useState<TelemetryResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [backendOn, setBackendOn] = useState(false);
  const [mounted,   setMounted]   = useState(false);
  const [theme,     setTheme]     = useState<"light" | "dark">("light");
  const [speed,     setSpeed]     = useState<number>(1);

  const [running,   setRunning]   = useState(false);
  const [phase,     setPhase]     = useState(0);

  const [visible,   setVisible]   = useState(0);
  const [selTok,    setSelTok]    = useState(0);
  const [scanCell,  setScanCell]  = useState<{ r: number; c: number } | null>(null);
  const [selHead,   setSelHead]   = useState<number | "avg">("avg");
  const [streaming, setStreaming] = useState<string[]>([]);
  const [genStep,   setGenStep]   = useState(0);

  const abort = useRef(false);
  const speedRef = useRef(1);

  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  // ── Sync Theme ─────────────────────────────────────────────────────────────
  useEffect(() => {
    setMounted(true);
    const savedTheme = localStorage.getItem("transformer_theme") as "light" | "dark" | null;
    if (savedTheme === "dark" || savedTheme === "light") {
      setTheme(savedTheme);
    } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
      setTheme("dark");
    }
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next = prev === "light" ? "dark" : "light";
      localStorage.setItem("transformer_theme", next);
      return next;
    });
  }, []);

  // ── Data ──────────────────────────────────────────────────────────────────
  const load = useCallback(async (text: string) => {
    if (!text.trim()) return null;
    setIsLoading(true);
    try {
      const d = await fetchTelemetry(text);
      setTelemetry(d);
      setBackendOn(true);
      return d;
    } catch {
      const d = generateLocalFallback(text);
      setTelemetry(d);
      setBackendOn(false);
      return d;
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    load(inputText);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Reset ─────────────────────────────────────────────────────────────────
  const reset = useCallback(() => {
    abort.current = true;
    setRunning(false);
    setPhase(0);
    setVisible(0);
    setSelTok(0);
    setScanCell(null);
    setSelHead("avg");
    setStreaming([]);
    setGenStep(0);
    setInputText("");
    setTelemetry(null);
    abort.current = false;
  }, []);

  // ── Load Sample ───────────────────────────────────────────────────────────
  const loadSample = useCallback(() => {
    abort.current = true;
    setRunning(false);
    setPhase(0);
    setVisible(0);
    setSelTok(0);
    setScanCell(null);
    setSelHead("avg");
    setStreaming([]);
    setGenStep(0);
    setInputText(SAMPLE_TEXT);
    setTimeout(() => {
      abort.current = false;
      load(SAMPLE_TEXT);
    }, 50);
  }, [load]);

  // ── Run Full Inference ────────────────────────────────────────────────────
  const run = useCallback(async () => {
    if (running) return;

    let textToRun = inputText.trim();
    if (!textToRun) {
      textToRun = SAMPLE_TEXT;
      setInputText(SAMPLE_TEXT);
    }

    abort.current = false;
    setRunning(true);
    setVisible(0); setScanCell(null); setStreaming([]); setGenStep(0); setSelTok(0);

    const data = await load(textToRun);
    if (!data || abort.current) { setRunning(false); return; }

    const delay = (ms: number) => sleep(Math.round(ms / (speedRef.current || 1)));
    const N = data.tokens.length;

    // Phase 0 — Tokenize
    setPhase(0);
    for (let i = 1; i <= N; i++) {
      if (abort.current) break;
      setVisible(i);
      await delay(24);
    }
    await delay(350);

    // Phase 1 — Embed + PE
    setPhase(1);
    for (let p = 0; p < Math.min(N, 10); p++) {
      if (abort.current) break;
      setSelTok(p);
      await delay(110);
    }
    await delay(350);

    // Phase 2 — Encoder MHA scan
    setPhase(2);
    const K = Math.min(N, 14);
    for (let r = 0; r < K; r++) {
      for (let c = 0; c < K; c++) {
        if (abort.current) break;
        setScanCell({ r, c });
        await delay(11);
      }
    }
    setScanCell(null);
    await delay(350);

    // Phase 3 — Decoder
    setPhase(3);
    await delay(900);

    // Phase 4 — Synthesis: stream words
    setPhase(4);
    const finalWords = data.final_output_text?.trim().split(/\s+/).filter(Boolean) ?? [];
    const seqLen     = data.generated_sequence.length;
    const words      = finalWords.length > 0 ? finalWords : data.generated_sequence.map((s) => s.word);

    for (let s = 0; s < seqLen; s++) {
      if (abort.current) break;
      setGenStep(s);
      setStreaming((prev) => [...prev, words[s] ?? data.generated_sequence[s].word]);
      await delay(165);
    }

    setRunning(false);
  }, [running, inputText, load]);

  const summaryText    = telemetry?.final_output_text ?? null;
  const showStreaming  = streaming.length > 0 || (running && phase === 4);
  const wordCount      = inputText.trim().split(/\s+/).filter(Boolean).length;
  const isDark         = theme === "dark";

  return (
    <div
      data-theme={theme}
      style={{
        height: "100vh",
        width: "100vw",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        background: "var(--bg)",
        color: "var(--text)",
        fontFamily: "var(--sans)",
        transition: "background-color 0.25s ease, color 0.25s ease",
      }}
    >
      {/* ═══════════════════════════════════════════════════════════════════
          NEUMORPHIC HEADER
      ═══════════════════════════════════════════════════════════════════ */}
      <header
        className="neu-flat"
        style={{
          height: 48,
          margin: "8px 12px 4px 12px",
          padding: "0 16px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexShrink: 0,
          borderRadius: 10,
        }}
      >
        {/* Left brand & status */}
        <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div
              className="neu-btn"
              style={{
                width: 24,
                height: 24,
                borderRadius: 6,
                background: "var(--accent)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                border: "none",
              }}
            >
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <rect x="1" y="1" width="4" height="4" rx="1" fill="white" opacity="0.95" />
                <rect x="7" y="1" width="4" height="4" rx="1" fill="white" opacity="0.65" />
                <rect x="1" y="7" width="4" height="4" rx="1" fill="white" opacity="0.65" />
                <rect x="7" y="7" width="4" height="4" rx="1" fill="white" opacity="0.95" />
              </svg>
            </div>
            <T size={13.5} weight={700}>Transformer Telemetry</T>
          </div>

          {/* Engine indicator */}
          <div style={{ display: "flex", alignItems: "center", gap: 6, paddingLeft: 14, borderLeft: "1px solid var(--border)" }}>
            <div
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: !mounted ? "var(--text-3)" : backendOn ? "var(--teal)" : "#d97706",
                flexShrink: 0,
              }}
            />
            <T size={12} weight={600} mono color={!mounted ? "text-3" : backendOn ? "teal" : "text-2"}>
              {!mounted ? "Connecting…" : backendOn ? "Falconsai T5 · Seq2Seq" : "Simulation Fallback"}
            </T>
          </div>

          {telemetry && (
            <div style={{ display: "flex", gap: 12, paddingLeft: 14, borderLeft: "1px solid var(--border)" }}>
              {[`d_model = ${telemetry.d_model}`, `heads = ${telemetry.num_heads}`, "vocab = 32,128"].map((s) => (
                <T key={s} size={11} weight={600} mono color="text-3">{s}</T>
              ))}
            </div>
          )}
        </div>

        {/* Right actions */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {/* Neumorphic Speed Selector */}
          <div
            className="neu-inset"
            style={{
              display: "flex",
              alignItems: "center",
              padding: "2px",
              gap: 2,
            }}
          >
            <span style={{ fontSize: 9.5, fontWeight: 700, color: "var(--text-3)", padding: "0 5px" }}>SPEED</span>
            {[0.5, 1, 1.5, 2].map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`neu-btn ${speed === s ? "neu-btn-active" : ""}`}
                style={{
                  padding: "3px 7px",
                  borderRadius: 5,
                  fontSize: 11,
                  fontFamily: "var(--mono)",
                  fontWeight: speed === s ? 700 : 500,
                  color: speed === s ? "var(--accent)" : "var(--text-2)",
                }}
              >
                {s}x
              </button>
            ))}
          </div>

          {/* Theme Toggle */}
          <NeuButton
            onClick={toggleTheme}
            ariaLabel={`Switch to ${isDark ? "light" : "dark"} mode`}
            style={{ padding: "5px 10px", fontSize: 12, fontWeight: 600 }}
          >
            {isDark ? (
              <>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="5" />
                  <line x1="12" y1="1" x2="12" y2="3" />
                  <line x1="12" y1="21" x2="12" y2="23" />
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                  <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                  <line x1="1" y1="12" x2="3" y2="12" />
                  <line x1="21" y1="12" x2="23" y2="12" />
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                  <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                </svg>
                <span>Light</span>
              </>
            ) : (
              <>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                </svg>
                <span>Dark</span>
              </>
            )}
          </NeuButton>

          {/* Load Sample Button */}
          <NeuButton onClick={loadSample} style={{ padding: "5px 12px", fontSize: 12.5 }}>
            Sample
          </NeuButton>

          {/* Reset / Clear Button */}
          <NeuButton onClick={reset} style={{ padding: "5px 12px", fontSize: 12.5 }}>
            Reset
          </NeuButton>

          {/* Run Button */}
          <button
            onClick={run}
            disabled={running || isLoading}
            className="neu-btn"
            style={{
              display: "flex",
              alignItems: "center",
              gap: 7,
              background: "var(--accent)",
              color: "#ffffff",
              fontWeight: 700,
              fontSize: 12.5,
              padding: "6px 16px",
              border: "none",
              borderRadius: 7,
              cursor: running || isLoading ? "not-allowed" : "pointer",
              opacity: running || isLoading ? 0.75 : 1,
            }}
          >
            {running ? (
              <>
                <div className="anim-spin" style={{ width: 10, height: 10, borderRadius: "50%", border: "2px solid rgba(255,255,255,0.3)", borderTopColor: "white" }} />
                Synthesizing ({speed}x)…
              </>
            ) : (
              <>
                <span>▶</span>
                Run Full Transformer
              </>
            )}
          </button>
        </div>
      </header>

      {/* ═══════════════════════════════════════════════════════════════════
          PIPELINE TRANSITION NAVIGATOR (With Clickable Step Advance Arrows)
      ═══════════════════════════════════════════════════════════════════ */}
      <nav
        className="neu-flat"
        aria-label="Transformer pipeline phases"
        style={{
          height: 52,
          margin: "4px 12px",
          padding: "0 12px",
          display: "flex",
          alignItems: "center",
          gap: 6,
          flexShrink: 0,
          borderRadius: 10,
        }}
      >
        {PHASES.map((p, i) => {
          const isActive = phase === i;
          const isDone   = phase > i;

          return (
            <React.Fragment key={p.id}>
              <button
                onClick={() => { if (!running) setPhase(i); }}
                className={`neu-btn ${isActive ? "neu-btn-active" : ""}`}
                style={{
                  flex: 1,
                  minWidth: 0,
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "7px 12px",
                  borderRadius: 8,
                  borderColor: isActive ? "var(--accent)" : isDone ? "var(--teal)" : undefined,
                  textAlign: "left",
                }}
              >
                <span
                  style={{
                    fontSize: 10.5,
                    fontFamily: "var(--mono)",
                    fontWeight: 800,
                    color: isActive ? "var(--accent)" : isDone ? "var(--teal)" : "var(--text-3)",
                    letterSpacing: "0.04em",
                    flexShrink: 0,
                  }}
                >
                  0{i + 1}
                </span>
                <div style={{ minWidth: 0 }}>
                  <div
                    style={{
                      fontSize: 12,
                      fontWeight: 700,
                      color: isActive ? "var(--accent)" : isDone ? "var(--teal)" : "var(--text)",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      lineHeight: 1.1,
                    }}
                  >
                    {p.label}
                  </div>
                  <div
                    style={{
                      fontSize: 10,
                      fontWeight: 500,
                      color: "var(--text-3)",
                      marginTop: 2,
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {p.sub}
                  </div>
                </div>
              </button>

              {/* Clickable transition connector arrow: advances to next phase */}
              {i < PHASES.length - 1 && (
                <button
                  onClick={() => { if (!running) setPhase(i + 1); }}
                  title={`Transition to Phase 0${i + 2}: ${PHASES[i + 1].label}`}
                  className="neu-btn"
                  style={{
                    width: 24,
                    height: 24,
                    borderRadius: "50%",
                    fontSize: 11,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                    color: isDone ? "var(--teal)" : "var(--text-3)",
                    padding: 0,
                  }}
                >
                  →
                </button>
              )}
            </React.Fragment>
          );
        })}
      </nav>

      {/* ═══════════════════════════════════════════════════════════════════
          MAIN WORKSPACE
      ═══════════════════════════════════════════════════════════════════ */}
      <main
        style={{
          flex: 1,
          display: "grid",
          gridTemplateRows: "186px 1fr",
          overflow: "hidden",
          minHeight: 0,
          padding: "4px 12px 12px 12px",
          gap: 10,
        }}
      >
        {/* ── ROW 1: Source + Summary (Neumorphic Inset Wells) ─────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, minHeight: 0 }}>
          {/* Source Document */}
          <NeuPanel style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexShrink: 0 }}>
              <SectionLabel>Source Document</SectionLabel>
              <T size={11} weight={600} mono color="text-3">
                {wordCount} words{telemetry ? ` · ${telemetry.tokens.length} tokens` : ""}
              </T>
            </div>
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              disabled={running}
              aria-label="Source document input"
              className="neu-inset"
              style={{
                flex: 1,
                padding: "10px 12px",
                fontSize: 13,
                fontWeight: 500,
                color: "var(--text)",
                lineHeight: 1.65,
                resize: "none",
                outline: "none",
                fontFamily: "var(--sans)",
              }}
              placeholder="Paste or type any document here, or click 'Sample' to load the demo text..."
            />
          </NeuPanel>

          {/* Abstractive Summary */}
          <NeuPanel style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexShrink: 0 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                <div style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--teal)" }} />
                <SectionLabel>Abstractive Summary Output</SectionLabel>
              </div>
              <T size={11} weight={600} mono color="text-3">
                {showStreaming
                  ? `${streaming.length} tokens`
                  : summaryText
                  ? `${summaryText.trim().split(/\s+/).filter(Boolean).length} words · ${telemetry?.generated_sequence.length ?? 0} tokens`
                  : "Awaiting run"}
              </T>
            </div>

            <div className="neu-inset" style={{ flex: 1, padding: "10px 14px", overflowY: "auto" }}>
              {!showStreaming && summaryText && (
                <div className="anim-fadein">
                  <p style={{ fontSize: 13.5, fontWeight: 500, color: "var(--text)", lineHeight: 1.75, margin: 0 }}>
                    {summaryText}
                  </p>
                  <p style={{ fontSize: 11, fontWeight: 600, color: "var(--text-3)", fontFamily: "var(--mono)", marginTop: 8, marginBottom: 0 }}>
                    T5 decoded output · Click "Run Full Transformer" to replay step-by-step
                  </p>
                </div>
              )}

              {/* Streaming tokens — each token is clickable and jumps to Phase 5 synthesis */}
              {showStreaming && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 5, alignContent: "flex-start" }}>
                  {streaming.map((w, i) => {
                    const isCurrent = i === streaming.length - 1 && running;
                    return (
                      <button
                        key={i}
                        onClick={() => {
                          setPhase(4);
                          setGenStep(i);
                        }}
                        className={`neu-btn anim-token ${isCurrent ? "neu-btn-active" : ""}`}
                        title={`Click to inspect Step #${i + 1} synthesis`}
                        style={{
                          animationDelay: `${Math.min(i * 12, 180)}ms`,
                          padding: "3px 8px",
                          borderRadius: 5,
                          fontSize: 12.5,
                          fontFamily: "var(--mono)",
                          fontWeight: isCurrent ? 700 : 500,
                          color: isCurrent ? "var(--accent)" : "var(--text)",
                          borderColor: isCurrent ? "var(--accent)" : undefined,
                        }}
                      >
                        {w}
                      </button>
                    );
                  })}
                  {running && phase === 4 && (
                    <span className="anim-blink" style={{ display: "inline-block", width: 2, height: 18, background: "var(--accent)", alignSelf: "center" }} />
                  )}
                </div>
              )}

              {!summaryText && !showStreaming && (
                <div style={{ display: "flex", alignItems: "center", height: "100%" }}>
                  <T size={13} weight={500} color="text-3" as="p" style={{ margin: 0 }}>
                    Click "Run Full Transformer" or "Sample" to synthesize a summary.
                  </T>
                </div>
              )}
            </div>
          </NeuPanel>
        </div>

        {/* ── ROW 2: Three-Column Inspection Studio ─────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "220px 1fr 260px", gap: 10, overflow: "hidden", minHeight: 0 }}>
          {/* ── COL A: Token List ───────────────────────────────────────── */}
          <NeuPanel style={{ display: "flex", flexDirection: "column", gap: 10, overflow: "hidden" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexShrink: 0 }}>
              <SectionLabel>Token Stream</SectionLabel>
              <T size={10.5} weight={700} mono color="text-3">d={telemetry?.d_model ?? 512}</T>
            </div>

            {telemetry && telemetry.tokens.length > 0 ? (
              <div className="scroll-y" style={{ flex: 1, display: "flex", flexDirection: "column", gap: 3 }}>
                {telemetry.tokens.map((t, i) => {
                  const isVis = visible === 0 || i < visible;
                  const isSel = selTok === i;
                  return (
                    <button
                      key={i}
                      onClick={() => {
                        setSelTok(i);
                        // If in Phase 1 or Phase 2, selecting automatically highlights it
                      }}
                      className={`neu-btn ${isSel ? "neu-btn-active" : ""}`}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        padding: "6px 10px",
                        opacity: isVis ? 1 : 0.45,
                        borderColor: isSel ? "var(--accent)" : undefined,
                        flexShrink: 0,
                        textAlign: "left",
                      }}
                    >
                      <div style={{ display: "flex", gap: 8, alignItems: "center", minWidth: 0 }}>
                        <T size={10} weight={600} mono color="text-3" style={{ flexShrink: 0 }}>{String(i).padStart(2, "0")}</T>
                        <T size={12} weight={isSel ? 700 : 500} mono color={isSel ? "accent" : "text"}
                          style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {t.word}
                        </T>
                      </div>
                      <T size={10} weight={600} mono color={isSel ? "accent" : "text-3"} style={{ flexShrink: 0 }}>{t.id}</T>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", textAlign: "center", padding: 12 }}>
                <T size={12} weight={500} color="text-3">No active tokens.<br />Enter text to inspect.</T>
              </div>
            )}
          </NeuPanel>

          {/* ── COL B: Phase Visualization Studio ───────────────────────── */}
          <NeuPanel style={{ display: "flex", flexDirection: "column", gap: 10, overflow: "hidden" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexShrink: 0 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent)", flexShrink: 0 }} />
                <SectionLabel>
                  {[
                    "Phase 1 — SentencePiece BPE Tokenization",
                    "Phase 2 — Learned Embedding + Vaswani Sinusoidal PE",
                    "Phase 3 — Encoder Multi-Head Self-Attention (Layer 0)",
                    "Phase 4 — Decoder (Causal Mask + Cross-Attention Bridge)",
                    "Phase 5 — Softmax Projection & Autoregressive Synthesis",
                  ][phase]}
                </SectionLabel>
              </div>

              {isLoading && (
                <div className="anim-spin" style={{ width: 16, height: 16, borderRadius: "50%", border: "2px solid var(--border)", borderTopColor: "var(--accent)", flexShrink: 0 }} />
              )}
            </div>

            <div style={{ flex: 1, overflow: "hidden", minHeight: 0 }}>
              {telemetry ? (
                <>
                  {phase === 0 && (
                    <Phase1Tokenizer
                      telemetry={telemetry}
                      visible={visible}
                      sel={selTok}
                      onSel={setSelTok}
                      onJumpToEmbed={() => setPhase(1)}
                    />
                  )}
                  {phase === 1 && (
                    <Phase2Embed telemetry={telemetry} sel={selTok} onSel={setSelTok} />
                  )}
                  {phase === 2 && (
                    <Phase3Encoder telemetry={telemetry} scan={scanCell} head={selHead} onHead={setSelHead} isDark={isDark} />
                  )}
                  {phase === 3 && (
                    <Phase4Decoder telemetry={telemetry} isDark={isDark} />
                  )}
                  {phase === 4 && (
                    <Phase5Synthesis
                      telemetry={telemetry}
                      step={genStep}
                      streaming={streaming}
                      running={running}
                      phase={phase}
                      onSelectStep={(s) => setGenStep(s)}
                    />
                  )}
                </>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: 14, textAlign: "center", padding: 24 }}>
                  <div className="neu-btn" style={{ width: 48, height: 48, borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="2" y="2" width="20" height="8" rx="2" />
                      <rect x="2" y="14" width="20" height="8" rx="2" />
                      <line x1="6" y1="6" x2="6.01" y2="6" />
                      <line x1="6" y1="18" x2="6.01" y2="18" />
                    </svg>
                  </div>
                  <div>
                    <T size={16} weight={700} as="div">Transformer Based Text Summarizer</T>
                    <T size={13} weight={500} color="text-2" as="div" style={{ maxWidth: 440, marginTop: 6, lineHeight: 1.6 }}>
                      Workspace is ready. Type custom text above or click <strong>Load Sample</strong> to inspect tokenization, positional frequencies, attention tensors, and softmax projections.
                    </T>
                  </div>
                  <NeuButton onClick={loadSample} style={{ padding: "8px 18px", color: "var(--accent)", fontWeight: 700 }}>
                    Load Sample Paragraph
                  </NeuButton>
                </div>
              )}
            </div>
          </NeuPanel>

          {/* ── COL C: Softmax Probability Distribution ─────────────────── */}
          <NeuPanel style={{ display: "flex", flexDirection: "column", gap: 10, overflow: "hidden" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexShrink: 0 }}>
              <SectionLabel>Softmax Distribution</SectionLabel>
              <T size={11} weight={700} mono color="text-3">Step #{genStep + 1}</T>
            </div>

            {telemetry?.generated_sequence[genStep] ? (
              <div className="scroll-y" style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
                {telemetry.generated_sequence[genStep].candidates.map((c, i) => {
                  const isWin  = i === 0;
                  const maxP   = telemetry.generated_sequence[genStep].candidates[0].prob;
                  const pct    = clamp((c.prob / (maxP || 1)) * 100, 2, 100);
                  return (
                    <div key={i} className="neu-flat" style={{ padding: "8px 10px", borderRadius: 8 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                        <div style={{ display: "flex", gap: 7, alignItems: "baseline", minWidth: 0 }}>
                          <T size={11} weight={700} mono color={isWin ? "accent" : "text-3"}>#{i + 1}</T>
                          <T size={12.5} weight={isWin ? 700 : 500} mono color={isWin ? "accent" : "text"} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            &ldquo;{c.word}&rdquo;
                          </T>
                          <T size={10} weight={600} mono color="text-3" style={{ flexShrink: 0 }}>{c.id}</T>
                        </div>
                        <T size={12} weight={isWin ? 700 : 600} mono color={isWin ? "accent" : "text"} style={{ flexShrink: 0, marginLeft: 4 }}>
                          {(c.prob * 100).toFixed(1)}%
                        </T>
                      </div>
                      <div style={{ height: 6, background: "var(--border)", borderRadius: 3, overflow: "hidden" }}>
                        <div
                          className="anim-bar"
                          style={{
                            height: "100%",
                            width: `${pct.toFixed(1)}%`,
                            background: isWin ? "var(--accent)" : "var(--teal)",
                            borderRadius: 3,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", textAlign: "center" }}>
                <T size={12} weight={500} color="text-3" as="p">
                  Run the transformer<br />to watch step-by-step<br />candidate selection
                </T>
              </div>
            )}

            <div style={{ borderTop: "1px solid var(--border)", paddingTop: 10, flexShrink: 0, display: "flex", flexDirection: "column", gap: 4 }}>
              <T size={11} weight={600} mono color="text-3">Vocab: 32,128 · Beam Search</T>
              {telemetry?.generated_sequence[genStep] && (
                <T size={11} weight={600} mono color="text-2">
                  Selected:{" "}
                  <span style={{ color: "var(--teal)", fontWeight: 700 }}>
                    &ldquo;{telemetry.generated_sequence[genStep].word}&rdquo;
                  </span>
                  {" "}({(telemetry.generated_sequence[genStep].prob * 100).toFixed(2)}%)
                </T>
              )}
            </div>
          </NeuPanel>
        </div>
      </main>
    </div>
  );
}
