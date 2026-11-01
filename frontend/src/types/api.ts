export interface TokenItem {
  word: string;
  id: number;
  pos: number;
}

export interface CandidateToken {
  word: string;
  id: number;
  prob: number;
}

export interface GeneratedTokenStep {
  step: number;
  word: string;
  id: number;
  prob: number;
  candidates: CandidateToken[];
}

export interface TelemetryResponse {
  tokens: TokenItem[];
  d_model: number;
  num_heads: number;
  embedding_slices: number[][];
  pe_slices: number[][];
  heads_attention: number[][][];
  avg_attention: number[][];
  generated_sequence: GeneratedTokenStep[];
  final_output_text: string;
}

export interface InspectRequest {
  text: string;
}
