// The shapes the server sends back (see server/anvil).

export interface Session {
  logged_in: boolean;
  username: string | null;
  has_accounts: boolean;
}

export interface HfStatus {
  connected: boolean;
  user: string | null;
  source: "account" | "env" | null;
}

export interface Account {
  username: string;
  hf: HfStatus;
}

export type Status = "ok" | "info" | "warn" | "bad";

export interface CheckItem {
  status: Status;
  label: string;
  detail: string;
}

export interface Report {
  verdict: "ok" | "warn" | "bad";
  message: string;
  kind: "model" | "decision" | "adapter" | "unknown";
  architecture: string | null;
  params: number | null;
  size: number;
  base_model: string | null;
  items: CheckItem[];
}

export interface HubModel {
  id: string;
  downloads: number;
  likes: number;
  licence: string | null;
  permissive: boolean;
  gated: boolean;
  params: number | null;
  pipeline: string | null;
  custom_code: boolean;
  gguf: boolean;
}

export interface Gpu {
  index: number;
  name: string;
  arch: string | null;
  pci: string | null;
  total_gb: number;
  used_gb: number | null;
  display: boolean;
  busy: boolean;
}

export interface GpuList {
  runtime: { kind: "rocm" | "cuda" | "none"; label: string; error: string | null };
  gpus: Gpu[];
}

export interface FolderList {
  path: string | null;
  parent: string | null;
  dirs: string[];
  files?: { name: string; size: number }[];
  roots: string[];
}

// ---- datasets

export type Slot = "train" | "validation" | "test";

export type DataSource =
  | { kind: "file"; path: string; name: string }
  | { kind: "hub"; repo: string; config: string | null; split: string; name: string };

export type DataFormat = "decision" | "chat" | "pairs" | "text";

export interface Problem {
  row: number;
  reason: string;
}

export interface DatasetInfo {
  rows: number;
  columns: string[];
  format: DataFormat | null;
  format_label: string | null;
  preview: Record<string, string>[];
  checked: number;
  invalid: number | null;
  examples: Problem[];
}

export interface PrepareReport {
  format: DataFormat;
  format_label: string;
  rows: Record<Slot, number>;
  invalid: Partial<Record<Slot, number>>;
  examples: Partial<Record<Slot, Problem[]>>;
  removed: Partial<Record<Slot, { exact: number; near: number; conflicts: number }>>;
  leaks: Partial<Record<Slot, number>>;
  given: { validation: boolean; test: boolean };
  notes: string[];
  seconds: number;
}

export interface PrepareJob {
  id: string;
  fingerprint: string;
  state: "running" | "done" | "failed";
  stage: string;
  progress: number;
  report: PrepareReport | null;
  error: string | null;
}

// ---- runs

export type RunState = "starting" | "loading" | "running" | "saving" | "done" | "failed" | "stopped";

export interface RunSummary {
  id: string;
  name: string;
  train_as: "lm" | "decision";
  method: string;
  model: string;
  gpu: number;
  created: number;
  state: RunState;
  stage: string;
  step: number;
  total: number | null;
  updated: number | null;
  last_eval: Record<string, number> | null;
  error: string | null;
}

export interface RunStatus {
  state: RunState;
  stage: string;
  step: number;
  total?: number;
  epoch?: number;
  loss?: number;
  tokens_per_s?: number;
  eta_s?: number | null;
  peak_mem_gb?: number | null;
  skipped?: number;
  started?: number;
  finished?: number;
  stopped?: number | null;
  error?: string | null;
  items?: number;
  trainable_m?: number;
  last_eval?: Record<string, number>;
}

export interface RunMetric {
  type: "train" | "eval" | "resume";
  step: number;
  [key: string]: number | string;
}

export interface RunDetail {
  config: {
    name: string;
    train_as: "lm" | "decision";
    method: string;
    training_type: string;
    model: { id: string; params: number | null };
    data: { format: string; rows: Record<string, number> };
    gpu: number;
    hyper: Record<string, number | string | boolean>;
    checkpoints: { every: number; pickBy: string };
    created: number;
  };
  status: RunStatus;
  metrics: RunMetric[];
  log: string;
  checkpoints: string[];
  model: string | null;
}

export interface HubDataset {
  id: string;
  downloads: number;
  licence: string | null;
  permissive: boolean;
  gated: boolean;
  private: boolean;
  size: string | null;
}
