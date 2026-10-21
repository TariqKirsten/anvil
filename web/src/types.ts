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
  // Decision data: the questions in train, which is what a training step's batch counts (rows can hold several).
  items?: { train: number };
  // Decision data: each question the validation set asks, its type and how many rows ask it.
  questions?: Record<string, { type: QuestionType; rows: number }>;
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

export type QuestionType = "noul" | "choice" | "score";

// One question's scores at a check (server/anvil/tasks.py, decision_metrics). answers: how often each answer was the
// gold one ("yes" and "no", a choice's keys, a score's levels). Runs from before these were kept have AUROC, catch
// rate and false alarms only.
export interface QuestionScore {
  type: QuestionType;
  rows?: number;
  answers?: Record<string, number>;
  auroc?: number;
  catch_rate?: number;
  false_alarms?: number;
  accuracy?: number;
  macro_f1?: number;
  mae?: number;
}

// A check's scores: numbers by name, and each question's own under "questions".
export type Scores = { [key: string]: number | string | Record<string, QuestionScore> | undefined } & {
  questions?: Record<string, QuestionScore>;
};

export type RunState = "starting" | "loading" | "running" | "saving" | "done" | "failed" | "stopped";

export interface RunSummary {
  id: string;
  name: string;
  train_as: "lm" | "decision";
  method: string;
  model: string;
  gpu: number;
  gpus: number[];
  created: number;
  state: RunState;
  stopping: boolean; // Stop was pressed and the trainer hasn't stopped yet
  stage: string;
  step: number;
  total: number | null;
  updated: number | null;
  last_eval: Scores | null;
  error: string | null;
  best: Leaderboard["entries"][number] | null;
  size: number; // bytes on disk
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
  updated?: number;
  error?: string | null;
  items?: number;
  world?: number;
  steps_per_epoch?: number;
  items_per_step?: number;
  head_m?: number; // Clef's head, in millions of parameters
  head_first?: number; // steps the new head trains alone
  trainable_m?: number;
  last_eval?: Scores;
  best_step?: number | null;
  early_stop?: string | null;
  final_test?: (Scores & { step: number; rows?: number }) | null;
  training_finished?: boolean;
  warm_until?: number; // the model has warmed up by this step: early stopping counts the checks after it
  crashes?: number; // times the trainer crashed in native code and started again (anvil.supervisor)
  last_crash?: { at: number; step: number; from: number | null } | null;
  saving_since?: number | null; // the trainer is saving a checkpoint (or the model): Stop never cuts that off
  // While Stop is under way: when it was pressed, and when the trainer is ended outright if it hasn't stopped by itself
  // (ending: that time has come; Stop now brings it forward).
  stopping?: { requested: number; end_by: number; ending: boolean };
  // Stopped without saving: the trainer was ended (or died) after Stop before it got to its checkpoint. from: where
  // Resume picks up (null: the beginning).
  ended?: { at: number; step: number; from: number | null } | null;
}

export type RunMetric = Scores & {
  type: "train" | "eval" | "resume" | "test" | "early_stop";
  step: number;
};

export interface RunDetail {
  config: {
    name: string;
    train_as: "lm" | "decision";
    method: string;
    training_type: string;
    model: { id: string; params: number | null };
    data: { format: string; rows: Record<string, number> };
    gpu: number;
    gpus?: number[];
    judge_gpu?: number | null;
    hyper: Record<string, number | string | boolean>;
    checkpoints: {
      every: number;
      keepBest: number;
      patience: number;
      pickBy: string;
      falseAlarmPercent?: number;
      countFrom?: number; // where Keep training picked up: early stopping counts the checks after it
    };
    created: number;
  };
  status: RunStatus;
  metrics: RunMetric[];
  log: string;
  checkpoints: string[];
  model: string | null;
  leaderboard: Leaderboard;
  judge: JudgeStatus | null;
  size: number; // bytes on disk: checkpoints, model, metrics and logs
  // What can be exported as GGUF or uploaded (the finished model, kept checkpoints), and the exports so far. sizes:
  // the weights' bytes, whole (merged) and LoRA's adapter alone, where known.
  versions: {
    version: string;
    step: number | null;
    score: number | null;
    metric: string | null;
    sizes?: { merged: number | null; adapter: number | null };
  }[];
  exports: ExportStatus[];
  export_params: number | null; // weights in the finished model (and Clef's head), for size estimates
  uploads?: UploadStatus[]; // uploads to Hugging Face, newest first
}

// server/anvil/upload.py: one upload of a version, to one repo or two (the weights', and the GGUF files').
export interface UploadStatus {
  id: string;
  version: string;
  weights: "merged" | "adapter" | null;
  gguf: string[];
  repo?: string | null;
  gguf_repo?: string | null;
  private: boolean;
  state: "starting" | "running" | "done" | "failed" | "cancelled";
  stage: string;
  bytes_total?: number;
  bytes_done?: number;
  repos?: {
    id: string;
    url: string;
    state: "waiting" | "running" | "done" | "failed";
    created?: boolean; // made by this upload
    made?: "public" | "private" | null; // an existing repo whose visibility this upload changed
    card?: "written" | "kept"; // kept: someone had edited it on Hugging Face
  }[];
  error?: string | null;
}

// Whose the Hugging Face token is, and the namespaces it can upload to (none for a read token).
export interface HubMe {
  connected: boolean;
  name: string | null;
  namespaces: string[];
  write: boolean;
}

export interface ExportStatus {
  version: string;
  state: "starting" | "running" | "done" | "failed";
  stage: string;
  quants?: string[];
  files?: { name: string; size: number }[];
  error?: string | null;
}

// server/anvil/picking.py: one entry per validation check; best_step is the checkpoint that becomes the model.
export interface Leaderboard {
  entries: { step: number; score: number | null; metric: string; direction: "max" | "min" }[];
  best_step: number | null;
}

export interface JudgeStatus {
  state: "starting" | "loading" | "checking" | "waiting" | "done" | "failed";
  stage: string;
  gpu: number | null;
  step?: number | null;
  error?: string | null;
  updated?: number;
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
