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
  roots: string[];
}
