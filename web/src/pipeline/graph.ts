// A pipeline as the server keeps it (anvil/pipelines.py), and what the canvas works out from it: the columns a step
// can use, the Model it asks, what may connect to what, and where steps go when the graph is tidied. The server checks
// the whole graph whenever it's saved; this only mirrors what the screen needs to offer the right choices.

export type StepType = "model" | "seed" | "list" | "generate" | "question" | "filter" | "dedupe" | "save";
export type EdgeKind = "rows" | "model" | "check";

export interface ModelConfig {
  name: string;
  base_url: string;
  model: string;
  temperature: number | null;
  max_tokens: number | null;
  thinking: "default" | "on" | "off";
  parallel: number | null;
  timeout: number | null;
  extra: Record<string, unknown>;
}

export interface SeedConfig {
  source: ({ kind: "file"; path: string; name?: string } | { kind: "hub"; repo: string; config: string | null;
    split: string; name?: string }) | null;
  columns: string[];
  order: "shuffled" | "in_order";
  limit: number | null;
}

export interface ListConfig {
  column: string;
  values: string[];
  pick: "random" | "turns";
}

export interface GenerateConfig {
  column: string;
  prompt: string;
  system: string;
  per_reply: number | null;
  temperature: number | null;
}

export type QuestionType = "noul" | "choice" | "score";

export interface QuestionConfig {
  name: string;
  type: QuestionType;
  instructions: string;
  yes: string;
  no: string;
  options: { name: string; meaning: string }[];
  levels: string[];
  about: string;
  context: string;
  answer: "model" | "column";
  from_column: string;
  mapping: Record<string, string>;
  sure_only: boolean;
  on_disagree: "leave_out" | "drop";
}

export type FilterOp = "is" | "is_not" | "contains" | "not_contains" | "matches" | "longer" | "shorter" | "empty"
  | "not_empty";

export interface FilterRule {
  column: string;
  op: FilterOp;
  value: string;
}

export interface FilterConfig {
  rules: FilterRule[];
  match: "all" | "any";
}

export interface DedupeConfig {
  column: string;
  near: boolean;
}

export type SaveFormat = "decision" | "chat" | "pairs" | "text";

export interface SaveConfig {
  name: string;
  balance: string;
  rows: number | null;
  format: SaveFormat;
  state_key: string;
  state_column: string;
  system: string;
  user_column: string;
  assistant_column: string;
  prompt_column: string;
  completion_column: string;
  text_column: string;
}

export interface Configs {
  model: ModelConfig;
  seed: SeedConfig;
  list: ListConfig;
  generate: GenerateConfig;
  question: QuestionConfig;
  filter: FilterConfig;
  dedupe: DedupeConfig;
  save: SaveConfig;
}

export interface PipelineNode<T extends StepType = StepType> {
  id: string;
  type: T;
  x: number;
  y: number;
  note?: string; // why the step is there: a template's explanation, or what you wrote
  config: Configs[T];
}

export interface PipelineEdge {
  id: string;
  from: string;
  to: string;
  kind: EdgeKind;
}

export interface PipelineProblem {
  node: string | null;
  field: string | null;
  level: "error" | "warn";
  message: string;
}

export type PipelineRunState = "starting" | "running" | "stopping" | "stopped" | "done" | "failed";

export const RUNNING: PipelineRunState[] = ["starting", "running", "stopping"];

// What one step did in a run: rows in and out, the requests it made, and why rows stopped there.
export interface StepStats {
  in: number;
  done: number; // finished with, passed on or dropped: the rest of "in" were under way, or let go at the end
  out: number;
  dropped: number;
  errors: number;
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  seconds: number;
  cut: number;
  left_out: number;
}

export interface SaveStatus {
  name: string;
  target: number;
  kept: number;
  closed: boolean;
  dataset?: string;
  balance?: string | null;
  quota?: Record<string, number>;
  by_value?: Record<string, number>;
}

export interface PipelineRun {
  id: string;
  state: PipelineRunState;
  stage?: string;
  preview?: boolean;
  started?: number;
  updated?: number;
  ended?: number;
  error?: string | null;
  saves?: Record<string, SaveStatus>;
  steps?: Record<string, StepStats>;
  reasons?: Record<string, Record<string, number>>; // why each step dropped rows, by kind
  blanks?: Record<string, Record<string, number>>; // why each Question left answers out
  rows_started?: number;
  rate?: number | null; // rows kept a second, lately
  eta_s?: number | null;
  resumed?: number;
}

// One row of a preview, or a dropped row of a run: where it ended, why, and (in a preview) what each step did.
export interface TraceStep {
  node: string;
  asked?: { model: string; prompt: string; reply: string; seconds: number }[];
  answer?: unknown;
  left_out?: string;
}

export interface RowEntry {
  id: string;
  outcome: "kept" | "dropped" | "error";
  node: string;
  reason?: string;
  row: Record<string, unknown>;
  record?: Record<string, unknown>;
  trace?: TraceStep[];
}

export interface Pipeline {
  id: string;
  name: string;
  about: string; // what the pipeline is for: a template's guide, or your own notes
  created: number;
  updated: number;
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  keys: string[];
  problems: PipelineProblem[];
  runs: PipelineRun[];
  preview: PipelineRun | null;
  assistant_key_at: string | null; // the assistant's address, when it has a key: Model steps there use it
}

// The pipeline assistant (anvil/assistant.py): a conversation, one turn at a time. It talks a new pipeline through
// before it builds it, and makes the changes it's asked for.
export type TurnState = "starting" | "working" | "waiting" | "done" | "failed" | "stopped";
export const TURN_WORKING: TurnState[] = ["starting", "working", "waiting"];

export interface AssistantTurn {
  turn: number;
  state: TurnState;
  doing?: string | null; // what it's doing now: Thinking, Previewing: 2 of 5 rows…
  error?: string | null;
  // what it waits for: training on the GPUs its server uses, or a run at its address with another model
  waiting_for?: { training?: string; pipeline?: string; name?: string; run?: string; model?: string;
    base_url?: string } | null;
}

export interface AssistantMessage {
  role: "user" | "assistant" | "action" | "note"; // action: a line about a tool it used
  text: string;
  at?: number;
  ok?: boolean;
  tool?: string;
}

export interface Conversation {
  messages: AssistantMessage[];
  turn: AssistantTurn | null;
  undo: number | null; // the newest turn whose changes Undo puts back
  model: string;
  ready: boolean; // the assistant's model is chosen in Settings
}

export interface PipelineSummary {
  id: string;
  name: string;
  updated: number;
  steps: number;
  latest: PipelineRun | null;
}

// A pipeline to start from (anvil/templates.py), as New pipeline lists it.
export interface Template {
  id: string;
  name: string;
  summary: string;
  learn: string[];
  makes: string;
  steps: number;
}

export interface StepDefaults {
  steps: { [T in StepType]: Configs[T] };
  filter_ops: Record<FilterOp, string>;
  leave_out: string;
}

// The steps that ask a model, so a Model can be plugged into them.
export const ASKS_MODEL: StepType[] = ["generate", "question"];

export const QUESTION_TYPES: Record<QuestionType, string> = { noul: "Yes or no", choice: "Choice", score: "Score" };

export const FORMATS: Record<SaveFormat, string> = {
  decision: "Decision",
  chat: "Chat",
  pairs: "Prompt and completion",
  text: "Plain text",
};

// A column name a prompt can use as {{ name }}: "Text Message" becomes text_message (the server's ident).
export function ident(name: string): string {
  const slug = String(name).toLowerCase().replace(/[^a-z0-9_]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 64);
  if (!slug) return "column";
  return /^[0-9]/.test(slug) ? `c_${slug}`.slice(0, 64) : slug;
}

// An endpoint's address the way the server compares them (llm.base_url): no endpoint path, no trailing slash.
export function address(url: string): string {
  let base = url.trim().replace(/\/+$/, "");
  for (const path of ["/chat/completions", "/completions", "/models"]) {
    if (base.endsWith(path)) base = base.slice(0, -path.length).replace(/\/+$/, "");
  }
  return base;
}

// A list step's values: one per line as typed, blank lines dropped.
export function values(list: string[]): string[] {
  return list.map((v) => v.trim()).filter(Boolean);
}

// The columns a step adds to each row.
export function made(node: PipelineNode): string[] {
  const cfg = node.config as unknown as Record<string, unknown>;
  if (node.type === "list" || node.type === "generate") return cfg.column ? [String(cfg.column)] : [];
  if (node.type === "question") return cfg.name ? [String(cfg.name)] : [];
  if (node.type === "seed") return (cfg.columns as string[]).map(ident);
  return [];
}

export class Graph {
  readonly byId: Map<string, PipelineNode>;
  readonly parent = new Map<string, string>();
  readonly children = new Map<string, string[]>();
  readonly plugged = new Map<string, string>(); // step -> the Model plugged into it
  readonly second = new Map<string, string>(); // question -> the Model giving a second opinion

  constructor(readonly nodes: PipelineNode[], readonly edges: PipelineEdge[]) {
    this.byId = new Map(nodes.map((n) => [n.id, n]));
    for (const n of nodes) this.children.set(n.id, []);
    for (const e of edges) {
      if (!this.byId.has(e.from) || !this.byId.has(e.to)) continue;
      if (e.kind === "rows") {
        if (!this.parent.has(e.to)) {
          this.parent.set(e.to, e.from);
          this.children.get(e.from)!.push(e.to);
        }
      } else (e.kind === "model" ? this.plugged : this.second).set(e.to, e.from);
    }
  }

  // The steps from the top of the line down to this one (just this one if its line goes round in a circle).
  path(id: string): string[] {
    const steps = [id];
    const seen = new Set([id]);
    for (let here = this.parent.get(id); here; here = this.parent.get(here)) {
      if (seen.has(here)) return [id];
      seen.add(here);
      steps.push(here);
    }
    return steps.reverse();
  }

  root(id: string): PipelineNode | undefined {
    return this.byId.get(this.path(id)[0]);
  }

  // The columns this step can use: every column the steps above it add, each once, top first.
  columnsAbove(id: string): string[] {
    const found: string[] = [];
    for (const step of this.path(id).slice(0, -1)) {
      for (const column of made(this.byId.get(step)!)) if (!found.includes(column)) found.push(column);
    }
    return found;
  }

  above(id: string, type: StepType): PipelineNode[] {
    return this.path(id).slice(0, -1).map((s) => this.byId.get(s)!).filter((n) => n.type === type);
  }

  // The Model a step asks: the one plugged into it, or the one its line starts from.
  modelFor(id: string): PipelineNode<"model"> | null {
    const plugged = this.plugged.get(id);
    if (plugged) return this.byId.get(plugged) as PipelineNode<"model">;
    const top = this.root(id);
    return top && top.type === "model" ? (top as PipelineNode<"model">) : null;
  }

  // The Model whose saved key this Model uses: its own, else another's at the same address (the server does the same).
  keyFrom(id: string, keys: Set<string>): PipelineNode<"model"> | null {
    const model = this.byId.get(id) as PipelineNode<"model"> | undefined;
    if (!model) return null;
    if (keys.has(id)) return model;
    const here = address(model.config.base_url);
    return (this.nodes.find((n) => n.type === "model" && keys.has(n.id)
      && address((n.config as ModelConfig).base_url) === here) as PipelineNode<"model"> | undefined) ?? null;
  }

  secondFor(id: string): PipelineNode<"model"> | null {
    const second = this.second.get(id);
    return second ? (this.byId.get(second) as PipelineNode<"model">) : null;
  }

  below(id: string): string[] {
    const found: string[] = [];
    const todo = [...(this.children.get(id) ?? [])];
    while (todo.length) {
      const here = todo.pop()!;
      if (found.includes(here)) continue;
      found.push(here);
      todo.push(...(this.children.get(here) ?? []));
    }
    return found;
  }

  // The newest text a step could work on: the column the nearest Generate above writes, else the first seed column.
  nearestText(id: string): string {
    const path = this.path(id).slice(0, -1).reverse().map((s) => this.byId.get(s)!);
    const written = path.find((n) => n.type === "generate" && (n.config as GenerateConfig).column);
    if (written) return (written.config as GenerateConfig).column;
    const seed = path.find((n) => n.type === "seed" && (n.config as SeedConfig).columns.length);
    return seed ? ident((seed.config as SeedConfig).columns[0]) : "";
  }
}

// Whether an arrow may go from one step to another, into the given handle of the target (rows, model or check).
export function canConnect(graph: Graph, from: string, to: string, kind: EdgeKind): boolean {
  const source = graph.byId.get(from);
  const target = graph.byId.get(to);
  if (!source || !target || from === to) return false;
  if (kind === "rows") {
    if (source.type === "save" || target.type === "model") return false;
    return !graph.path(from).includes(to); // no circles: the target can't be above the source
  }
  if (source.type !== "model") return false;
  return kind === "model" ? ASKS_MODEL.includes(target.type) : target.type === "question";
}

export const NODE_WIDTH = 248;
const ROW_GAP = 150;
const COLUMN_GAP = 40;

// Tidy positions: every line top to bottom, its branches side by side, the lines next to each other, and each
// plugged-in Model to the right of the first step it serves.
export function tidy(graph: Graph): Map<string, { x: number; y: number }> {
  const at = new Map<string, { x: number; y: number }>();
  let next = 0; // the next free column, in node widths

  const place = (id: string, depth: number): number => {
    const kids = graph.children.get(id) ?? [];
    let x: number;
    if (!kids.length) x = next++;
    else {
      const xs = kids.map((k) => place(k, depth + 1));
      x = (xs[0] + xs[xs.length - 1]) / 2;
    }
    at.set(id, { x: x * (NODE_WIDTH + COLUMN_GAP), y: depth * ROW_GAP });
    return x;
  };

  const plugs = new Set([...graph.plugged.values(), ...graph.second.values()]);
  const tops = graph.nodes.filter((n) => !graph.parent.has(n.id) && !(n.type === "model" && plugs.has(n.id)
    && !(graph.children.get(n.id) ?? []).length));
  for (const top of tops) {
    if (!at.has(top.id)) place(top.id, 0);
  }
  for (const n of graph.nodes) {
    if (at.has(n.id)) continue;
    const served = graph.nodes.find((s) => graph.plugged.get(s.id) === n.id || graph.second.get(s.id) === n.id);
    const there = served && at.get(served.id);
    if (there) {
      // Beside the step it serves; a second Model for the same row moves further right.
      let x = there.x + NODE_WIDTH + 80;
      while ([...at.values()].some((p) => Math.abs(p.y - there.y) < 60 && Math.abs(p.x - x) < NODE_WIDTH)) {
        x += NODE_WIDTH + COLUMN_GAP;
      }
      at.set(n.id, { x, y: there.y });
    } else at.set(n.id, { x: next++ * (NODE_WIDTH + COLUMN_GAP), y: 0 });
  }
  return at;
}
