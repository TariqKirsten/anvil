// What each kind of step is called and looks like, and a step's card on the canvas: rows come in at the top and leave
// at the bottom; a step that asks a model has a model handle on its right, and a Question a second-opinion one below
// it.

import {
  IconCopyOff, IconCpu, IconDatabase, IconDeviceFloppy, IconFilter, IconList, IconMessageQuestion, IconSparkles,
} from "@tabler/icons-react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { createContext, useContext, type ReactNode } from "react";
import {
  ASKS_MODEL, FORMATS, QUESTION_TYPES, RUNNING, values, type Configs, type Graph, type PipelineNode,
  type PipelineProblem, type PipelineRun, type StepType,
} from "./graph";

export const STEPS: Record<StepType, { label: string; about: string; icon: ReactNode }> = {
  model: { label: "Model", about: "Connects to an OpenAI-compatible endpoint. Every line starts at one.",
    icon: <IconCpu size={15} aria-hidden /> },
  seed: { label: "Seed data", about: "Starts from rows you already have: a file, or a Hugging Face dataset.",
    icon: <IconDatabase size={15} aria-hidden /> },
  list: { label: "List", about: "Gives each row a value from a list you type: topics, languages, tones.",
    icon: <IconList size={15} aria-hidden /> },
  generate: { label: "Generate", about: "The model writes text from a prompt, one row per reply or several.",
    icon: <IconSparkles size={15} aria-hidden /> },
  question: { label: "Question", about: "Labels each row with a decision question, answered by a model or a column.",
    icon: <IconMessageQuestion size={15} aria-hidden /> },
  filter: { label: "Filter", about: "Keeps only the rows that pass its rules.",
    icon: <IconFilter size={15} aria-hidden /> },
  dedupe: { label: "Dedupe", about: "Drops copies and near-copies, by the rule Prepare data uses.",
    icon: <IconCopyOff size={15} aria-hidden /> },
  save: { label: "Save", about: "Ends a line: writes its rows as an anvil dataset.",
    icon: <IconDeviceFloppy size={15} aria-hidden /> },
};

export const ORDER: StepType[] = ["model", "seed", "list", "generate", "question", "filter", "dedupe", "save"];

export type StepData = { step: StepType; config: Configs[StepType]; note: string };
export type StepNodeType = Node<StepData, "step">;

// What every card needs to know about the rest of the pipeline, shared rather than copied into each node.
export const CanvasContext = createContext<{
  graph: Graph;
  problems: Map<string, PipelineProblem[]>;
  keys: Set<string>;
  run: PipelineRun | null; // whose counts the steps show
} | null>(null);

function host(url: string) {
  try {
    return new URL(url).host;
  } catch {
    return url || "no address yet";
  }
}

export function modelName(model: PipelineNode<"model"> | null): string {
  if (!model) return "no Model";
  return model.config.name.trim() || model.config.model.split("/").pop() || "its Model";
}

// The card's title and the line under it.
export function describe(node: PipelineNode, graph: Graph, keys: Set<string>): { title: string; detail: string } {
  const c = node.config;
  switch (node.type) {
    case "model": {
      const m = c as Configs["model"];
      return {
        title: m.name.trim() || m.model.split("/").pop() || "Model",
        detail: m.model ? `${host(m.base_url)}${keys.has(node.id) ? " · key saved" : graph.keyFrom(node.id, keys)
          ? " · key shared" : ""}` : "Pick its endpoint and model",
      };
    }
    case "seed": {
      const s = c as Configs["seed"];
      if (!s.source) return { title: "Seed data", detail: "Pick the dataset to start from" };
      return { title: s.source.name || ("path" in s.source ? s.source.path.split("/").pop()! : s.source.repo),
        detail: `${s.columns.length} column${s.columns.length === 1 ? "" : "s"} · ${
          s.order === "shuffled" ? "shuffled" : "in order"}${s.limit ? ` · first ${s.limit.toLocaleString()}` : ""}` };
    }
    case "list": {
      const l = c as Configs["list"];
      const n = values(l.values).length;
      return { title: l.column || "List",
        detail: `${n} value${n === 1 ? "" : "s"} · ${l.pick === "turns" ? "taking turns" : "at random"}` };
    }
    case "generate": {
      const g = c as Configs["generate"];
      const per = g.per_reply ?? 1;
      return { title: g.column || "Generate",
        detail: `${per} per reply · asks ${modelName(graph.modelFor(node.id))}` };
    }
    case "question": {
      const q = c as Configs["question"];
      const kind = q.type === "choice" ? `choice of ${q.options.filter((o) => o.name.trim()).length}`
        : q.type === "score" ? (values(q.levels).length > 1 ? `score 0–${values(q.levels).length - 1}` : "score")
          : QUESTION_TYPES.noul.toLowerCase();
      const from = q.answer === "model" ? `asks ${modelName(graph.modelFor(node.id))}` : `from ${q.from_column || "a column"}`;
      return { title: q.name || "Question", detail: `${kind} · ${from}${graph.secondFor(node.id) ? " · 2nd opinion" : ""}` };
    }
    case "filter": {
      const f = c as Configs["filter"];
      const first = f.rules[0];
      if (!first) return { title: "Filter", detail: "No rules yet" };
      const value = first.op === "empty" || first.op === "not_empty" ? "" : ` ${first.value}`;
      const ops: Record<string, string> = { is: "is", is_not: "isn't", contains: "contains", not_contains: "lacks",
        matches: "matches", longer: ">", shorter: "<", empty: "is empty", not_empty: "isn't empty" };
      return { title: "Filter", detail: `${first.column || "?"} ${ops[first.op] ?? first.op}${value}${
        f.rules.length > 1 ? ` ${f.match === "all" ? "and" : "or"} ${f.rules.length - 1} more` : ""}` };
    }
    case "dedupe": {
      const d = c as Configs["dedupe"];
      return { title: d.column || "Dedupe", detail: d.near ? "copies and near-copies" : "exact copies" };
    }
    case "save": {
      const s = c as Configs["save"];
      return { title: s.name || "Save", detail: `${(s.rows ?? 0).toLocaleString()} rows · ${FORMATS[s.format]}` };
    }
  }
}

// What the step did in the run being looked at: rows in and out and dropped, or a Save's rows against its target.
// Rows a step took but didn't finish were under way while it runs, and let go once every Save below had its rows.
function Counts({ id, step, run }: { id: string; step: StepType; run: PipelineRun | null }) {
  const s = run?.steps?.[id];
  if (step === "model" || !s || !s.in) return null;
  const open = Math.max(0, s.in - (s.done ?? s.in));
  const going = run && RUNNING.includes(run.state);
  if (step === "save") {
    const save = run?.saves?.[id];
    return save ? <div className="step-counts">{save.kept.toLocaleString()} / {save.target.toLocaleString()} kept
      {s.dropped ? <span className="dropped"> · {s.dropped.toLocaleString()} dropped</span> : null}</div> : null;
  }
  return (
    <div className="step-counts">
      {s.in.toLocaleString()} in · {s.out.toLocaleString()} out
      {s.dropped ? <span className="dropped"> · {s.dropped.toLocaleString()} dropped</span> : null}
      {s.left_out ? <span className="dropped"> · {s.left_out.toLocaleString()} left out</span> : null}
      {open ? <span title={going ? "Rows this step is working on" : "Rows on their way when every Save below was full"}>
        {" "}· {open.toLocaleString()} {going ? "under way" : "let go"}</span> : null}
    </div>
  );
}

export function StepNode({ id, data, selected }: NodeProps<StepNodeType>) {
  const canvas = useContext(CanvasContext);
  if (!canvas) return null;
  const node = canvas.graph.byId.get(id);
  if (!node) return null;
  const info = STEPS[data.step];
  const { title, detail } = describe(node, canvas.graph, canvas.keys);
  const problems = canvas.problems.get(id) ?? [];
  const errors = problems.filter((p) => p.level === "error");
  const worst = errors.length ? "error" : problems.length ? "warn" : null;
  const shown = errors.length ? errors : problems;
  const asks = ASKS_MODEL.includes(data.step);
  // A handle's name shows on a picked or hovered step, while nothing is plugged into it.
  const free = (kind: "model" | "check") => !(kind === "model" ? canvas.graph.plugged : canvas.graph.second).has(id);
  return (
    <div className={`step-node step-${data.step}${selected ? " selected" : ""}${worst ? ` has-${worst}` : ""}`}>
      {data.step !== "model" && (
        <Handle type="target" position={Position.Top} id="rows" className="handle-rows" title="Rows come in here" />
      )}
      <div className="step-head">
        <span className="step-icon">{info.icon}</span>
        <span className="step-kind">{info.label}</span>
        {worst && (
          <span className={`step-flag flag-${worst}`} title={shown.map((p) => p.message).join("\n")}
            aria-label={`${shown.length} ${worst === "error" ? "to fix" : "to look at"}`}>
            {shown.length}
          </span>
        )}
      </div>
      <div className="step-title" title={title}>{title}</div>
      <div className="step-detail" title={detail}>{detail}</div>
      <Counts id={id} step={data.step} run={canvas.run} />
      {asks && (
        <>
          <Handle type="target" position={Position.Right} id="model" className="handle-plug"
            style={{ top: data.step === "question" ? 30 : "50%" }} title="Plug a Model in here to use it instead" />
          {free("model") && <span className="plug-label" style={{ top: data.step === "question" ? 30 : "50%" }}>model</span>}
        </>
      )}
      {data.step === "question" && (
        <>
          <Handle type="target" position={Position.Right} id="check" className="handle-plug" style={{ top: 62 }}
            title="Plug a second Model in here for a second opinion" />
          {free("check") && <span className="plug-label" style={{ top: 62 }}>2nd opinion</span>}
        </>
      )}
      {data.step === "model" && (
        <Handle type="source" position={Position.Left} id="plug" className="handle-plug"
          title="Drag to a step's model handle to use this Model there" />
      )}
      {data.step !== "save" && (
        <Handle type="source" position={Position.Bottom} id="out" className="handle-out"
          title={data.step === "model" ? "Drag down to start a line" : "Rows go on from here"} />
      )}
    </div>
  );
}
