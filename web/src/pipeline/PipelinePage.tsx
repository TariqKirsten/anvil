// A pipeline's page: its steps on a canvas, and beside it the picked step's settings (or, with nothing picked, what
// still needs doing). Every change is saved a moment later, and the server answers with what still stops the pipeline
// from running, shown on the steps and under their settings.

import {
  IconAlertCircle, IconAlertTriangle, IconArrowLeft, IconChevronDown, IconEye, IconNotes, IconPlayerPlay,
  IconPlayerStop, IconPlus, IconSitemap, IconTrash, IconX,
} from "@tabler/icons-react";
import {
  Background, BackgroundVariant, Controls, MarkerType, Panel, ReactFlow, ReactFlowProvider, useEdgesState,
  useNodesState, useReactFlow, type Connection, type Edge, type EdgeChange, type NodeChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { Topbar } from "../Topbar";
import { Badge } from "../ui";
import {
  canConnect, Graph, made, NODE_WIDTH, RUNNING, tidy, type Configs, type EdgeKind, type Pipeline, type PipelineEdge,
  type PipelineNode, type PipelineProblem, type PipelineRun, type StepDefaults, type StepType,
} from "./graph";
import { PreviewView, RunsView } from "./RunPanel";
import { StepForm } from "./StepForm";
import { CanvasContext, ORDER, STEPS, StepNode, type StepNodeType } from "./steps";

const nodeTypes = { step: StepNode };
const ROW_GAP = 136;
const CARD_HEIGHT = 96;

let stepDefaults: Promise<StepDefaults> | null = null; // the same for every pipeline: asked once

function edgeOf(e: PipelineEdge): Edge {
  const plug = e.kind !== "rows";
  return {
    id: e.id, source: e.from, target: e.to, sourceHandle: plug ? "plug" : "out", targetHandle: e.kind,
    type: plug ? "default" : "smoothstep", className: `edge-${e.kind}`,
    markerEnd: plug ? undefined : { type: MarkerType.ArrowClosed, width: 18, height: 18, color: "var(--edge)" },
  };
}

function toFlow(p: Pipeline): { nodes: StepNodeType[]; edges: Edge[] } {
  return {
    nodes: p.nodes.map((n) => ({ id: n.id, type: "step", position: { x: n.x, y: n.y },
      data: { step: n.type, config: n.config, note: n.note ?? "" } })),
    edges: p.edges.map(edgeOf),
  };
}

function fromFlow(nodes: StepNodeType[], edges: Edge[]): { nodes: PipelineNode[]; edges: PipelineEdge[] } {
  return {
    nodes: nodes.map((n) => ({ id: n.id, type: n.data.step, x: Math.round(n.position.x),
      y: Math.round(n.position.y), note: n.data.note, config: n.data.config }) as PipelineNode),
    edges: edges.map((e) => ({ id: e.id, from: e.source, to: e.target, kind: (e.targetHandle ?? "rows") as EdgeKind })),
  };
}

function message(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

export function PipelinePage({ id, onLogout }: { id: string; onLogout: () => void }) {
  const [pipeline, setPipeline] = useState<Pipeline | null>(null);
  const [defaults, setDefaults] = useState<StepDefaults | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    stepDefaults ??= api<StepDefaults>("/api/pipeline-steps");
    Promise.all([api<{ pipeline: Pipeline }>(`/api/pipelines/${id}`), stepDefaults])
      .then(([found, steps]) => {
        setPipeline(found.pipeline);
        setDefaults(steps);
      })
      .catch((e) => {
        stepDefaults = null;
        setError(message(e));
      });
  }, [id]);

  if (!pipeline || !defaults) {
    return (
      <div className="page">
        <Topbar onLogout={onLogout} />
        {error ? <p className="field-error page-note">{error}</p> : <p className="hint page-note">Loading…</p>}
        {error && <a className="back" href="#/pipelines"><IconArrowLeft size={14} aria-hidden /> All pipelines</a>}
      </div>
    );
  }
  return (
    <ReactFlowProvider>
      <Editor pipeline={pipeline} defaults={defaults} onLogout={onLogout} />
    </ReactFlowProvider>
  );
}

type Saving = { state: "saved" | "waiting" | "saving" | "failed"; error?: string };

function Editor({ pipeline, defaults, onLogout }: { pipeline: Pipeline; defaults: StepDefaults; onLogout: () => void }) {
  const first = useMemo(() => toFlow(pipeline), [pipeline]);
  const [nodes, setNodes, onNodesChangeFlow] = useNodesState<StepNodeType>(first.nodes);
  const [edges, setEdges, onEdgesChangeFlow] = useEdgesState<Edge>(first.edges);
  const [name, setName] = useState(pipeline.name);
  const [about, setAbout] = useState(pipeline.about ?? "");
  const [selected, setSelected] = useState<string | null>(null);
  const [problems, setProblems] = useState<PipelineProblem[]>(pipeline.problems);
  const [keys, setKeys] = useState(() => new Set(pipeline.keys));
  const [saving, setSaving] = useState<Saving>({ state: "saved" });
  const [version, setVersion] = useState(0);
  const touch = useCallback(() => setVersion((v) => v + 1), []);
  const flow = useReactFlow();
  const [preview, setPreview] = useState<PipelineRun | null>(pipeline.preview);
  const [runs, setRuns] = useState<PipelineRun[]>(pipeline.runs);
  const [tab, setTab] = useState<"about" | "preview" | "runs">("about");
  const [openRun, setOpenRun] = useState<string | null>(pipeline.runs[0]?.id ?? null);
  const [startError, setStartError] = useState<string | null>(null);

  // The start of every line, at a size to read: the whole graph if it fits, else its top, zoomed out only so far.
  const showStart = useCallback(() => {
    const box = document.querySelector(".pipe-canvas")?.getBoundingClientRect();
    const all = flow.getNodes();
    if (!box || !all.length) return;
    const xs = all.map((n) => n.position.x);
    const ys = all.map((n) => n.position.y);
    const left = Math.min(...xs), right = Math.max(...xs) + NODE_WIDTH;
    const top = Math.min(...ys), bottom = Math.max(...ys) + CARD_HEIGHT;
    const room = { width: box.width - 64, height: box.height - 96 };
    const zoom = Math.min(1, Math.max(0.7, Math.min(room.width / (right - left), room.height / (bottom - top))));
    const fits = (bottom - top) * zoom <= room.height;
    const wide = (right - left) * zoom > room.width; // a phone: the lines start at the left edge
    void flow.setViewport({
      x: wide ? 28 - left * zoom : (box.width - (right - left) * zoom) / 2 - left * zoom,
      y: fits ? 60 + (room.height - (bottom - top) * zoom) / 2 - top * zoom : 64 - top * zoom,
      zoom,
    });
  }, [flow]);

  const plain = useMemo(() => fromFlow(nodes, edges), [nodes, edges]);
  const graph = useMemo(() => new Graph(plain.nodes, plain.edges), [plain]);
  const byNode = useMemo(() => {
    const found = new Map<string, PipelineProblem[]>();
    for (const p of problems) if (p.node) found.set(p.node, [...(found.get(p.node) ?? []), p]);
    return found;
  }, [problems]);

  // A preview or run under way is asked how it's doing every second, until it ends.
  const going = [...(preview && RUNNING.includes(preview.state) ? [preview] : []),
    ...runs.filter((r) => RUNNING.includes(r.state))];
  const watched = going.map((r) => `${r.id}:${r.state}`).join(" ");
  useEffect(() => {
    if (!watched) return;
    const ids = watched.split(" ").map((w) => w.split(":")[0]);
    const timer = setInterval(() => {
      for (const id of ids) {
        api<{ run: PipelineRun }>(`/api/pipelines/${pipeline.id}/runs/${id}`)
          .then(({ run }) => (id === "preview" ? setPreview(run) : setRuns((rs) => rs.map((r) => (r.id === id ? run : r)))))
          .catch(() => null); // the next second tries again
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [watched, pipeline.id]);

  // The counts on the steps: the run being looked at, else the one under way.
  const shown = tab === "preview" && !selected ? preview
    : tab === "runs" && !selected ? runs.find((r) => r.id === openRun) ?? null
      : going[0] ?? null;
  const canvas = useMemo(() => ({ graph, problems: byNode, keys, run: shown }), [graph, byNode, keys, shown]);

  // ------------------------------------------------------------ saving: a moment after the last change, one at a time
  const latest = useRef({ name, about, ...plain });
  latest.current = { name, about, ...plain };
  const inFlight = useRef<Promise<void> | null>(null);
  const again = useRef(false);

  const flush = useCallback(async () => {
    if (inFlight.current) {
      again.current = true;
      return;
    }
    setSaving({ state: "saving" });
    const work = (async () => {
      try {
        const answer = await api<{ problems: PipelineProblem[] }>(`/api/pipelines/${pipeline.id}`,
          { method: "PUT", json: latest.current });
        setProblems(answer.problems);
        setSaving({ state: "saved" });
      } catch (e) {
        setSaving({ state: "failed", error: message(e) });
      }
    })();
    inFlight.current = work;
    await work;
    inFlight.current = null;
    if (again.current) {
      again.current = false;
      await flush();
    }
  }, [pipeline.id]);

  useEffect(() => {
    if (!version) return;
    setSaving((s) => (s.state === "saving" ? s : { state: "waiting" }));
    const timer = setTimeout(flush, 600);
    return () => clearTimeout(timer);
  }, [version, flush]);

  // Leaving the page saves what's waiting; closing the tab first asks.
  const pending = useRef(false);
  pending.current = saving.state === "waiting" || saving.state === "saving";
  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => {
      if (pending.current) e.preventDefault();
    };
    window.addEventListener("beforeunload", warn);
    return () => {
      window.removeEventListener("beforeunload", warn);
      if (pending.current) void flush();
    };
  }, [flush]);

  // ------------------------------------------------------------ preview and run

  // Save what's on screen first, so the run is of the pipeline as it's shown.
  async function saveNow(): Promise<boolean> {
    while (inFlight.current) await inFlight.current;
    await flush();
    while (inFlight.current) await inFlight.current;
    return true;
  }

  async function start(previewing: boolean) {
    setStartError(null);
    await saveNow();
    try {
      const { id } = await api<{ id: string }>(`/api/pipelines/${pipeline.id}/runs`, { json: { preview: previewing } });
      const { run } = await api<{ run: PipelineRun }>(`/api/pipelines/${pipeline.id}/runs/${id}`);
      if (previewing) setPreview(run);
      else {
        setRuns((rs) => [run, ...rs]);
        setOpenRun(run.id);
      }
      pick(null);
      setTab(previewing ? "preview" : "runs");
    } catch (e) {
      setStartError(message(e));
    }
  }

  async function stop(run: PipelineRun) {
    await api(`/api/pipelines/${pipeline.id}/runs/${run.id}/stop`, { method: "POST" }).catch(() => null);
    refresh(run.id);
  }

  function refresh(id: string, gone = false) {
    if (gone) {
      setRuns((rs) => rs.filter((r) => r.id !== id));
      return;
    }
    void api<{ run: PipelineRun }>(`/api/pipelines/${pipeline.id}/runs/${id}`)
      .then(({ run }) => (id === "preview" ? setPreview(run) : setRuns((rs) => rs.map((r) => (r.id === id ? run : r)))))
      .catch(() => null);
  }

  // ------------------------------------------------------------ editing the graph

  const forgetKey = useCallback((id: string) => {
    if (!keys.has(id)) return;
    void api(`/api/pipelines/${pipeline.id}/keys/${id}`, { method: "DELETE" }).catch(() => null);
    setKeys((k) => new Set([...k].filter((x) => x !== id)));
  }, [keys, pipeline.id]);

  const onNodesChange = useCallback((changes: NodeChange<StepNodeType>[]) => {
    onNodesChangeFlow(changes);
    let changed = false;
    for (const c of changes) {
      if (c.type === "position" && c.dragging === false) changed = true;
      if (c.type === "remove") {
        changed = true;
        forgetKey(c.id);
        setSelected((s) => (s === c.id ? null : s));
      }
    }
    if (changed) touch();
  }, [onNodesChangeFlow, forgetKey, touch]);

  const onEdgesChange = useCallback((changes: EdgeChange<Edge>[]) => {
    onEdgesChangeFlow(changes);
    if (changes.some((c) => c.type === "remove")) touch();
  }, [onEdgesChangeFlow, touch]);

  const isValidConnection = useCallback((c: Connection | Edge) =>
    canConnect(graph, c.source, c.target, (c.targetHandle ?? "rows") as EdgeKind), [graph]);

  // A new arrow into a handle replaces the one already there: rows come from one place, and one Model per handle.
  const onConnect = useCallback((c: Connection) => {
    const kind = (c.targetHandle ?? "rows") as EdgeKind;
    setEdges((es) => [...es.filter((e) => !(e.target === c.target && (e.targetHandle ?? "rows") === kind)),
      edgeOf({ id: `${c.source}-${kind}-${c.target}`, from: c.source, to: c.target, kind })]);
    touch();
  }, [setEdges, touch]);

  const changeNote = useCallback((id: string, note: string) => {
    setNodes((ns) => ns.map((n) => (n.id === id ? { ...n, data: { ...n.data, note } } : n)));
    touch();
  }, [setNodes, touch]);

  const change = useCallback((id: string, patch: Partial<Configs[StepType]>) => {
    setNodes((ns) => ns.map((n) => (n.id === id
      ? { ...n, data: { ...n.data, config: { ...n.data.config, ...patch } as Configs[StepType] } } : n)));
    touch();
  }, [setNodes, touch]);

  function newId(type: StepType) {
    for (;;) {
      const id = `${type}-${Math.random().toString(36).slice(2, 7)}`;
      if (!graph.byId.has(id)) return id;
    }
  }

  // A new step goes after the one picked (and is joined to it), with names filled in that fit where it lands.
  function addStep(type: StepType) {
    const after = selected ? graph.byId.get(selected) : undefined;
    const joins = after && after.type !== "save" && type !== "model" ? after : undefined;
    const have = joins ? [...graph.columnsAbove(joins.id), ...made(joins)] : [];
    const free = (base: string) => {
      let name = base;
      for (let n = 2; have.includes(name); n++) name = `${base}_${n}`;
      return name;
    };
    const text = !joins ? "" : joins.type === "generate" ? (joins.config as Configs["generate"]).column
      : joins.type === "seed" ? made(joins)[0] ?? graph.nearestText(joins.id) : graph.nearestText(joins.id);
    const cfg = structuredClone(defaults.steps[type]) as Configs[StepType];
    if (type === "list") Object.assign(cfg, { column: free("topic") });
    if (type === "generate") Object.assign(cfg, { column: free("message") });
    if (type === "question") Object.assign(cfg, { about: text });
    if (type === "dedupe") Object.assign(cfg, { column: text });
    if (type === "save") {
      const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60) || "dataset";
      const taken = plain.nodes.filter((n) => n.type === "save").map((n) => (n.config as Configs["save"]).name);
      let dataset = slug;
      for (let n = 2; taken.includes(dataset); n++) dataset = `${slug}-${n}`;
      Object.assign(cfg, { name: dataset, state_column: text, user_column: text, text_column: text });
    }

    // After a step with one step below it, the new one goes in between and everything below moves down a row (a
    // Save ends a line, so it starts a branch instead).
    const kids = joins ? graph.children.get(joins.id) ?? [] : [];
    const between = joins && kids.length === 1 && type !== "save" ? kids[0] : null;
    const moved = new Set(between ? [between, ...graph.below(between)] : []);
    for (const n of plain.nodes) { // Models plugged into the steps that move go with them
      if (n.type === "model" && !graph.children.get(n.id)?.length
        && [...graph.plugged, ...graph.second].some(([step, model]) => model === n.id && moved.has(step))) moved.add(n.id);
    }
    let position: { x: number; y: number };
    if (type === "model") {
      const right = Math.max(0, ...plain.nodes.map((n) => n.x));
      position = { x: right + NODE_WIDTH + 80, y: after?.y ?? 0 };
    } else if (joins) {
      position = { x: joins.x + (between ? 0 : kids.length * (NODE_WIDTH + 40)), y: joins.y + ROW_GAP };
    } else {
      const box = document.querySelector(".pipe-canvas")?.getBoundingClientRect();
      const middle = box ? flow.screenToFlowPosition({ x: box.left + box.width / 2, y: box.top + box.height / 3 })
        : { x: 0, y: 0 };
      position = { x: middle.x - NODE_WIDTH / 2, y: middle.y };
    }
    const id = newId(type);
    setNodes((ns) => [...ns.map((n) => ({ ...n, selected: false,
      position: moved.has(n.id) ? { x: n.position.x, y: n.position.y + ROW_GAP } : n.position })),
      { id, type: "step", position, data: { step: type, config: cfg, note: "" }, selected: true }]);
    if (joins) {
      setEdges((es) => [
        ...es.filter((e) => !(between && e.source === joins.id && e.target === between && (e.targetHandle ?? "rows") === "rows")),
        edgeOf({ id: `${joins.id}-rows-${id}`, from: joins.id, to: id, kind: "rows" }),
        ...(between ? [edgeOf({ id: `${id}-rows-${between}`, from: id, to: between, kind: "rows" })] : []),
      ]);
    }
    setSelected(id);
    touch();
    reveal(position);
  }

  // Bring a spot on the canvas into view, if it isn't already, at the same zoom.
  function reveal(at: { x: number; y: number }) {
    const box = document.querySelector(".pipe-canvas")?.getBoundingClientRect();
    if (!box) return;
    const corner = flow.flowToScreenPosition(at);
    const far = flow.flowToScreenPosition({ x: at.x + NODE_WIDTH, y: at.y + CARD_HEIGHT });
    if (corner.x >= box.left && corner.y >= box.top + 56 && far.x <= box.right && far.y <= box.bottom) return;
    void flow.setCenter(at.x + NODE_WIDTH / 2, at.y + CARD_HEIGHT / 2, { zoom: flow.getZoom(), duration: 300 });
  }

  function removeStep(id: string) {
    setNodes((ns) => ns.filter((n) => n.id !== id));
    setEdges((es) => es.filter((e) => e.source !== id && e.target !== id));
    forgetKey(id);
    setSelected(null);
    touch();
  }

  function tidyUp() {
    const at = tidy(graph);
    setNodes((ns) => ns.map((n) => ({ ...n, position: at.get(n.id) ?? n.position })));
    touch();
    requestAnimationFrame(showStart);
  }

  function pick(id: string | null) {
    setSelected(id);
    setNodes((ns) => ns.map((n) => (n.selected !== (n.id === id) ? { ...n, selected: n.id === id } : n)));
    const node = id ? flow.getNode(id) : undefined;
    if (node) flow.fitView({ nodes: [node], padding: 1.2, maxZoom: 1, duration: 300 });
  }

  const node = selected ? graph.byId.get(selected) : undefined;
  const errors = problems.filter((p) => p.level === "error");
  const warnings = problems.filter((p) => p.level === "warn");
  const blocked = errors.length ? "Fix what's marked in red first." : going.length
    ? "Wait for the preview or run under way, or stop it." : null;
  return (
    <div className="pipe-page">
      <Topbar onLogout={onLogout} right={
        <>
          <SaveBadge saving={saving} />
          {errors.length ? <Badge tone="bad">{errors.length} to fix</Badge>
            : warnings.length ? <Badge tone="warn">{warnings.length} to look at</Badge> : <Badge tone="ok">Ready</Badge>}
        </>
      }>
        <input className="run-name" aria-label="Pipeline name" value={name}
          onChange={(e) => {
            setName(e.target.value);
            touch();
          }} />
      </Topbar>
      <div className="pipe-main">
        <div className="pipe-canvas">
          <CanvasContext.Provider value={canvas}>
            <ReactFlow<StepNodeType, Edge> nodes={nodes} edges={edges} nodeTypes={nodeTypes}
              onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect}
              isValidConnection={isValidConnection}
              onSelectionChange={({ nodes: picked }) => setSelected(picked.length === 1 ? picked[0].id : null)}
              onInit={showStart} minZoom={0.2} maxZoom={1.75}
              deleteKeyCode={["Backspace", "Delete"]} connectionRadius={28}
              connectionLineStyle={{ stroke: "var(--accent)", strokeWidth: 2 }}>
              <Background variant={BackgroundVariant.Dots} gap={18} size={1.2} />
              <Controls showInteractive={false} position="bottom-left" />
              <Panel position="top-left" className="canvas-tools">
                <AddStep onAdd={addStep} after={node && node.type !== "save" ? STEPS[node.type].label : null} />
                <button type="button" onClick={tidyUp} title="Line the steps up">
                  <IconSitemap size={16} aria-hidden /> Tidy up
                </button>
              </Panel>
              <Panel position="top-right" className="canvas-tools run-tools">
                {going.length ? (
                  <button type="button" onClick={() => going.forEach(stop)}
                    disabled={going.every((r) => r.state === "stopping")}>
                    <IconPlayerStop size={16} aria-hidden /> Stop {going[0].id === "preview" ? "preview" : "run"}
                  </button>
                ) : (
                  <>
                    <button type="button" onClick={() => start(true)} disabled={!!blocked} title={blocked ?? "Make a few rows and see each step's work"}>
                      <IconEye size={16} aria-hidden /> Preview
                    </button>
                    <button type="button" className="primary" onClick={() => start(false)} disabled={!!blocked}
                      title={blocked ?? "Make every Save's rows"}>
                      <IconPlayerPlay size={16} aria-hidden /> Run
                    </button>
                  </>
                )}
              </Panel>
            </ReactFlow>
          </CanvasContext.Provider>
        </div>
        <aside className="pipe-side" aria-label={node ? `${STEPS[node.type].label} settings` : "About this pipeline"}>
          {startError && <p className="error-box side-error">{startError}</p>}
          {!node && (
            <div className="side-tabs" role="tablist" aria-label="About, preview and runs">
              {(["about", "preview", "runs"] as const).map((t) => (
                <button key={t} type="button" role="tab" aria-selected={tab === t} className={tab === t ? "on" : ""}
                  onClick={() => setTab(t)}>
                  {t === "about" ? "About" : t === "preview" ? "Preview" : `Runs${runs.length ? ` ${runs.length}` : ""}`}
                  {((t === "preview" && preview && RUNNING.includes(preview.state))
                    || (t === "runs" && runs.some((r) => RUNNING.includes(r.state)))) && <span className="live-dot" aria-label="running" />}
                </button>
              ))}
            </div>
          )}
          {node ? (
            <Inspector key={node.id} node={node} graph={graph} problems={byNode.get(node.id) ?? []} defaults={defaults}
              pipelineId={pipeline.id} keys={keys}
              onKeysChange={(id, saved) => setKeys((k) => {
                const next = new Set(k);
                if (saved) next.add(id);
                else next.delete(id);
                return next;
              })}
              change={(patch) => change(node.id, patch)} onRemove={() => removeStep(node.id)} onClose={() => pick(null)}
              onNote={(note) => changeNote(node.id, note)} />
          ) : tab === "preview" ? (
            <PreviewView pipelineId={pipeline.id} preview={preview} graph={graph} blocked={blocked}
              onStart={() => start(true)} onStop={() => preview && stop(preview)} />
          ) : tab === "runs" ? (
            <RunsView pipelineId={pipeline.id} runs={runs} graph={graph} blocked={blocked} open={openRun}
              onOpen={setOpenRun} onStart={() => start(false)} onChanged={refresh} />
          ) : (
            <Overview problems={problems} graph={graph} onPick={pick} pipelineId={pipeline.id} name={name}
              about={about} onAbout={(text) => {
                setAbout(text);
                touch();
              }} />
          )}
        </aside>
      </div>
    </div>
  );
}

function SaveBadge({ saving }: { saving: Saving }) {
  if (saving.state === "failed") return <Badge tone="bad" title={saving.error}>Not saved</Badge>;
  if (saving.state === "saved") return <Badge className="save-state">Saved</Badge>;
  return <Badge className="save-state">Saving…</Badge>;
}

function AddStep({ onAdd, after }: { onAdd: (type: StepType) => void; after: string | null }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: Event) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !box.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("pointerdown", close);
      document.removeEventListener("keydown", close);
    };
  }, [open]);
  return (
    <div className="add-step" ref={box}>
      <button type="button" className="primary" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <IconPlus size={16} aria-hidden /> Add a step <IconChevronDown size={14} aria-hidden />
      </button>
      {open && (
        <div className="step-menu" role="menu">
          <p className="hint">{after ? `Goes after the ${after} step you picked.` : "Pick a step first to add one after it."}</p>
          {ORDER.map((type) => (
            <button key={type} type="button" role="menuitem" className="step-menu-item" onClick={() => {
              setOpen(false);
              onAdd(type);
            }}>
              <span className={`step-icon step-${type}`}>{STEPS[type].icon}</span>
              <span className="step-menu-text">
                <span className="step-menu-name">{STEPS[type].label}</span>
                <span className="hint">{STEPS[type].about}</span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function Inspector(props: {
  node: PipelineNode; graph: Graph; problems: PipelineProblem[]; defaults: StepDefaults; pipelineId: string;
  keys: Set<string>; onKeysChange: (node: string, saved: boolean) => void;
  change: (patch: Partial<Configs[StepType]>) => void; onRemove: () => void; onClose: () => void;
  onNote: (note: string) => void;
}) {
  const info = STEPS[props.node.type];
  const general = props.problems.filter((p) => !p.field);
  return (
    <div className="side-body">
      <div className="side-head">
        <span className={`step-icon step-${props.node.type}`}>{info.icon}</span>
        <h2>{info.label}</h2>
        <button type="button" className="icon" aria-label="Remove this step" title="Remove this step"
          onClick={props.onRemove}>
          <IconTrash size={16} aria-hidden />
        </button>
        <button type="button" className="icon" aria-label="Close the settings" title="Close the settings"
          onClick={props.onClose}>
          <IconX size={16} aria-hidden />
        </button>
      </div>
      <p className="hint">{info.about}</p>
      <NoteBox text={props.node.note ?? ""} onChange={props.onNote} label="This step's note" callout
        empty="Add a note to this step" />
      {general.length > 0 && (
        <ul className="side-problems">
          {general.map((p, i) => (
            <li key={i} className={p.level === "error" ? "field-error" : "warn-text"}>{p.message}</li>
          ))}
        </ul>
      )}
      <StepForm node={props.node} graph={props.graph} problems={props.problems} defaults={props.defaults}
        pipelineId={props.pipelineId} keys={props.keys} onKeysChange={props.onKeysChange} change={props.change} />
    </div>
  );
}

// Words about a pipeline or a step: a template's explanation, or your own. Shown as paragraphs, edited in place.
function NoteBox({ text, onChange, label, empty, callout }: { text: string; onChange: (text: string) => void;
  label: string; empty: string; callout?: boolean }) {
  const [editing, setEditing] = useState(false);
  if (editing) {
    return (
      <div className="note-edit">
        <textarea aria-label={label} value={text} autoFocus
          rows={Math.min(14, Math.max(4, text.split("\n").length + 2))} onChange={(e) => onChange(e.target.value)} />
        <div className="form-actions">
          <button type="button" onClick={() => setEditing(false)}>Done</button>
        </div>
      </div>
    );
  }
  if (!text.trim()) {
    return (
      <button type="button" className="link add-note" onClick={() => setEditing(true)}>
        <IconNotes size={15} aria-hidden /> {empty}
      </button>
    );
  }
  return (
    <div className={callout ? "note-box" : "note-text"}>
      {text.split(/\n{2,}/).map((paragraph, i) => <p key={i}>{paragraph}</p>)}
      <button type="button" className="link" onClick={() => setEditing(true)}>Edit</button>
    </div>
  );
}

function Overview({ problems, graph, onPick, pipelineId, name, about, onAbout }: { problems: PipelineProblem[];
  graph: Graph; onPick: (id: string | null) => void; pipelineId: string; name: string; about: string;
  onAbout: (text: string) => void }) {
  const sorted = [...problems].sort((a, b) => (a.level === b.level ? 0 : a.level === "error" ? -1 : 1));
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  async function remove() {
    setDeleting(true);
    try {
      await api(`/api/pipelines/${pipelineId}`, { method: "DELETE" });
      window.location.hash = "#/pipelines";
    } catch (e) {
      setDeleteError(message(e));
      setDeleting(false);
    }
  }

  return (
    <div className="side-body">
      <a className="back" href="#/pipelines"><IconArrowLeft size={14} aria-hidden /> All pipelines</a>
      {about.trim() && <h2>About this pipeline</h2>}
      <NoteBox text={about} onChange={onAbout} label="About this pipeline" empty="Add a note about this pipeline" />
      <h2>How it works</h2>
      <p className="side-text">Every line starts at a Model. Rows flow down the arrows: each step adds a column to every
        row, or drops some, and Save writes them as a dataset that New run can train on.</p>
      <ul className="side-tips">
        <li><strong>Add a step</strong> with the button on the canvas. With a step picked, the new one goes after it.</li>
        <li><strong>Connect</strong> two steps by dragging from one's bottom handle to the next one's top.</li>
        <li><strong>Another model</strong> for one step: drag from that Model to the step's model handle, on its right.</li>
        <li><strong>Remove</strong> a step or an arrow: pick it, then press Delete.</li>
      </ul>
      <p className="side-text"><a href="#/docs/pipelines">More in the Docs</a>: each step, second opinions, balance,
        and what a saved row holds.</p>
      <h2>{sorted.length ? "Still to do" : "Nothing to fix"}</h2>
      {sorted.length ? (
        <ul className="todo-list">
          {sorted.map((p, i) => {
            const at = p.node ? graph.byId.get(p.node) : undefined;
            return (
              <li key={i}>
                <button type="button" className={`todo todo-${p.level}`} onClick={() => onPick(at ? at.id : null)}
                  disabled={!at}>
                  {p.level === "error" ? <IconAlertCircle size={15} aria-hidden /> : <IconAlertTriangle size={15} aria-hidden />}
                  <span>
                    {at && <span className="todo-step">{STEPS[at.type].label}: </span>}
                    {p.message}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="side-text">Every step is set up: Preview shows a few rows go through, and Run makes them all.</p>
      )}
      <div className="side-end">
        {confirming ? (
          <div className="confirm-box" role="alertdialog" aria-label="Delete this pipeline?">
            <p>Delete <strong>{name}</strong> and its runs for good? The datasets it made stay where they are.</p>
            <div className="confirm-actions">
              <button type="button" className="danger-solid" disabled={deleting} onClick={remove}>
                <IconTrash size={16} aria-hidden /> {deleting ? "Deleting…" : "Delete pipeline"}
              </button>
              <button type="button" disabled={deleting} onClick={() => setConfirming(false)}>Cancel</button>
            </div>
          </div>
        ) : (
          <button type="button" className="danger" onClick={() => setConfirming(true)}>
            <IconTrash size={16} aria-hidden /> Delete pipeline
          </button>
        )}
        {deleteError && <p className="error-box">{deleteError}</p>}
      </div>
    </div>
  );
}
