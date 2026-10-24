// Making the rows: a preview makes a few and shows each one's way through the steps; a run makes them all in the
// background and writes the datasets, with Stop and Resume. Both views follow the status the page polls.

import {
  IconChevronDown, IconChevronRight, IconEye, IconLoader2, IconPlayerPlay, IconPlayerStop, IconRefresh, IconSchool,
  IconTrash,
} from "@tabler/icons-react";
import { useEffect, useState } from "react";
import { api } from "../api";
import { loadForm, saveForm } from "../form";
import { duration } from "../format";
import { PIPELINE_STATE, pipelineTone } from "../Pipelines";
import { when } from "../Runs";
import { Badge } from "../ui";
import {
  FORMATS, RUNNING, type Configs, type Graph, type PipelineNode, type PipelineRun, type RowEntry, type SaveFormat,
  type TraceStep,
} from "./graph";
import { modelName, STEPS } from "./steps";

const PREVIEW_ROWS = 5; // the server's generate.PREVIEW_ROWS

function message(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

// A cell the way people read it: yes and no for a question's answer, a dash for none.
export function show(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (value === true) return "yes";
  if (value === false) return "no";
  return typeof value === "string" ? value : JSON.stringify(value);
}

function stepLabel(graph: Graph, id: string) {
  const node = graph.byId.get(id);
  return node ? STEPS[node.type].label : "A step that's gone";
}

// Every Save's rows so far, with each value's share when it balances.
export function Progress({ run }: { run: PipelineRun }) {
  const saves = Object.entries(run.saves ?? {});
  if (!saves.length) return null;
  return (
    <div className="save-progress">
      {saves.map(([id, s]) => (
        <div key={id} className="save-line">
          <div className="save-line-head">
            <span className="save-name" title={s.name}>{s.name}</span>
            <span className="save-count">
              {s.kept.toLocaleString()} / {s.target.toLocaleString()}
              {s.closed && s.kept < s.target ? " · seed data used up" : ""}
            </span>
          </div>
          <span className="bar"><span style={{ width: `${s.target ? Math.min(100, (100 * s.kept) / s.target) : 0}%` }} /></span>
          {s.balance && s.quota && (
            <div className="share-list">
              {Object.entries(s.quota).map(([value, share]) => (
                <span key={value} className="share">
                  {show(value === "true" ? true : value === "false" ? false : value)}{" "}
                  <span className="hint">{(s.by_value?.[value] ?? 0).toLocaleString()}/{share.toLocaleString()}</span>
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function StatusLine({ run }: { run: PipelineRun }) {
  const going = RUNNING.includes(run.state);
  return (
    <div className="run-status">
      <Badge tone={pipelineTone(run.state)}>{PIPELINE_STATE[run.state] ?? run.state}</Badge>
      <span className="hint">
        {going && run.state !== "stopping" && (run.rate
          ? `${run.rate >= 10 ? Math.round(run.rate) : run.rate.toFixed(1)} rows a second${
            run.eta_s ? ` · about ${duration(run.eta_s)} left` : ""}`
          : `${(run.rows_started ?? 0).toLocaleString()} rows started`)}
        {run.state === "stopping" && "Finishing the requests under way"}
        {!going && run.started && run.ended && `Took ${duration(run.ended - run.started)}`}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------- preview

export function PreviewView({ pipelineId, preview, graph, blocked, onStart, onStop }: {
  pipelineId: string; preview: PipelineRun | null; graph: Graph; blocked: string | null;
  onStart: () => void; onStop: () => void;
}) {
  const going = !!preview && RUNNING.includes(preview.state);
  const [entries, setEntries] = useState<RowEntry[] | null>(null);

  // The rows, once the preview has finished, and again for each new one.
  useEffect(() => {
    setEntries(null);
    if (!preview || going) return;
    let alive = true;
    api<{ rows: RowEntry[] }>(`/api/pipelines/${pipelineId}/runs/preview/rows?which=preview&limit=200`)
      .then((r) => alive && setEntries(r.rows))
      .catch(() => alive && setEntries([]));
    return () => {
      alive = false;
    };
  }, [pipelineId, preview?.started, going]); // not preview itself: each poll brings a new object

  const kept = entries?.filter((e) => e.outcome === "kept") ?? [];
  const dropped = entries?.filter((e) => e.outcome !== "kept") ?? [];
  return (
    <div className="side-body">
      <p className="side-text">A preview makes {PREVIEW_ROWS} rows for each Save and shows each row's way through the
        steps: what was asked, what the model answered, and why rows were dropped. It makes no dataset.</p>
      <div className="start-row">
        {going ? (
          <button type="button" onClick={onStop} disabled={preview?.state === "stopping"}>
            <IconPlayerStop size={16} aria-hidden /> Stop
          </button>
        ) : (
          <button type="button" className="primary" onClick={onStart} disabled={!!blocked}>
            <IconEye size={16} aria-hidden /> {preview ? "Preview again" : "Make a preview"}
          </button>
        )}
        {blocked && !going && <span className="hint">{blocked}</span>}
      </div>
      {preview && (
        <>
          <StatusLine run={preview} />
          <Progress run={preview} />
          {preview.error && <p className="error-box">{preview.error}</p>}
        </>
      )}
      {going && <p className="hint"><IconLoader2 size={14} className="spin" aria-hidden /> The rows show here when it's done.</p>}
      {entries && kept.length > 0 && (
        <>
          <h2>Kept <span className="count">{kept.length}</span></h2>
          {kept.map((e, i) => <RowCard key={`k${i}`} entry={e} graph={graph} />)}
        </>
      )}
      {entries && dropped.length > 0 && (
        <>
          <h2>Dropped <span className="count">{dropped.length}</span></h2>
          {dropped.map((e, i) => <RowCard key={`d${i}`} entry={e} graph={graph} />)}
        </>
      )}
      {entries && !entries.length && <p className="hint">No rows came through.</p>}
    </div>
  );
}

// A row: the list values that made it, the text it holds, its answers, where it ended and why, and in a preview what
// every step did with it.
export function RowCard({ entry, graph }: { entry: RowEntry; graph: Graph }) {
  const path = graph.byId.has(entry.node) ? graph.path(entry.node) : [];
  const nodes = path.map((id) => graph.byId.get(id)!);
  const lists = nodes.filter((n) => n.type === "list").map((n) => (n.config as Configs["list"]).column);
  const asked = nodes.filter((n) => n.type === "question").map((n) => (n.config as Configs["question"]).name);
  const known = new Set([...lists, ...asked]);
  const texts = Object.entries(entry.row).filter(([column]) => !known.has(column));
  return (
    <div className={`row-card outcome-${entry.outcome}`}>
      <div className="row-card-head">
        {entry.outcome === "kept" ? <Badge tone="ok">Kept</Badge>
          : <Badge tone={entry.outcome === "error" ? "bad" : "warn"}>
            {entry.outcome === "error" ? "Failed" : "Dropped"} at {stepLabel(graph, entry.node)}
          </Badge>}
        {lists.filter((c) => c in entry.row).map((c) => (
          <span key={c} className="row-chip" title={`${c}: ${show(entry.row[c])}`}>{show(entry.row[c])}</span>
        ))}
      </div>
      {texts.map(([column, value]) => (
        <div key={column} className="row-text">
          <span className="row-column">{column}</span>
          <p>{show(value)}</p>
        </div>
      ))}
      {asked.filter((c) => c in entry.row).length > 0 && (
        <p className="row-answers">
          {asked.filter((c) => c in entry.row).map((c) => (
            <span key={c}><span className="row-column">{c}</span> {entry.row[c] === null ? <em>left out</em> : show(entry.row[c])}</span>
          ))}
        </p>
      )}
      {entry.reason && <p className={entry.outcome === "error" ? "field-error" : "warn-text"}>{entry.reason}</p>}
      {entry.trace && entry.trace.length > 0 && (
        <details className="row-trace">
          <summary>What each step did</summary>
          <Trace steps={entry.trace} graph={graph} row={entry.row} />
        </details>
      )}
    </div>
  );
}

function Trace({ steps, graph, row }: { steps: TraceStep[]; graph: Graph; row: Record<string, unknown> }) {
  return (
    <ol className="trace">
      {steps.map((s, i) => {
        const node = graph.byId.get(s.node);
        const name = node ? describeStep(node) : "A step that's gone";
        const picked = node?.type === "list" ? row[(node.config as Configs["list"]).column] : undefined;
        return (
          <li key={i}>
            <span className="trace-step">{name}{picked !== undefined && <span className="hint"> → {show(picked)}</span>}</span>
            {s.asked?.map((a, j) => (
              <div key={j} className="trace-ask">
                <span className="hint">
                  Asked {modelName((graph.byId.get(a.model) as PipelineNode<"model">) ?? null)} · {a.seconds.toFixed(1)} s
                </span>
                <details>
                  <summary>The prompt</summary>
                  <pre>{a.prompt}</pre>
                </details>
                <pre className="trace-reply">{a.reply || "(an empty reply)"}</pre>
              </div>
            ))}
            {"answer" in s && (
              <span className="hint">Answer: {s.answer === null ? `left out${s.left_out ? ` (${s.left_out})` : ""}` : show(s.answer)}</span>
            )}
          </li>
        );
      })}
    </ol>
  );
}

function describeStep(node: PipelineNode): string {
  const c = node.config as unknown as Record<string, unknown>;
  const name = c.column || c.name || "";
  return `${STEPS[node.type].label}${name ? ` · ${name}` : ""}`;
}

// ---------------------------------------------------------------- runs

export function RunsView({ pipelineId, runs, graph, blocked, open, onOpen, onStart, onChanged }: {
  pipelineId: string; runs: PipelineRun[]; graph: Graph; blocked: string | null; open: string | null;
  onOpen: (id: string | null) => void; onStart: () => void; onChanged: (id: string, gone?: boolean) => void;
}) {
  const saves = graph.nodes.filter((n) => n.type === "save");
  const total = saves.reduce((n, s) => n + ((s.config as Configs["save"]).rows ?? 0), 0);
  return (
    <div className="side-body">
      <p className="side-text">A run makes every Save's rows, {total.toLocaleString()} in all, in the background: leaving
        the page doesn't stop it. Each Save becomes a dataset that New run lists, under datasets/generated.</p>
      <div className="start-row">
        <button type="button" className="primary" onClick={onStart} disabled={!!blocked}>
          <IconPlayerPlay size={16} aria-hidden /> Start a run
        </button>
        {blocked && <span className="hint">{blocked}</span>}
      </div>
      {runs.length === 0 && <p className="hint">No runs yet.</p>}
      {runs.map((run) => (
        <RunCard key={run.id} pipelineId={pipelineId} run={run} graph={graph} open={open === run.id}
          onOpen={() => onOpen(open === run.id ? null : run.id)} onChanged={onChanged} />
      ))}
    </div>
  );
}

function RunCard({ pipelineId, run, graph, open, onOpen, onChanged }: {
  pipelineId: string; run: PipelineRun; graph: Graph; open: boolean; onOpen: () => void;
  onChanged: (id: string, gone?: boolean) => void;
}) {
  const going = RUNNING.includes(run.state);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);

  async function act(what: "stop" | "resume" | "delete") {
    setBusy(true);
    setError(null);
    try {
      if (what === "delete") await api(`/api/pipelines/${pipelineId}/runs/${run.id}`, { method: "DELETE" });
      else await api(`/api/pipelines/${pipelineId}/runs/${run.id}/${what}`, { method: "POST" });
      onChanged(run.id, what === "delete");
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
      setConfirming(false);
    }
  }

  const datasets = Object.entries(run.saves ?? {}).filter(([, s]) => s.dataset);
  return (
    <div className={`run-card${open ? " open" : ""}`}>
      <button type="button" className="run-card-head" aria-expanded={open} onClick={onOpen}>
        {open ? <IconChevronDown size={15} aria-hidden /> : <IconChevronRight size={15} aria-hidden />}
        <span className="run-card-when">{when(run.started)}</span>
        {run.resumed ? <span className="hint">resumed</span> : null}
      </button>
      <StatusLine run={run} />
      <Progress run={run} />
      {run.error && <p className="error-box">{run.error}</p>}
      {error && <p className="error-box">{error}</p>}
      {datasets.length > 0 && !going && (
        <div className="datasets">
          {datasets.map(([id, s]) => (
            <div key={id} className="dataset-row">
              <code title={s.dataset}>{s.dataset!.split("/").pop()}</code>
              <TrainOn path={s.dataset!} format={(graph.byId.get(id)?.config as Configs["save"] | undefined)?.format} />
            </div>
          ))}
        </div>
      )}
      {confirming ? (
        <div className="confirm-box" role="alertdialog" aria-label="Delete this run?">
          <p>Delete this run's rows and counts? The datasets it wrote stay where they are.</p>
          <div className="confirm-actions">
            <button type="button" className="danger-solid" disabled={busy} onClick={() => act("delete")}>
              <IconTrash size={16} aria-hidden /> Delete run
            </button>
            <button type="button" disabled={busy} onClick={() => setConfirming(false)}>Cancel</button>
          </div>
        </div>
      ) : (
        <div className="run-card-actions">
          {going && (
            <button type="button" onClick={() => act("stop")} disabled={busy || run.state === "stopping"}>
              <IconPlayerStop size={16} aria-hidden /> {run.state === "stopping" ? "Stopping…" : "Stop"}
            </button>
          )}
          {(run.state === "stopped" || run.state === "failed") && (
            <button type="button" onClick={() => act("resume")} disabled={busy}>
              <IconRefresh size={16} aria-hidden /> Resume
            </button>
          )}
          {!going && (
            <button type="button" className="danger" onClick={() => setConfirming(true)} disabled={busy}>
              <IconTrash size={16} aria-hidden /> Delete
            </button>
          )}
        </div>
      )}
      {open && <RunDetail pipelineId={pipelineId} run={run} graph={graph} />}
    </div>
  );
}

// A dataset straight into New run: chosen as its train data, with the kind of model that trains on it.
function TrainOn({ path, format }: { path: string; format?: SaveFormat }) {
  return (
    <button type="button" className="train-on" title="Open New run with this dataset as its train data"
      onClick={() => {
        const form = loadForm();
        saveForm({ ...form, trainAs: format === "decision" || !format ? "decision" : "lm",
          data: { ...form.data, train: { kind: "file", path, name: path.split("/").pop()! }, validation: null, test: null } });
        window.location.hash = "#/";
      }}>
      <IconSchool size={16} aria-hidden /> Train on it
    </button>
  );
}

function RunDetail({ pipelineId, run, graph }: { pipelineId: string; run: PipelineRun; graph: Graph }) {
  const steps = graph.nodes.filter((n) => n.type !== "model" && run.steps?.[n.id]?.in);
  const why = Object.entries(run.reasons ?? {});
  const blanks = Object.entries(run.blanks ?? {});
  const [which, setWhich] = useState<string | null>(null);
  const [log, setLog] = useState<string | null>(null);
  return (
    <div className="run-detail">
      {steps.length > 0 && (
        <div className="table-scroll" data-scrolls>
          <table className="table step-table">
            <thead>
              <tr><th>Step</th><th>In</th><th>Out</th><th>Dropped</th><th title="Requests to a model">Asked</th></tr>
            </thead>
            <tbody>
              {steps.map((n) => {
                const s = run.steps![n.id];
                return (
                  <tr key={n.id}>
                    <td title={describeStep(n)}>{describeStep(n)}</td>
                    <td>{s.in.toLocaleString()}</td>
                    <td>{s.out.toLocaleString()}</td>
                    <td className={s.dropped ? "warn-text" : undefined}>{s.dropped.toLocaleString()}</td>
                    <td title={s.requests ? `${s.completion_tokens.toLocaleString()} tokens written, ${
                      (s.seconds / s.requests).toFixed(1)} s a request` : undefined}>{s.requests.toLocaleString()}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {(why.length > 0 || blanks.length > 0) && (
        <ul className="reason-list">
          {why.flatMap(([id, kinds]) => Object.entries(kinds).map(([kind, n]) => (
            <li key={`${id}${kind}`}><strong>{stepLabel(graph, id)}</strong> dropped {n.toLocaleString()}: {kind}</li>
          )))}
          {blanks.flatMap(([id, kinds]) => Object.entries(kinds).map(([kind, n]) => (
            <li key={`b${id}${kind}`}><strong>{stepLabel(graph, id)}</strong> left {n.toLocaleString()} answers out: {kind}</li>
          )))}
        </ul>
      )}
      <div className="detail-buttons">
        {Object.entries(run.saves ?? {}).map(([id, s]) => (
          <button key={id} type="button" className={`chip${which === id ? " on" : ""}`}
            onClick={() => setWhich(which === id ? null : id)}>{s.name} rows</button>
        ))}
        <button type="button" className={`chip${which === "dropped" ? " on" : ""}`}
          onClick={() => setWhich(which === "dropped" ? null : "dropped")}>Dropped rows</button>
        <button type="button" className={`chip${log !== null ? " on" : ""}`} onClick={() => {
          if (log !== null) return setLog(null);
          api<{ log: string }>(`/api/pipelines/${pipelineId}/runs/${run.id}/log`).then((r) => setLog(r.log))
            .catch((e) => setLog(message(e)));
        }}>Log</button>
      </div>
      {which && <RowsBrowser key={which} pipelineId={pipelineId} run={run} which={which} graph={graph} />}
      {log !== null && <pre className="run-log">{log || "Nothing in the log."}</pre>}
    </div>
  );
}

// A Save's rows (as its dataset has them) or the dropped ones, ten at a time.
function RowsBrowser({ pipelineId, run, which, graph }: { pipelineId: string; run: PipelineRun; which: string;
  graph: Graph }) {
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const more = (offset: number) =>
    api<{ rows: Record<string, unknown>[]; total: number }>(
      `/api/pipelines/${pipelineId}/runs/${run.id}/rows?which=${encodeURIComponent(which)}&offset=${offset}&limit=10`)
      .then((r) => {
        setRows((had) => [...had.slice(0, offset), ...r.rows]); // a page asked for twice still shows once
        setTotal(r.total);
      })
      .catch((e) => setError(message(e)));
  useEffect(() => {
    void more(0);
  }, []); // the browser is keyed by what it shows, so it loads once
  if (error) return <p className="error-box">{error}</p>;
  if (total === null) return <p className="hint">Loading…</p>;
  if (!total) return <p className="hint">No rows.</p>;
  const save = graph.byId.get(which);
  return (
    <div className="rows-browser">
      <p className="hint">{rows.length.toLocaleString()} of {total.toLocaleString()}{which === "dropped"
        ? " (the first of each step's dropped rows are kept to look at)" : save
          ? ` · ${FORMATS[(save.config as Configs["save"]).format]}` : ""}</p>
      {which === "dropped" ? rows.map((r, i) => <RowCard key={i} entry={r as unknown as RowEntry} graph={graph} />)
        : rows.map((r, i) => <SavedRow key={i} row={r} />)}
      {rows.length < total && (
        <button type="button" className="add-row" onClick={() => more(rows.length)}>Show more</button>
      )}
    </div>
  );
}

// A row as its dataset holds it: the state and answers, the chat, the prompt and completion, or the text.
function SavedRow({ row }: { row: Record<string, unknown> }) {
  const parse = (v: unknown) => {
    try {
      return typeof v === "string" ? JSON.parse(v) : v;
    } catch {
      return v;
    }
  };
  let body: { label: string; text: string }[];
  if ("gold" in row) {
    const state = parse(row.state) as Record<string, unknown>;
    const gold = parse(row.gold) as Record<string, unknown>;
    body = [...Object.entries(state).map(([k, v]) => ({ label: k, text: show(v) })),
      { label: "answers", text: Object.entries(gold).map(([k, v]) => `${k}: ${v === null ? "left out" : show(v)}`).join(" · ") }];
  } else if ("messages" in row) {
    body = (row.messages as { role: string; content: string }[]).map((m) => ({ label: m.role, text: m.content }));
  } else if ("completion" in row) {
    body = [{ label: "prompt", text: show(row.prompt) }, { label: "completion", text: show(row.completion) }];
  } else {
    body = [{ label: "text", text: show(row.text) }];
  }
  return (
    <div className="row-card">
      {body.map((b, i) => (
        <div key={i} className="row-text">
          <span className="row-column">{b.label}</span>
          <p>{b.text}</p>
        </div>
      ))}
    </div>
  );
}
