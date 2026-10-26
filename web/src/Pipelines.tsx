// Every pipeline, the one changed last first, with how its newest run went, and a bin to delete it. New pipeline
// starts from a description (the assistant talks it through and builds it), a template to learn from
// (anvil/templates.py) or nothing, makes the pipeline and opens it.

import { IconLoader2, IconPlus, IconSparkles, IconTrash, IconX } from "@tabler/icons-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { api } from "./api";
import type { PipelineRunState, PipelineSummary, Template } from "./pipeline/graph";
import type { AssistantSettings } from "./types";
import { when } from "./Runs";
import { Topbar } from "./Topbar";
import { Badge } from "./ui";

export const PIPELINE_STATE: Record<PipelineRunState, string> = {
  starting: "Starting",
  running: "Running",
  stopping: "Stopping",
  stopped: "Stopped",
  done: "Done",
  failed: "Failed",
};

export function pipelineTone(state: PipelineRunState): "ok" | "warn" | "bad" | "accent" {
  return state === "done" ? "ok" : state === "failed" ? "bad" : state === "stopped" || state === "stopping" ? "warn"
    : "accent";
}

// graph.ts's RUNNING, written out so this page doesn't pull the editor's code into the first download.
const GOING: PipelineRunState[] = ["starting", "running", "stopping"];

export function Pipelines({ onLogout }: { onLogout: () => void }) {
  const [pipelines, setPipelines] = useState<PipelineSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [templates, setTemplates] = useState<Template[] | null>(null);
  const [choosing, setChoosing] = useState(false);
  const [making, setMaking] = useState<string | null>(null); // a template's id, "" for an empty one, or "describe"
  const [assistantReady, setAssistantReady] = useState(false);
  const deletes = useRef(0); // a list asked for before a delete went through still has the pipeline

  useEffect(() => {
    api<{ templates: Template[] }>("/api/pipeline-templates")
      .then((r) => setTemplates(r.templates))
      .catch((e: Error) => setError(e.message));
    api<{ settings: AssistantSettings }>("/api/assistant")
      .then((r) => setAssistantReady(!!r.settings.model))
      .catch(() => null);
  }, []);

  useEffect(() => {
    let alive = true;
    const load = () => {
      const asked = deletes.current;
      api<{ pipelines: PipelineSummary[] }>("/api/pipelines")
        .then((r) => alive && asked === deletes.current && setPipelines(r.pipelines))
        .catch((e: Error) => alive && setError(e.message));
    };
    load();
    const timer = setInterval(load, 5000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  async function make(template: string | null) {
    setMaking(template ?? "");
    try {
      const made = await api<{ pipeline: { id: string } }>("/api/pipelines", { json: { template } });
      window.location.hash = `#/pipelines/${made.pipeline.id}`;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setMaking(null);
    }
  }

  // A pipeline from a description: an empty one, with the description as the first message to the assistant, which
  // talks it through before it builds it.
  async function describe(text: string) {
    setMaking("describe");
    try {
      const made = await api<{ pipeline: { id: string } }>("/api/pipelines", { json: { template: null } });
      await api(`/api/pipelines/${made.pipeline.id}/assistant`, { json: { message: text } });
      window.location.hash = `#/pipelines/${made.pipeline.id}`;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setMaking(null);
    }
  }

  function forget(id: string) {
    deletes.current += 1;
    setPipelines((ps) => ps && ps.filter((p) => p.id !== id));
  }

  const none = pipelines !== null && pipelines.length === 0;

  return (
    <div className="page">
      <Topbar onLogout={onLogout} />
      <div className="page-head">
        <h1>Pipelines</h1>
        {!none && (
          <button type="button" onClick={() => setChoosing((c) => !c)} aria-expanded={choosing}>
            <IconPlus size={16} aria-hidden /> New pipeline
          </button>
        )}
      </div>
      {error && <p className="field-error page-note">{error}</p>}
      {none && (
        <p className="side-text page-note">
          A pipeline makes a dataset with a model: it writes rows, labels them, drops the ones that don't fit, and saves
          the rest for New run to train on. Start from a template to see how one works.
        </p>
      )}
      {(choosing || none) && (
        <StartFrom templates={templates} making={making} onMake={make} onDescribe={describe} ready={assistantReady}
          onClose={none ? undefined : () => setChoosing(false)} />
      )}
      {pipelines && pipelines.length > 0 && (
        <div className="section run-list">
          {pipelines.map((p) => <PipelineRow key={p.id} pipeline={p} onDeleted={() => forget(p.id)} />)}
        </div>
      )}
    </div>
  );
}

// A row opens its pipeline. The bin asks first, then deletes it with its runs; the datasets it made stay where they
// are. A pipeline with a run going has no bin (the server refuses a preview's too, and says so).
function PipelineRow({ pipeline: p, onDeleted }: { pipeline: PipelineSummary; onDeleted: () => void }) {
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const latest = p.latest;
  const going = !!latest && GOING.includes(latest.state);
  const saves = latest?.saves ? Object.values(latest.saves) : [];
  const kept = saves.reduce((n, s) => n + s.kept, 0);
  const target = saves.reduce((n, s) => n + s.target, 0);

  async function remove() {
    setDeleting(true);
    setError(null);
    try {
      await api(`/api/pipelines/${p.id}`, { method: "DELETE" });
      onDeleted();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setDeleting(false);
    }
  }

  return (
    <div className="pipeline-row">
      <a className="run-row" href={`#/pipelines/${p.id}`}>
        <span className="run-main">
          <span className="run-title" title={p.name}>{p.name}</span>
          <span className="run-meta">
            {p.steps} step{p.steps === 1 ? "" : "s"} · changed {when(p.updated)}
          </span>
        </span>
        <span className="run-progress">
          {latest ? (
            <>
              <span className="run-steps">
                {kept.toLocaleString()} / {target.toLocaleString()} rows · {when(latest.started)}
              </span>
              <span className="bar"><span style={{ width: `${target ? Math.min(100, (kept / target) * 100) : 0}%` }} /></span>
            </>
          ) : <span className="run-steps">Not run yet</span>}
        </span>
        {latest ? <Badge tone={pipelineTone(latest.state)}>{PIPELINE_STATE[latest.state] ?? latest.state}</Badge>
          : <span />}
      </a>
      {!going && (
        <button type="button" className="icon row-bin" aria-label={`Delete ${p.name}`} title="Delete pipeline"
          aria-expanded={confirming} onClick={() => setConfirming((c) => !c)}>
          <IconTrash size={16} aria-hidden />
        </button>
      )}
      {confirming && !going && (
        <div className="confirm-box" role="alertdialog" aria-label="Delete this pipeline?">
          <p>Delete <strong>{p.name}</strong> and its runs for good? The datasets it made stay where they are.</p>
          <div className="confirm-actions">
            <button type="button" className="danger-solid" disabled={deleting} onClick={remove}>
              <IconTrash size={16} aria-hidden /> {deleting ? "Deleting…" : "Delete pipeline"}
            </button>
            <button type="button" disabled={deleting} onClick={() => setConfirming(false)}>Cancel</button>
          </div>
          {error && <p className="field-error">{error}</p>}
        </div>
      )}
    </div>
  );
}

// A description for the assistant, an empty pipeline, or a template: a working pipeline with a note on every step about
// why it's there.
function StartFrom({ templates, making, onMake, onDescribe, ready, onClose }: { templates: Template[] | null;
  making: string | null; onMake: (template: string | null) => void; onDescribe: (text: string) => void; ready: boolean;
  onClose?: () => void }) {
  const [text, setText] = useState("");
  const busy = (id: string) => making === id && <IconLoader2 size={15} className="spin" aria-hidden />;

  function submit(event: FormEvent) {
    event.preventDefault();
    if (ready && text.trim() && making === null) onDescribe(text.trim());
  }

  return (
    <div className="section start-from">
      <div className="section-head">
        <h2>Start from</h2>
        {onClose && (
          <button type="button" className="icon" aria-label="Close" title="Close" onClick={onClose}>
            <IconX size={16} aria-hidden />
          </button>
        )}
      </div>
      <form className="describe" onSubmit={submit}>
        <label htmlFor="describe-text"><IconSparkles size={15} aria-hidden /> Describe the data you want</label>
        <textarea id="describe-text" rows={2} value={text} disabled={!ready}
          placeholder="Short messages bank customers send, labelled with whether they're about fraud"
          onChange={(e) => setText(e.target.value)} />
        <div className="form-actions">
          <p className="hint">{ready ? "The assistant talks it through with you, then builds it when you say so."
            : <>Choose the assistant's model in <a href="#/settings">Settings</a> first.</>}</p>
          <button type="submit" className="primary" disabled={!ready || !text.trim() || making !== null}>
            {busy("describe")} Talk it through
          </button>
        </div>
      </form>
      <p className="hint">Or start from a template. They run on radiance's Gemma: with the assistant's key saved in
        Settings they're ready to preview, otherwise paste the key into their Model. Every step has a note on why it's
        there and what to try.</p>
      <div className="template-grid">
        <button type="button" className="template-card" onClick={() => onMake(null)} disabled={making !== null}>
          <span className="template-makes">Your own</span>
          <span className="template-name">{busy("")} Empty pipeline</span>
          <span className="template-summary">Just a Model, to build the rest yourself.</span>
        </button>
        {templates?.map((t) => (
          <button key={t.id} type="button" className="template-card" onClick={() => onMake(t.id)}
            disabled={making !== null}>
            <span className="template-makes">{t.makes} · {t.steps} steps</span>
            <span className="template-name">{busy(t.id)} {t.name}</span>
            <span className="template-summary">{t.summary}</span>
            <span className="template-learn">
              {t.learn.map((item) => <span key={item} className="learn-chip">{item}</span>)}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
