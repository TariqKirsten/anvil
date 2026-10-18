// One run, live: progress, speed, the loss and validation charts, checkpoints, the log, and Stop / Resume.

import { IconArrowLeft, IconPlayerPlay, IconPlayerStop } from "@tabler/icons-react";
import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import { Chart } from "./Chart";
import { STATE_LABEL, stateTone, when } from "./Runs";
import { Topbar } from "./Topbar";
import type { RunDetail, RunMetric } from "./types";
import { Badge, Section } from "./ui";

const ACTIVE = new Set(["starting", "loading", "running", "saving"]);

function duration(seconds: number | null | undefined) {
  if (seconds == null || !Number.isFinite(seconds)) return "–";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

// An exponential moving average, so the trend shows through batch-to-batch noise.
function smooth(points: { x: number; y: number }[], weight = 0.85) {
  let value: number | null = null;
  return points.map((p) => {
    value = value === null ? p.y : weight * value + (1 - weight) * p.y;
    return { x: p.x, y: value };
  });
}

function num(v: unknown, digits = 3) {
  return typeof v === "number" ? v.toFixed(digits) : "–";
}

export function RunPage({ id, onLogout }: { id: string; onLogout: () => void }) {
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [acting, setActing] = useState(false);
  const logRef = useRef<HTMLPreElement>(null);
  const active = run ? ACTIVE.has(run.status.state) : true;

  useEffect(() => {
    let alive = true;
    const load = () =>
      api<RunDetail>(`/api/runs/${id}`)
        .then((r) => {
          if (!alive) return;
          setRun(r);
          setError(null);
        })
        .catch((e: Error) => alive && setError(e.message));
    load();
    const timer = setInterval(load, active ? 2000 : 15000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [id, active]);

  useEffect(() => {
    const el = logRef.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 80) el.scrollTop = el.scrollHeight;
  }, [run?.log]);

  async function act(what: "stop" | "resume") {
    setActing(true);
    try {
      await api(`/api/runs/${id}/${what}`, { method: "POST" });
      setRun(await api<RunDetail>(`/api/runs/${id}`));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setActing(false);
    }
  }

  if (!run)
    return (
      <div className="page">
        <Topbar onLogout={onLogout} />
        {error ? <p className="field-error notice">{error}</p> : <p className="hint notice">Loading…</p>}
      </div>
    );

  const { config: cfg, status: st } = run;
  const decision = cfg.train_as === "decision";
  const train = run.metrics.filter((m): m is RunMetric => m.type === "train");
  const evals = run.metrics.filter((m) => m.type === "eval");
  const pct = st.total ? Math.min(100, (st.step / st.total) * 100) : 0;
  const elapsed = st.started ? (st.finished ?? st.stopped ?? (ACTIVE.has(st.state) ? Date.now() / 1000 : st.started)) - st.started : null;
  const perQuestion = decision && evals.length
    ? Object.keys(evals[evals.length - 1]).filter((k) => k.startsWith("auroc_")).map((k) => k.slice(6))
    : [];

  return (
    <div className="page">
      <Topbar onLogout={onLogout} />
      <a className="back" href="#/runs">
        <IconArrowLeft size={16} aria-hidden /> All runs
      </a>

      <section className="section">
        <div className="run-head">
          <div>
            <h1>{cfg.name}</h1>
            <p className="hint">
              {cfg.model.id} · {decision ? `decision model (${cfg.method})` : `language model (${cfg.method})`} ·{" "}
              {cfg.training_type === "lora" ? "LoRA" : "full fine-tuning"} · GPU {cfg.gpu} · started {when(cfg.created)}
            </p>
          </div>
          <div className="run-actions">
            <Badge tone={stateTone(st.state)}>{STATE_LABEL[st.state] ?? st.state}</Badge>
            {ACTIVE.has(st.state) && (
              <button type="button" onClick={() => act("stop")}>
                <IconPlayerStop size={16} aria-hidden /> {acting ? "Stopping…" : "Stop"}
              </button>
            )}
            {(st.state === "stopped" || st.state === "failed") && (
              <button type="button" className="primary" onClick={() => act("resume")}>
                <IconPlayerPlay size={16} aria-hidden /> {acting ? "Resuming…" : "Resume"}
              </button>
            )}
          </div>
        </div>

        <div className="run-bar">
          <div className="bar big"><div style={{ width: `${pct}%` }} /></div>
          <span>
            {st.total ? `Step ${st.step.toLocaleString()} of ${st.total.toLocaleString()}` : st.stage}
            {ACTIVE.has(st.state) && st.total ? ` · ${st.stage}` : ""}
          </span>
        </div>

        <div className="stats">
          <div><span className="stat-label">Time so far</span><span className="stat-value">{duration(elapsed)}</span></div>
          <div><span className="stat-label">Time left</span><span className="stat-value">{ACTIVE.has(st.state) ? duration(st.eta_s) : "–"}</span></div>
          <div><span className="stat-label">Speed</span><span className="stat-value">{st.tokens_per_s ? `${Math.round(st.tokens_per_s).toLocaleString()} tok/s` : "–"}</span></div>
          <div><span className="stat-label">GPU memory (peak)</span><span className="stat-value">{st.peak_mem_gb ? `${st.peak_mem_gb} GB` : "–"}</span></div>
        </div>

        {st.state === "failed" && st.error && <p className="error-box">{st.error}</p>}
        {st.state === "done" && run.model && (
          <p className="ok-text">
            Finished. The model is in <code>{run.model}</code>
            {cfg.training_type === "lora" ? " (adapter/ and merged/)" : " (merged/)"}.
          </p>
        )}
        {st.state === "stopped" && <p className="hint">Stopped at step {st.step}. Resume continues from the last checkpoint.</p>}
        {!!st.skipped && <p className="warn-text">{st.skipped} batches were skipped because their loss wasn't a number.</p>}
      </section>

      <div className="chart-grid-2">
        <Section title="Training loss">
          <Chart empty="The first loss shows up after the first steps."
            series={[
              { label: "each logged step", points: train.map((m) => ({ x: m.step, y: Number(m.loss) })), tone: "muted" },
              { label: "smoothed", points: smooth(train.map((m) => ({ x: m.step, y: Number(m.loss) }))), tone: "accent" },
            ]} />
        </Section>
        <Section title={decision ? "Validation: mean AUROC and accuracy" : "Validation loss"}>
          <Chart empty="Validation runs before training starts and at every checkpoint."
            series={decision
              ? [
                  { label: "mean AUROC", points: evals.filter((m) => m.auroc !== undefined).map((m) => ({ x: m.step, y: Number(m.auroc) })), tone: "ok" },
                  { label: "accuracy", points: evals.map((m) => ({ x: m.step, y: Number(m.accuracy) })), tone: "accent" },
                ]
              : [{ label: "validation loss", points: evals.map((m) => ({ x: m.step, y: Number(m.val_loss) })), tone: "ok" }]} />
        </Section>
      </div>

      {evals.length > 0 && (
        <Section title="Validation checks">
          <div className="preview-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Val loss</th>
                  {decision ? <th>Accuracy</th> : <th>Perplexity</th>}
                  {decision && <th>Mean AUROC</th>}
                  {perQuestion.map((q) => <th key={q}>{q}</th>)}
                </tr>
              </thead>
              <tbody>
                {evals.map((m) => (
                  <tr key={m.step}>
                    <td>{m.step === 0 ? "0 (before)" : m.step.toLocaleString()}</td>
                    <td>{num(m.val_loss)}</td>
                    <td>{decision ? num(m.accuracy) : num(m.perplexity, 2)}</td>
                    {decision && <td>{num(m.auroc)}</td>}
                    {perQuestion.map((q) => <td key={q}>{num(m[`auroc_${q}`])}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="hint">Validation uses up to 2,000 rows of the validation set; yes/no questions are asked both ways round and averaged.</p>
        </Section>
      )}

      <Section title="Checkpoints">
        {run.checkpoints.length ? (
          <p className="plain">{run.checkpoints.map((c) => c.replace("step-", "step ").replace(/ 0+(\d)/, " $1")).join(" · ")}</p>
        ) : (
          <p className="hint">Saved every {cfg.checkpoints.every.toLocaleString()} steps; the newest two are kept.</p>
        )}
      </Section>

      <details className="section log-box" open={st.state === "failed"}>
        <summary>Log</summary>
        <pre ref={logRef}>{run.log || "Nothing yet."}</pre>
      </details>
      {error && <p className="field-error">{error}</p>}
    </div>
  );
}
