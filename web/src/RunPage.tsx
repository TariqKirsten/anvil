// One run, live: progress, speed, the loss and validation charts, which checkpoint is best, the final test, the
// checkpoints kept, the log, and Stop / Resume.

import { IconArrowLeft, IconPlayerPlay, IconPlayerStop } from "@tabler/icons-react";
import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import { Chart } from "./Chart";
import { METRIC_LABEL, STATE_LABEL, stateTone, when } from "./Runs";
import { Topbar } from "./Topbar";
import type { RunDetail, RunMetric } from "./types";
import { Badge, Section } from "./ui";

const ACTIVE = new Set(["starting", "loading", "running", "saving"]);

// The "pick best by" setting -> the validation number it ranks by (server/anvil/picking.py).
const PICK_METRIC: Record<string, string> = {
  auroc: "auroc",
  catch_rate: "catch_rate",
  calibration: "ece",
  accuracy: "accuracy",
  val_loss: "val_loss",
};

function stepOf(checkpoint: string) {
  return Number(checkpoint.replace("step-", ""));
}

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
  // One row per step: if a step was ever checked twice (the judge died mid-check), the later check counts.
  const evals = [...new Map(run.metrics.filter((m) => m.type === "eval").map((m) => [m.step, m])).values()]
    .sort((a, b) => a.step - b.step);
  const board = run.leaderboard;
  const best = board.best_step ?? st.best_step ?? null;
  const metric = board.entries.at(-1)?.metric ?? PICK_METRIC[cfg.checkpoints.pickBy] ?? "val_loss";
  const scores = new Map(board.entries.map((e) => [e.step, e.score]));
  const newest = run.checkpoints.length ? Math.max(...run.checkpoints.map(stepOf)) : null;
  const judge = run.judge;
  // Checkpoints the judge hasn't got to yet (not counting the one it's checking right now).
  const waiting = Math.max(0, run.checkpoints.map(stepOf).filter((s) => !scores.has(s)).length - (judge?.state === "checking" ? 1 : 0));
  const test = st.final_test;
  const pct = st.total ? Math.min(100, (st.step / st.total) * 100) : 0;
  const elapsed = st.started ? (st.finished ?? st.stopped ?? (ACTIVE.has(st.state) ? Date.now() / 1000 : st.started)) - st.started : null;
  const columns: [string, string, string][] = decision
    ? [
        ["val_loss", "Val loss", "Validation loss"],
        ["accuracy", "Accuracy", "Share of questions answered right"],
        ["auroc", "Mean AUROC", "Mean AUROC over the yes/no questions"],
        ["catch_rate", "Catch rate", "Real yes cases caught when only 1% of no cases may be flagged"],
        ["false_alarms", "False alarms", "Share of no cases flagged at 50%"],
        ["ece", "Calib. error", "Calibration error: confidence vs how often it's right (lower is better)"],
      ]
    : [
        ["val_loss", "Val loss", "Validation loss"],
        ["perplexity", "Perplexity", "e to the validation loss"],
      ];
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
              {cfg.training_type === "lora" ? "LoRA" : "full fine-tuning"} ·{" "}
              {cfg.judge_gpu != null ? `trains on GPU ${cfg.gpu}, checks on GPU ${cfg.judge_gpu}` : `GPU ${cfg.gpu}`} ·
              started {when(cfg.created)}
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
          <div><span className="stat-label">{ACTIVE.has(st.state) ? "Time so far" : "Time taken"}</span><span className="stat-value">{duration(elapsed)}</span></div>
          <div><span className="stat-label">Time left</span><span className="stat-value">{ACTIVE.has(st.state) ? duration(st.eta_s) : "–"}</span></div>
          <div><span className="stat-label">Speed</span><span className="stat-value">{st.tokens_per_s ? `${Math.round(st.tokens_per_s).toLocaleString()} tok/s` : "–"}</span></div>
          <div><span className="stat-label">GPU memory (peak)</span><span className="stat-value">{st.peak_mem_gb ? `${st.peak_mem_gb} GB` : "–"}</span></div>
        </div>

        {judge && (ACTIVE.has(st.state) || judge.state === "failed") && (
          <p className={judge.state === "failed" ? "warn-text" : "hint"}>
            Checks on GPU {judge.gpu}: {judge.state === "failed" ? judge.error ?? "the judge stopped." : judge.stage}
            {judge.state !== "failed" && waiting > 0 && ` (${waiting} more checkpoint${waiting === 1 ? "" : "s"} waiting)`}
          </p>
        )}
        {judge && judge.state !== "failed" && ACTIVE.has(st.state) && waiting >= 3 && (
          <p className="hint">
            Each check takes longer than training to the next checkpoint, so checks are falling behind. Nothing is lost,
            but early stopping reacts late and the run waits for them at the end. Next time, save less often (a bigger
            "Save every").
          </p>
        )}
        {st.state === "failed" && st.error && <p className="error-box">{st.error}</p>}
        {st.state === "done" && run.model && (
          <p className="ok-text">
            Finished. The model is the step {best?.toLocaleString()} checkpoint, the best by{" "}
            {METRIC_LABEL[metric] ?? metric}. It's in <code>{run.model}</code>
            {cfg.training_type === "lora" ? " (adapter/ and merged/)" : " (merged/)"}.
          </p>
        )}
        {st.early_stop && (
          <p className="hint">Stopped early at step {st.step.toLocaleString()}: {st.early_stop}.</p>
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
        <Section title={`Validation: ${METRIC_LABEL[metric] ?? metric}`}>
          <Chart empty="Validation runs before training starts and at every checkpoint."
            marker={best != null ? { x: best, label: "best" } : undefined}
            series={[{
              label: METRIC_LABEL[metric] ?? metric,
              points: evals.filter((m) => typeof m[metric] === "number").map((m) => ({ x: m.step, y: Number(m[metric]) })),
              tone: "ok",
            }]} />
        </Section>
      </div>

      {test && (
        <Section title="Final test">
          <div className="stats">
            {(decision
              ? [["Accuracy", test.accuracy], ["Mean AUROC", test.auroc], ["Catch rate at 1% false alarms", test.catch_rate],
                 ["False alarms", test.false_alarms], ["Calibration error", test.ece]]
              : [["Test loss", test.val_loss], ["Perplexity", test.perplexity]]
            ).filter(([, v]) => typeof v === "number").map(([label, v]) => (
              <div key={label as string}>
                <span className="stat-label">{label}</span>
                <span className="stat-value">{num(v, label === "Perplexity" ? 2 : 3)}</span>
              </div>
            ))}
          </div>
          <p className="hint">
            The step {test.step.toLocaleString()} checkpoint on {test.rows?.toLocaleString()} rows of the test set. The
            test set played no part in picking it, so this is the fair number to quote.
          </p>
        </Section>
      )}

      {evals.length > 0 && (
        <Section title="Validation checks">
          <div className="preview-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Step</th>
                  {columns.map(([key, label, title]) => (
                    <th key={key} title={title} className={key === metric ? "picked" : undefined}>{label}</th>
                  ))}
                  {perQuestion.map((q) => <th key={q} title={`AUROC for ${q}`}>{q}</th>)}
                </tr>
              </thead>
              <tbody>
                {evals.map((m) => (
                  <tr key={m.step} className={m.step === best ? "best-row" : undefined}>
                    <td>
                      {m.step === 0 ? "0 (before)" : m.step.toLocaleString()}
                      {m.step === best && <Badge tone="ok">best</Badge>}
                    </td>
                    {columns.map(([key]) => (
                      <td key={key} className={key === metric ? "picked" : undefined}>{num(m[key], key === "perplexity" ? 2 : 3)}</td>
                    ))}
                    {perQuestion.map((q) => <td key={q}>{num(m[`auroc_${q}`])}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="hint">
            Up to 2,000 rows of the validation set; yes/no questions are asked both ways round and averaged. The
            highlighted column picks the best checkpoint.
            {decision && " Catch rate: the share of real yes cases caught when only 1 in 100 no cases may be flagged. False alarms: the share of no cases flagged at 50%. Calibration error: how far the confidence is from how often it's right (lower is better)."}
          </p>
        </Section>
      )}

      <Section title="Checkpoints">
        {run.checkpoints.length > 0 && (
          <div className="ckpt-list">
            {run.checkpoints.map(stepOf).map((s) => (
              <span key={s} className={`ckpt${s === best ? " best" : ""}`}>
                <span>step {s.toLocaleString()}</span>
                <span className="ckpt-score">
                  {scores.has(s) ? `${METRIC_LABEL[metric] ?? metric} ${num(scores.get(s))}` : "not checked yet"}
                </span>
                {s === best && <Badge tone="ok">best</Badge>}
                {s === newest && s !== best && <Badge>newest</Badge>}
              </span>
            ))}
          </div>
        )}
        <p className="hint">
          Saved every {cfg.checkpoints.every.toLocaleString()} steps. Kept: the best {cfg.checkpoints.keepBest ?? 3}, the
          newest (Resume starts there), and any not checked yet.{" "}
          {cfg.checkpoints.patience
            ? `Training stops early if the ${METRIC_LABEL[metric] ?? metric} doesn't improve for ${cfg.checkpoints.patience} checks in a row.`
            : "Early stopping is off."}
        </p>
      </Section>

      <details className="section log-box" open={st.state === "failed"}>
        <summary>Log</summary>
        <pre ref={logRef}>{run.log || "Nothing yet."}</pre>
      </details>
      {error && <p className="field-error">{error}</p>}
    </div>
  );
}
