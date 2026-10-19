// One run, live: progress, speed, the loss and validation charts, which checkpoint is best, the final test, the
// checkpoints kept, the log, and Stop / Resume.

import { IconArrowLeft, IconPlayerPlay, IconPlayerStop, IconTrash } from "@tabler/icons-react";
import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import { Chart } from "./Chart";
import { ExportSection } from "./ExportSection";
import { bytes } from "./format";
import { QuestionScores, questionsOf, type ScoredCheck } from "./QuestionScores";
import { METRIC_LABEL, STATE_LABEL, stateTone, when } from "./Runs";
import { Topbar } from "./Topbar";
import type { RunDetail, RunMetric } from "./types";
import { Badge, Section } from "./ui";

const ACTIVE = new Set(["starting", "loading", "running", "saving"]);

// The "pick best by" setting -> the validation number it ranks by (server/anvil/picking.py).
const PICK_METRIC: Record<string, string> = {
  overall: "overall",
  auroc: "auroc",
  catch_rate: "catch_rate",
  macro_f1: "macro_f1",
  mae: "mae",
  calibration: "ece",
  accuracy: "accuracy",
  val_loss: "val_loss",
};

// The averages a check has for each question type it scored: yes/no, choice and score questions.
const TYPE_MEANS = ["auroc", "macro_f1", "mae"];

function stepOf(checkpoint: string) {
  return Number(checkpoint.replace("step-", ""));
}

// "GPU 1", "GPUs 1 and 0"
function cardNames(cards: number[]) {
  return cards.length === 1 ? `GPU ${cards[0]}` : `GPUs ${cards.slice(0, -1).join(", ")} and ${cards[cards.length - 1]}`;
}

// Where the run is in its passes over the training data: "Epoch 1 of 2 · 40% through it".
function epochLine(st: RunDetail["status"]) {
  const perEpoch = st.steps_per_epoch;
  if (!perEpoch || !st.total) return null;
  const epochs = Math.max(1, Math.ceil(st.total / perEpoch));
  const current = Math.min(epochs, Math.floor(st.step / perEpoch) + 1);
  const pct = (100 * (st.step - (current - 1) * perEpoch)) / perEpoch;
  return `Epoch ${current} of ${epochs} · ${pct > 0 && pct < 1 ? pct.toFixed(1) : Math.round(pct)}% through it`;
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
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const logRef = useRef<HTMLPreElement>(null);
  const active = run ? ACTIVE.has(run.status.state) : true;
  const exporting = !!run?.exports.some((e) => e.state === "starting" || e.state === "running");
  const polling = active || exporting;

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
    const timer = setInterval(load, polling ? 2000 : 15000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [id, polling]);

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

  // Deleting removes the run's folder for good: checkpoints, the finished model, metrics and the log.
  async function remove() {
    setDeleting(true);
    setDeleteError(null);
    try {
      await api(`/api/runs/${id}`, { method: "DELETE" });
      window.location.hash = "#/runs";
    } catch (e) {
      setDeleteError(e instanceof Error ? e.message : String(e));
      setDeleting(false);
      setConfirming(false);
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
  const cards = cfg.gpus ?? [cfg.gpu];
  const epoch = epochLine(st);
  const pct = st.total ? Math.min(100, (st.step / st.total) * 100) : 0;
  // A failed run never gets a finish time, so its last update stands in for one.
  const end = st.finished ?? st.stopped ?? (ACTIVE.has(st.state) ? Date.now() / 1000 : st.updated);
  const elapsed = st.started && end ? end - st.started : null;
  const allowed = `${cfg.checkpoints.falseAlarmPercent ?? 1}%`; // the catch rate's false alarms
  const has = (key: string) => evals.some((m) => typeof m[key] === "number");
  // Overall is worth a column when the data mixes question types; with one type it's that type's average.
  const mixed = (scores: Record<string, unknown>[]) =>
    TYPE_MEANS.filter((k) => scores.some((m) => typeof m[k] === "number")).length > 1;
  const columns: [string, string, string][] = [["val_loss", "Val loss", "Validation loss"]];
  if (decision) {
    columns.push(["accuracy", "Accuracy", "Share of questions answered right"]);
    if (mixed(evals) || metric === "overall")
      columns.push(["overall", "Overall", "Every question's own score, averaged: AUROC for yes/no, macro-F1 for choice, and 1 minus the absolute error over the levels' range for score questions"]);
    if (has("auroc"))
      columns.push(["auroc", "Mean AUROC", "Mean AUROC over the yes/no questions"],
        ["catch_rate", "Catch rate", `Real yes cases caught when only ${allowed} of no cases may be flagged`],
        ["false_alarms", "False alarms", "Share of no cases flagged at 50%"]);
    if (has("macro_f1")) columns.push(["macro_f1", "Macro-F1", "Mean macro-F1 over the choice questions"]);
    if (has("mae"))
      columns.push(["mae", "MAE", "Mean absolute error over the score questions: levels off, on average (lower is better)"]);
    columns.push(["ece", "Calib. error", "Calibration error: confidence vs how often it's right (lower is better)"]);
  } else {
    columns.push(["perplexity", "Perplexity", "e to the validation loss"]);
  }
  // Each question's scores: at the final test, and at every check, newest first.
  const scored: ScoredCheck[] = decision
    ? [
        ...(test ? [{ key: "test", label: `Final test · step ${test.step.toLocaleString()}`, scores: test }] : []),
        ...[...evals].reverse().map((m) => ({
          key: `step-${m.step}`,
          label: `Validation · step ${m.step === 0 ? "0 (before training)" : m.step.toLocaleString()}${m.step === best ? " · best" : ""}`,
          scores: m,
        })),
      ].filter((c) => questionsOf(c.scores))
    : [];
  const preferred = test ? "test" : best != null ? `step-${best}` : evals.length ? `step-${evals[evals.length - 1].step}` : "";

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
              {cfg.model.id} · {decision ? `decision model (${cfg.method === "clef" ? "Clef head" : cfg.method})` : `language model (${cfg.method})`} ·{" "}
              {st.head_m ? `head ${st.head_m}M parameters · ` : ""}
              {cfg.training_type === "lora" ? "LoRA" : "full fine-tuning"} ·{" "}
              {cfg.judge_gpu != null ? `trains on ${cardNames(cards)}, checks on GPU ${cfg.judge_gpu}`
                : cards.length > 1 ? `trains on ${cardNames(cards)}` : `GPU ${cards[0]}`} ·
              started {when(cfg.created)} · {bytes(run.size)} on disk
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
            {!ACTIVE.has(st.state) && !exporting && !confirming && (
              <button type="button" className="danger" onClick={() => setConfirming(true)}>
                <IconTrash size={16} aria-hidden /> Delete
              </button>
            )}
          </div>
        </div>
        {confirming && (
          <div className="confirm-box" role="alertdialog" aria-label="Delete this run?">
            <p>
              Delete <strong>{cfg.name}</strong> for good? Its checkpoints, finished model, metrics and log go too,
              freeing {bytes(run.size)}. This can't be undone.
            </p>
            <div className="confirm-actions">
              <button type="button" className="danger-solid" disabled={deleting} onClick={remove}>
                <IconTrash size={16} aria-hidden /> {deleting ? "Deleting…" : "Delete run"}
              </button>
              <button type="button" disabled={deleting} onClick={() => setConfirming(false)}>Cancel</button>
            </div>
          </div>
        )}
        {deleteError && <p className="error-box">{deleteError}</p>}

        <div className="run-bar">
          <div className="bar big"><div style={{ width: `${pct}%` }} /></div>
          <span>
            {st.total ? `Step ${st.step.toLocaleString()} of ${st.total.toLocaleString()}` : st.stage}
            {ACTIVE.has(st.state) && st.total ? ` · ${st.stage}` : ""}
          </span>
          {epoch && (
            <span>
              {epoch}
              {st.items_per_step ? ` · ${st.items_per_step.toLocaleString()} rows per step` : ""}
              {(st.world ?? 1) > 1 ? ` on ${st.world} cards` : ""}
            </span>
          )}
        </div>

        <div className="stats">
          <div><span className="stat-label">{ACTIVE.has(st.state) ? "Time so far" : "Time taken"}</span><span className="stat-value">{duration(elapsed)}</span></div>
          <div><span className="stat-label">Time left</span><span className="stat-value">{ACTIVE.has(st.state) ? duration(st.eta_s) : "–"}</span></div>
          <div><span className="stat-label">Speed</span><span className="stat-value">{st.tokens_per_s ? `${Math.round(st.tokens_per_s).toLocaleString()} tok/s` : "–"}</span></div>
          <div><span className="stat-label">{cards.length > 1 ? "GPU memory (peak, per card)" : "GPU memory (peak)"}</span><span className="stat-value">{st.peak_mem_gb ? `${st.peak_mem_gb} GB` : "–"}</span></div>
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
        <Section title={metric === "val_loss" ? "Validation loss" : `Validation: ${METRIC_LABEL[metric] ?? metric}`}>
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
              ? [...(mixed([test]) || metric === "overall" ? [["Overall", test.overall]] : []),
                 ["Accuracy", test.accuracy], ["Mean AUROC", test.auroc],
                 [`Catch rate at ${allowed} false alarms`, test.catch_rate], ["False alarms", test.false_alarms],
                 ["Mean macro-F1", test.macro_f1], ["Mean absolute error", test.mae], ["Calibration error", test.ece]]
              : [["Test loss", test.val_loss], ["Perplexity", test.perplexity]]
            ).filter(([, v]) => typeof v === "number").map(([label, v]) => (
              <div key={label as string}>
                <span className="stat-label">{label as string}</span>
                <span className="stat-value">{num(v, label === "Perplexity" ? 2 : 3)}</span>
              </div>
            ))}
          </div>
          <p className="hint">
            The step {test.step.toLocaleString()} checkpoint on {test.rows?.toLocaleString()}{" "}
            {decision ? "questions" : "rows"} from the test set. The test set played no part in picking it, so this is
            the fair number to quote.{decision && " Each question's scores are under By question."}
          </p>
        </Section>
      )}

      {scored.length > 0 && (
        <QuestionScores checks={scored} preferred={preferred} falseAlarmPercent={cfg.checkpoints.falseAlarmPercent ?? 1} />
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
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="hint">
            Up to 2,000 rows of the validation set{decision && cfg.method !== "clef" && "; every question is asked with its options in both orders and averaged"}. The
            highlighted column picks the best checkpoint.
            {decision && " The averages are over the questions of each type; By question has each one's own."}
            {has("auroc") && ` Catch rate: the share of real yes cases caught when only ${allowed} of no cases may be flagged. False alarms: the share of no cases flagged at 50%.`}
            {has("mae") && " MAE: how many levels off a score question's answer is, on average."}
            {decision && " Calibration error: how far the confidence is from how often it's right (lower is better)."}
          </p>
        </Section>
      )}

      {!ACTIVE.has(st.state) && run.versions.length > 0 && (
        <ExportSection run={run} id={id} best={best}
          onStarted={() => api<RunDetail>(`/api/runs/${id}`).then(setRun).catch(() => {})} />
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
