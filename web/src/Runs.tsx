// Every run, newest first. Running ones refresh on their own.

import { IconPlus } from "@tabler/icons-react";
import { useEffect, useState } from "react";
import { api } from "./api";
import { bytes } from "./format";
import { Topbar } from "./Topbar";
import type { RunState, RunSummary } from "./types";
import { Badge } from "./ui";

export const STATE_LABEL: Record<RunState, string> = {
  starting: "Starting",
  loading: "Loading",
  running: "Training",
  saving: "Saving",
  done: "Done",
  failed: "Failed",
  stopped: "Stopped",
};

// Validation numbers by the name the server gives them.
export const METRIC_LABEL: Record<string, string> = {
  overall: "overall score",
  auroc: "mean AUROC",
  catch_rate: "catch rate",
  macro_f1: "mean macro-F1",
  mae: "mean absolute error",
  ece: "calibration error",
  accuracy: "accuracy",
  val_loss: "validation loss",
};

// Shorter names for the list of runs, where space is tight.
const SHORT_LABEL: Record<string, string> = {
  overall: "overall",
  auroc: "AUROC",
  catch_rate: "catch rate",
  macro_f1: "macro-F1",
  mae: "MAE",
  ece: "calib. error",
  accuracy: "accuracy",
  val_loss: "val loss",
};

export function stateTone(state: RunState): "ok" | "warn" | "bad" | "accent" | "neutral" {
  if (state === "done") return "ok";
  if (state === "failed") return "bad";
  if (state === "stopped") return "warn";
  return "accent";
}

export function when(seconds: number | null | undefined) {
  if (!seconds) return "";
  return new Date(seconds * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function Runs({ onLogout }: { onLogout: () => void }) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () =>
      api<{ runs: RunSummary[] }>("/api/runs")
        .then((r) => alive && setRuns(r.runs))
        .catch((e: Error) => alive && setError(e.message));
    load();
    const timer = setInterval(load, 5000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <div className="page">
      <Topbar onLogout={onLogout} />
      <div className="page-head">
        <h1>Runs</h1>
        <a className="button-link" href="#/">
          <IconPlus size={16} aria-hidden /> New run
        </a>
      </div>
      {error && <p className="field-error">{error}</p>}
      {runs && runs.length === 0 && (
        <div className="section empty">
          <p className="plain">No runs yet. Set one up under New run, and it shows up here.</p>
        </div>
      )}
      {runs && runs.length > 0 && (
        <div className="section run-list">
          {runs.map((r) => {
            const pct = r.total ? Math.min(100, Math.round((r.step / r.total) * 100)) : 0;
            const best = r.best && r.best.score !== null ? r.best : null;
            return (
              <a key={r.id} className="run-row" href={`#/runs/${r.id}`}>
                <span className="run-main">
                  <span className="run-title">{r.name}</span>
                  <span className="run-meta">
                    {r.model} · {r.train_as === "decision" ? "decision" : "language"} model ·{" "}
                    {r.gpus.length > 1 ? `GPUs ${r.gpus.join(", ")}` : `GPU ${r.gpu}`} · {when(r.created)} ·{" "}
                    {bytes(r.size)}
                  </span>
                </span>
                <span className="run-progress">
                  <span className="run-steps">
                    {r.total ? `${r.step.toLocaleString()} / ${r.total.toLocaleString()} steps` : r.stage}
                  </span>
                  <span className="bar"><span style={{ width: `${pct}%` }} /></span>
                  {best && (
                    <span className="run-steps">
                      best {SHORT_LABEL[best.metric] ?? best.metric} {best.score!.toFixed(3)} · step {best.step.toLocaleString()}
                    </span>
                  )}
                </span>
                <Badge tone={stateTone(r.state)}>{STATE_LABEL[r.state] ?? r.state}</Badge>
              </a>
            );
          })}
        </div>
      )}
    </div>
  );
}
