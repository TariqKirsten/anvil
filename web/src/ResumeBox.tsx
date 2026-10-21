// What Resume and Keep training ask first: how many epochs the run makes in all (more lets a finished run train on), and
// for Keep training when to stop. A run that Max steps cut short can stay at that length or take all its epochs.

import { IconPlayerPlay } from "@tabler/icons-react";
import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { RunDetail } from "./types";
import { Field, NumberInput, Segmented } from "./ui";

// 1 -> "1 epoch", 1.5 -> "1.5 epochs"
function epochCount(n: number) {
  const text = String(Math.round(n * 100) / 100);
  return `${text} epoch${text === "1" ? "" : "s"}`;
}

export function ResumeBox({ run, id, mode, newest, onStarted, onCancel }: {
  run: RunDetail;
  id: string;
  mode: "resume" | "continue"; // Resume a stopped or failed run, or Keep training a finished one
  newest: number | null; // the newest checkpoint's step, where it picks up
  onStarted: () => Promise<void>;
  onCancel: () => void;
}) {
  const { config: cfg, status: st } = run;
  const had = Number(cfg.hyper.epochs) || 1;
  const maxSteps = Number(cfg.hyper.maxSteps) || 0;
  const perEpoch = st.steps_per_epoch ?? null;
  const from = newest ?? 0;
  // A finished run that trained all its steps (rather than stopping early before them) needs more of them: at least the
  // next whole epoch.
  const fewest = perEpoch ? Math.floor(from / perEpoch) + 1 : 1;
  const trainedAll = mode === "continue" && !(st.early_stop && st.total && st.step < st.total);
  const [length, setLength] = useState<"steps" | "epochs">(maxSteps && !trainedAll ? "steps" : "epochs");
  // Epochs that would end at or before where it is (all of them trained, or Max steps past them) start at the next one.
  const [epochs, setEpochs] = useState(perEpoch && had * perEpoch <= from ? Math.max(had, fewest) : had);
  const [toEnd, setToEnd] = useState(false);
  const [more, setMore] = useState(3);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const boxRef = useRef<HTMLFormElement>(null);

  // Focus goes into the box, to the epochs or else the button that starts it, so Enter starts it and Escape closes it.
  useEffect(() => {
    const box = boxRef.current;
    (box?.querySelector<HTMLElement>("input") ?? box?.querySelector<HTMLElement>("button[type=submit]"))?.focus();
  }, []);

  // The epochs are sent only when they change the run's length: they also lift a Max steps limit.
  const newLength = length === "epochs" && (epochs !== had || maxSteps > 0);
  const total = newLength && perEpoch ? perEpoch * epochs : st.total ?? null;
  // Resuming a run stopped at its last step just finishes it, but a new length, or Keep training, needs steps to go.
  const lengthError = length === "epochs" && !(Number.isInteger(epochs) && epochs >= 1 && epochs <= 100)
    ? "Use a whole number from 1 to 100."
    : (newLength || trainedAll) && total != null && total <= from
      ? length === "epochs" && perEpoch
        ? `It has trained ${epochCount(from / perEpoch)} already: give it at least ${fewest}.`
        : `It has trained all its ${from.toLocaleString()} steps: pick Epochs to give it more.`
      : undefined;
  const moreError = mode === "continue" && !toEnd && !(Number.isInteger(more) && more >= 1 && more <= 1000)
    ? "Use a whole number from 1 to 1000, or pick To the end." : undefined;
  const epochsHint = perEpoch
    ? `${perEpoch.toLocaleString()} steps each${total ? `: it ends at step ${total.toLocaleString()}` : ""}${from ? `, and has trained ${epochCount(from / perEpoch)}` : ""}.`
    : "Passes over the training data, in all.";
  // Checks fall on every `every` steps, and early stopping counts the ones after where training picks up (or after the
  // model has warmed up, if that's later), so with no gain the stop comes `more` checks after that.
  const every = cfg.checkpoints.every;
  const stopsAt = (Math.floor(Math.max(from, st.warm_until ?? 0) / every) + more) * every;
  const moreHint = total != null && stopsAt < total
    ? `A check every ${every.toLocaleString()} steps: with no gain, it stops around step ${stopsAt.toLocaleString()}. A gain starts the count again.`
    : `A check every ${every.toLocaleString()} steps: that many reaches the last step, so it trains to the end either way.`;
  const invalid = !!lengthError || !!moreError;

  async function start() {
    setBusy(true);
    setError(null);
    const body = { ...(mode === "continue" ? { patience: toEnd ? 0 : more } : {}), ...(newLength ? { epochs } : {}) };
    try {
      await api(`/api/runs/${id}/resume`, Object.keys(body).length ? { json: body } : { method: "POST" });
      await onStarted();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  const where = newest != null ? `the newest checkpoint (step ${newest.toLocaleString()})` : null;
  const towards = total ? ` towards step ${total.toLocaleString()}` : "";
  return (
    <form ref={boxRef} className="resume-box" aria-label={mode === "resume" ? "Resume" : "Keep training"}
      onKeyDown={(e) => e.key === "Escape" && !busy && onCancel()}
      onSubmit={(e) => {
        e.preventDefault();
        if (!invalid && !busy) start();
      }}>
      <p>
        {mode === "resume"
          ? where ? `Resume from ${where}${towards}.` : `Resume from the beginning${towards}: there's no checkpoint yet.`
          : `Train on from ${where ?? "the newest checkpoint"}${towards}. The finished model is picked again at the end, from every check so far.`}
      </p>
      {maxSteps > 0 && (
        <div className="field">
          <span className="label">Length</span>
          <Segmented label="Length" value={length} onChange={setLength}
            options={[{ value: "steps", label: `Max steps (${maxSteps.toLocaleString()})` }, { value: "epochs", label: "Epochs" }]} />
        </div>
      )}
      {length === "epochs" ? (
        <Field id="resume-epochs" label="Epochs" error={lengthError} hint={epochsHint}>
          <NumberInput id="resume-epochs" value={epochs} min={1} step={1} invalid={!!lengthError} onChange={setEpochs} />
        </Field>
      ) : lengthError && <p className="field-error" role="alert">{lengthError}</p>}
      {newLength && !lengthError && newest != null && (
        <p className="hint">
          The learning rate follows its schedule to the new last step. The warmup stays as it was, and where the schedule
          is higher than where the run left off, it climbs back over the warmup's steps instead of jumping.
        </p>
      )}
      {mode === "continue" && (
        <>
          <Segmented label="When to stop" value={toEnd ? "end" : "early"} onChange={(v) => setToEnd(v === "end")}
            options={[{ value: "early", label: "Stop early" }, { value: "end", label: "To the end" }]} />
          {toEnd ? (
            <p className="hint">No early stopping: it trains all the way{total ? ` to step ${total.toLocaleString()}` : ""}.</p>
          ) : (
            <Field id="more-checks" label="More checks without a gain" error={moreError} hint={moreHint}>
              <NumberInput id="more-checks" value={more} min={1} step={1} invalid={!!moreError} onChange={setMore} />
            </Field>
          )}
        </>
      )}
      {error && <p className="field-error" role="alert">{error}</p>}
      <div className="confirm-actions">
        <button type="submit" className="primary" disabled={busy || invalid}>
          <IconPlayerPlay size={16} aria-hidden /> {busy ? "Starting…" : mode === "resume" ? "Resume" : "Keep training"}
        </button>
        <button type="button" disabled={busy} onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}
