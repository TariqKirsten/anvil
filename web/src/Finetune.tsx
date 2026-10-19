// The fine-tuning screen: every section, the header, and the Start button that checks the whole form.

import { IconPlayerPlay } from "@tabler/icons-react";
import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { loadForm, prepareRequest, saveForm, validate, type FormState } from "./form";
import { CheckpointSection } from "./sections/Checkpoints";
import { DataSection } from "./sections/Data";
import { GpuSection, pickedGpus } from "./sections/Gpus";
import { HyperSection } from "./sections/Hyperparameters";
import { ModelSection } from "./sections/Model";
import { TrainAs } from "./sections/TrainAs";
import { Topbar } from "./Topbar";
import type { GpuList, PrepareReport, Report } from "./types";
import { Badge } from "./ui";

export function Finetune({ onLogout }: { onLogout: () => void }) {
  const [form, setForm] = useState<FormState>(loadForm);
  const [gpus, setGpus] = useState<GpuList | null>(null);
  const [gpuError, setGpuError] = useState<string | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [dataReady, setDataReady] = useState(false);
  const [prepared, setPrepared] = useState<PrepareReport | null>(null);
  const onDataReady = useCallback((ready: boolean, report: PrepareReport | null) => {
    setDataReady(ready);
    setPrepared(report);
  }, []);
  const [starting, setStarting] = useState(false);
  const [outcome, setOutcome] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => saveForm(form), [form]);

  // Memory use changes all the time, so the GPU list refreshes every 5 seconds.
  useEffect(() => {
    let alive = true;
    const load = () =>
      api<GpuList>("/api/gpus")
        .then((list) => {
          if (!alive) return;
          setGpus(list);
          setGpuError(null);
        })
        .catch((e: Error) => alive && setGpuError(e.message));
    load();
    const timer = setInterval(load, 5000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  const errors = validate(form);
  const update = (change: (f: FormState) => FormState) => {
    setForm(change);
    setOutcome(null);
  };
  const props = { form, update, errors };

  async function start() {
    const chosenModel = form.model.source === "hub" ? form.model.repo : form.model.folder.trim();
    const problem = !gpus?.gpus.length
      ? "There's no GPU to train on."
      : !pickedGpus(form, gpus).length
        ? "Pick at least one GPU."
        : !chosenModel || !report
          ? "Pick a base model first."
          : report.verdict === "bad"
            ? "That base model can't be trained as it is. Pick another one."
            : !form.data.train
              ? "Choose a train dataset first."
              : Object.keys(errors).length
                ? "Fix the fields marked in red first."
                : !dataReady
                  ? "Prepare the data first (the button in the Data section)."
                  : null;
    if (problem) {
      setOutcome({ ok: false, text: problem });
      return;
    }
    setStarting(true);
    try {
      const run = await api<{ id: string }>("/api/runs", { json: { form, prepare: prepareRequest(form) } });
      window.location.hash = `#/runs/${run.id}`;
    } catch (e) {
      setOutcome({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally {
      setStarting(false);
    }
  }

  const found = gpus?.gpus.length ?? 0;
  return (
    <div className="page">
      <Topbar
        onLogout={onLogout}
        right={
          gpus && (
            <>
              <Badge className="runtime">{gpus.runtime.label}</Badge>
              <Badge tone={found ? "ok" : "bad"} title={gpus.runtime.label}>
                {found === 1 ? "1 GPU found" : `${found} GPUs found`}
              </Badge>
            </>
          )
        }
      >
        <input
          className="run-name"
          aria-label="Run name"
          value={form.name}
          aria-invalid={errors.name ? true : undefined}
          onChange={(e) => update((f) => ({ ...f, name: e.target.value }))}
        />
      </Topbar>

      <TrainAs {...props} />
      <ModelSection {...props} onReport={setReport} />
      <GpuSection {...props} list={gpus} error={gpuError} />
      <DataSection {...props} onReady={onDataReady} />
      <HyperSection {...props} cards={Math.max(1, pickedGpus(form, gpus).length)}
        trainItems={!prepared ? null
          : form.trainAs === "decision" && form.decisionMethod === "letters" ? (prepared.items?.train ?? prepared.rows.train)
          : prepared.rows.train} />
      <CheckpointSection {...props}
        types={prepared?.questions ? new Set(Object.values(prepared.questions).map((q) => q.type)) : null} />

      <footer className="footer">
        {outcome && (
          <p role="status" className={outcome.ok ? "ok-text" : "field-error"}>
            {outcome.text}
          </p>
        )}
        <button type="button" className="primary" onClick={start}>
          <IconPlayerPlay size={16} aria-hidden /> {starting ? "Starting…" : "Start training"}
        </button>
      </footer>
    </div>
  );
}
