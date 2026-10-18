// The fine-tuning screen: every section, the header, and the Start button that checks the whole form.

import { IconPlayerPlay } from "@tabler/icons-react";
import { useEffect, useState } from "react";
import { api } from "./api";
import { loadForm, saveForm, validate, type FormState } from "./form";
import { CheckpointSection } from "./sections/Checkpoints";
import { DataSection } from "./sections/Data";
import { GpuSection, pickedGpus } from "./sections/Gpus";
import { HyperSection } from "./sections/Hyperparameters";
import { ModelSection } from "./sections/Model";
import { TrainAs } from "./sections/TrainAs";
import { Topbar } from "./Topbar";
import type { GpuList, Report } from "./types";
import { Badge } from "./ui";

export function Finetune({ onLogout }: { onLogout: () => void }) {
  const [form, setForm] = useState<FormState>(loadForm);
  const [gpus, setGpus] = useState<GpuList | null>(null);
  const [gpuError, setGpuError] = useState<string | null>(null);
  const [report, setReport] = useState<Report | null>(null);
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

  function start() {
    const chosenModel = form.model.source === "hub" ? form.model.repo : form.model.folder.trim();
    const problem = !gpus?.gpus.length
      ? "There's no GPU to train on."
      : !pickedGpus(form, gpus).length
        ? "Pick at least one GPU."
        : !chosenModel || !report
          ? "Pick a base model first."
          : report.verdict === "bad"
            ? "That base model can't be trained as it is. Pick another one."
            : Object.keys(errors).length
              ? "Fix the fields marked in red first."
              : null;
    setOutcome(problem ? { ok: false, text: problem } : { ok: true, text: "Everything checks out. Training arrives in step 3." });
  }

  const found = gpus?.gpus.length ?? 0;
  return (
    <div className="page">
      <Topbar
        onLogout={onLogout}
        right={
          gpus && (
            <>
              <Badge>{gpus.runtime.label}</Badge>
              <Badge tone={found ? "ok" : "bad"}>{found === 1 ? "1 GPU found" : `${found} GPUs found`}</Badge>
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
      <DataSection {...props} />
      <HyperSection {...props} />
      <CheckpointSection {...props} />

      <footer className="footer">
        {outcome && (
          <p role="status" className={outcome.ok ? "ok-text" : "field-error"}>
            {outcome.text}
          </p>
        )}
        <button type="button" className="primary" onClick={start}>
          <IconPlayerPlay size={16} aria-hidden /> Start training
        </button>
      </footer>
    </div>
  );
}
