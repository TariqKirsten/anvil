import { IconChevronDown, IconChevronUp } from "@tabler/icons-react";
import { useState } from "react";
import type { FormState, SectionProps } from "../form";
import { Field, NumberInput, Section } from "../ui";

type Hyper = FormState["hyper"];

export function HyperSection({ form, update, errors }: SectionProps) {
  const [advanced, setAdvanced] = useState(false);
  const h = form.hyper;
  const lora = form.trainingType !== "full";
  const setH = (change: Partial<Hyper>) => update((f) => ({ ...f, hyper: { ...f.hyper, ...change } }));
  const num = (key: keyof Hyper, label: string, step = 1, hint?: string) => (
    <Field id={key} label={label} error={errors[key]} hint={hint}>
      <NumberInput id={key} value={h[key] as number} step={step} invalid={!!errors[key]}
        onChange={(value) => setH({ [key]: value } as Partial<Hyper>)} />
    </Field>
  );

  return (
    <Section title="Hyperparameters">
      <div className="grid-4">
        <Field id="learningRate" label="Learning rate" error={errors.learningRate}>
          <input id="learningRate" value={h.learningRate} aria-invalid={errors.learningRate ? true : undefined}
            onChange={(e) => setH({ learningRate: e.target.value })} />
        </Field>
        {num("epochs", "Epochs")}
        {num("batchSize", "Batch size")}
        {lora && num("loraRank", "LoRA rank")}
      </div>
      <button type="button" className="link toggle" aria-expanded={advanced} onClick={() => setAdvanced((a) => !a)}>
        {advanced ? <IconChevronUp size={16} aria-hidden /> : <IconChevronDown size={16} aria-hidden />} Advanced
      </button>
      {advanced ? (
        <div className="grid-4">
          {num("warmupPercent", "Warmup %")}
          <Field id="scheduler" label="Schedule">
            <select id="scheduler" value={h.scheduler}
              onChange={(e) => setH({ scheduler: e.target.value as Hyper["scheduler"] })}>
              <option value="cosine">cosine</option>
              <option value="linear">linear</option>
              <option value="constant">constant</option>
            </select>
          </Field>
          {num("weightDecay", "Weight decay", 0.01)}
          {num("gradAccum", "Gradient accumulation", 1, "batches per step")}
          {num("maxLength", "Max length (tokens)", 128)}
          {num("seed", "Seed")}
          <Field id="precision" label="Precision">
            <select id="precision" value={h.precision}
              onChange={(e) => setH({ precision: e.target.value as Hyper["precision"] })}>
              <option value="bf16">bf16</option>
              <option value="fp16">fp16</option>
            </select>
          </Field>
          <Field id="gradCheckpointing" label="Gradient checkpointing" hint="less memory, a bit slower">
            <select id="gradCheckpointing" value={h.gradCheckpointing ? "on" : "off"}
              onChange={(e) => setH({ gradCheckpointing: e.target.value === "on" })}>
              <option value="on">on</option>
              <option value="off">off</option>
            </select>
          </Field>
          {lora && num("loraAlpha", "LoRA alpha", 1, "usually 2 × rank")}
          {num("maxSteps", "Max steps", 1, "0 = all epochs; set it for quick test runs")}
        </div>
      ) : (
        <p className="hint">
          Warmup {h.warmupPercent}% · {h.scheduler} · weight decay {h.weightDecay} · max length {h.maxLength} ·{" "}
          {h.precision} · seed {h.seed}{h.maxSteps ? ` · stops after ${h.maxSteps} steps` : ""}
        </p>
      )}
    </Section>
  );
}
