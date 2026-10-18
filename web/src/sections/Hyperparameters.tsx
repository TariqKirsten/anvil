import { IconChevronDown, IconChevronUp } from "@tabler/icons-react";
import { useState } from "react";
import type { FormState, SectionProps } from "../form";
import { Field, NumberInput, Section } from "../ui";

type Hyper = FormState["hyper"];

// What one step and one epoch mean for these settings: each card takes a batch, so a step learns from
// batch size × cards × gradient accumulation rows, and an epoch is one pass over all of train.
function percent(part: number, whole: number) {
  const p = (100 * part) / whole;
  return `${p < 1 ? p.toFixed(1) : Math.round(p)}%`;
}

export function HyperSection({ form, update, errors, cards, trainItems }: SectionProps & {
  cards: number;
  trainItems: number | null;
}) {
  const [advanced, setAdvanced] = useState(false);
  const h = form.hyper;
  const clef = form.trainAs === "decision" && form.decisionMethod === "clef";
  const perStep = h.batchSize * cards * h.gradAccum;
  const stepsPerEpoch = trainItems && perStep > 0 ? Math.ceil(trainItems / perStep) : null;
  const epochHint = stepsPerEpoch
    ? h.epochs === 1
      ? `1 epoch = ${stepsPerEpoch.toLocaleString()} steps`
      : `${h.epochs} epochs = ${(stepsPerEpoch * h.epochs).toLocaleString()} steps`
    : "passes over the training data";
  const cut = h.maxSteps && stepsPerEpoch ? `${percent(h.maxSteps, stepsPerEpoch)} of an epoch` : null;
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
        {num("epochs", "Epochs", 1, epochHint)}
        {num("batchSize", cards > 1 ? "Batch size (per card)" : "Batch size", 1,
          `${perStep.toLocaleString()} rows per step${cards > 1 ? ` on ${cards} cards` : ""}`)}
        {lora && num("loraRank", "LoRA rank")}
        {clef && (
          <Field id="headLearningRate" label="Head learning rate" error={errors.headLearningRate}
            hint="the new head; it trains alone for the first 10% of steps">
            <input id="headLearningRate" value={h.headLearningRate}
              aria-invalid={errors.headLearningRate ? true : undefined}
              onChange={(e) => setH({ headLearningRate: e.target.value })} />
          </Field>
        )}
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
          {num("maxLength", "Max length (tokens)", 128, clef ? "the whole row: its state and every question" : undefined)}
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
          {num("maxSteps", "Max steps", 1, cut ? `${cut}; 0 = all epochs` : "0 = all epochs; set it for quick test runs")}
        </div>
      ) : (
        <p className="hint">
          Warmup {h.warmupPercent}% · {h.scheduler} · weight decay {h.weightDecay} · max length {h.maxLength} ·{" "}
          {h.precision} · seed {h.seed}{h.maxSteps ? ` · stops after ${h.maxSteps} steps${cut ? ` (${cut})` : ""}` : ""}
        </p>
      )}
    </Section>
  );
}
