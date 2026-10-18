// Everything the fine-tuning screen asks for, its defaults, and the checks it must pass before a run can start.
// The form is kept in this browser's localStorage, so a reload doesn't lose it.

import type { DataSource } from "./types";

export type TrainAs = "lm" | "decision";

export interface FormState {
  name: string;
  trainAs: TrainAs;
  lmFormat: "chat" | "text";
  decisionMethod: "clef" | "letters";
  trainingType: "lora" | "qlora" | "full";
  model: { source: "hub" | "folder"; repo: string | null; folder: string };
  // null until you pick: then every card that isn't busy is used.
  gpus: number[] | null;
  // A free card you didn't tick checks the checkpoints. Off: the training cards do, and that card stays free.
  spareChecks: boolean;
  data: {
    train: DataSource | null;
    validation: DataSource | null;
    test: DataSource | null;
    dedupe: boolean;
    nearCopies: boolean;
    valPercent: number;
    testPercent: number;
    keepTogether: "input" | "source" | "nothing";
  };
  hyper: {
    learningRate: string;
    headLearningRate: string; // Clef's head only
    epochs: number;
    batchSize: number;
    loraRank: number;
    loraAlpha: number;
    warmupPercent: number;
    scheduler: "cosine" | "linear" | "constant";
    weightDecay: number;
    gradAccum: number;
    maxLength: number;
    seed: number;
    precision: "bf16" | "fp16";
    gradCheckpointing: boolean;
    maxSteps: number;
  };
  checkpoints: { every: number; keepBest: number; patience: number; pickBy: string };
}

// What each section gets: the form, and one way to change it.
export interface SectionProps {
  form: FormState;
  update: (change: (form: FormState) => FormState) => void;
  errors: Errors;
}

// The same names as server/anvil/picking.py.
export const PICK_BY: Record<TrainAs, { value: string; label: string }[]> = {
  decision: [
    { value: "auroc", label: "Mean AUROC" },
    { value: "catch_rate", label: "Catch rate at 1% false alarms" },
    { value: "calibration", label: "Calibration (lowest error)" },
    { value: "accuracy", label: "Accuracy" },
    { value: "val_loss", label: "Validation loss" },
  ],
  lm: [{ value: "val_loss", label: "Validation loss" }],
};

// The model's learning rate that suits each way of training. Clef from a plain model learned nothing at letters'
// 2e-4 (the model's features moved faster than the new head could follow); 5e-5 worked.
export function defaultLearningRate(form: Pick<FormState, "trainAs" | "decisionMethod">) {
  return form.trainAs === "decision" && form.decisionMethod === "clef" ? "5e-5" : "2e-4";
}

// Switching the way of training moves the learning rate to the new default, unless you've set your own.
export function withMethod(form: FormState, change: Partial<Pick<FormState, "trainAs" | "decisionMethod">>): FormState {
  const next = { ...form, ...change };
  const untouched = form.hyper.learningRate.trim() === defaultLearningRate(form);
  return untouched ? { ...next, hyper: { ...next.hyper, learningRate: defaultLearningRate(next) } } : next;
}

export const DEFAULTS: FormState = {
  name: "new-run",
  trainAs: "decision",
  lmFormat: "chat",
  decisionMethod: "letters", // the measured baseline; Clef becomes the default if it wins a bake-off
  trainingType: "lora",
  model: { source: "hub", repo: null, folder: "" },
  gpus: null,
  spareChecks: true,
  data: {
    train: null,
    validation: null,
    test: null,
    dedupe: true,
    nearCopies: true,
    valPercent: 5,
    testPercent: 5,
    keepTogether: "input",
  },
  hyper: {
    learningRate: "2e-4",
    headLearningRate: "1e-4",
    epochs: 1,
    batchSize: 8,
    loraRank: 32,
    loraAlpha: 64,
    warmupPercent: 3,
    scheduler: "cosine",
    weightDecay: 0.01,
    gradAccum: 1,
    maxLength: 2048,
    seed: 3407,
    precision: "bf16",
    gradCheckpointing: true,
    maxSteps: 0,
  },
  checkpoints: { every: 500, keepBest: 3, patience: 3, pickBy: "auroc" },
};

const KEY = "anvil.finetune.v1";

export function loadForm(): FormState {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) ?? "null");
    if (!saved || typeof saved !== "object") return DEFAULTS;
    // Saved forms from older versions miss newer fields, so fill them from the defaults.
    return {
      ...DEFAULTS,
      ...saved,
      model: { ...DEFAULTS.model, ...saved.model },
      data: { ...DEFAULTS.data, ...saved.data },
      hyper: { ...DEFAULTS.hyper, ...saved.hyper },
      checkpoints: { ...DEFAULTS.checkpoints, ...saved.checkpoints },
    };
  } catch {
    return DEFAULTS;
  }
}

export function saveForm(form: FormState) {
  try {
    localStorage.setItem(KEY, JSON.stringify(form));
  } catch {
    // Private windows and blocked storage: the form just isn't remembered.
  }
}

// What the server needs to prepare the data (see server/anvil/datasets.py). The same request always gives the same
// result, so it doubles as the key for "is the prepared data still current?".
export function prepareRequest(form: FormState) {
  const d = form.data;
  if (!d.train) return null;
  const source = (s: DataSource | null) =>
    s && (s.kind === "file" ? { kind: s.kind, path: s.path } : { kind: s.kind, repo: s.repo, config: s.config, split: s.split });
  return {
    train: source(d.train),
    validation: source(d.validation),
    test: source(d.test),
    dedupe: d.dedupe,
    near_copies: d.nearCopies,
    val_percent: d.valPercent,
    test_percent: d.testPercent,
    keep_together: d.keepTogether,
    seed: form.hyper.seed,
  };
}

export type Errors = Partial<Record<string, string>>;

function wholeNumber(value: number, min: number, max: number) {
  return Number.isInteger(value) && value >= min && value <= max;
}

// Field errors, keyed by the field's id on the screen.
export function validate(form: FormState): Errors {
  const errors: Errors = {};
  const h = form.hyper;
  const lr = Number(h.learningRate);
  if (!h.learningRate.trim() || !Number.isFinite(lr) || lr <= 0 || lr >= 1)
    errors.learningRate = "Use a number between 0 and 1, like 2e-4.";
  const headLr = Number(h.headLearningRate);
  if (form.trainAs === "decision" && form.decisionMethod === "clef" &&
      (!h.headLearningRate.trim() || !Number.isFinite(headLr) || headLr <= 0 || headLr >= 1))
    errors.headLearningRate = "Use a number between 0 and 1, like 1e-4.";
  if (!wholeNumber(h.epochs, 1, 100)) errors.epochs = "Use a whole number from 1 to 100.";
  if (!wholeNumber(h.batchSize, 1, 4096)) errors.batchSize = "Use a whole number of at least 1.";
  if (form.trainingType !== "full" && !wholeNumber(h.loraRank, 1, 1024)) errors.loraRank = "Use 1 to 1024.";
  if (form.trainingType !== "full" && !(h.loraAlpha > 0)) errors.loraAlpha = "Use a number above 0.";
  if (!(h.warmupPercent >= 0 && h.warmupPercent <= 50)) errors.warmupPercent = "Use 0 to 50%.";
  if (!(h.weightDecay >= 0 && h.weightDecay <= 1)) errors.weightDecay = "Use 0 to 1.";
  if (!wholeNumber(h.gradAccum, 1, 1024)) errors.gradAccum = "Use a whole number of at least 1.";
  if (!wholeNumber(h.maxLength, 16, 1_000_000)) errors.maxLength = "Use at least 16 tokens.";
  if (!wholeNumber(h.seed, 0, 2 ** 31 - 1)) errors.seed = "Use a whole number of 0 or more.";
  if (!wholeNumber(h.maxSteps, 0, 100_000_000)) errors.maxSteps = "Use 0 (no limit) or a whole number of steps.";
  const d = form.data;
  if (!(d.valPercent >= 0 && d.valPercent <= 40)) errors.valPercent = "Use 0 to 40%.";
  if (!(d.testPercent >= 0 && d.testPercent <= 40)) errors.testPercent = "Use 0 to 40%.";
  const c = form.checkpoints;
  if (!wholeNumber(c.every, 10, 1_000_000)) errors.every = "Use at least 10 steps.";
  if (!wholeNumber(c.keepBest, 1, 100)) errors.keepBest = "Use 1 to 100.";
  if (!wholeNumber(c.patience, 0, 1000)) errors.patience = "Use 0 (never stop early) or a whole number.";
  if (!form.name.trim()) errors.name = "Give the run a name.";
  return errors;
}
