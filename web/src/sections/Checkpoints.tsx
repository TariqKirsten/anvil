import { useEffect } from "react";
import { pickByOptions, type FormState, type SectionProps } from "../form";
import type { QuestionType } from "../types";
import { Field, NumberInput, Section } from "../ui";

type Checkpoints = FormState["checkpoints"];

// types: the question types in the prepared validation data (null until the data is prepared).
export function CheckpointSection({ form, update, errors, types }: SectionProps & { types: Set<QuestionType> | null }) {
  const c = form.checkpoints;
  const options = pickByOptions(form.trainAs, types, c.falseAlarmPercent);
  const setC = (change: Partial<Checkpoints>) =>
    update((f) => ({ ...f, checkpoints: { ...f.checkpoints, ...change } }));
  const num = (key: "every" | "keepBest" | "patience" | "falseAlarmPercent", label: string, step: number,
    hint?: string) => (
    <Field id={key} label={label} error={errors[key]} hint={hint}>
      <NumberInput id={key} value={c[key]} step={step} invalid={!!errors[key]} onChange={(v) => setC({ [key]: v })} />
    </Field>
  );
  // A setting the data can't score (AUROC without yes/no questions, say) moves to the first one it can.
  const available = options.map((o) => o.value).join(",");
  useEffect(() => {
    if (!options.some((o) => o.value === c.pickBy)) setC({ pickBy: options[0].value });
  }, [available, c.pickBy]);
  const yesNo = form.trainAs === "decision" && (!types || types.has("noul"));

  return (
    <Section title="Checkpoints">
      <div className="grid-4">
        {num("every", "Save every (steps)", 50)}
        {num("keepBest", "Keep best", 1)}
        {num("patience", "Stop after no gain for", 1, c.patience === 0 ? "never stops early" : "checks in a row")}
        <Field id="pickBy" label="Pick best by">
          <select id="pickBy" value={options.some((o) => o.value === c.pickBy) ? c.pickBy : options[0].value}
            onChange={(e) => setC({ pickBy: e.target.value })}>
            {options.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </Field>
        {(yesNo || errors.falseAlarmPercent) && num("falseAlarmPercent", "False alarms allowed (%)", 0.1,
          "for the catch rate")}
      </div>
      <p className="hint">
        Each checkpoint is scored on the validation set, and the best one becomes the model. A free GPU you didn't
        tick does the checks, so training doesn't pause for them; otherwise the training cards pause briefly and
        check together. The test set is used once, at the end, on the best checkpoint.
      </p>
      {form.trainAs === "decision" && (
        <p className="hint">
          {types ? "Pick best by offers what this data can score." : "Prepare the data to see what it can score."}
          {(!types || types.size > 1) && " Overall averages every question's own score (AUROC for yes/no, macro-F1 for choice, absolute error for score questions), so each question counts the same."}
          {yesNo && ` Catch rate: the share of real yes cases caught when at most ${Number.isFinite(c.falseAlarmPercent) ? c.falseAlarmPercent : 1}% of no cases may be flagged. Keep it low for a guard; raise it where a false alarm costs little.`}
        </p>
      )}
    </Section>
  );
}
