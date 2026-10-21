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
  const num = (key: "every" | "keepBest" | "patience" | "falseAlarmPercent" | "minGain", label: string, step: number,
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
  // The default for this data is the first option; the hint names it without its "(every question)" or "(lowest)".
  const defaultName = options[0].label.replace(/ \(.+\)$/, "");

  return (
    <Section title="Checkpoints">
      <div className="grid-checkpoints">
        {num("every", "Save every (steps)", 50)}
        {num("keepBest", "Keep best", 1)}
        {num("patience", "Stop after no gain for", 1, c.patience === 0 ? "never stops early" : "checks in a row")}
        {num("minGain", "Least gain that counts", 0.001,
          c.minGain === 0 || Number.isNaN(c.minGain) ? "any rise counts" : "in the score's own units")}
        <Field id="pickBy" label="Pick best by" className="wide">
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
        check together. The test set is used once, at the end, on the best checkpoint. A check beats the best only by
        more than the least gain that counts, and only then does the no-gain count start again: the run page shows
        each score with ± its noise, and a gain within that is a toss-up.
      </p>
      {form.trainAs === "decision" && (
        <p className="hint">
          {types ? "Pick best by offers what this data can score." : "Prepare the data to see what it can score."}
          {(!types || types.size > 1) && " Overall averages every question's own score (AUROC for yes/no, macro-F1 for choice, absolute error for score questions), so each question counts the same."}
          {` Not sure? Use the default, ${defaultName}. Validation loss is for when you'll use the model's probabilities, not just its answers.`}
          {yesNo && ` Catch rate: the share of real yes cases caught when at most ${Number.isFinite(c.falseAlarmPercent) ? c.falseAlarmPercent : 1}% of no cases may be flagged. Keep it low for a guard; raise it where a false alarm costs little.`}
          {" "}<a href="#/docs">What each score means</a>, and when to use it.
        </p>
      )}
    </Section>
  );
}
