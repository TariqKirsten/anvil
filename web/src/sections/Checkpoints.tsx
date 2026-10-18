import { PICK_BY, type FormState, type SectionProps } from "../form";
import { Field, NumberInput, Section } from "../ui";

type Checkpoints = FormState["checkpoints"];

export function CheckpointSection({ form, update, errors }: SectionProps) {
  const c = form.checkpoints;
  const options = PICK_BY[form.trainAs];
  const setC = (change: Partial<Checkpoints>) =>
    update((f) => ({ ...f, checkpoints: { ...f.checkpoints, ...change } }));
  const num = (key: "every" | "keepBest" | "patience", label: string, step: number, hint?: string) => (
    <Field id={key} label={label} error={errors[key]} hint={hint}>
      <NumberInput id={key} value={c[key]} step={step} invalid={!!errors[key]} onChange={(v) => setC({ [key]: v })} />
    </Field>
  );

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
      </div>
      <p className="hint">
        Each checkpoint is scored on the validation set, and the best one becomes the model. With a spare GPU the
        checks run there, so training doesn't pause for them. The test set is used once, at the end, on the best
        checkpoint.
      </p>
    </Section>
  );
}
