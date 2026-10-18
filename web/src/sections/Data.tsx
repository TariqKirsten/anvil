// Train, validation and test data. Validation and test are optional: whatever you don't provide is cut from train.
// Duplicates are removed by default, and rows that share an input stay on one side of the split.
// Choosing files arrives in step 2; the settings here already work.

import { IconUpload } from "@tabler/icons-react";
import { useState } from "react";
import type { FormState, SectionProps } from "../form";
import { Field, NumberInput, Section, Segmented } from "../ui";

const FILES = [
  { key: "train", label: "Train", placeholder: "Choose a file" },
  { key: "validation", label: "Validation (optional)", placeholder: "Optional" },
  { key: "test", label: "Test (optional)", placeholder: "Optional" },
];

export function DataSection({ form, update, errors }: SectionProps) {
  const [notice, setNotice] = useState<string | null>(null);
  const decision = form.trainAs === "decision";
  const d = form.data;
  const setData = (change: Partial<FormState["data"]>) => update((f) => ({ ...f, data: { ...f.data, ...change } }));
  const input = decision ? "a state" : "a prompt";

  return (
    <Section
      title="Data"
      aside={<span className="hint">{decision ? "Columns: state, questions, gold" : "Columns: messages or text"}</span>}
    >
      <div className="grid-3">
        {FILES.map((file) => (
          <div className="field" key={file.key}>
            <span className="label">{file.label}</span>
            <button type="button" className="drop" onClick={() => setNotice("Choosing datasets arrives in step 2.")}>
              <IconUpload size={16} aria-hidden /> {file.placeholder}
            </button>
          </div>
        ))}
      </div>
      {notice && <p className="hint notice">{notice}</p>}

      <div className="panel">
        <div className="panel-row">
          <span className="panel-label">Duplicates</span>
          <Segmented
            label="Duplicates"
            value={d.dedupe ? "remove" : "keep"}
            options={[
              { value: "remove", label: "Remove" },
              { value: "keep", label: "Keep" },
            ]}
            onChange={(v) => setData({ dedupe: v === "remove" })}
          />
          {d.dedupe && (
            <select
              aria-label="What counts as a duplicate"
              className="auto"
              value={d.nearCopies ? "near" : "exact"}
              onChange={(e) => setData({ nearCopies: e.target.value === "near" })}
            >
              <option value="near">exact and near-copies</option>
              <option value="exact">exact copies only</option>
            </select>
          )}
        </div>
        <div className="panel-row">
          <span className="panel-label">Split from train</span>
          <div className="split-fields">
            <Field id="valPercent" label="Validation %" error={errors.valPercent}>
              <NumberInput id="valPercent" value={d.valPercent} min={0} invalid={!!errors.valPercent}
                onChange={(valPercent) => setData({ valPercent })} />
            </Field>
            <Field id="testPercent" label="Test %" error={errors.testPercent}>
              <NumberInput id="testPercent" value={d.testPercent} min={0} invalid={!!errors.testPercent}
                onChange={(testPercent) => setData({ testPercent })} />
            </Field>
            <Field id="keepTogether" label="Keep together">
              <select
                id="keepTogether"
                value={d.keepTogether}
                onChange={(e) => setData({ keepTogether: e.target.value as FormState["data"]["keepTogether"] })}
              >
                <option value="input">{decision ? "same state" : "same prompt"}</option>
                <option value="source">same source</option>
                <option value="nothing">nothing</option>
              </select>
            </Field>
          </div>
        </div>
        <div className="panel-notes">
          <p>Only what you don't provide is cut from train. A validation or test file you give is used as it is.</p>
          {d.dedupe ? (
            <p>
              {d.nearCopies ? "Exact and near-copies" : "Exact copies"} are removed from train, and validation or test
              rows that also appear in train are dropped.
            </p>
          ) : (
            <p className="warn-text">Duplicates are kept, so scores can look better than the model really is.</p>
          )}
          {d.keepTogether === "nothing" && (
            <p className="warn-text">Rows that share {input} can land on both sides of the split.</p>
          )}
        </div>
      </div>
    </Section>
  );
}
