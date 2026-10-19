// Train, validation and test data. Validation and test are optional: whatever you don't provide is cut from train.
// Choosing a dataset shows its size, format and first rows at once. "Prepare data" then checks every row, removes
// duplicates (by default), keeps rows that share an input on one side of the split, and reports what it did.

import { IconCircleCheck, IconDatabase, IconFile, IconX } from "@tabler/icons-react";
import { useEffect, useRef, useState } from "react";
import { api, isAbort } from "../api";
import { prepareRequest, type FormState, type SectionProps } from "../form";
import type { DataSource, DatasetInfo, PrepareJob, PrepareReport, Slot } from "../types";
import { Field, NumberInput, Section, Segmented } from "../ui";
import { DataPicker } from "./DataPicker";

const SLOTS: { slot: Slot; label: string; empty: string }[] = [
  { slot: "train", label: "Train", empty: "Choose a dataset" },
  { slot: "validation", label: "Validation (optional)", empty: "Optional" },
  { slot: "test", label: "Test (optional)", empty: "Optional" },
];

const LM_FORMATS = ["chat", "pairs", "text"];

function message(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

type Look = DatasetInfo | { error: string } | "loading";

export function DataSection({ form, update, errors, onReady }: SectionProps & {
  onReady: (ready: boolean, report: PrepareReport | null) => void;
}) {
  const d = form.data;
  const decision = form.trainAs === "decision";
  const [picking, setPicking] = useState<Slot | null>(null);
  const [looks, setLooks] = useState<Partial<Record<Slot, Look>>>({});
  const setData = (change: Partial<FormState["data"]>) => update((f) => ({ ...f, data: { ...f.data, ...change } }));
  const choose = (slot: Slot, source: DataSource | null) => setData({ [slot]: source } as Partial<FormState["data"]>);

  // A quick look at every chosen dataset: rows, format, first rows.
  const sources = JSON.stringify([d.train, d.validation, d.test]);
  useEffect(() => {
    const controller = new AbortController();
    for (const { slot } of SLOTS) {
      const source = d[slot];
      if (!source) {
        setLooks((l) => ({ ...l, [slot]: undefined }));
        continue;
      }
      setLooks((l) => ({ ...l, [slot]: "loading" }));
      api<DatasetInfo>("/api/datasets/inspect", { json: { source }, signal: controller.signal })
        .then((info) => setLooks((l) => ({ ...l, [slot]: info })))
        .catch((e) => !isAbort(e) && setLooks((l) => ({ ...l, [slot]: { error: message(e) } })));
    }
    return () => controller.abort();
  }, [sources]); // d is read through `sources`, its JSON form, so new objects with the same content don't refetch

  const train = looks.train && typeof looks.train === "object" && "rows" in looks.train ? looks.train : null;
  const wrongKind =
    train?.format && (decision ? train.format !== "decision" : !LM_FORMATS.includes(train.format))
      ? `This is ${train.format_label?.toLowerCase()} data, but you're training a ${decision ? "decision" : "language"} model.`
      : null;

  return (
    <Section title="Data" aside={<span className="hint">{decision ? "Columns: state, questions, gold" : "Columns: messages, prompt and completion, or text"}</span>}>
      <div className="grid-3">
        {SLOTS.map(({ slot, label, empty }) => (
          <div className="field" key={slot}>
            <span className="label">{label}</span>
            <SlotCard source={d[slot]} look={looks[slot]} empty={empty} active={picking === slot}
              onPick={() => setPicking(slot)} onClear={() => choose(slot, null)} />
          </div>
        ))}
      </div>
      {wrongKind && <p className="warn-text notice">{wrongKind}</p>}
      {picking && (
        <DataPicker slot={picking} onClose={() => setPicking(null)}
          onChoose={(source) => {
            choose(picking, source);
            setPicking(null);
          }} />
      )}
      {train && train.preview.length > 0 && <Preview info={train} />}

      <div className="panel">
        <div className="panel-row">
          <span className="panel-label">Duplicates</span>
          <Segmented label="Duplicates" value={d.dedupe ? "remove" : "keep"}
            options={[{ value: "remove", label: "Remove" }, { value: "keep", label: "Keep" }]}
            onChange={(v) => setData({ dedupe: v === "remove" })} />
          {d.dedupe && (
            <select aria-label="What counts as a duplicate" className="auto" value={d.nearCopies ? "near" : "exact"}
              onChange={(e) => setData({ nearCopies: e.target.value === "near" })}>
              <option value="near">exact and near-copies</option>
              <option value="exact">exact copies only</option>
            </select>
          )}
        </div>
        <div className="panel-row">
          <span className="panel-label">Split from train</span>
          <div className="split-fields">
            {d.validation ? (
              <div className="field">
                <span className="label">Validation</span>
                <span className="given">your file</span>
              </div>
            ) : (
              <Field id="valPercent" label="Validation %" error={errors.valPercent}>
                <NumberInput id="valPercent" value={d.valPercent} min={0} invalid={!!errors.valPercent}
                  onChange={(valPercent) => setData({ valPercent })} />
              </Field>
            )}
            {d.test ? (
              <div className="field">
                <span className="label">Test</span>
                <span className="given">your file</span>
              </div>
            ) : (
              <Field id="testPercent" label="Test %" error={errors.testPercent}>
                <NumberInput id="testPercent" value={d.testPercent} min={0} invalid={!!errors.testPercent}
                  onChange={(testPercent) => setData({ testPercent })} />
              </Field>
            )}
            <Field id="keepTogether" label="Keep together">
              <select id="keepTogether" value={d.keepTogether}
                onChange={(e) => setData({ keepTogether: e.target.value as FormState["data"]["keepTogether"] })}>
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
              {d.nearCopies ? "Exact and near-copies" : "Exact copies"} are removed, and validation or test rows that
              share {decision ? "a state" : "a prompt"} with train are dropped.
            </p>
          ) : (
            <p className="warn-text">Duplicates are kept, so scores can look better than the model really is.</p>
          )}
          {d.keepTogether === "nothing" && (
            <p className="warn-text">Rows that share {decision ? "a state" : "a prompt"} can land on both sides of the split.</p>
          )}
        </div>
      </div>

      <Prepare form={form} invalid={!!(errors.valPercent || errors.testPercent || errors.seed)} onReady={onReady} />
    </Section>
  );
}

function SlotCard(props: {
  source: DataSource | null;
  look: Look | undefined;
  empty: string;
  active: boolean;
  onPick: () => void;
  onClear: () => void;
}) {
  const { source, look } = props;
  if (!source)
    return (
      <button type="button" className={`drop${props.active ? " active" : ""}`} onClick={props.onPick}>
        <IconDatabase size={16} aria-hidden /> {props.empty}
      </button>
    );
  const info = look && typeof look === "object" && "rows" in look ? look : null;
  return (
    <div className={`slot${props.active ? " active" : ""}`}>
      <button type="button" className="slot-main" onClick={props.onPick} title="Change">
        <IconFile size={16} aria-hidden />
        <span className="slot-text">
          <span className="slot-name">{source.name}</span>
          <span className="slot-meta">
            {look === "loading" && "Reading…"}
            {info && `${info.rows.toLocaleString()} rows · ${info.format_label ?? "unknown format"}`}
            {look && typeof look === "object" && "error" in look && <span className="bad-text">{look.error}</span>}
          </span>
          {info && !info.format && <span className="bad-text">The columns don't match a format anvil knows.</span>}
          {info && !!info.invalid && (
            <span className="warn-text">
              {info.invalid} of the first {info.checked.toLocaleString()} rows don't fit the format.
            </span>
          )}
        </span>
      </button>
      <button type="button" className="icon slot-clear" aria-label="Remove" title="Remove" onClick={props.onClear}>
        <IconX size={14} aria-hidden />
      </button>
    </div>
  );
}

function Preview({ info }: { info: DatasetInfo }) {
  const columns = Object.keys(info.preview[0]);
  return (
    <details className="preview">
      <summary>Train's first rows</summary>
      <div className="preview-scroll">
        <table>
          <thead>
            <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {info.preview.map((row, i) => (
              <tr key={i}>
                {columns.map((c) => (
                  <td key={c}>
                    <div className="cell" title={row[c]}>{row[c]}</div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {info.examples.length > 0 && (
        <ul className="problems">
          {info.examples.map((p) => <li key={p.row}>Row {p.row + 1}: {p.reason}</li>)}
        </ul>
      )}
    </details>
  );
}

function Prepare({ form, invalid, onReady }: {
  form: FormState;
  invalid: boolean;
  onReady: (ready: boolean, report: PrepareReport | null) => void;
}) {
  const request = prepareRequest(form);
  const key = request && !invalid ? JSON.stringify(request) : null;
  const [job, setJob] = useState<{ key: string; job: PrepareJob } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const keyRef = useRef(key);
  keyRef.current = key;

  // When the request changes, look for an earlier result without starting anything.
  useEffect(() => {
    setError(null);
    if (!key) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      api<{ job: PrepareJob | null }>("/api/datasets/prepare?run=false", { json: JSON.parse(key), signal: controller.signal })
        .then((r) => r.job && setJob({ key, job: r.job }))
        .catch((e) => !isAbort(e) && setError(message(e)));
    }, 400);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [key]);

  // Follow a running job.
  const running = job?.job.state === "running" ? job.job.id : null;
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => {
      api<{ job: PrepareJob }>(`/api/datasets/jobs/${running}`)
        .then((r) => setJob((j) => (j && j.job.id === r.job.id ? { ...j, job: r.job } : j)))
        .catch((e) => setError(message(e)));
    }, 1000);
    return () => clearInterval(timer);
  }, [running]);

  const current = job && job.key === key ? job.job : null;
  const ready = current?.state === "done";
  const report = ready ? current.report : null;
  useEffect(() => onReady(ready, report), [ready, report, onReady]);

  function start() {
    if (!key) return;
    setError(null);
    api<{ job: PrepareJob }>("/api/datasets/prepare", { json: JSON.parse(key) })
      .then((r) => keyRef.current === key && setJob({ key, job: r.job }))
      .catch((e) => setError(message(e)));
  }

  if (!request) return <p className="hint prepare-note">Choose a train dataset, then prepare the data.</p>;
  return (
    <div className="prepare">
      {current?.state === "running" ? (
        <div className="prepare-running">
          <span>{current.stage}…</span>
          <div className="bar"><div style={{ width: `${Math.round(current.progress * 100)}%` }} /></div>
        </div>
      ) : ready && current.report ? (
        <Summary report={current.report} />
      ) : (
        <div className="prepare-row">
          <button type="button" onClick={start}>Prepare data</button>
          <span className="hint">
            {job && job.key !== key ? "Settings changed since the data was last prepared." :
              "Checks every row, removes duplicates and makes the split. Large datasets take a minute or two."}
          </span>
        </div>
      )}
      {current?.state === "failed" && <p className="field-error">{current.error}</p>}
      {error && <p className="field-error">{error}</p>}
    </div>
  );
}

function Summary({ report }: { report: PrepareReport }) {
  const rows: { slot: Slot; label: string }[] = [
    { slot: "train", label: "Train" },
    { slot: "validation", label: report.given.validation ? "Validation (your file)" : "Validation (cut from train)" },
    { slot: "test", label: report.given.test ? "Test (your file)" : "Test (cut from train)" },
  ];
  const removed = (slot: Slot) => {
    const r = report.removed[slot];
    return r ? r.exact + r.near : 0;
  };
  const conflicts = Object.values(report.removed).reduce((n, r) => n + (r?.conflicts ?? 0), 0);
  const broken = Object.values(report.invalid).reduce((n: number, v) => n + (v ?? 0), 0);
  return (
    <div className="summary">
      <p className="ok-text summary-head">
        <IconCircleCheck size={16} aria-hidden /> Prepared in {Math.round(report.seconds)} s · {report.format_label}
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th></th>
              <th>Rows</th>
              <th>Copies removed</th>
              <th>Shared with train</th>
              <th>Broken rows</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ slot, label }) => (
              <tr key={slot}>
                <td>{label}</td>
                <td>{report.rows[slot].toLocaleString()}</td>
                <td>{report.removed[slot] ? removed(slot).toLocaleString() : "–"}</td>
                <td>{report.leaks[slot] !== undefined ? report.leaks[slot]!.toLocaleString() : "–"}</td>
                <td>{report.invalid[slot] !== undefined ? report.invalid[slot]!.toLocaleString() : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {conflicts > 0 && (
        <p className="warn-text">
          {conflicts.toLocaleString()} of the removed copies had a different answer than the row that was kept, so some
          labels disagree.
        </p>
      )}
      {broken > 0 && (
        <ul className="problems">
          {(Object.entries(report.examples) as [Slot, { row: number; reason: string }[]][]).flatMap(([slot, list]) =>
            list.slice(0, 3).map((p) => <li key={`${slot}-${p.row}`}>{slot} row {p.row + 1}: {p.reason}</li>))}
        </ul>
      )}
      {(report.questions || report.notes.length > 0) && (
        <div>
          {report.questions && <p className="hint">{askedLine(report.questions)}</p>}
          {report.notes.map((n) => <p key={n} className="hint">{n}</p>)}
        </div>
      )}
    </div>
  );
}

// "Validation asks 8 questions: 5 yes/no, 2 choice and 1 score."
function askedLine(questions: NonNullable<PrepareReport["questions"]>) {
  const all = Object.values(questions);
  const parts = ([["noul", "yes/no"], ["choice", "choice"], ["score", "score"]] as const)
    .map(([type, name]) => [all.filter((q) => q.type === type).length, name] as const)
    .filter(([n]) => n > 0)
    .map(([n, name]) => `${n} ${name}`);
  const list = parts.length > 1 ? `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}` : parts[0];
  return `Validation asks ${all.length === 1 ? "1 question" : `${all.length} questions`}: ${list}.`;
}
