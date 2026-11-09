// Train, validation and test data. Validation and test are optional: whatever you don't provide is cut from train.
// Choosing a dataset shows its size, format and first rows at once. "Prepare data" then checks every row, removes
// exact copies (by default), keeps rows that share an input (or a column's value) on one side of the split, gives
// each answer its share of validation and test (decision data), and reports what it did.

import { IconCircleCheck, IconDatabase, IconFile, IconX } from "@tabler/icons-react";
import { useEffect, useRef, useState } from "react";
import { api, isAbort } from "../api";
import { prepareRequest, type FormState, type SectionProps } from "../form";
import type { DataSource, DatasetInfo, GroupChoice, PrepareJob, PrepareReport, Slot } from "../types";
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
  // Columns that only say the answer again aren't offered: every row with one answer would land on the same side.
  const columns: GroupChoice[] = (train?.groups ?? []).filter((g) => !g.answer);
  const keptBy = d.keepTogether.startsWith("column:") ? d.keepTogether.slice("column:".length) : null;
  const inputWord = decision ? "state" : "prompt";
  const cutting = !(d.validation && d.test);
  const balancing = decision && d.balance && cutting;
  // The smaller share cut from train, and whether one value of the chosen column holds more rows than that on average.
  const shares = [d.validation ? 0 : d.valPercent, d.test ? 0 : d.testPercent].filter((p) => p > 0);
  const share = shares.length ? Math.min(...shares) : 0;
  const kept = columns.find((c) => c.name === keptBy);
  const coarse = kept && share > 0 && kept.values * share < 100;

  // A column chosen for another dataset: back to the same input once this one turns out not to have it.
  useEffect(() => {
    if (train && keptBy && !(train.groups ?? []).some((g) => g.name === keptBy && !g.answer))
      setData({ keepTogether: "input" });
  }, [train, keptBy]); // not setData: it's a new function every render

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
              <select id="keepTogether" value={d.keepTogether} onChange={(e) => setData({ keepTogether: e.target.value })}>
                <option value="input">same {inputWord}</option>
                {columns.map((c) => <option key={c.name} value={`column:${c.name}`}>same {c.name}</option>)}
                {keptBy && !columns.some((c) => c.name === keptBy) && <option value={d.keepTogether}>same {keptBy}</option>}
                <option value="nothing">nothing</option>
              </select>
            </Field>
            {decision && cutting && (
              <Field id="balance" label="Balance by">
                <select id="balance" value={d.balance ? "answer" : "nothing"}
                  onChange={(e) => setData({ balance: e.target.value === "answer" })}>
                  <option value="answer">answer</option>
                  <option value="nothing">nothing</option>
                </select>
              </Field>
            )}
          </div>
        </div>
        <div className="panel-notes">
          <p>Only what you don't provide is cut from train. A validation or test file you give is used as it is.</p>
          {d.dedupe ? (
            <p>
              Exact copies are removed, and validation or test rows with the same {inputWord} as a train row are
              dropped.
            </p>
          ) : (
            <p className="warn-text">Duplicates are kept, so scores can look better than the model really is.</p>
          )}
          {keptBy && (
            <p>
              Rows with the same {keptBy} stay on one side, so validation and test only hold {keptBy} values train never
              saw.
              {coarse && ` With ${kept.values} of them, one holds more than ${share}% of the rows, so validation and test come out bigger than asked.`}
            </p>
          )}
          {d.keepTogether === "nothing" && (
            <p className="warn-text">Rows with the same {inputWord} can land on both sides of the split.</p>
          )}
          {balancing && <p>{balanceLine(train?.answers)}</p>}
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
    return r ? r.exact + (r.near ?? 0) : 0;
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
      {report.answers && Object.keys(report.answers).length > 0 && (
        <Answers answers={report.answers} />
      )}
    </div>
  );
}

// "true" and "false" are how a yes/no answer is stored.
function said(answer: string) {
  return answer === "true" ? "yes" : answer === "false" ? "no" : answer;
}

// Under the split settings: which answers balancing shares out, from the first rows of train.
function balanceLine(answers: DatasetInfo["answers"]) {
  const names = Object.keys(answers ?? {});
  if (!answers || names.length === 0) return "Each answer gets its share of validation and test.";
  const shown = names.slice(0, 3).map((q) => {
    const options = Object.keys(answers[q]);
    const listed = options.slice(0, 6).map(said).join(", ") + (options.length > 6 ? ", …" : "");
    return names.length === 1 ? listed : `${q} (${listed})`;
  });
  const more = names.length > 3 ? ` and ${names.length - 3} more question${names.length === 4 ? "" : "s"}` : "";
  return names.length === 1
    ? `Each answer of ${names[0]} (${shown[0]}) gets its share of validation and test.`
    : `Each answer gets its share of validation and test: ${shown.join(", ")}${more}.`;
}

// How many rows of each split have each answer, a question at a time. Open at once for one or two questions.
function Answers({ answers }: { answers: NonNullable<PrepareReport["answers"]> }) {
  const names = Object.keys(answers);
  return (
    <details className="answers" open={names.length <= 2}>
      <summary>Answers in each split</summary>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Answer</th>
              <th>Train</th>
              <th>Validation</th>
              <th>Test</th>
            </tr>
          </thead>
          {names.map((q) => (
            <tbody key={q}>
              <tr className="answers-question">
                <th colSpan={4}>{q}</th>
              </tr>
              {Object.entries(answers[q]).map(([answer, where]) => (
                <tr key={answer}>
                  <td>{said(answer)}</td>
                  <td className={where.train ? undefined : "warn-text"}>{where.train.toLocaleString()}</td>
                  <td className={where.validation ? undefined : "warn-text"}>{where.validation.toLocaleString()}</td>
                  <td className={where.test ? undefined : "warn-text"}>{where.test.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          ))}
        </table>
      </div>
    </details>
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
