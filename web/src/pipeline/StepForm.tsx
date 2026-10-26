// The settings of the step picked on the canvas, one form per kind of step. Every change goes straight into the graph
// (the page saves it); what the server says is wrong shows under the setting it's about.

import { IconKey, IconLoader2, IconPlugConnected, IconPlus, IconSend, IconX } from "@tabler/icons-react";
import { useEffect, useRef, useState, type ReactNode, type Ref, type RefObject } from "react";
import { api } from "../api";
import { DataPicker } from "../sections/DataPicker";
import type { DataSource } from "../types";
import { Badge, Field, NumberInput, Segmented } from "../ui";
import {
  address, FORMATS, ident, values, type Configs, type FilterOp, type FilterRule, type Graph, type PipelineNode,
  type PipelineProblem, type QuestionConfig, type SaveFormat, type StepDefaults, type StepType,
} from "./graph";
import { modelName } from "./steps";

export interface FormProps<T extends StepType> {
  node: PipelineNode<T>;
  graph: Graph;
  problems: PipelineProblem[];
  defaults: StepDefaults;
  pipelineId: string;
  keys: Set<string>;
  onKeysChange: (node: string, saved: boolean) => void;
  change: (patch: Partial<Configs[T]>) => void;
  assistantKeyAt: string | null; // the assistant's address, when it has a key: a Model there without one uses it
}

function message(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

// What the server said about one setting: the error first, else a warning.
function said(problems: PipelineProblem[], field: string) {
  const mine = problems.filter((p) => p.field === field);
  return { error: mine.find((p) => p.level === "error")?.message, warn: mine.find((p) => p.level === "warn")?.message };
}

function TextBox(props: { id: string; value: string; onChange: (v: string) => void; rows?: number; placeholder?: string;
  mono?: boolean; invalid?: boolean; boxRef?: Ref<HTMLTextAreaElement> }) {
  return (
    <textarea id={props.id} ref={props.boxRef} className={props.mono ? "mono" : undefined} rows={props.rows ?? 3}
      value={props.value} placeholder={props.placeholder} spellCheck={!props.mono}
      aria-invalid={props.invalid || undefined} onChange={(e) => props.onChange(e.target.value)} />
  );
}

// A column from above this step. One that's no longer there stays listed, so it's clear what needs picking again.
function ColumnSelect(props: { id: string; value: string; columns: string[]; onChange: (v: string) => void;
  invalid?: boolean; empty?: string }) {
  const gone = props.value && !props.columns.includes(props.value);
  return (
    <select id={props.id} value={props.value} aria-invalid={props.invalid || undefined}
      onChange={(e) => props.onChange(e.target.value)}>
      <option value="">{props.empty ?? (props.columns.length ? "Pick a column" : "No columns above this step yet")}</option>
      {gone && <option value={props.value}>{props.value} (missing)</option>}
      {props.columns.map((c) => <option key={c} value={c}>{c}</option>)}
    </select>
  );
}

// Buttons that put {{ column }} into a prompt where the cursor is.
function Slots({ columns, box, onInsert }: { columns: string[]; box: RefObject<HTMLTextAreaElement | null>;
  onInsert: (text: string) => void }) {
  if (!columns.length) return <p className="hint">Steps above this one add columns, such as a List's topic.</p>;
  const put = (column: string) => {
    const el = box.current;
    const slot = `{{ ${column} }}`;
    if (!el) return onInsert(slot);
    const start = el.selectionStart ?? el.value.length;
    const end = el.selectionEnd ?? start;
    onInsert(el.value.slice(0, start) + slot + el.value.slice(end));
    requestAnimationFrame(() => {
      el.focus();
      el.setSelectionRange(start + slot.length, start + slot.length);
    });
  };
  return (
    <div className="slot-chips">
      <span className="hint">Insert:</span>
      {columns.map((c) => (
        <button key={c} type="button" className="chip" onClick={() => put(c)}>{`{{ ${c} }}`}</button>
      ))}
    </div>
  );
}

export function StepForm(props: FormProps<StepType>) {
  const { node } = props;
  // Each form gets the props typed for its kind of step.
  const as = <T extends StepType>() => props as unknown as FormProps<T>;
  switch (node.type) {
    case "model":
      return <ModelForm key={node.id} {...as<"model">()} />;
    case "seed":
      return <SeedForm key={node.id} {...as<"seed">()} />;
    case "list":
      return <ListForm key={node.id} {...as<"list">()} />;
    case "generate":
      return <GenerateForm key={node.id} {...as<"generate">()} />;
    case "question":
      return <QuestionForm key={node.id} {...as<"question">()} />;
    case "filter":
      return <FilterForm key={node.id} {...as<"filter">()} />;
    case "dedupe":
      return <DedupeForm key={node.id} {...as<"dedupe">()} />;
    case "save":
      return <SaveForm key={node.id} {...as<"save">()} />;
  }
}

// ---------------------------------------------------------------- Model

function ModelForm({ node, graph, problems, pipelineId, keys, onKeysChange, change, assistantKeyAt }: FormProps<"model">) {
  const c = node.config;
  const id = (f: string) => `${node.id}-${f}`;
  const [models, setModels] = useState<string[] | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [connectError, setConnectError] = useState<string | null>(null);
  const [key, setKey] = useState("");
  const [replacing, setReplacing] = useState(false);
  const [keyError, setKeyError] = useState<string | null>(null);
  const saved = keys.has(node.id);
  const borrowed = saved ? null : graph.keyFrom(node.id, keys); // another Model's at this address
  const assistants = !saved && !borrowed && !!assistantKeyAt && address(c.base_url) === address(assistantKeyAt);

  async function connect() {
    setConnecting(true);
    setConnectError(null);
    try {
      const found = await api<{ base_url: string; models: string[] }>(
        `/api/pipelines/${pipelineId}/nodes/${node.id}/models`, { json: { base_url: c.base_url } });
      setModels(found.models);
      const patch: Partial<Configs["model"]> = {};
      if (found.base_url !== c.base_url) patch.base_url = found.base_url;
      if (!c.model && found.models.length === 1) patch.model = found.models[0];
      if (Object.keys(patch).length) change(patch);
    } catch (e) {
      setModels(null);
      setConnectError(message(e));
    } finally {
      setConnecting(false);
    }
  }

  async function saveKey() {
    setKeyError(null);
    try {
      await api(`/api/pipelines/${pipelineId}/keys/${node.id}`, { method: "PUT", json: { key } });
      onKeysChange(node.id, true);
      setKey("");
      setReplacing(false);
    } catch (e) {
      setKeyError(message(e));
    }
  }

  async function removeKey() {
    setKeyError(null);
    try {
      await api(`/api/pipelines/${pipelineId}/keys/${node.id}`, { method: "DELETE" });
      onKeysChange(node.id, false);
    } catch (e) {
      setKeyError(message(e));
    }
  }

  const url = said(problems, "base_url");
  const model = said(problems, "model");
  const listed = models && models.length > 0;
  return (
    <div className="step-form">
      <Field id={id("base_url")} label="Address" error={connectError ?? url.error}
        hint="The endpoint's base URL. Inside anvil's container this machine is host.docker.internal.">
        <div className="input-row">
          <input id={id("base_url")} value={c.base_url} spellCheck={false} aria-invalid={url.error ? true : undefined}
            placeholder="http://host.docker.internal:8000/v1"
            onChange={(e) => {
              setModels(null);
              change({ base_url: e.target.value });
            }} />
          <button type="button" onClick={connect} disabled={connecting || !c.base_url.trim()}>
            {connecting ? <IconLoader2 size={16} className="spin" aria-hidden /> : <IconPlugConnected size={16} aria-hidden />}
            Connect
          </button>
        </div>
      </Field>

      <Field id={id("key")} label="API key" error={keyError ?? undefined}
        hint={saved && !replacing ? "Kept on the server: anvil never shows it again."
          : borrowed && !replacing ? `The same address as ${modelName(borrowed)}, so its key is used.`
            : assistants && !replacing ? "The assistant's address (Settings), so the assistant's key is used."
              : "Optional, if the endpoint wants one. It stays on the server, and Models at the same address share it."}>
        {borrowed && !replacing ? (
          <div className="input-row key-saved">
            <Badge tone="ok"><IconKey size={13} aria-hidden /> {modelName(borrowed)}'s key</Badge>
            <button type="button" onClick={() => setReplacing(true)}>Use another key</button>
          </div>
        ) : assistants && !replacing ? (
          <div className="input-row key-saved">
            <Badge tone="ok"><IconKey size={13} aria-hidden /> The assistant's key</Badge>
            <button type="button" onClick={() => setReplacing(true)}>Use another key</button>
          </div>
        ) : saved && !replacing ? (
          <div className="input-row key-saved">
            <Badge tone="ok"><IconKey size={13} aria-hidden /> Saved</Badge>
            <button type="button" onClick={() => setReplacing(true)}>Replace</button>
            <button type="button" onClick={removeKey}>Remove</button>
          </div>
        ) : (
          <div className="input-row">
            <input id={id("key")} type="password" autoComplete="off" value={key} placeholder="sk-…"
              onChange={(e) => setKey(e.target.value)} onKeyDown={(e) => e.key === "Enter" && key.trim() && saveKey()} />
            <button type="button" onClick={saveKey} disabled={!key.trim()}>Save key</button>
            {replacing && (
              <button type="button" className="icon" aria-label="Keep the saved key" title="Keep the saved key"
                onClick={() => setReplacing(false)}>
                <IconX size={16} aria-hidden />
              </button>
            )}
          </div>
        )}
      </Field>

      <Field id={id("model")} label="Model" error={model.error}
        hint={listed ? `${models!.length} model${models!.length === 1 ? "" : "s"} at this endpoint.`
          : models ? "The endpoint lists no models: type the model's name." : "Connect lists the models the endpoint serves."}>
        {listed ? (
          <select id={id("model")} value={c.model} aria-invalid={model.error ? true : undefined}
            onChange={(e) => change({ model: e.target.value })}>
            {!models!.includes(c.model) && <option value={c.model}>{c.model || "Pick a model"}</option>}
            {models!.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        ) : (
          <input id={id("model")} value={c.model} spellCheck={false} aria-invalid={model.error ? true : undefined}
            placeholder="gemma4-31b" onChange={(e) => change({ model: e.target.value })} />
        )}
      </Field>

      <Field id={id("name")} label="Name on the canvas" hint="Optional: how the steps that use it name it.">
        <input id={id("name")} value={c.name} placeholder={c.model.split("/").pop() || "Gemma"}
          onChange={(e) => change({ name: e.target.value })} />
      </Field>

      <div className="form-grid">
        <Field id={id("temperature")} label="Temperature" error={said(problems, "temperature").error}>
          <NumberInput id={id("temperature")} value={c.temperature ?? Number.NaN} step={0.1} min={0}
            onChange={(v) => change({ temperature: Number.isNaN(v) ? null : v })} />
        </Field>
        <Field id={id("max_tokens")} label="Max tokens" error={said(problems, "max_tokens").error}>
          <NumberInput id={id("max_tokens")} value={c.max_tokens ?? Number.NaN} step={256} min={1}
            onChange={(v) => change({ max_tokens: Number.isNaN(v) ? null : v })} />
        </Field>
        <Field id={id("parallel")} label="Requests at once" error={said(problems, "parallel").error}>
          <NumberInput id={id("parallel")} value={c.parallel ?? Number.NaN} min={1}
            onChange={(v) => change({ parallel: Number.isNaN(v) ? null : v })} />
        </Field>
        <Field id={id("timeout")} label="Timeout (s)" error={said(problems, "timeout").error}>
          <NumberInput id={id("timeout")} value={c.timeout ?? Number.NaN} step={30} min={5}
            onChange={(v) => change({ timeout: Number.isNaN(v) ? null : v })} />
        </Field>
      </div>
      <p className="hint">An empty temperature leaves it to the server. Requests at once: how many rows ask it at the
        same time (radiance's Gemma serves 32).</p>

      <Field id={id("thinking")} label="Thinking"
        hint={c.thinking === "default" ? "Whatever the server does. OpenAI's own API needs this."
          : `Sends enable_thinking: ${c.thinking === "on"}, which Qwen, Gemma and DeepSeek servers understand.`}>
        <Segmented label="Thinking" value={c.thinking} onChange={(v) => change({ thinking: v })}
          options={[{ value: "default", label: "Server's default" }, { value: "off", label: "Off" },
            { value: "on", label: "On" }]} />
      </Field>

      <ExtraFields id={id("extra")} value={c.extra} error={said(problems, "extra").error}
        onChange={(extra) => change({ extra })} />

      <TryIt node={node} pipelineId={pipelineId} />
    </div>
  );
}

// Extra fields for every request, as JSON: kept as typed until it parses.
function ExtraFields({ id, value, error, onChange }: { id: string; value: Record<string, unknown>; error?: string;
  onChange: (v: Record<string, unknown>) => void }) {
  const [text, setText] = useState(() => (Object.keys(value).length ? JSON.stringify(value, null, 2) : ""));
  const [bad, setBad] = useState<string | null>(null);
  return (
    <details className="more" open={Object.keys(value).length > 0 || undefined}>
      <summary>Extra request fields</summary>
      <Field id={id} label="JSON added to every request" error={bad ?? error}
        hint='Anything else the endpoint takes, such as {"top_p": 0.95} or {"top_k": 40}.'>
        <TextBox id={id} value={text} rows={3} mono placeholder="{}" invalid={!!(bad ?? error)}
          onChange={(v) => {
            setText(v);
            if (!v.trim()) {
              setBad(null);
              onChange({});
              return;
            }
            try {
              const parsed = JSON.parse(v);
              if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("not an object");
              setBad(null);
              onChange(parsed);
            } catch {
              setBad("That isn't a JSON object yet.");
            }
          }} />
      </Field>
    </details>
  );
}

function TryIt({ node, pipelineId }: { node: PipelineNode<"model">; pipelineId: string }) {
  const [text, setText] = useState("Say hello in one short sentence.");
  const [busy, setBusy] = useState(false);
  const [reply, setReply] = useState<{ reply: string; seconds: number; prompt_tokens: number;
    completion_tokens: number; cut: boolean } | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function send() {
    setBusy(true);
    setError(null);
    try {
      setReply(await api(`/api/pipelines/${pipelineId}/nodes/${node.id}/try`,
        { json: { message: text, config: node.config } }));
    } catch (e) {
      setReply(null);
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="try-box">
      <Field id={`${node.id}-try`} label="Try it" error={error ?? undefined}
        hint="One message to the model as it's set up now, to check it answers and how fast.">
        <TextBox id={`${node.id}-try`} value={text} rows={2} onChange={setText} />
      </Field>
      <div className="form-actions">
        <button type="button" onClick={send} disabled={busy || !text.trim() || !node.config.model}>
          {busy ? <IconLoader2 size={16} className="spin" aria-hidden /> : <IconSend size={16} aria-hidden />} Send
        </button>
      </div>
      {reply && (
        <div className="try-reply">
          <p>{reply.reply}</p>
          <p className="hint">
            {reply.seconds.toFixed(1)} s · {reply.prompt_tokens.toLocaleString()} tokens in,{" "}
            {reply.completion_tokens.toLocaleString()} out
            {reply.cut && " · cut off at max tokens"}
          </p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- Seed data

function SeedForm({ node, problems, change }: FormProps<"seed">) {
  const c = node.config;
  const id = (f: string) => `${node.id}-${f}`;
  const dialog = useRef<HTMLDialogElement>(null);
  const [picking, setPicking] = useState(false);
  const [seen, setSeen] = useState<{ rows: number; columns: string[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fresh = useRef(false); // a dataset just chosen brings in all its columns

  useEffect(() => {
    if (picking) dialog.current?.showModal();
    else dialog.current?.close();
  }, [picking]);

  // The dataset's columns and size, to tick the columns to bring in.
  useEffect(() => {
    if (!c.source) return;
    let alive = true;
    api<{ rows: number; columns: string[] }>("/api/datasets/inspect", { json: { source: c.source } })
      .then((found) => {
        if (!alive) return;
        setSeen(found);
        if (fresh.current) {
          fresh.current = false;
          change({ columns: found.columns });
        }
      })
      .catch((e) => alive && setError(message(e)));
    return () => {
      alive = false;
    };
  }, [c.source]); // change is the page's, and stays the same

  function choose(source: DataSource) {
    setPicking(false);
    setSeen(null);
    setError(null);
    fresh.current = true;
    change({ source, columns: [] });
  }

  const source = said(problems, "source");
  const columns = said(problems, "columns");
  const all = seen?.columns ?? c.columns;
  return (
    <div className="step-form">
      <Field id={id("source")} label="Dataset" error={error ?? source.error}
        hint={seen ? `${seen.rows.toLocaleString()} rows.` : "A file on this machine or uploaded, or a Hugging Face split."}>
        <div className="input-row">
          <span className="chosen-source" title={c.source ? ("path" in c.source ? c.source.path : c.source.repo) : ""}>
            {c.source ? c.source.name || ("path" in c.source ? c.source.path : `${c.source.repo} · ${c.source.split}`)
              : "Nothing chosen yet"}
          </span>
          <button type="button" onClick={() => setPicking(true)}>{c.source ? "Change" : "Choose"}</button>
        </div>
      </Field>
      <dialog ref={dialog} className="picker-dialog" aria-label="Choose the seed data" onClose={() => setPicking(false)}
        onClick={(e) => e.target === e.currentTarget && setPicking(false)}>
        {picking && <DataPicker title="Choose the seed data" onChoose={choose} onClose={() => setPicking(false)} />}
      </dialog>

      {all.length > 0 && (
        <Field id={id("columns")} label="Columns to bring in" error={columns.error}
          hint="Each becomes a column of the row; prompts use it by the name on the right.">
          <div className="column-checks" id={id("columns")}>
            {all.map((col) => (
              <label key={col} className="check-row">
                <input type="checkbox" checked={c.columns.includes(col)}
                  onChange={(e) => change({ columns: e.target.checked ? all.filter((x) => x === col || c.columns.includes(x))
                    : c.columns.filter((x) => x !== col) })} />
                <span className="check-name">{col}</span>
                {ident(col) !== col && <span className="hint">as {ident(col)}</span>}
              </label>
            ))}
          </div>
        </Field>
      )}

      <Field id={id("order")} label="Order">
        <Segmented label="Order" value={c.order} onChange={(v) => change({ order: v })}
          options={[{ value: "shuffled", label: "Shuffled" }, { value: "in_order", label: "As they are" }]} />
      </Field>
      <Field id={id("limit")} label="Rows to use" error={said(problems, "limit").error}
        hint="0 uses them all. When they run out, the Saves below get no more rows.">
        <NumberInput id={id("limit")} value={c.limit ?? Number.NaN} min={0}
          onChange={(v) => change({ limit: Number.isNaN(v) ? null : v })} />
      </Field>
    </div>
  );
}

// ---------------------------------------------------------------- List

function ListForm({ node, problems, change }: FormProps<"list">) {
  const c = node.config;
  const id = (f: string) => `${node.id}-${f}`;
  const [text, setText] = useState(() => c.values.join("\n"));
  const n = values(c.values).length;
  const column = said(problems, "column");
  return (
    <div className="step-form">
      <Field id={id("column")} label="Column" error={column.error}
        hint={c.column ? `Prompts below use it as {{ ${c.column} }}.` : "The name prompts use for the value, such as topic."}>
        <input id={id("column")} value={c.column} spellCheck={false} aria-invalid={column.error ? true : undefined}
          onChange={(e) => change({ column: e.target.value })} />
      </Field>
      <Field id={id("values")} label="Values" error={said(problems, "values").error}
        hint={`${n} value${n === 1 ? "" : "s"}, one per line. A value listed twice comes up twice as often.`}>
        <TextBox id={id("values")} value={text} rows={8} placeholder={"a card that was blocked\na transfer that hasn't arrived"}
          onChange={(v) => {
            setText(v);
            change({ values: v.split("\n") });
          }} />
      </Field>
      <Field id={id("pick")} label="Pick"
        hint={c.pick === "turns" ? "Each value in turn, so they all come up as often." : "A value at random for each row."}>
        <Segmented label="Pick" value={c.pick} onChange={(v) => change({ pick: v })}
          options={[{ value: "random", label: "At random" }, { value: "turns", label: "Taking turns" }]} />
      </Field>
    </div>
  );
}

// ---------------------------------------------------------------- Generate

function GenerateForm({ node, graph, problems, change }: FormProps<"generate">) {
  const c = node.config;
  const id = (f: string) => `${node.id}-${f}`;
  const box = useRef<HTMLTextAreaElement>(null);
  const columns = graph.columnsAbove(node.id);
  const model = graph.modelFor(node.id);
  const plugged = graph.plugged.has(node.id);
  const column = said(problems, "column");
  const prompt = said(problems, "prompt");
  const per = c.per_reply ?? 1;
  return (
    <div className="step-form">
      <Field id={id("column")} label="Column" error={column.error} hint="The model's reply goes into this column.">
        <input id={id("column")} value={c.column} spellCheck={false} aria-invalid={column.error ? true : undefined}
          onChange={(e) => change({ column: e.target.value })} />
      </Field>
      <Field id={id("prompt")} label="Prompt" error={prompt.error}
        hint="{{ column }} puts in that column's value for the row.">
        <TextBox id={id("prompt")} boxRef={box} value={c.prompt} rows={7} invalid={!!prompt.error}
          placeholder="Write a short message that a {{ mood }} customer sends their bank about {{ topic }}."
          onChange={(v) => change({ prompt: v })} />
      </Field>
      <Slots columns={columns} box={box} onInsert={(prompt) => change({ prompt })} />
      <Field id={id("per_reply")} label="Rows per reply" error={said(problems, "per_reply").error}
        hint={per > 1 ? `The model is asked for a JSON list of ${per}, and each becomes a row. Several in one request is `
          + "faster, and they come out more varied." : "One reply, one row. Ask for several at once to go faster."}>
        <NumberInput id={id("per_reply")} value={c.per_reply ?? Number.NaN} min={1}
          onChange={(v) => change({ per_reply: Number.isNaN(v) ? null : v })} />
      </Field>
      <details className="more" open={!!c.system || c.temperature !== null || undefined}>
        <summary>System message and temperature</summary>
        <Field id={id("system")} label="System message" error={said(problems, "system").error}
          hint="Optional. It can use {{ column }} too.">
          <TextBox id={id("system")} value={c.system} rows={2} onChange={(v) => change({ system: v })} />
        </Field>
        <Field id={id("temperature")} label="Temperature" error={said(problems, "temperature").error}
          hint="Empty: the Model's own.">
          <NumberInput id={id("temperature")} value={c.temperature ?? Number.NaN} step={0.1} min={0}
            onChange={(v) => change({ temperature: Number.isNaN(v) ? null : v })} />
        </Field>
      </details>
      <AsksNote model={model} plugged={plugged} />
    </div>
  );
}

function AsksNote({ model, plugged }: { model: PipelineNode<"model"> | null; plugged: boolean }) {
  const named = model && (model.config.name.trim() || model.config.model);
  return (
    <p className="hint asks-note">
      {!model ? "No Model yet: start the line at one, or plug one into this step's model handle."
        : <>{plugged ? "Asks the Model plugged in here" : "Asks the Model its line starts from"}
          {named && <>, <strong>{modelName(model)}</strong></>}. To use another, drag from that Model's diamond handle to
          this step's.</>}
    </p>
  );
}

// ---------------------------------------------------------------- Question

function QuestionForm({ node, graph, problems, defaults, change }: FormProps<"question">) {
  const c = node.config;
  const id = (f: string) => `${node.id}-${f}`;
  const columns = graph.columnsAbove(node.id);
  const second = graph.secondFor(node.id);
  const name = said(problems, "name");
  const instructions = said(problems, "instructions");
  return (
    <div className="step-form">
      <Field id={id("name")} label="Name" error={name.error}
        hint="The question's name in the dataset, and the column its answer goes in, such as frustrated.">
        <input id={id("name")} value={c.name} spellCheck={false} aria-invalid={name.error ? true : undefined}
          onChange={(e) => change({ name: e.target.value })} />
      </Field>
      <Field id={id("type")} label="Kind of answer">
        <Segmented label="Kind of answer" value={c.type} onChange={(v) => change({ type: v, mapping: {} })}
          options={[{ value: "noul", label: "Yes or no" }, { value: "choice", label: "Choice" },
            { value: "score", label: "Score" }]} />
      </Field>
      <Field id={id("instructions")} label="Question" error={instructions.error}
        hint="Word for word: the model labelling the rows reads it, and so does the decision model that trains on them.">
        <TextBox id={id("instructions")} value={c.instructions} rows={2} invalid={!!instructions.error}
          placeholder="Is the customer frustrated?" onChange={(v) => change({ instructions: v })} />
      </Field>
      {c.type === "noul" && (
        <div className="form-grid two">
          <Field id={id("yes")} label="Yes means" hint="Optional.">
            <input id={id("yes")} value={c.yes} onChange={(e) => change({ yes: e.target.value })} />
          </Field>
          <Field id={id("no")} label="No means" hint="Optional.">
            <input id={id("no")} value={c.no} onChange={(e) => change({ no: e.target.value })} />
          </Field>
        </div>
      )}
      {c.type === "choice" && <Options c={c} id={id("options")} error={said(problems, "options").error} change={change} />}
      {c.type === "score" && <Levels c={c} id={id("levels")} error={said(problems, "levels").error} change={change} />}

      <Field id={id("answer")} label="Answers come from">
        <Segmented label="Answers come from" value={c.answer} onChange={(v) => change({ answer: v })}
          options={[{ value: "model", label: "A model" }, { value: "column", label: "A column" }]} />
      </Field>
      {(c.answer === "model" || second) && (
        <Field id={id("about")} label="About" error={said(problems, "about").error}
          hint="The column with the text the model judges.">
          <ColumnSelect id={id("about")} value={c.about} columns={columns} invalid={!!said(problems, "about").error}
            onChange={(about) => change({ about })} />
        </Field>
      )}
      {c.answer === "model" ? (
        <>
          <Field id={id("context")} label="Context" hint="Optional: who wrote the text and where, for the model.">
            <TextBox id={id("context")} value={c.context} rows={2}
              placeholder="These are messages customers send to their bank's chat assistant."
              onChange={(context) => change({ context })} />
          </Field>
          <label className="check-row">
            <input type="checkbox" checked={c.sure_only} onChange={(e) => change({ sure_only: e.target.checked })} />
            <span>Only keep answers the model is sure of</span>
          </label>
          <p className="hint">The others are left out (asked, not scored). Small models often say they aren't sure,
            so try a preview before turning this on.</p>
          <AsksNote model={graph.modelFor(node.id)} plugged={graph.plugged.has(node.id)} />
        </>
      ) : (
        <FromColumn c={c} node={node} graph={graph} problems={problems} defaults={defaults} change={change} />
      )}

      <div className="second-opinion">
        {second ? (
          <Field id={id("on_disagree")} label={`Second opinion: ${modelName(second)}`}
            hint={c.answer === "model" ? "Both models must give the same answer."
              : "The column's answer stands unless the second model is sure it's wrong."}>
            <Segmented label="When they disagree" value={c.on_disagree} onChange={(v) => change({ on_disagree: v })}
              options={[{ value: "leave_out", label: "Leave the answer out" }, { value: "drop", label: "Drop the row" }]} />
          </Field>
        ) : (
          <p className="hint">For a second opinion, plug another Model into this step's 2nd opinion handle.</p>
        )}
      </div>
    </div>
  );
}

function Options({ c, id, error, change }: { c: QuestionConfig; id: string; error?: string;
  change: (p: Partial<QuestionConfig>) => void }) {
  const set = (i: number, patch: Partial<{ name: string; meaning: string }>) =>
    change({ options: c.options.map((o, j) => (j === i ? { ...o, ...patch } : o)) });
  return (
    <Field id={id} label="Options" error={error} hint="Each option's name (the answer), and what it means.">
      <div className="list-editor" id={id}>
        {c.options.map((o, i) => (
          <div className="list-row" key={i}>
            <input aria-label={`Option ${i + 1}`} value={o.name} placeholder="cards" spellCheck={false}
              onChange={(e) => set(i, { name: e.target.value })} />
            <input aria-label={`What option ${i + 1} means`} value={o.meaning} placeholder="card trouble"
              onChange={(e) => set(i, { meaning: e.target.value })} />
            <button type="button" className="icon" aria-label={`Remove option ${i + 1}`} title="Remove"
              onClick={() => change({ options: c.options.filter((_, j) => j !== i) })}>
              <IconX size={15} aria-hidden />
            </button>
          </div>
        ))}
        <button type="button" className="add-row" onClick={() => change({ options: [...c.options, { name: "", meaning: "" }] })}>
          <IconPlus size={15} aria-hidden /> Add an option
        </button>
      </div>
    </Field>
  );
}

function Levels({ c, id, error, change }: { c: QuestionConfig; id: string; error?: string;
  change: (p: Partial<QuestionConfig>) => void }) {
  return (
    <Field id={id} label="Levels" error={error} hint="Lowest first. The answer is the level's number, from 0.">
      <div className="list-editor" id={id}>
        {c.levels.map((level, i) => (
          <div className="list-row level-row" key={i}>
            <span className="level-number">{i}</span>
            <input aria-label={`Level ${i}`} value={level} placeholder={["calm", "annoyed", "angry", "furious"][i] ?? ""}
              onChange={(e) => change({ levels: c.levels.map((l, j) => (j === i ? e.target.value : l)) })} />
            <button type="button" className="icon" aria-label={`Remove level ${i}`} title="Remove"
              onClick={() => change({ levels: c.levels.filter((_, j) => j !== i) })}>
              <IconX size={15} aria-hidden />
            </button>
          </div>
        ))}
        <button type="button" className="add-row" onClick={() => change({ levels: [...c.levels, ""] })}>
          <IconPlus size={15} aria-hidden /> Add a level
        </button>
      </div>
    </Field>
  );
}

// Answers taken from a column: a List's values can each be given an answer; other columns are read as they are.
function FromColumn({ c, node, graph, problems, defaults, change }: { c: QuestionConfig; node: PipelineNode<"question">;
  graph: Graph; problems: PipelineProblem[]; defaults: StepDefaults; change: (p: Partial<QuestionConfig>) => void }) {
  const id = (f: string) => `${node.id}-${f}`;
  const columns = graph.columnsAbove(node.id);
  const list = graph.above(node.id, "list").find((n) => (n.config as Configs["list"]).column === c.from_column);
  const listed = list ? [...new Set(values((list.config as Configs["list"]).values))] : [];
  const answers: { value: string; label: string }[] = c.type === "noul"
    ? [{ value: "yes", label: "yes" }, { value: "no", label: "no" }]
    : c.type === "choice" ? c.options.filter((o) => o.name.trim()).map((o) => ({ value: o.name, label: o.name }))
      : values(c.levels).map((l, i) => ({ value: String(i), label: `${i}: ${l}` }));
  const from = said(problems, "from_column");
  return (
    <>
      <Field id={id("from_column")} label="Column" error={from.error}
        hint="Such as the List that told the model what to write: a frustrated or a calm message.">
        <ColumnSelect id={id("from_column")} value={c.from_column} columns={columns} invalid={!!from.error}
          onChange={(from_column) => change({ from_column, mapping: {} })} />
      </Field>
      {listed.length > 0 ? (
        <Field id={id("mapping")} label="Each value's answer" error={said(problems, "mapping").error}
          hint="Left out: the row asks the question without an answer.">
          <div className="mapping" id={id("mapping")}>
            {listed.map((value) => (
              <label key={value} className="mapping-row">
                <span className="mapping-value" title={value}>{value}</span>
                <select value={c.mapping[value] ?? ""} onChange={(e) => {
                  const mapping = { ...c.mapping };
                  if (e.target.value) mapping[value] = e.target.value;
                  else delete mapping[value];
                  change({ mapping });
                }}>
                  <option value="">as it is</option>
                  {answers.map((a) => <option key={a.value} value={a.value}>{a.label}</option>)}
                  <option value={defaults.leave_out}>leave out</option>
                </select>
              </label>
            ))}
          </div>
        </Field>
      ) : c.from_column ? (
        <p className="hint">Values are read as they are: {c.type === "noul" ? "true or false, yes or no, 1 or 0"
          : c.type === "choice" ? "an option's name" : "a level's number"}. Others leave the answer out.</p>
      ) : null}
    </>
  );
}

// ---------------------------------------------------------------- Filter

function FilterForm({ node, graph, problems, defaults, change }: FormProps<"filter">) {
  const c = node.config;
  const columns = graph.columnsAbove(node.id);
  const ops = Object.entries(defaults.filter_ops) as [FilterOp, string][];
  const set = (i: number, patch: Partial<FilterRule>) =>
    change({ rules: c.rules.map((r, j) => (j === i ? { ...r, ...patch } : r)) });
  return (
    <div className="step-form">
      <Field id={`${node.id}-match`} label="Keep a row when it passes">
        <Segmented label="Keep a row when it passes" value={c.match} onChange={(v) => change({ match: v })}
          options={[{ value: "all", label: "Every rule" }, { value: "any", label: "Any rule" }]} />
      </Field>
      <Field id={`${node.id}-rules`} label="Rules" error={said(problems, "rules").error}
        hint="Text compares without minding capitals. A Question's answers read as true and false, or the option's name.">
        <div className="list-editor rules" id={`${node.id}-rules`}>
          {c.rules.map((r, i) => (
            <div className="rule-row" key={i}>
              <ColumnSelect id={`${node.id}-rule-${i}`} value={r.column} columns={columns} empty="Column"
                onChange={(column) => set(i, { column })} />
              <select aria-label={`Rule ${i + 1}'s test`} value={r.op} onChange={(e) => set(i, { op: e.target.value as FilterOp })}>
                {ops.map(([op, label]) => <option key={op} value={op}>{label}</option>)}
              </select>
              <button type="button" className="icon" aria-label={`Remove rule ${i + 1}`} title="Remove"
                onClick={() => change({ rules: c.rules.filter((_, j) => j !== i) })}>
                <IconX size={15} aria-hidden />
              </button>
              {r.op !== "empty" && r.op !== "not_empty" && (
                <input className="rule-value" aria-label={`Rule ${i + 1}'s value`} value={r.value}
                  placeholder={r.op === "matches" ? "A regular expression" : r.op === "longer" || r.op === "shorter"
                    ? "Characters" : "Value"}
                  onChange={(e) => set(i, { value: e.target.value })} />
              )}
            </div>
          ))}
          <button type="button" className="add-row"
            onClick={() => change({ rules: [...c.rules, { column: columns[columns.length - 1] ?? "", op: "is", value: "" }] })}>
            <IconPlus size={15} aria-hidden /> Add a rule
          </button>
        </div>
      </Field>
      <p className="hint">Rows that don't pass are dropped, and a run counts why.</p>
    </div>
  );
}

// ---------------------------------------------------------------- Dedupe

function DedupeForm({ node, graph, problems, change }: FormProps<"dedupe">) {
  const c = node.config;
  const column = said(problems, "column");
  return (
    <div className="step-form">
      <Field id={`${node.id}-column`} label="Compare" error={column.error} hint="The column whose copies are dropped.">
        <ColumnSelect id={`${node.id}-column`} value={c.column} columns={graph.columnsAbove(node.id)}
          invalid={!!column.error} onChange={(v) => change({ column: v })} />
      </Field>
      <label className="check-row">
        <input type="checkbox" checked={c.near} onChange={(e) => change({ near: e.target.checked })} />
        <span>Near-copies too</span>
      </label>
      <p className="hint">Two texts sharing at least 70% of their three-word runs count as copies, the rule Prepare data
        uses. The first one through is kept.</p>
    </div>
  );
}

// ---------------------------------------------------------------- Save

// An equal share of the rows for each value of a List's column or a Question's answer.
function Balance({ node, graph, problems, change }: { node: PipelineNode<"save">; graph: Graph;
  problems: PipelineProblem[]; change: (p: Partial<Configs["save"]>) => void }) {
  const c = node.config;
  const lists = graph.above(node.id, "list").map((n) => (n.config as Configs["list"]).column).filter(Boolean);
  const asked = graph.above(node.id, "question").map((n) => (n.config as Configs["question"]).name).filter(Boolean);
  const error = said(problems, "balance").error;
  const hint = !c.balance ? "Off: rows are kept as they come, so whatever the model writes most fills the most."
    : lists.includes(c.balance) ? `Each ${c.balance} gets an equal share, and rows are only made for the ones still short.`
      : `Each answer gets an equal share. Rows whose answer already has its share are dropped.`;
  return (
    <Field id={`${node.id}-balance`} label="Balance by" error={error} hint={hint}>
      <select id={`${node.id}-balance`} value={c.balance} aria-invalid={error ? true : undefined}
        onChange={(e) => change({ balance: e.target.value })}>
        <option value="">Don't balance</option>
        {lists.map((col) => <option key={`l${col}`} value={col}>{col} (a List)</option>)}
        {asked.map((col) => <option key={`q${col}`} value={col}>{col} (a Question's answer)</option>)}
        {c.balance && ![...lists, ...asked].includes(c.balance) && <option value={c.balance}>{c.balance} (missing)</option>}
      </select>
    </Field>
  );
}

function SaveForm({ node, graph, problems, change }: FormProps<"save">) {
  const c = node.config;
  const id = (f: string) => `${node.id}-${f}`;
  const columns = graph.columnsAbove(node.id);
  const questions = graph.above(node.id, "question");
  const pick = (field: keyof Configs["save"], label: string, hint?: string): ReactNode => {
    const s = said(problems, field);
    return (
      <Field id={id(field)} label={label} error={s.error} hint={hint}>
        <ColumnSelect id={id(field)} value={String(c[field] ?? "")} columns={columns} invalid={!!s.error}
          onChange={(v) => change({ [field]: v } as Partial<Configs["save"]>)} />
      </Field>
    );
  };
  const name = said(problems, "name");
  return (
    <div className="step-form">
      <Field id={id("name")} label="Dataset name" error={name.error}
        hint={c.name ? `A run writes datasets/generated/${c.name}-<when>.jsonl, which New run's data picker lists.`
          : "Such as bank-messages."}>
        <input id={id("name")} value={c.name} spellCheck={false} aria-invalid={name.error ? true : undefined}
          onChange={(e) => change({ name: e.target.value })} />
      </Field>
      <div className="form-grid two">
        <Field id={id("rows")} label="Rows" error={said(problems, "rows").error} hint="The run stops at this many.">
          <NumberInput id={id("rows")} value={c.rows ?? Number.NaN} min={1} step={100}
            onChange={(v) => change({ rows: Number.isNaN(v) ? null : v })} />
        </Field>
        <Field id={id("format")} label="Format">
          <select id={id("format")} value={c.format} onChange={(e) => change({ format: e.target.value as SaveFormat })}>
            {(Object.entries(FORMATS) as [SaveFormat, string][]).map(([f, label]) => (
              <option key={f} value={f}>{label}</option>
            ))}
          </select>
        </Field>
      </div>

      <Balance node={node} graph={graph} problems={problems} change={change} />

      {c.format === "decision" && (
        <>
          <div className="form-grid two">
            <Field id={id("state_key")} label="State field" error={said(problems, "state_key").error}
              hint={`The state is {"${c.state_key || "message"}": …}.`}>
              <input id={id("state_key")} value={c.state_key} spellCheck={false}
                onChange={(e) => change({ state_key: e.target.value })} />
            </Field>
            {pick("state_column", "Taken from")}
          </div>
          <div className="field">
            <span className="label">Questions</span>
            {questions.length ? (
              <ul className="question-list">
                {questions.map((q) => {
                  const qc = q.config as Configs["question"];
                  return <li key={q.id}><strong>{qc.name || "(no name yet)"}</strong> <span className="hint">
                    {qc.instructions || "no question yet"}</span></li>;
                })}
              </ul>
            ) : <p className="hint">No Question steps above this one yet.</p>}
            <p className="hint">Every Question above asks in every row; one left without an answer is asked, not scored.</p>
          </div>
        </>
      )}
      {c.format === "chat" && (
        <>
          <Field id={id("system")} label="System message" error={said(problems, "system").error}
            hint="Optional. It can use {{ column }}.">
            <TextBox id={id("system")} value={c.system} rows={2} onChange={(system) => change({ system })} />
          </Field>
          {pick("user_column", "User's turn")}
          {pick("assistant_column", "Assistant's reply")}
        </>
      )}
      {c.format === "pairs" && (
        <>
          {pick("prompt_column", "Prompt")}
          {pick("completion_column", "Completion")}
        </>
      )}
      {c.format === "text" && pick("text_column", "Text")}
      <p className="hint">Each row also keeps source (the pipeline and the List values that made it) and meta (all its
        columns).</p>
    </div>
  );
}
