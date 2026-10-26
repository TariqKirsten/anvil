// Settings: the Hugging Face token anvil uses for you, the pipeline assistant's model, the GPUs anvil shares with a
// model server, and your password.

import { IconArrowLeft, IconCircleCheck, IconInfoCircle, IconKey, IconLoader2, IconPlugConnected, IconX } from "@tabler/icons-react";
import { useEffect, useState, type FormEvent } from "react";
import { api } from "./api";
import { Topbar } from "./Topbar";
import type { Account, AssistantSettings, HfStatus, SharingSettings } from "./types";
import { Badge, Field, NumberInput, Section } from "./ui";

function message(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

export function Settings({ onLogout }: { onLogout: () => void }) {
  const [account, setAccount] = useState<Account | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<Account>("/api/account")
      .then(setAccount)
      .catch((e) => setError(message(e)));
  }, []);

  return (
    <div className="page">
      <Topbar onLogout={onLogout} />
      <a className="back" href="#/">
        <IconArrowLeft size={16} aria-hidden /> Back to fine-tuning
      </a>
      <div className="page-head">
        <h1>Settings</h1>
      </div>
      {error && <p className="field-error">{error}</p>}
      {account && (
        <>
          <HuggingFace hf={account.hf} onChange={(hf) => setAccount({ ...account, hf })} />
          <AssistantModel />
          <SharedGpus />
          <Password />
          <Section title="Account">
            <p className="plain">
              You're logged in as <strong>{account.username}</strong>.
            </p>
          </Section>
        </>
      )}
    </div>
  );
}

function HuggingFace({ hf, onChange }: { hf: HfStatus; onChange: (hf: HfStatus) => void }) {
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save(event: FormEvent) {
    event.preventDefault();
    const value = token.trim();
    if (!value) return setError("Paste a token first.");
    if (!value.startsWith("hf_")) return setError("Hugging Face tokens start with hf_.");
    setBusy(true);
    setError(null);
    try {
      const result = await api<{ hf: HfStatus }>("/api/account/hf-token", { method: "PUT", json: { token: value } });
      onChange(result.hf);
      setToken("");
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    try {
      onChange((await api<{ hf: HfStatus }>("/api/account/hf-token", { method: "DELETE" })).hf);
    } catch (e) {
      setError(message(e));
    }
  }

  const status =
    hf.source === "account"
      ? `Connected as ${hf.user ?? "an unknown user"}.`
      : hf.source === "env"
        ? "Using the token from .env (HF_TOKEN). Save your own below to use it instead."
        : "Not connected. Public models work without a token.";

  return (
    <Section title="Hugging Face">
      <p className={`status-line${hf.connected ? " connected" : ""}`}>
        {hf.connected ? <IconCircleCheck size={16} aria-hidden /> : <IconInfoCircle size={16} aria-hidden />} {status}
      </p>
      <form className="token-form" onSubmit={save}>
        <Field
          id="hf-token"
          label={hf.source === "account" ? "Replace the token" : "Access token"}
          error={error ?? undefined}
          hint="Needed for gated models like Llama, and for uploading your models from a run's page."
        >
          <input
            id="hf-token"
            type="password"
            autoComplete="off"
            spellCheck={false}
            placeholder="hf_…"
            value={token}
            aria-invalid={error ? true : undefined}
            onChange={(e) => {
              setToken(e.target.value);
              setError(null);
            }}
          />
        </Field>
        <div className="form-actions">
          <button className="primary" type="submit">
            {busy ? "Checking…" : "Save token"}
          </button>
          {hf.source === "account" && (
            <button type="button" onClick={remove}>
              Remove
            </button>
          )}
        </div>
      </form>
      <p className="hint">
        Make one at{" "}
        <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
          huggingface.co/settings/tokens
        </a>
        . A read token is enough for downloads; uploading needs one that can write. anvil checks it with Hugging Face,
        keeps it on the server, and never shows it again.
      </p>
    </Section>
  );
}

// A key that stays on the server: saved, replaced or removed, never shown again.
function KeyField({ id, saved, label, hint, onSave, onRemove }: { id: string; saved: boolean; label: string;
  hint: string; onSave: (key: string) => Promise<void>; onRemove: () => Promise<void> }) {
  const [key, setKey] = useState("");
  const [replacing, setReplacing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(work: () => Promise<void>) {
    setError(null);
    try {
      await work();
      setKey("");
      setReplacing(false);
    } catch (e) {
      setError(message(e));
    }
  }

  return (
    <Field id={id} label={label} error={error ?? undefined}
      hint={saved && !replacing ? "Kept on the server: anvil never shows it again." : hint}>
      {saved && !replacing ? (
        <div className="input-row key-saved">
          <Badge tone="ok"><IconKey size={13} aria-hidden /> Saved</Badge>
          <button type="button" onClick={() => setReplacing(true)}>Replace</button>
          <button type="button" onClick={() => run(onRemove)}>Remove</button>
        </div>
      ) : (
        <div className="input-row">
          <input id={id} type="password" autoComplete="off" value={key} placeholder="sk-…"
            onChange={(e) => setKey(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                if (key.trim()) void run(() => onSave(key.trim()));
              }
            }} />
          <button type="button" onClick={() => run(() => onSave(key.trim()))} disabled={!key.trim()}>Save key</button>
          {replacing && (
            <button type="button" className="icon" aria-label="Keep the saved key" title="Keep the saved key"
              onClick={() => setReplacing(false)}>
              <IconX size={16} aria-hidden />
            </button>
          )}
        </div>
      )}
    </Field>
  );
}

// The model the pipeline assistant talks with: any OpenAI-compatible one that takes tools.
function AssistantModel() {
  const [saved, setSaved] = useState<AssistantSettings | null>(null);
  const [form, setForm] = useState<AssistantSettings | null>(null);
  const [models, setModels] = useState<string[] | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  useEffect(() => {
    api<{ settings: AssistantSettings }>("/api/assistant")
      .then(({ settings }) => {
        setSaved(settings);
        setForm(settings);
      })
      .catch((e) => setError(message(e)));
  }, []);

  if (!form || !saved) return error ? <Section title="Pipeline assistant"><p className="field-error">{error}</p></Section> : null;
  const edit = (patch: Partial<AssistantSettings>) => {
    setForm({ ...form, ...patch });
    setDone(false);
  };

  async function connect() {
    if (!form) return;
    setConnecting(true);
    setError(null);
    try {
      const found = await api<{ base_url: string; models: string[] }>("/api/assistant/models",
        { json: { base_url: form.base_url } });
      setModels(found.models);
      edit({ base_url: found.base_url, model: form.model || (found.models.length === 1 ? found.models[0] : "") });
    } catch (e) {
      setModels(null);
      setError(message(e));
    } finally {
      setConnecting(false);
    }
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!form) return;
    setError(null);
    try {
      const { settings } = await api<{ settings: AssistantSettings }>("/api/assistant", { method: "PUT", json: {
        base_url: form.base_url, model: form.model, context: form.context, thinking: form.thinking,
        reasoning_effort: form.reasoning_effort } });
      setSaved(settings);
      setForm({ ...form, ...settings });
      setDone(true);
    } catch (e) {
      setError(message(e));
    }
  }

  const keyChanged = (settings: AssistantSettings) => {
    setSaved(settings);
    setForm({ ...form, has_key: settings.has_key });
  };
  const listed = models && models.length > 0;
  return (
    <Section title="Pipeline assistant">
      <p className="hint">It builds pipelines with you, from the Assistant tab beside a pipeline. Any OpenAI-compatible
        model that takes tools (function calling) will do; a long context helps it keep a long conversation.</p>
      <form className="assistant-form" onSubmit={save}>
        <Field id="assistant-url" label="Address" hint="The endpoint's base URL. Inside anvil's container this machine is host.docker.internal.">
          <div className="input-row">
            <input id="assistant-url" value={form.base_url} spellCheck={false}
              placeholder="http://host.docker.internal:8000/v1" onChange={(e) => {
                setModels(null);
                edit({ base_url: e.target.value });
              }} />
            <button type="button" onClick={connect} disabled={connecting || !form.base_url.trim()}>
              {connecting ? <IconLoader2 size={16} className="spin" aria-hidden /> : <IconPlugConnected size={16} aria-hidden />}
              Connect
            </button>
          </div>
        </Field>
        <KeyField id="assistant-key" label="API key" saved={saved.has_key}
          hint="Optional, if the endpoint wants one. Model steps at this address that have no key of their own use it too."
          onSave={async (key) => keyChanged((await api<{ settings: AssistantSettings }>("/api/assistant/key",
            { method: "PUT", json: { key } })).settings)}
          onRemove={async () => keyChanged((await api<{ settings: AssistantSettings }>("/api/assistant/key",
            { method: "DELETE" })).settings)} />
        <Field id="assistant-model" label="Model"
          hint={listed ? `${models!.length} model${models!.length === 1 ? "" : "s"} at this endpoint.`
            : "Connect lists the models the endpoint serves."}>
          {listed ? (
            <select id="assistant-model" value={form.model} onChange={(e) => edit({ model: e.target.value })}>
              {!models!.includes(form.model) && <option value={form.model}>{form.model || "Pick a model"}</option>}
              {models!.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          ) : (
            <input id="assistant-model" value={form.model} spellCheck={false} placeholder="qwen-assistant"
              onChange={(e) => edit({ model: e.target.value })} />
          )}
        </Field>
        <div className="grid-3">
          <Field id="assistant-context" label="Context (tokens)"
            hint="What the model holds; older parts of a long conversation are left out to fit.">
            <NumberInput id="assistant-context" value={form.context} min={4096} step={1024}
              onChange={(v) => edit({ context: Number.isNaN(v) ? saved.context : v })} />
          </Field>
          <Field id="assistant-thinking" label="Thinking" hint="Off answers sooner; default is the model's own.">
            <select id="assistant-thinking" value={form.thinking}
              onChange={(e) => edit({ thinking: e.target.value as AssistantSettings["thinking"] })}>
              <option value="default">Default</option>
              <option value="on">On</option>
              <option value="off">Off</option>
            </select>
          </Field>
          <Field id="assistant-effort" label="Reasoning effort"
            hint="More plans better but answers slower. Models take different levels: Qwen3.8 on radiance takes low, medium and xhigh.">
            <select id="assistant-effort" value={form.reasoning_effort}
              onChange={(e) => edit({ reasoning_effort: e.target.value as AssistantSettings["reasoning_effort"] })}>
              <option value="default">Default</option>
              <option value="none">None</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="xhigh">Extra high (xhigh)</option>
            </select>
          </Field>
        </div>
        <div className="form-actions">
          {error && <p className="field-error" role="alert">{error}</p>}
          {done && <p className="ok-text">Saved.</p>}
          <button type="submit" className="primary">Save</button>
        </div>
      </form>
    </Section>
  );
}

// A model server on the GPUs anvil trains on: anvil asks it to let go of them before training.
function SharedGpus() {
  const [saved, setSaved] = useState<SharingSettings | null>(null);
  const [url, setUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [freeing, setFreeing] = useState(false);

  useEffect(() => {
    api<{ sharing: SharingSettings }>("/api/sharing")
      .then(({ sharing }) => {
        setSaved(sharing);
        setUrl(sharing.url);
      })
      .catch((e) => setError(message(e)));
  }, []);

  if (!saved) return error ? <Section title="GPUs shared with a model server"><p className="field-error">{error}</p></Section> : null;

  async function save(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setNote(null);
    try {
      const { sharing } = await api<{ sharing: SharingSettings }>("/api/sharing", { method: "PUT", json: { url } });
      setSaved(sharing);
      setNote(sharing.url ? "Saved: training calls it first." : "Saved: training calls nothing first.");
    } catch (e) {
      setError(message(e));
    }
  }

  async function free() {
    setFreeing(true);
    setError(null);
    setNote(null);
    try {
      setNote((await api<{ note: string }>("/api/sharing/free", { method: "POST" })).note);
    } catch (e) {
      setError(message(e));
    } finally {
      setFreeing(false);
    }
  }

  return (
    <Section title="GPUs shared with a model server">
      <p className="hint">If a model server on this machine uses the GPUs anvil trains on, anvil can ask it to let go of
        them before training starts, by calling an address (a POST). While training runs, pipelines and the assistant
        don't ask models at that server, since answering would load a model back onto the cards.</p>
      <form className="assistant-form" onSubmit={save}>
        <Field id="sharing-url" label="Before training, call"
          hint="Such as a model server's unload address. Leave it empty to call nothing.">
          <input id="sharing-url" value={url} spellCheck={false}
            placeholder="http://host.docker.internal:8000/api/models/unload"
            onChange={(e) => {
              setUrl(e.target.value);
              setNote(null);
            }} />
        </Field>
        <KeyField id="sharing-key" label="API key" saved={saved.has_key}
          hint="Optional. Without one, a call to the assistant's server uses the assistant's key."
          onSave={async (key) => setSaved((await api<{ sharing: SharingSettings }>("/api/sharing/key",
            { method: "PUT", json: { key } })).sharing)}
          onRemove={async () => setSaved((await api<{ sharing: SharingSettings }>("/api/sharing/key",
            { method: "DELETE" })).sharing)} />
        <div className="form-actions">
          {error && <p className="field-error" role="alert">{error}</p>}
          {note && <p className="ok-text">{note}</p>}
          <button type="button" onClick={free} disabled={freeing || !saved.url || url !== saved.url}
            title={url !== saved.url ? "Save the address first" : "Call it now, as training would"}>
            {freeing ? "Freeing…" : "Free them now"}
          </button>
          <button type="submit" className="primary">Save</button>
        </div>
      </form>
    </Section>
  );
}

function Password() {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [busy, setBusy] = useState(false);

  async function save(event: FormEvent) {
    event.preventDefault();
    setDone(false);
    if (!current || !next) return setError("Fill in your current and new password.");
    if (next.length < 8) return setError("Use at least 8 characters.");
    if (next !== again) return setError("The new passwords don't match.");
    setBusy(true);
    setError(null);
    try {
      await api("/api/account/password", { method: "PUT", json: { current, new: next } });
      setCurrent("");
      setNext("");
      setAgain("");
      setDone(true);
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }

  const typing = (set: (v: string) => void) => (e: { target: { value: string } }) => {
    set(e.target.value);
    setError(null);
    setDone(false);
  };

  return (
    <Section title="Password">
      <form onSubmit={save}>
        <div className="grid-3">
          <Field id="current-password" label="Current password">
            <input id="current-password" type="password" autoComplete="current-password" value={current}
              onChange={typing(setCurrent)} />
          </Field>
          <Field id="new-password" label="New password" hint="at least 8 characters">
            <input id="new-password" type="password" autoComplete="new-password" value={next}
              onChange={typing(setNext)} />
          </Field>
          <Field id="new-password-again" label="New password again">
            <input id="new-password-again" type="password" autoComplete="new-password" value={again}
              onChange={typing(setAgain)} />
          </Field>
        </div>
        <div className="form-actions">
          {error && <p className="field-error" role="alert">{error}</p>}
          {done && <p className="ok-text">Password changed. Other browsers will need to log in again.</p>}
          <button type="submit">{busy ? "Saving…" : "Change password"}</button>
        </div>
      </form>
    </Section>
  );
}
