// Settings for your account: the Hugging Face token anvil uses for you, and your password.

import { IconArrowLeft, IconCircleCheck, IconInfoCircle } from "@tabler/icons-react";
import { useEffect, useState, type FormEvent } from "react";
import { api } from "./api";
import { Topbar } from "./Topbar";
import type { Account, HfStatus } from "./types";
import { Field, Section } from "./ui";

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
          hint="Needed for gated models like Llama, and later for uploading your models."
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
        . A read token is enough for downloads. anvil checks it with Hugging Face, keeps it on the server, and never
        shows it again.
      </p>
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
