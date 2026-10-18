import { IconHammer } from "@tabler/icons-react";
import { useState, type FormEvent } from "react";
import { api } from "./api";
import { Field } from "./ui";

export function Login({ hasAccounts, onLoggedIn }: { hasAccounts: boolean; onLoggedIn: (username: string) => void }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!username.trim() || !password) {
      setError("Enter your username and password first.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await api<{ username: string }>("/api/login", { json: { username: username.trim(), password } });
      onLoggedIn(result.username);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="center">
      <form className="login" onSubmit={submit}>
        <div className="brand">
          <IconHammer size={22} aria-hidden /> anvil
        </div>
        {hasAccounts ? (
          <>
            <Field id="username" label="Username">
              <input id="username" autoComplete="username" value={username}
                onChange={(e) => {
                  setUsername(e.target.value);
                  setError(null);
                }} />
            </Field>
            <Field id="password" label="Password" error={error ?? undefined}>
              <input
                id="password"
                type="password"
                autoFocus
                autoComplete="current-password"
                value={password}
                aria-invalid={error ? true : undefined}
                onChange={(e) => {
                  setPassword(e.target.value);
                  setError(null);
                }}
              />
            </Field>
            <button className="primary" type="submit">
              {busy ? "Logging in…" : "Log in"}
            </button>
          </>
        ) : (
          <p className="field-error">There's no account yet. Set ANVIL_PASSWORD in .env, then restart anvil.</p>
        )}
      </form>
    </div>
  );
}
