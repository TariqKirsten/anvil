import { useEffect, useState } from "react";
import { api, onLoggedOut } from "./api";
import { Finetune } from "./Finetune";
import { Login } from "./Login";
import { RunPage } from "./RunPage";
import { Runs } from "./Runs";
import { Settings } from "./Settings";
import type { Session } from "./types";

// The page comes from the address: #/settings, #/runs, #/runs/<id>, or the fine-tuning screen for anything else.
type Page = { name: "finetune" } | { name: "settings" } | { name: "runs" } | { name: "run"; id: string };

function pageFromAddress(): Page {
  const hash = window.location.hash;
  if (hash === "#/settings") return { name: "settings" };
  if (hash === "#/runs") return { name: "runs" };
  const run = hash.match(/^#\/runs\/([a-z0-9-]+)$/);
  if (run) return { name: "run", id: run[1] };
  return { name: "finetune" };
}

export function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState<Page>(pageFromAddress);

  useEffect(() => {
    onLoggedOut(() => setSession((s) => (s ? { ...s, logged_in: false, username: null } : s)));
    api<Session>("/api/session")
      .then(setSession)
      .catch((e: Error) => setError(e.message));
    const follow = () => {
      setPage(pageFromAddress());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", follow);
    return () => window.removeEventListener("hashchange", follow);
  }, []);

  if (error)
    return (
      <div className="center">
        <p className="field-error">Can't reach the anvil server. {error}</p>
      </div>
    );
  if (!session) return null;
  if (!session.logged_in)
    return (
      <Login
        hasAccounts={session.has_accounts}
        onLoggedIn={(username) => setSession({ ...session, logged_in: true, username })}
      />
    );

  const logout = async () => {
    await api("/api/logout", { method: "POST" }).catch(() => null);
    setSession({ ...session, logged_in: false, username: null });
  };
  if (page.name === "settings") return <Settings onLogout={logout} />;
  if (page.name === "runs") return <Runs onLogout={logout} />;
  if (page.name === "run") return <RunPage key={page.id} id={page.id} onLogout={logout} />;
  return <Finetune onLogout={logout} />;
}
