import { useEffect, useState } from "react";
import { api, onLoggedOut } from "./api";
import { Finetune } from "./Finetune";
import { Login } from "./Login";
import { Settings } from "./Settings";
import type { Session } from "./types";

// Two pages, picked by the address: #/settings, or the fine-tuning screen for anything else.
type Page = "finetune" | "settings";
const pageFromAddress = (): Page => (window.location.hash === "#/settings" ? "settings" : "finetune");

export function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState<Page>(pageFromAddress);

  useEffect(() => {
    onLoggedOut(() => setSession((s) => (s ? { ...s, logged_in: false, username: null } : s)));
    api<Session>("/api/session")
      .then(setSession)
      .catch((e: Error) => setError(e.message));
    const follow = () => setPage(pageFromAddress());
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
  return page === "settings" ? <Settings onLogout={logout} /> : <Finetune onLogout={logout} />;
}
