import { lazy, Suspense, useEffect, useState } from "react";
import { api, onLoggedOut } from "./api";
import { Docs } from "./Docs";
import { Finetune } from "./Finetune";
import { Login } from "./Login";
import { Pipelines } from "./Pipelines";
import { RunPage } from "./RunPage";
import { Runs } from "./Runs";
import { Settings } from "./Settings";
import type { Session } from "./types";

// The pipeline editor brings the graph library along, so it loads only when a pipeline is opened.
const PipelinePage = lazy(() => import("./pipeline/PipelinePage").then((m) => ({ default: m.PipelinePage })));

// The page comes from the address: #/settings, #/runs, #/runs/<id>, #/pipelines, #/pipelines/<id>, #/docs or
// #/docs/<part>, or the fine-tuning screen for anything else.
type Page =
  | { name: "finetune" }
  | { name: "settings" }
  | { name: "runs" }
  | { name: "run"; id: string }
  | { name: "pipelines" }
  | { name: "pipeline"; id: string }
  | { name: "docs"; part: string | null };

function pageFromAddress(): Page {
  const hash = window.location.hash;
  if (hash === "#/settings") return { name: "settings" };
  if (hash === "#/runs") return { name: "runs" };
  const run = hash.match(/^#\/runs\/([a-z0-9-]+)$/);
  if (run) return { name: "run", id: run[1] };
  if (hash === "#/pipelines") return { name: "pipelines" };
  const pipeline = hash.match(/^#\/pipelines\/([a-z0-9-]+)$/);
  if (pipeline) return { name: "pipeline", id: pipeline[1] };
  const docs = hash.match(/^#\/docs(?:\/([a-z0-9-]+))?$/);
  if (docs) return { name: "docs", part: docs[1] ?? null };
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
  if (page.name === "pipelines") return <Pipelines onLogout={logout} />;
  if (page.name === "pipeline") {
    return (
      <Suspense fallback={null}>
        <PipelinePage key={page.id} id={page.id} onLogout={logout} />
      </Suspense>
    );
  }
  if (page.name === "docs") return <Docs part={page.part} onLogout={logout} />;
  return <Finetune onLogout={logout} />;
}
