import { IconHammer, IconLogout, IconSettings } from "@tabler/icons-react";
import type { ReactNode } from "react";

// The bar at the top of every page: the name (a link home), the pages, whatever the page puts in the middle and on
// the right, Settings and Log out. On a phone the page's own part gets a row of its own.
export function Topbar({ children, right, onLogout }: { children?: ReactNode; right?: ReactNode; onLogout: () => void }) {
  const here = window.location.hash;
  const runs = here.startsWith("#/runs");
  const pipelines = here.startsWith("#/pipelines");
  const settings = here === "#/settings";
  const docs = here.startsWith("#/docs");
  const finetune = !runs && !pipelines && !settings && !docs;
  return (
    <header className="topbar">
      <a className="brand" href="#/">
        <IconHammer size={20} aria-hidden /> <span className="brand-name">anvil</span>
      </a>
      <nav className="nav" aria-label="Pages">
        <a href="#/" className={finetune ? "on" : ""} aria-current={finetune ? "page" : undefined}>New run</a>
        <a href="#/runs" className={runs ? "on" : ""} aria-current={runs ? "page" : undefined}>Runs</a>
        <a href="#/pipelines" className={pipelines ? "on" : ""} aria-current={pipelines ? "page" : undefined}>Pipelines</a>
        <a href="#/docs" className={docs ? "on" : ""} aria-current={docs ? "page" : undefined}>Docs</a>
      </nav>
      {(children || right) && (
        <div className="topbar-page">
          {children}
          {right && <div className="topbar-status">{right}</div>}
        </div>
      )}
      <div className="topbar-account">
        <a className={`icon-link${settings ? " on" : ""}`} href="#/settings" aria-label="Settings" title="Settings"
          aria-current={settings ? "page" : undefined}>
          <IconSettings size={18} aria-hidden />
        </a>
        <button type="button" className="icon" aria-label="Log out" title="Log out" onClick={onLogout}>
          <IconLogout size={18} aria-hidden />
        </button>
      </div>
    </header>
  );
}
