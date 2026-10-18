import { IconHammer, IconLogout, IconSettings } from "@tabler/icons-react";
import type { ReactNode } from "react";

// The bar at the top of every page: the name (a link home), the pages, whatever the page puts in the middle,
// Settings and Log out.
export function Topbar({ children, right, onLogout }: { children?: ReactNode; right?: ReactNode; onLogout: () => void }) {
  const here = window.location.hash;
  const runs = here.startsWith("#/runs");
  return (
    <header className="topbar">
      <a className="brand" href="#/">
        <IconHammer size={20} aria-hidden /> anvil
      </a>
      <nav className="nav" aria-label="Pages">
        <a href="#/" className={!runs && here !== "#/settings" ? "on" : ""}>New run</a>
        <a href="#/runs" className={runs ? "on" : ""}>Runs</a>
      </nav>
      {children}
      <div className="topbar-right">
        {right}
        <a className="icon-link" href="#/settings" aria-label="Settings" title="Settings">
          <IconSettings size={18} aria-hidden />
        </a>
        <button type="button" className="icon" aria-label="Log out" title="Log out" onClick={onLogout}>
          <IconLogout size={18} aria-hidden />
        </button>
      </div>
    </header>
  );
}
