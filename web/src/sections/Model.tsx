// The base model: search Hugging Face, or browse to a folder on this machine. Either way the server checks that the
// model can be trained (config, safetensors weights, tokenizer) and the result is shown as a checklist.

import { IconArrowUp, IconBox, IconFolder, IconLoader2, IconLock, IconSearch } from "@tabler/icons-react";
import { useEffect, useState } from "react";
import { api, isAbort } from "../api";
import type { SectionProps } from "../form";
import { count, params } from "../format";
import type { FolderList, HubModel, Report } from "../types";
import { Badge, Checklist, Section, Segmented } from "../ui";

export function ModelSection({ form, update, onReport }: SectionProps & { onReport: (r: Report | null) => void }) {
  const model = form.model;
  return (
    <Section
      title="Base model"
      aside={
        <Segmented
          label="Where the model comes from"
          value={model.source}
          options={[
            { value: "hub", label: "Hugging Face" },
            { value: "folder", label: "Local folder" },
          ]}
          onChange={(source) => update((f) => ({ ...f, model: { ...f.model, source } }))}
        />
      }
    >
      {model.source === "hub" ? (
        <HubPicker
          repo={model.repo}
          onPick={(repo) => update((f) => ({ ...f, model: { ...f.model, repo } }))}
          onReport={onReport}
        />
      ) : (
        <FolderPicker
          folder={model.folder}
          onChange={(folder) => update((f) => ({ ...f, model: { ...f.model, folder } }))}
          onReport={onReport}
        />
      )}
    </Section>
  );
}

function HubPicker(props: { repo: string | null; onPick: (repo: string) => void; onReport: (r: Report | null) => void }) {
  const { repo, onPick, onReport } = props;
  const [query, setQuery] = useState(repo ?? "");
  // The result list closes once you pick, so the checklist sits right under the search box.
  const [open, setOpen] = useState(!repo);
  const [results, setResults] = useState<HubModel[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [checking, setChecking] = useState(false);
  const [checkError, setCheckError] = useState<string | null>(null);

  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setResults(null);
      setSearchError(null);
      return;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => {
      setSearching(true);
      api<{ models: HubModel[] }>(`/api/hub/search?q=${encodeURIComponent(q)}`, { signal: controller.signal })
        .then((r) => {
          setResults(r.models);
          setSearchError(null);
        })
        .catch((e) => !isAbort(e) && setSearchError(e.message))
        .finally(() => setSearching(false));
    }, 300);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  useEffect(() => {
    setReport(null);
    setCheckError(null);
    onReport(null);
    if (!repo) return;
    const controller = new AbortController();
    setChecking(true);
    api<Report>(`/api/hub/check?repo=${encodeURIComponent(repo)}`, { signal: controller.signal })
      .then((r) => {
        setReport(r);
        onReport(r);
      })
      .catch((e) => !isAbort(e) && setCheckError(e.message))
      .finally(() => setChecking(false));
    return () => controller.abort();
  }, [repo, onReport]);

  return (
    <div className="picker">
      <div className="search">
        <IconSearch size={16} aria-hidden />
        <input
          aria-label="Search Hugging Face models"
          placeholder="qwen3.5"
          value={query}
          onFocus={() => setOpen(true)}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
        />
        {searching && <IconLoader2 size={16} className="spin" aria-hidden />}
      </div>
      {searchError && <p className="field-error">{searchError}</p>}
      {!results && !searchError && !repo && <p className="hint">Search Hugging Face by name, like qwen3.5 or llama.</p>}
      {open && results?.length === 0 && <p className="hint">No models match that search.</p>}
      {open && results && results.length > 0 && (
        <div className="results" role="listbox" aria-label="Models">
          {results.map((m) => (
            <ModelRow
              key={m.id}
              model={m}
              selected={m.id === repo}
              onPick={() => {
                onPick(m.id);
                setOpen(false);
              }}
            />
          ))}
        </div>
      )}
      {repo && (
        <div className="chosen">
          <div className="chosen-head">
            <span>
              Chosen: <code>{repo}</code>
            </span>
            {checking && <span className="hint">Checking…</span>}
          </div>
          {checkError && <p className="field-error">{checkError}</p>}
          {report && <Checklist report={report} />}
        </div>
      )}
    </div>
  );
}

function ModelRow({ model, selected, onPick }: { model: HubModel; selected: boolean; onPick: () => void }) {
  const meta = [
    model.params !== null && `${params(model.params)} params`,
    `${count(model.downloads)} downloads`,
    model.pipeline,
    model.gated && "gated",
  ].filter(Boolean);
  return (
    <button
      type="button"
      role="option"
      aria-selected={selected}
      className={`result${selected ? " selected" : ""}`}
      onClick={onPick}
    >
      {model.gated ? <IconLock size={16} aria-hidden /> : <IconBox size={16} aria-hidden />}
      <span className="result-main">
        <span className="result-id">{model.id}</span>
        <span className="result-meta">{meta.join(" · ")}</span>
      </span>
      {model.gguf && <Badge tone="warn">GGUF</Badge>}
      {model.custom_code && <Badge>custom code</Badge>}
      <Badge tone={model.permissive ? "ok" : "warn"}>{model.licence ?? "no licence"}</Badge>
    </button>
  );
}

function shortPath(path: string) {
  return path.split("/").filter(Boolean).slice(-2).join("/");
}

// A folder with no config.json is usually just a folder on the way to a model, not a broken model.
function onTheWay(report: Report) {
  return report.verdict === "bad" && report.items.length === 1 && report.items[0].label === "config.json";
}

function FolderPicker(props: { folder: string; onChange: (folder: string) => void; onReport: (r: Report | null) => void }) {
  const { folder, onChange, onReport } = props;
  const [roots, setRoots] = useState<string[]>([]);
  const [listing, setListing] = useState<FolderList | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The model folders are loaded on their own, so they stay clickable even when the typed path is wrong.
  useEffect(() => {
    api<FolderList>("/api/folders")
      .then((l) => setRoots(l.roots))
      .catch(() => setRoots([]));
  }, []);

  useEffect(() => {
    const path = folder.trim();
    const controller = new AbortController();
    const timer = setTimeout(() => {
      api<FolderList>(`/api/folders${path ? `?path=${encodeURIComponent(path)}` : ""}`, { signal: controller.signal })
        .then(setListing)
        .catch((e) => !isAbort(e) && setListing(null));
    }, 250);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [folder]);

  useEffect(() => {
    const path = folder.trim();
    setReport(null);
    setError(null);
    onReport(null);
    if (!path) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      setChecking(true);
      api<Report>("/api/folders/check", { json: { path }, signal: controller.signal })
        .then((r) => {
          setReport(r);
          onReport(onTheWay(r) ? null : r);
        })
        .catch((e) => !isAbort(e) && setError(e.message))
        .finally(() => setChecking(false));
    }, 400);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [folder, onReport]);

  return (
    <div className="picker">
      <div className="search">
        <IconFolder size={16} aria-hidden />
        <input
          aria-label="Model folder"
          placeholder="/srv/models/qwen3.5-2b"
          value={folder}
          onChange={(e) => onChange(e.target.value)}
        />
        {checking && <IconLoader2 size={16} className="spin" aria-hidden />}
      </div>
      <div className="roots">
        <span className="hint">Model folders:</span>
        {roots.map((root) => (
          <button type="button" className="chip" key={root} title={root} onClick={() => onChange(root)}>
            {shortPath(root)}
          </button>
        ))}
      </div>
      {listing?.path && (
        <div className="browser">
          {listing.parent && (
            <button type="button" className="dir" onClick={() => onChange(listing.parent!)}>
              <IconArrowUp size={16} aria-hidden /> Up
            </button>
          )}
          {listing.dirs.map((d) => (
            <button type="button" className="dir" key={d} onClick={() => onChange(`${listing.path}/${d}`)}>
              <IconFolder size={16} aria-hidden /> {d}
            </button>
          ))}
          {listing.dirs.length === 0 && <span className="hint">No folders inside.</span>}
        </div>
      )}
      {error && <p className="field-error">{error}</p>}
      {report && onTheWay(report) && <p className="hint">No model here. Open a folder that holds a config.json.</p>}
      {report && !onTheWay(report) && <Checklist report={report} />}
    </div>
  );
}
