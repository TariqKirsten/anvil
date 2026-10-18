// Choosing one dataset: a file on this machine, a file uploaded from this browser, or a split of a Hugging Face dataset.

import { IconArrowUp, IconDatabase, IconFile, IconFolder, IconLoader2, IconLock, IconSearch, IconUpload, IconX } from "@tabler/icons-react";
import { useEffect, useState } from "react";
import { api, isAbort, upload } from "../api";
import { count } from "../format";
import type { DataSource, FolderList, HubDataset, Slot } from "../types";
import { Badge, Segmented } from "../ui";

const SLOT_NAMES: Record<Slot, string> = { train: "train", validation: "validation", test: "test" };

export function size(bytes: number) {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${Math.round(bytes / 1e3)} kB`;
  return `${bytes} B`;
}

function message(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

export function DataPicker(props: { slot: Slot; onChoose: (source: DataSource) => void; onClose: () => void }) {
  const [where, setWhere] = useState<"files" | "upload" | "hub">("files");
  return (
    <div className="picker-panel">
      <div className="picker-head">
        <span className="picker-title">Choose the {SLOT_NAMES[props.slot]} data</span>
        <Segmented
          label="Where the data comes from"
          value={where}
          options={[
            { value: "files", label: "This machine" },
            { value: "upload", label: "Upload" },
            { value: "hub", label: "Hugging Face" },
          ]}
          onChange={setWhere}
        />
        <button type="button" className="icon" aria-label="Close" title="Close" onClick={props.onClose}>
          <IconX size={16} aria-hidden />
        </button>
      </div>
      {where === "files" && <Files onChoose={props.onChoose} />}
      {where === "upload" && <Upload onChoose={props.onChoose} />}
      {where === "hub" && <HubDatasets onChoose={props.onChoose} />}
    </div>
  );
}

// The folder you last looked in, so picking validation after train opens where train was (kept across reloads).
const FOLDER_KEY = "anvil.lastDataFolder";

function lastFolder(): string | null {
  try {
    return localStorage.getItem(FOLDER_KEY);
  } catch {
    return null;
  }
}

function Files({ onChoose }: { onChoose: (source: DataSource) => void }) {
  const [path, setPathState] = useState<string | null>(lastFolder);
  const setPath = (p: string | null) => {
    try {
      if (p) localStorage.setItem(FOLDER_KEY, p);
    } catch {
      // without storage the browser just starts at the dataset folders
    }
    setPathState(p);
  };
  const [listing, setListing] = useState<FolderList | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api<FolderList>(`/api/datasets/browse${path ? `?path=${encodeURIComponent(path)}` : ""}`, { signal: controller.signal })
      .then((l) => {
        setListing(l);
        setError(null);
      })
      .catch((e) => {
        if (isAbort(e)) return;
        if (path) setPathState(null); // a remembered folder that's gone: start at the dataset folders instead
        else setError(message(e));
      });
    return () => controller.abort();
  }, [path]);

  if (!listing) return error ? <p className="field-error">{error}</p> : <p className="hint">Loading…</p>;
  const roots = listing.roots;
  return (
    <div className="files">
      <div className="roots">
        <span className="hint">Dataset folders:</span>
        {roots.map((root) => (
          <button type="button" className={`chip${listing.path === root ? " on" : ""}`} key={root} title={root}
            onClick={() => setPath(root)}>
            {root.split("/").filter(Boolean).slice(-2).join("/")}
          </button>
        ))}
      </div>
      {error && <p className="field-error">{error}</p>}
      {listing.path && (
        <>
          <p className="hint path">{listing.path}</p>
          <div className="file-list">
            {listing.parent && (
              <button type="button" className="file-row" onClick={() => setPath(listing.parent)}>
                <IconArrowUp size={16} aria-hidden /> <span className="file-name">Up</span>
              </button>
            )}
            {listing.dirs.map((d) => (
              <button type="button" className="file-row" key={d} onClick={() => setPath(`${listing.path}/${d}`)}>
                <IconFolder size={16} aria-hidden /> <span className="file-name">{d}</span>
              </button>
            ))}
            {(listing.files ?? []).map((f) => (
              <button type="button" className="file-row data-file" key={f.name}
                onClick={() => onChoose({ kind: "file", path: `${listing.path}/${f.name}`, name: f.name })}>
                <IconFile size={16} aria-hidden /> <span className="file-name">{f.name}</span>
                <span className="file-size">{size(f.size)}</span>
              </button>
            ))}
            {!listing.dirs.length && !listing.files?.length && <p className="hint">Nothing here.</p>}
          </div>
        </>
      )}
      {!listing.path && <p className="hint">Pick a folder. anvil reads .parquet, .jsonl, .json and .csv files.</p>}
    </div>
  );
}

function Upload({ onChoose }: { onChoose: (source: DataSource) => void }) {
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  async function send(file: File | undefined) {
    if (!file || progress !== null) return;
    if (!/\.(parquet|jsonl|json|csv)$/i.test(file.name)) {
      setError("Upload a .parquet, .jsonl, .json or .csv file.");
      return;
    }
    setError(null);
    setProgress(0);
    try {
      const saved = await upload<{ path: string; name: string }>(
        `/api/datasets/upload?name=${encodeURIComponent(file.name)}`, file, setProgress);
      onChoose({ kind: "file", path: saved.path, name: saved.name });
    } catch (e) {
      setError(message(e));
    } finally {
      setProgress(null);
    }
  }

  return (
    <div className="upload">
      <label
        className={`upload-drop${dragging ? " dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          send(e.dataTransfer.files?.[0]);
        }}
      >
        <IconUpload size={22} aria-hidden />
        <span>
          {progress === null ? "Drop a file here, or click to choose one" : `Uploading… ${Math.round(progress * 100)}%`}
        </span>
        <span className="hint">.parquet, .jsonl, .json or .csv</span>
        <input type="file" accept=".parquet,.jsonl,.json,.csv" className="hidden-input"
          onChange={(e) => send(e.target.files?.[0])} />
      </label>
      {progress !== null && <div className="bar"><div style={{ width: `${Math.round(progress * 100)}%` }} /></div>}
      {error && <p className="field-error">{error}</p>}
      <p className="hint">The file is stored with anvil's data, so you can pick it again later from This machine.</p>
    </div>
  );
}

function HubDatasets({ onChoose }: { onChoose: (source: DataSource) => void }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<HubDataset[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [repo, setRepo] = useState<string | null>(null);
  const [configs, setConfigs] = useState<string[] | null>(null);
  const [config, setConfig] = useState<string | null>(null);
  const [splits, setSplits] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setResults(null);
      return;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => {
      setSearching(true);
      api<{ datasets: HubDataset[] }>(`/api/hub/datasets/search?q=${encodeURIComponent(q)}`, { signal: controller.signal })
        .then((r) => {
          setResults(r.datasets);
          setError(null);
        })
        .catch((e) => !isAbort(e) && setError(message(e)))
        .finally(() => setSearching(false));
    }, 300);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query]);

  function pick(id: string) {
    setRepo(id);
    setConfigs(null);
    setConfig(null);
    setSplits(null);
    setError(null);
    setLoading(true);
    api<{ configs: string[] }>(`/api/hub/datasets/configs?repo=${encodeURIComponent(id)}`)
      .then((r) => {
        setConfigs(r.configs);
        if (r.configs.length === 1) chooseConfig(id, r.configs[0]);
        else setLoading(false);
      })
      .catch((e) => {
        setError(message(e));
        setLoading(false);
      });
  }

  function chooseConfig(id: string, name: string) {
    setConfig(name);
    setSplits(null);
    setLoading(true);
    api<{ splits: string[] }>(`/api/hub/datasets/splits?repo=${encodeURIComponent(id)}&config=${encodeURIComponent(name)}`)
      .then((r) => setSplits(r.splits))
      .catch((e) => setError(message(e)))
      .finally(() => setLoading(false));
  }

  return (
    <div className="picker">
      <div className="search">
        <IconSearch size={16} aria-hidden />
        <input aria-label="Search Hugging Face datasets" placeholder="prompt injection" value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setRepo(null);
          }} />
        {searching && <IconLoader2 size={16} className="spin" aria-hidden />}
      </div>
      {error && <p className="field-error">{error}</p>}
      {!repo && results?.length === 0 && <p className="hint">No datasets match that search.</p>}
      {!repo && results && results.length > 0 && (
        <div className="results" role="listbox" aria-label="Datasets">
          {results.map((d) => (
            <button type="button" role="option" aria-selected={false} className="result" key={d.id} onClick={() => pick(d.id)}>
              {d.gated || d.private ? <IconLock size={16} aria-hidden /> : <IconDatabase size={16} aria-hidden />}
              <span className="result-main">
                <span className="result-id">{d.id}</span>
                <span className="result-meta">
                  {[`${count(d.downloads)} downloads`, d.size && `${d.size} rows`, d.gated && "gated", d.private && "private"]
                    .filter(Boolean).join(" · ")}
                </span>
              </span>
              <Badge tone={d.permissive ? "ok" : "warn"}>{d.licence ?? "no licence"}</Badge>
            </button>
          ))}
        </div>
      )}
      {repo && (
        <div className="chosen">
          <div className="chosen-head">
            <span>
              Dataset: <code>{repo}</code>
            </span>
            <button type="button" className="link" onClick={() => setRepo(null)}>Back to results</button>
          </div>
          {loading && <p className="hint">Looking at its splits…</p>}
          {configs && configs.length > 1 && (
            <div className="roots">
              <span className="hint">Subset:</span>
              {configs.map((c) => (
                <button type="button" key={c} className={`chip${c === config ? " on" : ""}`} onClick={() => chooseConfig(repo, c)}>
                  {c}
                </button>
              ))}
            </div>
          )}
          {splits && (
            <div className="roots">
              <span className="hint">Split:</span>
              {splits.map((s) => (
                <button type="button" key={s} className="chip"
                  onClick={() => onChoose({ kind: "hub", repo, config, split: s, name: `${repo} · ${s}` })}>
                  {s}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
