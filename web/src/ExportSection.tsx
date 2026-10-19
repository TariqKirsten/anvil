// Exporting a version of a run as GGUF (server/anvil/export.py): pick the finished model or a kept checkpoint and the
// types to make; the files appear below when they're ready, to download.

import { IconDownload, IconPackageExport } from "@tabler/icons-react";
import { useState } from "react";
import { api } from "./api";
import { bytes } from "./format";
import { METRIC_LABEL } from "./Runs";
import type { RunDetail } from "./types";
import { Badge, Field, Section } from "./ui";

// The types llama-quantize makes, with roughly how many bits each weight takes (for the size estimates).
const QUANTS: { name: string; bits: number; hint: string }[] = [
  { name: "BF16", bits: 16, hint: "full quality" },
  { name: "Q8_0", bits: 8.5, hint: "next to lossless" },
  { name: "Q6_K", bits: 6.56, hint: "very close" },
  { name: "Q5_K_M", bits: 5.69, hint: "close" },
  { name: "Q4_K_M", bits: 4.85, hint: "smallest, fine for most uses" },
];

type Version = RunDetail["versions"][number];

function versionLabel(v: Version, best: number | null) {
  const score = v.score != null && v.metric ? ` · ${METRIC_LABEL[v.metric] ?? v.metric} ${v.score.toFixed(3)}` : "";
  if (v.version === "final") return `Finished model${v.step != null ? ` · step ${v.step.toLocaleString()}` : ""}${score}`;
  return `Step ${(v.step ?? 0).toLocaleString()}${score}${v.step === best ? " · best" : ""}`;
}

function exportName(version: string) {
  return version === "final" ? "Finished model" : `Step ${Number(version.slice(5)).toLocaleString()}`;
}

export function ExportSection({ run, id, best, onStarted }: {
  run: RunDetail;
  id: string;
  best: number | null;
  onStarted: () => void;
}) {
  const [version, setVersion] = useState(run.versions[0]?.version ?? "final");
  const [quants, setQuants] = useState<Set<string>>(new Set(["Q8_0", "Q4_K_M"]));
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const running = run.exports.some((e) => e.state === "starting" || e.state === "running");
  const params = run.export_params ?? (run.config.model.params ?? 0) + (run.status.head_m ?? 0) * 1e6;
  const clef = run.config.train_as === "decision" && run.config.method === "clef";
  const toggle = (name: string) =>
    setQuants((q) => {
      const next = new Set(q);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });

  async function start() {
    setStarting(true);
    setError(null);
    try {
      await api(`/api/runs/${id}/exports`, { json: { version, quants: [...quants] } });
      onStarted();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  }

  return (
    <Section title="Export as GGUF">
      <div className="export-form">
        <Field id="export-version" label="Version">
          <select id="export-version" value={version} onChange={(e) => setVersion(e.target.value)}>
            {run.versions.map((v) => (
              <option key={v.version} value={v.version}>
                {versionLabel(v, best)}
              </option>
            ))}
          </select>
        </Field>
        <div className="field">
          <span className="label">Types</span>
          <div className="quant-list" role="group" aria-label="Types to export">
            {QUANTS.map((q) => (
              <label key={q.name} className={`quant${quants.has(q.name) ? " on" : ""}`}>
                <input type="checkbox" checked={quants.has(q.name)} onChange={() => toggle(q.name)} />
                <span className="quant-name">{q.name}</span>
                <span className="quant-size">{params ? `~${bytes((params * q.bits) / 8)}` : ""}</span>
                <span className="quant-hint">{q.hint}</span>
              </label>
            ))}
          </div>
        </div>
      </div>
      <div className="export-actions">
        <button type="button" className="primary" disabled={running || starting || quants.size === 0} onClick={start}>
          <IconPackageExport size={16} aria-hidden /> {starting ? "Starting…" : "Export"}
        </button>
        {running && <span className="hint">One export at a time: wait for this one to finish.</span>}
      </div>
      {error && <p className="field-error">{error}</p>}

      {run.exports.length > 0 && (
        <div className="export-list">
          {run.exports.map((e) => (
            <div key={e.version} className="export-item">
              <div className="export-head">
                <strong>{exportName(e.version)}</strong>
                <span className="hint">{(e.quants ?? []).join(", ")}</span>
                <Badge tone={e.state === "done" ? "ok" : e.state === "failed" ? "bad" : "accent"}>
                  {e.state === "done" ? "Ready" : e.state === "failed" ? "Failed" : e.stage}
                </Badge>
              </div>
              {e.state === "failed" && e.error && <p className="error-box">{e.error}</p>}
              {e.state === "done" && (
                <ul className="export-files">
                  {(e.files ?? []).map((f) => (
                    <li key={f.name}>
                      <a href={`/api/runs/${id}/exports/${e.version}/${f.name}`} download>
                        <IconDownload size={15} aria-hidden /> {f.name}
                      </a>
                      <span className="file-size">{bytes(f.size)}</span>
                    </li>
                  ))}
                  <li>
                    <a href={`/api/runs/${id}/exports/${e.version}/anvil.json`} download>
                      <IconDownload size={15} aria-hidden /> anvil.json
                    </a>
                    <span className="file-size">what it is, how to prompt it</span>
                  </li>
                </ul>
              )}
            </div>
          ))}
        </div>
      )}
      <p className="hint">
        Made with llama.cpp on the CPU (the GPUs stay free); a checkpoint is merged into the base model first. The files
        run in llama.cpp, LM Studio, Ollama and the like.{" "}
        {clef
          ? "A Clef export is a decision GGUF with its head: llama-server answers SystemOne requests with it at /v1/systemone."
          : run.config.train_as === "decision"
            ? "Prompt it the letters way; anvil.json says how."
            : ""}
      </p>
    </Section>
  );
}
