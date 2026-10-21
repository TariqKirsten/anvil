// Uploading a version of the run to Hugging Face (server/anvil/upload.py): its weights, its GGUF files, or both, to
// repos on the account whose token is in Settings, private or public. Each repo gets a model card saying what the model
// is, how it scored and how to use it.

import { IconCloudUpload, IconExternalLink, IconX } from "@tabler/icons-react";
import { useEffect, useState } from "react";
import { api } from "./api";
import { bytes } from "./format";
import type { HubMe, RunDetail, UploadStatus } from "./types";
import { Badge, Field, Section, Segmented } from "./ui";

const REPO = /^[\w.-]+\/[\w.-]+$/;

type Version = RunDetail["versions"][number];

// A repo name from the run's name: Hugging Face allows letters, digits, "-", "_" and ".".
function repoName(name: string) {
  return name.trim().replace(/[^\w.-]+/g, "-").replace(/-{2,}/g, "-").replace(/\.{2,}/g, ".")
    .replace(/^[-.]+|[-.]+$/g, "").slice(0, 80) || "model";
}

// "Q4_K_M" from "decision-0-8b-final-Q4_K_M.gguf"
function quantOf(file: string) {
  return file.slice(0, -5).split("-").pop() ?? file;
}

function versionLabel(v: Version, best: number | null) {
  if (v.version === "final") return `Finished model${v.step != null ? ` · step ${v.step.toLocaleString()}` : ""}`;
  return `Step ${(v.step ?? 0).toLocaleString()}${v.step === best ? " · best" : ""}`;
}

function message(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

// Whether a repo exists (as far as the token can see), asked once typing has paused.
function useRepo(repo: string, used: boolean) {
  const [found, setFound] = useState<{ repo: string; exists: boolean; private: boolean | null } | null>(null);
  useEffect(() => {
    if (!used || !REPO.test(repo)) return;
    let alive = true;
    const timer = setTimeout(() => {
      api<{ exists: boolean; private: boolean | null }>(`/api/hub/repo?repo=${encodeURIComponent(repo)}`)
        .then((r) => alive && setFound({ repo, ...r }))
        .catch(() => alive && setFound(null));
    }, 500);
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [repo, used]);
  return found?.repo === repo ? found : null;
}

// What an upload will do to a repo: make it, or add to it, and change its visibility if it isn't the one picked.
function repoNote(found: ReturnType<typeof useRepo>, makePrivate: boolean) {
  const want = makePrivate ? "private" : "public";
  if (!found) return { text: undefined, warn: false };
  if (!found.exists) return { text: `New: anvil makes it, ${want}.`, warn: false };
  if (found.private === makePrivate)
    return { text: `It exists (${want}): the upload adds these files, replacing any of the same name.`, warn: false };
  return { text: `It exists and is ${found.private ? "private" : "public"}: the upload adds these files and makes it ${want}.`, warn: true };
}

export function UploadSection({ run, id, best, onStarted }: {
  run: RunDetail;
  id: string;
  best: number | null;
  onStarted: () => void;
}) {
  const [me, setMe] = useState<HubMe | null>(null);
  const [meError, setMeError] = useState<string | null>(null);
  const [version, setVersion] = useState(run.versions[0]?.version ?? "final");
  const [sendWeights, setSendWeights] = useState(true);
  const [form, setForm] = useState<"merged" | "adapter">("merged");
  const [unticked, setUnticked] = useState<Set<string>>(new Set()); // GGUF files left out (the rest go)
  const [repo, setRepo] = useState<string | null>(null); // null: follow the suggestion
  const [ggufRepo, setGgufRepo] = useState<string | null>(null); // null: the weights' repo, with -GGUF
  const [makePrivate, setMakePrivate] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState<string | null>(null);

  useEffect(() => {
    api<HubMe>("/api/hub/me").then(setMe).catch((e) => setMeError(message(e)));
  }, []);

  const uploads = run.uploads ?? [];
  const going = uploads.some((u) => u.state === "starting" || u.state === "running");
  const exporting = run.exports.some((e) => e.state === "starting" || e.state === "running");
  const lora = run.config.training_type === "lora";
  const chosen = run.versions.find((v) => v.version === version) ?? run.versions[0];
  const files = run.exports.find((e) => e.version === chosen?.version && e.state === "done")?.files ?? [];
  const gguf = files.filter((f) => !unticked.has(f.name));
  const weightsSize = chosen?.sizes?.[lora ? form : "merged"] ?? null;
  const total = (sendWeights ? weightsSize ?? 0 : 0) + gguf.reduce((n, f) => n + f.size, 0);
  const owner = me?.namespaces[0] ?? me?.name ?? "";
  const suggested = `${owner ? `${owner}/` : ""}${repoName(run.config.name)}${chosen?.version === "final" || chosen?.step == null ? "" : `-step-${chosen.step}`}`;
  const weightsRepo = repo ?? suggested;
  const filesRepo = ggufRepo ?? `${weightsRepo}-GGUF`;
  const repoError = (value: string) => (REPO.test(value) ? undefined : "A repository looks like your-name/model-name.");
  const weightsError = sendWeights ? repoError(weightsRepo) : undefined;
  const filesError = gguf.length ? repoError(filesRepo) : undefined;
  const weightsFound = useRepo(weightsRepo, sendWeights);
  const filesFound = useRepo(filesRepo, gguf.length > 0);
  // A namespace the token can't write to: worth saying before Hugging Face refuses (the list can miss some, so it's
  // a warning rather than a refusal).
  const foreign = (value: string) => me?.write && REPO.test(value) && !me.namespaces.includes(value.split("/")[0])
    ? `Your token may not be able to write to ${value.split("/")[0]}: it can write to ${me.namespaces.join(", ")}.` : undefined;
  const nothing = !sendWeights && gguf.length === 0;
  const blocked = !me?.connected || me.write === false;
  const toggle = (name: string) =>
    setUnticked((u) => {
      const next = new Set(u);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });

  async function start() {
    if (!chosen) return;
    setStarting(true);
    setError(null);
    try {
      await api(`/api/runs/${id}/uploads`, { json: {
        version: chosen.version, weights: sendWeights ? (lora ? form : "merged") : null, gguf: gguf.map((f) => f.name),
        repo: sendWeights ? weightsRepo : null, gguf_repo: gguf.length ? filesRepo : null, private: makePrivate,
      } });
      onStarted();
    } catch (e) {
      setError(message(e));
    } finally {
      setStarting(false);
    }
  }

  async function cancel(upload: UploadStatus) {
    setCancelling(upload.id);
    try {
      await api(`/api/runs/${id}/uploads/${upload.id}/cancel`, { method: "POST" });
      onStarted();
    } catch (e) {
      setError(message(e));
    } finally {
      setCancelling(null);
    }
  }

  const weightsNote = repoNote(weightsFound, makePrivate);
  const filesNote = repoNote(filesFound, makePrivate);
  return (
    <Section title="Upload to Hugging Face" aside={me?.connected && me.name && <span className="hint">Connected as {me.name}</span>}>
      {!me && !meError && <p className="hint">Checking your Hugging Face account…</p>}
      {meError && <p className="field-error">{meError}</p>}
      {me && !me.connected && (
        <p className="hint">
          Connect your Hugging Face account first: save a token that can write in <a href="#/settings">Settings</a>.
        </p>
      )}
      {me?.connected && !me.write && (
        <p className="warn-text">
          Your Hugging Face token can only read. Make one that can write at{" "}
          <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">huggingface.co/settings/tokens</a>{" "}
          and save it in <a href="#/settings">Settings</a>.
        </p>
      )}
      {me?.connected && me.write && chosen && (
        <>
          <div className="export-form">
            <Field id="upload-version" label="Version">
              <select id="upload-version" value={chosen.version} onChange={(e) => {
                setVersion(e.target.value);
                setUnticked(new Set());
              }}>
                {run.versions.map((v) => <option key={v.version} value={v.version}>{versionLabel(v, best)}</option>)}
              </select>
            </Field>
            <div className="field">
              <span className="label">What</span>
              <div className="quant-list" role="group" aria-label="What to upload">
                <label className={`quant${sendWeights ? " on" : ""}`}>
                  <input type="checkbox" checked={sendWeights} onChange={() => setSendWeights((s) => !s)} />
                  <span className="quant-name">Weights</span>
                  <span className="quant-size">{weightsSize ? bytes(weightsSize) : ""}</span>
                  <span className="quant-hint">
                    {lora && form === "adapter" ? "the LoRA adapter alone" : "safetensors, the whole model"}
                  </span>
                </label>
                {files.map((f) => (
                  <label key={f.name} className={`quant${!unticked.has(f.name) ? " on" : ""}`}>
                    <input type="checkbox" checked={!unticked.has(f.name)} onChange={() => toggle(f.name)} />
                    <span className="quant-name">{quantOf(f.name)}</span>
                    <span className="quant-size">{bytes(f.size)}</span>
                    <span className="quant-hint upload-file" title={f.name}><span dir="ltr">{f.name}</span></span>
                  </label>
                ))}
              </div>
              {files.length === 0 && (
                <p className="hint">Export this version as GGUF (above) to upload its quantized files too.</p>
              )}
            </div>
            {lora && sendWeights && (
              <Field id="upload-form" label="Weights as" hint={form === "merged"
                ? "The LoRA merged into the base model: loads on its own, like any model."
                : "Small, but it needs the base model to run on, loaded with PEFT."}>
                <Segmented label="Weights as" value={form} onChange={setForm}
                  options={[{ value: "merged", label: "Whole model" }, { value: "adapter", label: "Adapter alone" }]} />
              </Field>
            )}
            {sendWeights && (
              <Field id="upload-repo" label={gguf.length ? "Repository for the weights" : "Repository"}
                error={weightsError} warn={foreign(weightsRepo) ?? (weightsNote.warn ? weightsNote.text : undefined)}
                hint={weightsNote.text}>
                <input id="upload-repo" value={weightsRepo} spellCheck={false} autoComplete="off"
                  aria-invalid={weightsError ? true : undefined} onChange={(e) => setRepo(e.target.value.trim())} />
              </Field>
            )}
            {gguf.length > 0 && (
              <Field id="upload-gguf-repo" label={sendWeights ? "Repository for the GGUF files" : "Repository"}
                error={filesError} warn={foreign(filesRepo) ?? (filesNote.warn ? filesNote.text : undefined)}
                hint={filesNote.text ?? (sendWeights && ggufRepo == null
                  ? "A repo of their own, Hugging Face's usual way; the weights' repo works too." : undefined)}>
                <input id="upload-gguf-repo" value={filesRepo} spellCheck={false} autoComplete="off"
                  aria-invalid={filesError ? true : undefined} onChange={(e) => setGgufRepo(e.target.value.trim())} />
              </Field>
            )}
            <Field id="upload-visibility" label="Visibility" hint={makePrivate
              ? "Only you can see it (and your organisation, for an organisation's repo)."
              : "Anyone can find it and download it."}>
              <Segmented label="Visibility" value={makePrivate ? "private" : "public"} onChange={(v) => setMakePrivate(v === "private")}
                options={[{ value: "private", label: "Private" }, { value: "public", label: "Public" }]} />
            </Field>
          </div>
          <div className="export-actions">
            <button type="button" className="primary" onClick={start}
              disabled={starting || going || exporting || nothing || blocked || !!weightsError || !!filesError}>
              <IconCloudUpload size={16} aria-hidden /> {starting ? "Starting…" : total ? `Upload ${bytes(total)}` : "Upload"}
            </button>
            {nothing && <span className="hint">Pick the weights, a GGUF file, or both.</span>}
            {going && <span className="hint">One upload at a time: wait for this one to finish.</span>}
            {!going && exporting && <span className="hint">Wait for the export to finish.</span>}
          </div>
          {error && <p className="field-error">{error}</p>}
        </>
      )}

      {uploads.length > 0 && (
        <div className="export-list">
          {uploads.slice(0, 5).map((u) => {
            const live = u.state === "starting" || u.state === "running";
            const fraction = u.bytes_total ? Math.min(1, (u.bytes_done ?? 0) / u.bytes_total) : 0;
            const what = [u.weights ? (u.weights === "adapter" ? "the adapter" : "the weights") : null,
              u.gguf?.length ? `${u.gguf.length} GGUF file${u.gguf.length === 1 ? "" : "s"}` : null].filter(Boolean).join(" and ");
            const repos = u.repos ?? [...new Set([u.repo, u.gguf_repo].filter((r): r is string => !!r))]
              .map((r) => ({ id: r, url: `https://huggingface.co/${r}`, state: "waiting" as const, made: null, created: undefined, card: undefined }));
            return (
              <div key={u.id} className="export-item">
                <div className="upload-head">
                  <strong>{u.version === "final" ? "Finished model" : `Step ${Number(u.version.slice(5)).toLocaleString()}`}</strong>
                  <Badge tone={u.state === "done" ? "ok" : u.state === "failed" ? "bad" : u.state === "cancelled" ? "warn" : "accent"}>
                    {u.state === "done" ? "Uploaded" : u.state === "failed" ? "Failed" : u.state === "cancelled" ? "Cancelled" : "Uploading"}
                  </Badge>
                </div>
                <p className="hint upload-what">{what}, {u.private ? "private" : "public"}</p>
                <ul className="export-files upload-repos">
                  {repos.map((r) => (
                    <li key={r.id}>
                      <a href={r.url} target="_blank" rel="noreferrer" title={r.id}>
                        <IconExternalLink size={15} aria-hidden />
                        <span className="export-name"><span dir="ltr">{r.id}</span></span>
                      </a>
                      <span className="file-note">
                        {r.state === "done"
                          ? [r.created ? "made" : r.made ? `made ${r.made}` : "updated", r.card === "kept" ? "your card kept" : null].filter(Boolean).join(", ")
                          : r.state === "running" ? "uploading" : r.state === "failed" ? "not changed" : live ? "next" : ""}
                      </span>
                    </li>
                  ))}
                </ul>
                {live && (
                  <div className="upload-progress">
                    <div className="bar"><div style={{ width: `${fraction * 100}%` }} /></div>
                    <span className="hint">
                      {u.stage}{u.bytes_total ? ` · ${bytes(u.bytes_done ?? 0)} of ${bytes(u.bytes_total)}` : ""}
                    </span>
                    <button type="button" disabled={cancelling === u.id} onClick={() => cancel(u)}>
                      <IconX size={16} aria-hidden /> {cancelling === u.id ? "Cancelling…" : "Cancel"}
                    </button>
                  </div>
                )}
                {u.state === "failed" && u.error && <p className="error-box">{u.error}</p>}
              </div>
            );
          })}
        </div>
      )}
      <p className="hint">
        Each repo gets a model card: what the model is, how it scored on the test set, how to use it and how it was
        trained. A card you've edited on Hugging Face stays as you left it. A repo's files change only once its upload
        has finished, so a cancelled or failed one leaves them as they were.
      </p>
    </Section>
  );
}
