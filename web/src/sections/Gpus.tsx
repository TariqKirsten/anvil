// The GPUs to train on. Works for any number of cards: the note under the cards is worked out from what you pick,
// including where the judge runs (a spare card means no pauses) and the edge cases worth knowing about.

import type { FormState, SectionProps } from "../form";
import { gb } from "../format";
import type { Gpu, GpuList } from "../types";
import { Badge, Section } from "../ui";

// The cards a run would use. Until you pick, that's every card that isn't busy.
export function pickedGpus(form: FormState, list: GpuList | null): number[] {
  const gpus = list?.gpus ?? [];
  if (form.gpus === null) return gpus.filter((g) => !g.busy).map((g) => g.index);
  return form.gpus.filter((i) => gpus.some((g) => g.index === i));
}

function note(gpus: Gpu[], picked: Set<number>) {
  const chosen = gpus.filter((g) => picked.has(g.index));
  if (!chosen.length) return { main: "Pick at least one card.", bad: true, warnings: [] as string[], info: [] as string[] };
  const spare = gpus.find((g) => !picked.has(g.index) && !g.busy);
  const trains = chosen.length === 1 ? "1 card trains" : `${chosen.length} cards train at once`;
  const main = spare
    ? `${trains}. GPU ${spare.index} judges checkpoints, so training never pauses.`
    : `${trains}. Checkpoint checks pause training briefly.`;
  const warnings: string[] = [];
  const sizes = [...new Set(chosen.map((g) => Math.round(g.total_gb)))];
  if (sizes.length > 1)
    warnings.push(`The cards differ in size. The ${Math.min(...sizes)} GB card limits the batch, and the slowest card sets the pace.`);
  for (const g of chosen.filter((g) => g.busy))
    warnings.push(`GPU ${g.index} is busy (${g.used_gb} GB in use), so training there may run out of memory.`);
  const info = chosen
    .filter((g) => g.display)
    .map((g) => `GPU ${g.index} also drives your screen, so the desktop may stutter while it trains.`);
  return { main, bad: false, warnings, info };
}

export function GpuSection({ form, update, list, error }: SectionProps & { list: GpuList | null; error: string | null }) {
  const gpus = list?.gpus ?? [];
  const picked = new Set(pickedGpus(form, list));
  const toggle = (index: number) => {
    const next = new Set(picked);
    if (next.has(index)) next.delete(index);
    else next.add(index);
    update((f) => ({ ...f, gpus: [...next].sort((a, b) => a - b) }));
  };
  const n = note(gpus, picked);

  return (
    <Section title="GPUs">
      {error && <p className="field-error">{error}</p>}
      {!list && !error && <p className="hint">Looking for GPUs…</p>}
      {list && gpus.length === 0 && (
        <p className="field-error">
          No GPUs found{list.runtime.error ? ` (${list.runtime.error})` : ""}. Training would fall back to the CPU, which
          is far too slow for real runs.
        </p>
      )}
      {gpus.length > 0 && (
        <>
          <div className="gpu-grid">
            {gpus.map((g) => (
              <label key={g.index} className={`gpu${picked.has(g.index) ? " selected" : ""}`}>
                <input
                  type="checkbox"
                  aria-label={`Train on GPU ${g.index}`}
                  checked={picked.has(g.index)}
                  onChange={() => toggle(g.index)}
                />
                <span className="gpu-main">
                  <span className="gpu-name">
                    GPU {g.index} · {g.name}
                  </span>
                  <span className="gpu-meta">
                    {gb(g.total_gb)}
                    {g.arch && ` · ${g.arch}`}
                    {g.used_gb !== null && ` · ${g.used_gb} GB in use`}
                  </span>
                  {(g.display || g.busy) && (
                    <span className="gpu-tags">
                      {g.display && <Badge>drives your screen</Badge>}
                      {g.busy && <Badge tone="warn">busy</Badge>}
                    </span>
                  )}
                </span>
              </label>
            ))}
          </div>
          <div className="gpu-note">
            <p className={n.bad ? "field-error" : undefined}>{n.main}</p>
            {n.warnings.map((w) => (
              <p key={w} className="warn-text">
                {w}
              </p>
            ))}
            {picked.size > 1 && (
              <p className="hint">For now a run trains on one card, preferring one that doesn't drive a screen. Several at once arrives in step 5.</p>
            )}
            {n.info.map((i) => (
              <p key={i} className="hint">
                {i}
              </p>
            ))}
          </div>
        </>
      )}
    </Section>
  );
}
