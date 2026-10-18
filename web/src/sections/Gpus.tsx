// The GPUs to train on. Works for any number of cards: the note under the cards says what each picked card will do,
// including where the checkpoint checks run (a spare card means no pauses), and the edge cases worth knowing about.

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

// What a run does with the cards: the same choice the server makes (runs._choose_gpus). One card trains for now
// (several at once is step 5), preferring one that doesn't drive a screen. Another free card, a ticked one first,
// checks the checkpoints, so training never pauses for them.
function plan(gpus: Gpu[], picked: Set<number>) {
  const trains = gpus
    .filter((g) => picked.has(g.index))
    .sort((a, b) => Number(a.display) - Number(b.display) || a.index - b.index)[0];
  const checks = gpus
    .filter((g) => g !== trains && !g.busy)
    .sort((a, b) => Number(!picked.has(a.index)) - Number(!picked.has(b.index)) || a.index - b.index)[0];
  return { trains, checks };
}

function note(gpus: Gpu[], picked: Set<number>) {
  const { trains, checks } = plan(gpus, picked);
  if (!trains) return { main: "Pick at least one card.", bad: true, warnings: [] as string[], info: [] as string[] };
  const main = checks
    ? `GPU ${trains.index} trains. GPU ${checks.index} checks the checkpoints, so training never pauses for them.`
    : `GPU ${trains.index} trains, and pauses briefly at each checkpoint to check it.`;
  const warnings = trains.busy
    ? [`GPU ${trains.index} is busy (${trains.used_gb} GB in use), so training there may run out of memory.`]
    : [];
  const info: string[] = [];
  if (checks && !picked.has(checks.index)) info.push(`GPU ${checks.index} isn't ticked, but it's free, so it does the checks.`);
  if (picked.size > 1) info.push("For now a run trains on one card. Training on several at once arrives in step 5.");
  if (trains.display) info.push(`GPU ${trains.index} also drives your screen, so the desktop may stutter while it trains.`);
  if (checks?.display) info.push(`GPU ${checks.index} drives your screen, so the desktop may stutter during checks.`);
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
