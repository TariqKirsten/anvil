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

// What a run does with the cards: the same choice the server makes (runs._choose_gpus). Every ticked card trains, each
// on its share of every batch, led by one that doesn't drive a screen. A free card you didn't tick checks the
// checkpoints, so training never pauses for them; without one, the training cards pause and check together.
function plan(gpus: Gpu[], picked: Set<number>) {
  const trains = gpus
    .filter((g) => picked.has(g.index))
    .sort((a, b) => Number(a.display) - Number(b.display) || a.index - b.index);
  const checks = gpus.filter((g) => !picked.has(g.index) && !g.busy).sort((a, b) => a.index - b.index)[0];
  return { trains, checks };
}

// "GPU 1", "GPUs 1 and 0", "GPUs 1, 2 and 0"
function names(cards: Gpu[]) {
  const ids = cards.map((g) => String(g.index));
  return ids.length === 1 ? `GPU ${ids[0]}` : `GPUs ${ids.slice(0, -1).join(", ")} and ${ids[ids.length - 1]}`;
}

function note(gpus: Gpu[], picked: Set<number>) {
  const { trains, checks } = plan(gpus, picked);
  if (!trains.length) return { main: "Pick at least one card.", bad: true, warnings: [] as string[], info: [] as string[] };
  const together = trains.length > 1;
  const main =
    (together ? `${names(trains)} train together, each on its share of every batch.` : `${names(trains)} trains.`) +
    (checks
      ? ` GPU ${checks.index} checks the checkpoints, so training never pauses for them.`
      : together
        ? " At each checkpoint they pause briefly and check it together."
        : " It pauses briefly at each checkpoint to check it.");
  const warnings = trains
    .filter((g) => g.busy)
    .map((g) => `GPU ${g.index} is busy (${g.used_gb} GB in use), so training there may run out of memory.`);
  const sizes = [...new Set(trains.map((g) => Math.round(g.total_gb)))];
  if (sizes.length > 1)
    warnings.push(`The cards differ in memory: the batch has to fit on the ${Math.min(...sizes)} GB card, and the slowest card sets the pace.`);
  const info: string[] = [];
  if (checks && together) info.push(`GPU ${checks.index} isn't ticked but it's free, so it does the checks.`);
  else if (checks) info.push(`GPU ${checks.index} isn't ticked but it's free, so it does the checks. Tick it to train on it too.`);
  else if (together) info.push("To check without pausing, untick a card: a free card you didn't tick does the checks.");
  for (const g of trains.filter((g) => g.display))
    info.push(`GPU ${g.index} also drives your screen, so the desktop may stutter while it trains. Close video calls and games on it until the run ends.`);
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
