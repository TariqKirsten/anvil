// Under the hood of a run as it trains: tiles for the stats row (the gradient norm, tokens trained on, padding, rows
// cut to fit max length) and a second row of charts (the gradient norm before clipping, the learning rate over the
// steps). Runs from before these numbers were kept show what they have: the learning rate, and no tiles.

import { Chart, type Series } from "./Chart";
import type { RunDetail, RunMetric, RunStatus } from "./types";
import { Section } from "./ui";

// 1234 -> "1,234", 1234567 -> "1.23M", 2.5e9 -> "2.50B"
function compact(n: number) {
  if (n < 1e6) return n.toLocaleString();
  if (n < 1e9) return `${(n / 1e6).toFixed(n < 1e7 ? 2 : 1)}M`;
  return `${(n / 1e9).toFixed(2)}B`;
}

// A learning rate the way the settings write it: 5e-5, 2.5e-5.
export function rate(v: number) {
  return v === 0 ? "0" : Number(v.toPrecision(2)).toExponential().replace("e+0", "");
}

function percent(share: number) {
  return `${(100 * share).toFixed(share < 0.1 ? 1 : 0)}%`;
}

export function MoreStats({ status: st }: { status: RunStatus }) {
  if (st.tokens_seen == null) return null; // a run from before these were kept
  const cut = st.truncated ?? 0;
  // Of the rows seen so far: a step takes items_per_step rows (skipped steps aside).
  const seen = st.items_per_step && st.step ? st.step * st.items_per_step : null;
  return (
    <>
      <div>
        <span className="stat-label">Gradient norm</span>
        <span className="stat-value">{st.grad_norm != null ? st.grad_norm.toFixed(st.grad_norm >= 10 ? 1 : 2) : "–"}</span>
      </div>
      <div>
        <span className="stat-label">Tokens trained on</span>
        <span className="stat-value">{compact(st.tokens_seen)}</span>
      </div>
      <div>
        <span className="stat-label">Padding</span>
        <span className="stat-value">{st.padding != null ? percent(st.padding) : "–"}</span>
      </div>
      <div>
        <span className="stat-label">Rows cut by max length</span>
        <span className="stat-value">{cut.toLocaleString()}{seen && cut ? ` · ${percent(cut / seen)}` : ""}</span>
      </div>
    </>
  );
}

export function TrainingCharts({ train, head, hyper }: {
  train: RunMetric[];
  head: boolean; // a Clef head, with its own learning rate and gradient norm
  hyper: RunDetail["config"]["hyper"];
}) {
  const has = (key: string) => train.some((m) => typeof m[key] === "number");
  const points = (key: string) =>
    train.filter((m) => typeof m[key] === "number").map((m) => ({ x: m.step, y: Number(m[key]) }));
  if (!train.length) return null;
  // The largest step of each log line goes on the chart while it's within reach of the mean; a spike far above the
  // rest would squash the mean line flat, so that peak is named instead.
  const means = points("grad_norm");
  const maxes = points("grad_norm_max");
  const peak = maxes.reduce((top, p) => (p.y > top.y ? p : top), { x: 0, y: 0 });
  const typical = Math.max(...means.map((p) => p.y), 0);
  const drawMax = peak.y <= 5 * typical;
  const norms: Series[] = [
    ...(drawMax ? [{ label: "largest step", points: maxes, tone: "muted" as const }] : []),
    { label: "mean", points: means, tone: "accent" },
  ];
  if (head && has("head_grad_norm")) norms.push({ label: "head", points: points("head_grad_norm"), tone: "warn" });
  const rates: Series[] = [{ label: "model", points: points("lr"), tone: "accent" }];
  if (head && has("head_lr")) rates.push({ label: "head", points: points("head_lr"), tone: "warn" });
  const warmup = Number(hyper.warmupPercent);
  const schedule = `${Number.isFinite(warmup) && warmup > 0 ? `${warmup}% warmup, then ` : ""}${hyper.scheduler ?? "cosine"}`;
  return (
    <div className="chart-grid-2">
      {has("grad_norm") && (
        <Section title="Gradient norm">
          <Chart empty="The first norms show up with the first loss." series={norms} />
          <p className="hint">
            The size of the gradient before it's clipped to 1, for each log line: the mean over its steps
            {drawMax ? " and the largest step" : ""}. Over 1 means the update was clipped. A spike is a batch that pushed
            hard; a norm that climbs late in a run says the learning rate is high for it.
            {head && " The head's is clipped on its own."}
            {!drawMax && ` The largest single step reached ${peak.y >= 100 ? Math.round(peak.y).toLocaleString() : peak.y.toFixed(1)} at step ${peak.x.toLocaleString()}, too far above the rest to draw.`}
          </p>
        </Section>
      )}
      <Section title="Learning rate">
        <Chart empty="The first steps show up with the first loss." series={rates} format={rate} />
        <p className="hint">
          {schedule.charAt(0).toUpperCase() + schedule.slice(1)}
          {hyper.scheduler === "cosine" || hyper.scheduler === "linear" ? ", down to 0 at the last step." : "."}
          {head && " The head has its own rate; the model's is 0 while the head trains alone."}
        </p>
      </Section>
    </div>
  );
}
