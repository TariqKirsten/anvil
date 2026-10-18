// A small line chart in plain SVG: one or more series over training steps.

export interface Series {
  label: string;
  points: { x: number; y: number }[];
  tone?: "accent" | "ok" | "warn" | "muted";
}

const WIDTH = 520;
const HEIGHT = 170;
const PAD = { left: 44, right: 12, top: 12, bottom: 24 };

function tidy(v: number) {
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

export function Chart({ series, empty }: { series: Series[]; empty: string }) {
  const all = series.flatMap((s) => s.points);
  if (all.length === 0) return <p className="hint chart-empty">{empty}</p>;
  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs, Math.min(...xs) + 1)];
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (y0 === y1) [y0, y1] = [y0 - 0.5, y1 + 0.5];
  const sx = (x: number) => PAD.left + ((x - x0) / (x1 - x0)) * (WIDTH - PAD.left - PAD.right);
  const sy = (y: number) => HEIGHT - PAD.bottom - ((y - y0) / (y1 - y0)) * (HEIGHT - PAD.top - PAD.bottom);
  const grid = [y0, (y0 + y1) / 2, y1];

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={series.map((s) => s.label).join(", ")}>
        {grid.map((g) => (
          <g key={g}>
            <line x1={PAD.left} x2={WIDTH - PAD.right} y1={sy(g)} y2={sy(g)} className="chart-grid" />
            <text x={PAD.left - 6} y={sy(g) + 4} textAnchor="end" className="chart-label">{tidy(g)}</text>
          </g>
        ))}
        <text x={PAD.left} y={HEIGHT - 6} className="chart-label">step {x0}</text>
        <text x={WIDTH - PAD.right} y={HEIGHT - 6} textAnchor="end" className="chart-label">{x1}</text>
        {series.map((s) => (
          <g key={s.label} className={`chart-series chart-${s.tone ?? "accent"}`}>
            <polyline fill="none" points={s.points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")} />
            {s.points.length <= 40 && s.tone !== "muted" && s.points.map((p) => <circle key={p.x} cx={sx(p.x)} cy={sy(p.y)} r={2.5} />)}
          </g>
        ))}
      </svg>
      {series.length > 1 && (
        <div className="chart-legend">
          {series.map((s) => (
            <span key={s.label} className={`chart-key chart-${s.tone ?? "accent"}`}>{s.label}</span>
          ))}
        </div>
      )}
    </div>
  );
}
