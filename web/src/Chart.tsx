// A small line chart in plain SVG: one or more series over training steps. It's drawn at the width it's shown at, so
// the labels keep their size on any screen instead of shrinking with the picture.

import { useEffect, useState } from "react";

export interface Series {
  label: string;
  points: { x: number; y: number }[];
  tone?: "accent" | "ok" | "warn" | "muted";
}

const HEIGHT = 160;
const PAD = { left: 44, right: 12, top: 12, bottom: 24 };

// The decimals an axis needs, from its range, so all its labels have the same number of them.
function decimals(span: number) {
  if (span >= 100) return 0;
  if (span >= 10) return 1;
  if (span >= 0.1) return 2;
  return 3;
}

export function Chart({ series, empty, marker }: { series: Series[]; empty: string; marker?: { x: number; label: string } }) {
  const [box, setBox] = useState<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(520);
  useEffect(() => {
    if (!box) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(240, Math.round(entry.contentRect.width))));
    observer.observe(box);
    return () => observer.disconnect();
  }, [box]);

  const all = series.flatMap((s) => s.points);
  if (all.length === 0) return <p className="hint chart-empty">{empty}</p>;
  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs, Math.min(...xs) + 1)];
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (y0 === y1) [y0, y1] = [y0 - 0.5, y1 + 0.5];
  const sx = (x: number) => PAD.left + ((x - x0) / (x1 - x0)) * (width - PAD.left - PAD.right);
  const sy = (y: number) => HEIGHT - PAD.bottom - ((y - y0) / (y1 - y0)) * (HEIGHT - PAD.top - PAD.bottom);
  const grid = [y0, (y0 + y1) / 2, y1];
  const digits = decimals(y1 - y0);
  const flip = sx(marker?.x ?? 0) > width - 80; // a marker near the right edge gets its label on its left

  return (
    <div className="chart" ref={setBox}>
      <svg viewBox={`0 0 ${width} ${HEIGHT}`} role="img" aria-label={series.map((s) => s.label).join(", ")}>
        {grid.map((g) => (
          <g key={g}>
            <line x1={PAD.left} x2={width - PAD.right} y1={sy(g)} y2={sy(g)} className="chart-grid" />
            <text x={PAD.left - 6} y={sy(g) + 4} textAnchor="end" className="chart-label">{g.toFixed(digits)}</text>
          </g>
        ))}
        {marker && marker.x >= x0 && marker.x <= x1 && (
          <g className="chart-marker">
            <line x1={sx(marker.x)} x2={sx(marker.x)} y1={PAD.top} y2={HEIGHT - PAD.bottom} />
            <text x={sx(marker.x) + (flip ? -4 : 4)} y={PAD.top + 10} textAnchor={flip ? "end" : "start"}
              className="chart-label">{marker.label}</text>
          </g>
        )}
        <text x={PAD.left} y={HEIGHT - 6} className="chart-label">step {x0.toLocaleString()}</text>
        {Math.max(...xs) > x0 && (
          <text x={width - PAD.right} y={HEIGHT - 6} textAnchor="end" className="chart-label">{x1.toLocaleString()}</text>
        )}
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
