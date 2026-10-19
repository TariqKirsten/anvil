// A small line chart in plain SVG: one or more series over training steps. It's drawn at the width it's shown at, so
// the labels keep their size on any screen instead of shrinking with the picture. Pointing at it (or touching it) shows
// the values at the nearest step.

import { useEffect, useLayoutEffect, useRef, useState, type PointerEvent } from "react";

export interface Series {
  label: string;
  points: { x: number; y: number }[];
  tone?: "accent" | "ok" | "warn" | "muted";
}

const HEIGHT = 160;
const PAD = { left: 44, right: 12, top: 12, bottom: 24 };
const TIP_GAP = 10; // between the line at the step pointed at and the box of values

// The decimals an axis needs, from its range, so all its labels have the same number of them.
function decimals(span: number) {
  if (span >= 100) return 0;
  if (span >= 10) return 1;
  if (span >= 0.1) return 2;
  return 3;
}

// The step nearest to x, from steps in order.
function nearest(steps: number[], x: number) {
  let [lo, hi] = [0, steps.length - 1];
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (steps[mid] <= x) lo = mid;
    else hi = mid;
  }
  return steps[hi] - x < x - steps[lo] ? steps[hi] : steps[lo];
}

export function Chart({ series, empty, marker }: { series: Series[]; empty: string; marker?: { x: number; label: string } }) {
  const [box, setBox] = useState<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(520);
  const [hover, setHover] = useState<number | null>(null); // the step whose values show
  const tipRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!box) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(240, Math.round(entry.contentRect.width))));
    observer.observe(box);
    return () => observer.disconnect();
  }, [box]);

  // A finger doesn't hover, so the values it brought up stay until a touch anywhere else.
  const showing = hover != null;
  useEffect(() => {
    if (!showing || !box) return;
    const away = (e: globalThis.PointerEvent) => {
      if (!box.contains(e.target as Node)) setHover(null);
    };
    document.addEventListener("pointerdown", away);
    return () => document.removeEventListener("pointerdown", away);
  }, [showing, box]);

  // The box of values goes beside the line, on whichever side has room, level with the points. On a chart too narrow
  // for that (a phone), it goes above the points or below them, so it never covers them. Its size is only known once
  // it's drawn, so it's placed here, before it shows.
  useLayoutEffect(() => {
    const tip = tipRef.current;
    if (!tip || !box) return;
    const room = box.clientWidth;
    const scale = room / width; // the picture shrinks to fit below 240px
    const [x, high, low] = [tip.dataset.x, tip.dataset.high, tip.dataset.low].map((v) => Number(v) * scale);
    const [w, h] = [tip.offsetWidth, tip.offsetHeight];
    let [left, top] = [x + TIP_GAP, (high + low - h) / 2];
    if (left + w > room) left = x - TIP_GAP - w;
    if (left < 0) [left, top] = [x - w / 2, high - TIP_GAP - h >= 0 ? high - TIP_GAP - h : low + TIP_GAP];
    tip.style.left = `${Math.max(0, Math.min(left, room - w))}px`;
    tip.style.top = `${Math.max(0, Math.min(top, HEIGHT * scale - h))}px`;
  });

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

  const steps = [...new Set(xs)].sort((a, b) => a - b);
  const hits = series.flatMap((s) => {
    const p = hover == null ? undefined : s.points.find((q) => q.x === hover);
    return p ? [{ s, p }] : [];
  });
  const at = hits[0]?.p.x; // the step the values are at (hover, while it's still in the data)

  function point(e: PointerEvent<SVGSVGElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * width;
    setHover(nearest(steps, x0 + ((px - PAD.left) / (width - PAD.left - PAD.right)) * (x1 - x0)));
  }

  return (
    <div className="chart" ref={setBox}>
      <svg viewBox={`0 0 ${width} ${HEIGHT}`} role="img" aria-label={series.map((s) => s.label).join(", ")}
        onPointerMove={point}
        // A touch shows the values when it lifts, so a finger scrolling the page past the chart doesn't flash them.
        onPointerDown={(e) => e.pointerType !== "touch" && point(e)}
        onPointerUp={point}
        onPointerLeave={(e) => e.pointerType !== "touch" && setHover(null)}
        onPointerCancel={() => setHover(null)}>
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
        {at != null && (
          <g className="chart-hover">
            <line x1={sx(at)} x2={sx(at)} y1={PAD.top} y2={HEIGHT - PAD.bottom} />
            {hits.map(({ s, p }) => (
              <circle key={s.label} className={`chart-${s.tone ?? "accent"}`} cx={sx(p.x)} cy={sy(p.y)} r={4} />
            ))}
          </g>
        )}
      </svg>
      {series.length > 1 && (
        <div className="chart-legend">
          {series.map((s) => (
            <span key={s.label} className={`chart-key chart-${s.tone ?? "accent"}`}>{s.label}</span>
          ))}
        </div>
      )}
      {at != null && (
        <div className="chart-tip" ref={tipRef} aria-hidden data-x={sx(at)}
          data-high={Math.min(...hits.map(({ p }) => sy(p.y)))} data-low={Math.max(...hits.map(({ p }) => sy(p.y)))}>
          <div className="chart-tip-step">
            Step {at.toLocaleString()}{at === 0 ? " (before training)" : ""}{marker?.x === at ? ` · ${marker.label}` : ""}
          </div>
          {hits.map(({ s, p }) => (
            <div key={s.label} className="chart-tip-row">
              <span className={`chart-key chart-${s.tone ?? "accent"}`}>{s.label}</span>
              <span className="chart-tip-value">{p.y.toFixed(Math.min(4, digits + 1))}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
