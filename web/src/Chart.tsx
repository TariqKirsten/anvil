// A small line chart in plain SVG: one or more series over training steps. It's drawn at the width it's shown at, so
// the labels keep their size on any screen instead of shrinking with the picture. Pointing at it (or touching it) shows
// the values at the nearest step. ChartSection puts one in a section that opens it bigger, over the page.

import { IconArrowsMaximize, IconX } from "@tabler/icons-react";
import { useEffect, useLayoutEffect, useRef, useState, type MouseEvent, type PointerEvent, type ReactNode } from "react";
import { Section } from "./ui";

export interface Series {
  label: string;
  points: { x: number; y: number }[];
  tone?: "accent" | "ok" | "warn" | "muted";
}

const HEIGHT = 160;
const PAD = { left: 44, right: 12, top: 12, bottom: 24 };
const TIP_GAP = 10; // between the line at the step pointed at and the box of values
const TAP = 6; // a press that moves less than this many pixels is a click (or a tap), not a slide along the values

// The decimals an axis needs, from its range, so all its labels have the same number of them.
function decimals(span: number) {
  if (span >= 100) return 0;
  if (span >= 10) return 1;
  if (span >= 0.1) return 2;
  return 3;
}

// Round steps between x0 and x1, 1, 2 or 5 times a power of ten apart: as many as fit in `count`, and at least two.
function ticks(x0: number, x1: number, count: number) {
  const between = (gap: number) => Math.floor(x1 / gap) - Math.ceil(x0 / gap) + 1;
  const gaps = [];
  for (let power = 10 ** Math.floor(Math.log10(Math.max(1, (x1 - x0) / 50))); power <= Math.max(1, x1); power *= 10)
    gaps.push(power, 2 * power, 5 * power);
  let fits = gaps.findIndex((g) => between(g) <= Math.max(2, count));
  if (fits === -1) fits = gaps.length - 1;
  else if (fits > 0 && between(gaps[fits]) < 2) fits -= 1; // a single mark says little: one round step closer
  const gap = gaps[fits] ?? Math.max(1, x1 - x0);
  const found = [];
  for (let t = Math.ceil(x0 / gap) * gap; t <= x1; t += gap) found.push(t);
  return found;
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

export interface ChartProps {
  series: Series[];
  empty: string;
  marker?: { x: number; label: string };
  // How a value reads on the axis and in the box of values, for numbers the decimals of the range don't suit (a
  // learning rate of 5e-5, say).
  format?: (y: number) => string;
  height?: number; // drawn this tall, in pixels
  detailed?: boolean; // more lines across, and the steps along the bottom (the big chart)
  onOpen?: () => void; // a click or a tap opens it bigger
}

export function Chart({ series, empty, marker, format, height = HEIGHT, detailed = false, onOpen }: ChartProps) {
  const [box, setBox] = useState<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(520);
  const [hover, setHover] = useState<number | null>(null); // the step whose values show
  const tipRef = useRef<HTMLDivElement>(null);
  const pressed = useRef<{ x: number; y: number } | null>(null); // where a press started, to tell a click from a slide
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
    tip.style.top = `${Math.max(0, Math.min(top, height * scale - h))}px`;
  });

  const all = series.flatMap((s) => s.points);
  if (all.length === 0) return <p className="hint chart-empty">{empty}</p>;
  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs, Math.min(...xs) + 1)];
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (y0 === y1) [y0, y1] = [y0 - 0.5, y1 + 0.5];
  const sx = (x: number) => PAD.left + ((x - x0) / (x1 - x0)) * (width - PAD.left - PAD.right);
  const sy = (y: number) => height - PAD.bottom - ((y - y0) / (y1 - y0)) * (height - PAD.top - PAD.bottom);
  const grid = (detailed ? [0, 0.25, 0.5, 0.75, 1] : [0, 0.5, 1]).map((f) => y0 + f * (y1 - y0));
  const digits = decimals(y1 - y0);
  const flip = sx(marker?.x ?? 0) > width - 80; // a marker near the right edge gets its label on its left
  // The big chart marks round steps along the bottom, up to one per 90 pixels, instead of only its first and last.
  const marks = detailed && Math.max(...xs) > x0 ? ticks(x0, x1, Math.floor((width - PAD.left - PAD.right) / 90)) : [];

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

  function tapped(e: { clientX: number; clientY: number }) {
    const from = pressed.current;
    return !!from && Math.hypot(e.clientX - from.x, e.clientY - from.y) < TAP;
  }

  // With somewhere bigger to open, a click or a tap opens it; sliding along the chart still shows the values.
  function open(e: MouseEvent<SVGSVGElement>) {
    if (!onOpen || !tapped(e)) return;
    setHover(null);
    onOpen();
  }

  return (
    <div className={`chart${onOpen ? " chart-opens" : ""}`} ref={setBox}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={series.map((s) => s.label).join(", ")}
        onPointerMove={point}
        // A touch shows the values when it lifts, so a finger scrolling the page past the chart doesn't flash them.
        onPointerDown={(e) => {
          pressed.current = { x: e.clientX, y: e.clientY };
          if (e.pointerType !== "touch") point(e);
        }}
        onPointerUp={(e) => !(onOpen && e.pointerType === "touch" && tapped(e)) && point(e)}
        onClick={open}
        onPointerLeave={(e) => e.pointerType !== "touch" && setHover(null)}
        onPointerCancel={() => setHover(null)}>
        {grid.map((g) => (
          <g key={g}>
            <line x1={PAD.left} x2={width - PAD.right} y1={sy(g)} y2={sy(g)} className="chart-grid" />
            <text x={PAD.left - 6} y={sy(g) + 4} textAnchor="end" className="chart-label">{format ? format(g) : g.toFixed(digits)}</text>
          </g>
        ))}
        {marks.map((t) => (
          <line key={t} x1={sx(t)} x2={sx(t)} y1={PAD.top} y2={height - PAD.bottom} className="chart-grid chart-grid-x" />
        ))}
        {marker && marker.x >= x0 && marker.x <= x1 && (
          <g className="chart-marker">
            <line x1={sx(marker.x)} x2={sx(marker.x)} y1={PAD.top} y2={height - PAD.bottom} />
            <text x={sx(marker.x) + (flip ? -4 : 4)} y={PAD.top + 10} textAnchor={flip ? "end" : "start"}
              className="chart-label">{marker.label}</text>
          </g>
        )}
        {marks.length > 0 ? (
          marks.map((t) => (
            // Centred under its line, but kept inside the picture at either end.
            <text key={t} x={sx(t)} y={height - 6} className="chart-label"
              textAnchor={sx(t) - PAD.left < 24 ? "start" : width - PAD.right - sx(t) < 24 ? "end" : "middle"}>
              {t === marks[0] ? `step ${t.toLocaleString()}` : t.toLocaleString()}
            </text>
          ))
        ) : (
          <>
            <text x={PAD.left} y={height - 6} className="chart-label">step {x0.toLocaleString()}</text>
            {Math.max(...xs) > x0 && (
              <text x={width - PAD.right} y={height - 6} textAnchor="end" className="chart-label">{x1.toLocaleString()}</text>
            )}
          </>
        )}
        {series.map((s) => (
          <g key={s.label} className={`chart-series chart-${s.tone ?? "accent"}`}>
            <polyline fill="none" points={s.points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")} />
            {s.points.length <= 40 && s.tone !== "muted" && s.points.map((p) => <circle key={p.x} cx={sx(p.x)} cy={sy(p.y)} r={2.5} />)}
          </g>
        ))}
        {at != null && (
          <g className="chart-hover">
            <line x1={sx(at)} x2={sx(at)} y1={PAD.top} y2={height - PAD.bottom} />
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
              <span className="chart-tip-value">{format ? format(p.y) : p.y.toFixed(Math.min(4, digits + 1))}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// The big chart's height: most of the window, so it reads closely, with room for its title and legend.
function bigHeight() {
  return Math.round(Math.max(220, Math.min(560, window.innerHeight - 200)));
}

// A chart in its own section, which opens bigger in a dialog over the page: with a click or a tap on it, or its
// button. The big one follows the run live, like the small one. children: what goes under the small one (a hint).
export function ChartSection({ title, children, ...chart }: ChartProps & { title: string; children?: ReactNode }) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [open, setOpen] = useState(false);
  const [height, setHeight] = useState(bigHeight);
  const drawn = chart.series.some((s) => s.points.length > 0);

  useEffect(() => {
    if (!open) return;
    const fit = () => setHeight(bigHeight());
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, [open]);

  // The dialog itself takes the focus, rather than its close button: Escape closes it, and Tab gets to the button.
  function show() {
    setHeight(bigHeight());
    setOpen(true);
    dialogRef.current?.showModal();
    dialogRef.current?.focus();
  }

  return (
    <Section title={title} aside={drawn && (
      <button type="button" className="chart-expand" aria-label={`Open the ${title.toLowerCase()} chart bigger`}
        title="Open bigger" onClick={show}>
        <IconArrowsMaximize size={16} aria-hidden />
      </button>
    )}>
      <Chart {...chart} onOpen={drawn ? show : undefined} />
      {children}
      <dialog ref={dialogRef} className="chart-dialog" aria-label={title} tabIndex={-1} onClose={() => setOpen(false)}
        // A click beside the box, on the dimmed page, closes it too.
        onClick={(e) => e.target === e.currentTarget && dialogRef.current?.close()}>
        <div className="chart-dialog-body">
          <div className="chart-dialog-head">
            <h2>{title}</h2>
            <button type="button" className="icon" aria-label="Close" title="Close" onClick={() => dialogRef.current?.close()}>
              <IconX size={16} aria-hidden />
            </button>
          </div>
          {open && <Chart {...chart} height={height} detailed />}
        </div>
      </dialog>
    </Section>
  );
}
