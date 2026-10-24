// Short human numbers: 2274069824 params -> "2.27B", 7892933 downloads -> "7.9M".

export function params(n: number | null) {
  if (n === null) return "unknown size";
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 1 : 2)}B`;
  if (n >= 1e6) return `${Math.round(n / 1e6)}M`;
  return `${Math.round(n / 1e3)}k`;
}

export function count(n: number) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${Math.round(n / 1e3)}k`;
  return String(n);
}

export function gb(n: number) {
  return `${Math.round(n)} GB`;
}

// How long, roughly: "45s", "12 min", "2 h 5 min"; a dash when it isn't known.
export function duration(seconds: number | null | undefined) {
  if (seconds == null || !Number.isFinite(seconds)) return "–";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

export function bytes(n: number) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.max(1, Math.round(n / 1e3))} kB`;
}
