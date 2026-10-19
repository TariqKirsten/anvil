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

export function bytes(n: number) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.max(1, Math.round(n / 1e3))} kB`;
}
