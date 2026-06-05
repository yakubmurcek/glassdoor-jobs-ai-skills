"use client";

/**
 * Lightweight overlapping-density chart for comparing distributions.
 * Bins values into a histogram and renders as filled area paths in SVG.
 * No D3, no recharts — just SVG paths.
 *
 * The viewBox width tracks the container width via ResizeObserver so text
 * labels render at native size (no horizontal stretching).
 */

import { useEffect, useRef, useState } from "react";

interface Series {
  label: string;
  color: string;
  values: number[];
}

interface Props {
  series: Series[];
  /** Domain override; auto if undefined */
  domain?: [number, number];
  bins?: number;
  height?: number;
  formatX?: (v: number) => string;
  /** Limit max value (e.g. clip salary tail) */
  clip?: [number, number];
}

function clipValues(values: number[], lo?: number, hi?: number): number[] {
  if (lo === undefined && hi === undefined) return values;
  return values.filter((v) => (lo === undefined || v >= lo) && (hi === undefined || v <= hi));
}

function makeBins(values: number[], bins: number, lo: number, hi: number): number[] {
  const counts = new Array(bins).fill(0);
  if (hi === lo) return counts;
  const step = (hi - lo) / bins;
  for (const v of values) {
    if (v < lo || v > hi) continue;
    let idx = Math.floor((v - lo) / step);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    counts[idx] += 1;
  }
  return counts;
}

function smooth(arr: number[], window: number = 3): number[] {
  const n = arr.length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i += 1) {
    let sum = 0;
    let count = 0;
    for (let k = -window; k <= window; k += 1) {
      const j = i + k;
      if (j < 0 || j >= n) continue;
      sum += arr[j];
      count += 1;
    }
    out[i] = count === 0 ? 0 : sum / count;
  }
  return out;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const m = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[m - 1] + sorted[m]) / 2 : sorted[m];
}

export function DensityChart({
  series,
  domain,
  bins = 36,
  height = 220,
  formatX = (v) => v.toFixed(0),
  clip,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState<number>(600);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    function update() {
      if (el) setWidth(Math.max(280, Math.round(el.getBoundingClientRect().width)));
    }
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const cleanedSeries = series.map((s) => ({
    ...s,
    values: clip ? clipValues(s.values, clip[0], clip[1]) : s.values,
  }));

  // Auto-domain: 2nd–98th percentile pooled, with 3% padding
  const allValues = cleanedSeries.flatMap((s) => s.values);
  if (allValues.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-md border bg-muted/30 text-xs text-muted-foreground"
        style={{ height }}
      >
        No data
      </div>
    );
  }
  let lo: number;
  let hi: number;
  if (domain) {
    [lo, hi] = domain;
  } else {
    const sorted = [...allValues].sort((a, b) => a - b);
    lo = sorted[Math.floor(sorted.length * 0.02)];
    hi = sorted[Math.ceil(sorted.length * 0.98) - 1] ?? sorted[sorted.length - 1];
    const span = hi - lo;
    lo -= span * 0.03;
    hi += span * 0.03;
  }

  const W = width;
  const H = height;
  const PADL = 8;
  const PADR = 8;
  const PADT = 8;
  const PADB = 28;

  const innerW = W - PADL - PADR;
  const innerH = H - PADT - PADB;

  // Compute smoothed densities per series, scaled to maxFreq across all
  const smoothed = cleanedSeries.map((s) => {
    const counts = makeBins(s.values, bins, lo, hi);
    const sm = smooth(counts, 2);
    return { ...s, dens: sm };
  });
  const maxDens = Math.max(...smoothed.map((s) => Math.max(...s.dens, 0)), 1);

  const xAt = (i: number) => PADL + (i / Math.max(1, bins - 1)) * innerW;
  const yAt = (d: number) => PADT + innerH - (d / maxDens) * innerH;

  return (
    <div ref={containerRef} style={{ width: "100%", height: H }}>
      <svg
        width={W}
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ display: "block", maxWidth: "100%" }}
      >
        {/* baseline */}
        <line
          x1={PADL}
          x2={W - PADR}
          y1={PADT + innerH}
          y2={PADT + innerH}
          stroke="#e5e7eb"
        />
        {/* grid x-ticks (5) */}
        {Array.from({ length: 5 }, (_, i) => {
          const t = i / 4;
          const x = PADL + t * innerW;
          const v = lo + t * (hi - lo);
          return (
            <g key={i}>
              <line x1={x} x2={x} y1={PADT + innerH} y2={PADT + innerH + 4} stroke="#9ca3af" />
              <text x={x} y={PADT + innerH + 16} fontSize={10} fill="#6b7280" textAnchor="middle">
                {formatX(v)}
              </text>
            </g>
          );
        })}
        {/* density paths */}
        {smoothed.map((s) => {
          let d = `M ${xAt(0)} ${yAt(0)}`;
          for (let i = 0; i < s.dens.length; i += 1) {
            d += ` L ${xAt(i)} ${yAt(s.dens[i])}`;
          }
          d += ` L ${xAt(s.dens.length - 1)} ${yAt(0)} Z`;
          return (
            <g key={s.label}>
              <path d={d} fill={s.color} fillOpacity={0.22} stroke={s.color} strokeWidth={1.6} />
            </g>
          );
        })}
        {/* median tickmarks */}
        {smoothed.map((s) => {
          if (s.values.length === 0) return null;
          const med = median(s.values);
          if (med < lo || med > hi) return null;
          const x = PADL + ((med - lo) / Math.max(1e-9, hi - lo)) * innerW;
          return (
            <g key={`m-${s.label}`}>
              <line
                x1={x}
                x2={x}
                y1={PADT + innerH - 6}
                y2={PADT + innerH + 4}
                stroke={s.color}
                strokeWidth={2}
              />
            </g>
          );
        })}
      </svg>
      {/* legend */}
      <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-muted-foreground">
        {smoothed.map((s) => (
          <div key={s.label} className="flex items-center gap-1.5">
            <span
              className="inline-block size-2.5 rounded-sm"
              style={{ background: s.color, opacity: 0.7 }}
            />
            <span>{s.label}</span>
            <span className="tabular-nums opacity-70">
              n={s.values.length.toLocaleString()} · med {formatX(median(s.values))}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
