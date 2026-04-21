"use client";

import { useMemo } from "react";
import type { CoefRow } from "@/lib/data/types";
import { COUNTRIES, COUNTRY_FLAGS, type Country } from "@/lib/constants";
import { cn } from "@/lib/utils";

interface Props {
  data: Record<Country, CoefRow[]>;
  onCellClick?: (label: string, country: Country) => void;
  activeLabel?: string | null;
}

function interpolate(v: number, min: number, max: number): string {
  // Diverging scale: red (neg) → white → blue (pos)
  if (v === 0) return "#ffffff";
  if (v > 0) {
    const t = Math.min(1, v / max);
    const r = Math.round(255 - (255 - 60) * t);
    const g = Math.round(255 - (255 - 110) * t);
    const b = Math.round(255 - (255 - 168) * t);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    const t = Math.min(1, v / min);
    const r = Math.round(255 - (255 - 184) * t);
    const g = Math.round(255 - (255 - 74) * t);
    const b = Math.round(255 - (255 - 74) * t);
    return `rgb(${r}, ${g}, ${b})`;
  }
}

export function ClusterHeatmap({ data, onCellClick, activeLabel }: Props) {
  // Collect union of labels across all countries, sorted by US coefficient desc
  const labels = useMemo(() => {
    const seen = new Map<string, number>();
    for (const c of COUNTRIES) {
      for (const row of data[c] ?? []) {
        if (!seen.has(row.label)) {
          const usRow = data.US?.find((r) => r.label === row.label);
          seen.set(row.label, usRow?.b ?? row.b ?? 0);
        }
      }
    }
    return [...seen.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([label]) => label);
  }, [data]);

  const { min, max } = useMemo(() => {
    let lo = 0;
    let hi = 0;
    for (const c of COUNTRIES) {
      for (const row of data[c] ?? []) {
        if (row.b !== null) {
          if (row.b < lo) lo = row.b;
          if (row.b > hi) hi = row.b;
        }
      }
    }
    // symmetric domain for diverging
    const m = Math.max(Math.abs(lo), Math.abs(hi));
    return { min: -m, max: m };
  }, [data]);

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] border-collapse text-sm">
        <thead>
          <tr>
            <th className="sticky left-0 bg-background p-2 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Skill cluster
            </th>
            {COUNTRIES.map((c) => (
              <th
                key={c}
                className="p-2 text-center text-xs font-medium uppercase tracking-wider text-muted-foreground"
              >
                {COUNTRY_FLAGS[c]} {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {labels.map((label) => {
            const isActive = activeLabel === label;
            return (
              <tr
                key={label}
                className={cn(
                  "border-t transition-colors",
                  isActive && "bg-primary/5",
                )}
              >
                <td
                  className={cn(
                    "sticky left-0 bg-background px-3 py-1.5 text-left font-medium",
                    isActive && "text-primary",
                  )}
                >
                  {label}
                </td>
                {COUNTRIES.map((c) => {
                  const row = data[c]?.find((r) => r.label === label);
                  if (!row || row.b === null) {
                    return (
                      <td key={c} className="px-1 py-1">
                        <div className="mx-auto h-10 w-full max-w-[110px] rounded-md border border-dashed bg-muted/30 text-center text-xs leading-10 text-muted-foreground">
                          —
                        </div>
                      </td>
                    );
                  }
                  const bg = row.sig === "ns" ? "#eef1f4" : interpolate(row.b, min, max);
                  const textColor = Math.abs(row.b) > Math.max(Math.abs(min), max) * 0.55 && row.sig !== "ns" ? "#fff" : "#222";
                  const ame = (row.b * 100).toFixed(1);
                  return (
                    <td key={c} className="px-1 py-1">
                      <button
                        type="button"
                        onClick={() => onCellClick?.(label, c)}
                        title={`${label} (${c}): AME ${ame} pp · ${row.sig}`}
                        className={cn(
                          "relative mx-auto flex h-10 w-full max-w-[110px] items-center justify-center rounded-md border text-xs font-semibold tabular-nums transition-all hover:scale-[1.03] hover:shadow",
                          row.sig === "ns" && "bg-[repeating-linear-gradient(45deg,#e9ecef_0_4px,transparent_4px_8px)]",
                        )}
                        style={{ backgroundColor: bg, color: textColor }}
                      >
                        {row.b > 0 ? "+" : ""}{ame}
                        <span className="ml-1 text-[10px] opacity-80">
                          {row.sig === "ns" ? "ns" : row.sig}
                        </span>
                      </button>
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="mt-4 flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
        <span className="font-medium">Legend:</span>
        <div className="flex items-center gap-2">
          <span className="inline-block h-3 w-6 rounded" style={{ backgroundColor: "rgb(60,110,168)" }} />
          <span>Positive AME</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block h-3 w-6 rounded" style={{ backgroundColor: "rgb(184,74,74)" }} />
          <span>Negative AME</span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className="inline-block h-3 w-6 rounded border"
            style={{
              backgroundImage:
                "repeating-linear-gradient(45deg,#e9ecef 0 4px,transparent 4px 8px)",
            }}
          />
          <span>Not significant (p ≥ 0.05)</span>
        </div>
        <div className="ml-auto">*** p&lt;0.001 · ** p&lt;0.01 · * p&lt;0.05</div>
      </div>
    </div>
  );
}
