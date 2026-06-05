"use client";

import { useMemo, useState } from "react";
import {
  COUNTRIES,
  COUNTRY_COLORS,
  COUNTRY_FLAGS,
  type Country,
} from "@/lib/constants";
import { useRows, type CompactRow, SENIORITY_ORDER } from "@/lib/data/rows-store";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { formatNumber, formatPct } from "@/lib/utils";
import { wilsonCi } from "@/lib/stats";

type Metric = "ai_share" | "applied_share";

interface Cell {
  family: string;
  seniority: NonNullable<CompactRow["sen"]>;
  n: number;
  ai: number;
  applied: number;
}

const FAMILIES_DEFAULT = [
  "Software Engineer",
  "Software Developer",
  "Senior Software Engineer",
  "Data & AI",
  "DevOps & Cloud",
  "Management",
  "Other",
];

function buildMatrix(rows: readonly CompactRow[], country: Country): Cell[] {
  const map = new Map<string, Cell>();
  for (const r of rows) {
    if (r.co !== country) continue;
    if (!r.jf || !r.sen || r.t === null) continue;
    const key = `${r.jf}__${r.sen}`;
    let cell = map.get(key);
    if (!cell) {
      cell = { family: r.jf, seniority: r.sen, n: 0, ai: 0, applied: 0 };
      map.set(key, cell);
    }
    cell.n += 1;
    if (r.t >= 1) cell.ai += 1;
    if (r.t === 2) cell.applied += 1;
  }
  return [...map.values()];
}

function colorFor(value: number, max: number, accent: string): string {
  if (max === 0) return "rgba(0,0,0,0.04)";
  const t = Math.min(1, value / max);
  // mix with white based on intensity
  const opacity = 0.08 + t * 0.85;
  return `${accent}${Math.round(opacity * 255).toString(16).padStart(2, "0")}`;
}

interface Props {
  defaultCountry?: Country;
  defaultMetric?: Metric;
}

export function SeniorityFamilyHeatmap({ defaultCountry = "US", defaultMetric = "ai_share" }: Props) {
  const { rows, loading, error } = useRows();
  const [country, setCountry] = useState<Country>(defaultCountry);
  const [metric, setMetric] = useState<Metric>(defaultMetric);
  const [hovered, setHovered] = useState<Cell | null>(null);

  const cells = useMemo(() => (rows ? buildMatrix(rows, country) : []), [rows, country]);

  const { families, max } = useMemo(() => {
    const famSet = new Set<string>();
    let m = 0;
    for (const c of cells) {
      famSet.add(c.family);
      if (c.n < 10) continue;
      const v = metric === "ai_share" ? c.ai / c.n : c.applied / c.n;
      if (v > m) m = v;
    }
    const ordered = FAMILIES_DEFAULT.filter((f) => famSet.has(f)).concat(
      [...famSet].filter((f) => !FAMILIES_DEFAULT.includes(f)),
    );
    return { families: ordered, max: m };
  }, [cells, metric]);

  const cellMap = useMemo(() => {
    const m = new Map<string, Cell>();
    for (const c of cells) m.set(`${c.family}__${c.seniority}`, c);
    return m;
  }, [cells]);

  if (loading) {
    return <div className="flex h-44 items-center justify-center text-xs text-muted-foreground">Loading…</div>;
  }
  if (error) {
    return <div className="text-xs text-destructive">{error}</div>;
  }

  const accent = COUNTRY_COLORS[country];
  const seniorities = SENIORITY_ORDER.filter(Boolean) as NonNullable<CompactRow["sen"]>[];

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-2">
          <ToggleGroup
            type="single"
            value={country}
            onValueChange={(v: string) => v && setCountry(v as Country)}
          >
            {COUNTRIES.map((c) => (
              <ToggleGroupItem key={c} value={c}>
                {COUNTRY_FLAGS[c]} {c}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <ToggleGroup
            type="single"
            value={metric}
            onValueChange={(v: string) => v && setMetric(v as Metric)}
          >
            <ToggleGroupItem value="ai_share">AI share</ToggleGroupItem>
            <ToggleGroupItem value="applied_share">Applied/Core</ToggleGroupItem>
          </ToggleGroup>
        </div>
        <div className="text-xs text-muted-foreground">
          {hovered ? (
            <span className="rounded-md bg-card px-2 py-1 shadow-sm">
              <strong>{hovered.family}</strong> · <em>{hovered.seniority}</em> ·{" "}
              {formatNumber(hovered.n)} postings · AI {formatPct((hovered.ai / Math.max(1, hovered.n)) * 100, 1)} ·
              Applied/Core {formatPct((hovered.applied / Math.max(1, hovered.n)) * 100, 1)}
            </span>
          ) : (
            "Hover any cell · grey = n < 10"
          )}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr>
              <th className="w-44 px-2 pb-2 text-left text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Job family ↓ / Seniority →
              </th>
              {seniorities.map((s) => (
                <th key={s} className="px-2 pb-2 text-center text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  {s}
                </th>
              ))}
              <th className="px-2 pb-2 text-center text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Row n
              </th>
            </tr>
          </thead>
          <tbody>
            {families.map((fam) => {
              const rowN = seniorities.reduce(
                (s, sen) => s + (cellMap.get(`${fam}__${sen}`)?.n ?? 0),
                0,
              );
              return (
                <tr key={fam}>
                  <td className="px-2 py-1 font-medium">{fam}</td>
                  {seniorities.map((sen) => {
                    const c = cellMap.get(`${fam}__${sen}`);
                    if (!c || c.n < 10) {
                      return (
                        <td key={sen} className="p-1">
                          <div
                            className="flex h-12 items-center justify-center rounded-md border text-[10px] text-muted-foreground"
                            style={{ background: "rgba(0,0,0,0.025)" }}
                            title={`${c?.n ?? 0} postings — too few`}
                          >
                            {c?.n ?? 0}
                          </div>
                        </td>
                      );
                    }
                    const pctVal = metric === "ai_share" ? c.ai / c.n : c.applied / c.n;
                    return (
                      <td key={sen} className="p-1">
                        <div
                          onMouseEnter={() => setHovered(c)}
                          onMouseLeave={() => setHovered((cur) => (cur === c ? null : cur))}
                          className="flex h-12 cursor-pointer flex-col items-center justify-center rounded-md border text-[11px] font-semibold transition-shadow hover:shadow"
                          style={{ background: colorFor(pctVal, max, accent) }}
                          title={`${fam} · ${sen} — ${formatPct(pctVal * 100, 1)} (n=${c.n})`}
                        >
                          <span className="tabular-nums">{formatPct(pctVal * 100, 1)}</span>
                          <span className="text-[9px] font-normal opacity-70">n={formatNumber(c.n)}</span>
                        </div>
                      </td>
                    );
                  })}
                  <td className="px-2 py-1 text-right text-[10px] tabular-nums text-muted-foreground">
                    {formatNumber(rowN)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="flex flex-wrap items-center gap-3 text-[10px] text-muted-foreground">
        <span>Cells coloured by {metric === "ai_share" ? "AI mention rate" : "Applied/Core share"}, scaled to country max</span>
        <span className="ml-auto inline-flex items-center gap-1.5">
          0%
          <span className="h-2 w-24 rounded" style={{ background: `linear-gradient(to right, ${accent}10, ${accent})` }} />
          {formatPct(max * 100, 0)}
        </span>
      </div>
    </div>
  );
}
