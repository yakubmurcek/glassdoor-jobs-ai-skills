"use client";

import { Suspense, useMemo } from "react";
import { useQueryState, parseAsString, parseAsStringLiteral, parseAsBoolean } from "nuqs";
import { Loader2, X } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { SlicerBar } from "@/components/slicer/slicer-bar";
import { ClusterHeatmap } from "@/components/charts/cluster-heatmap";
import { getG5, getClusters } from "@/lib/data/loaders";
import {
  AI_TIER_ORDER,
  COUNTRIES,
  COUNTRY_FLAGS,
  COUNTRY_LABELS,
  TIER_COLORS,
  type Country,
} from "@/lib/constants";
import type { CoefRow } from "@/lib/data/types";
import { CLUSTER_KEYS, CLUSTER_LABELS, filterRows, useRows, type CompactRow } from "@/lib/data/rows-store";
import { useSlicer } from "@/lib/state/slicer";
import { wilsonCi } from "@/lib/stats";
import { formatNumber, formatPct, sigColor } from "@/lib/utils";

const SORT_MODES = ["us_abs", "disagreement", "sig", "alpha"] as const;
type SortMode = (typeof SORT_MODES)[number];
const sortParser = parseAsStringLiteral(SORT_MODES).withDefault("us_abs");

const SORT_LABEL: Record<SortMode, string> = {
  us_abs: "US |AME| ↓",
  disagreement: "Cross-country disagreement",
  sig: "Significance (US)",
  alpha: "Alphabetical",
};

export default function ClustersPage() {
  return (
    <Suspense fallback={<Loading />}>
      <ClustersContent />
    </Suspense>
  );
}

function Loading() {
  return (
    <div className="flex min-h-[400px] items-center justify-center gap-2 text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin" />
      Loading cluster data…
    </div>
  );
}

function normalizeLabel(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function ClustersContent() {
  const g5 = getG5();
  const clusterFreq = getClusters();
  const { rows } = useRows();
  const slicer = useSlicer();
  const [activeLabel, setActiveLabel] = useQueryState("cluster", parseAsString);
  const [sort, setSort] = useQueryState("sort", sortParser);
  const [onlySig, setOnlySig] = useQueryState("sig", parseAsBoolean.withDefault(false));

  const filtered = useMemo(() => (rows ? filterRows(rows, slicer.value) : []), [rows, slicer.value]);

  // Build ordered list of labels
  const orderedLabels = useMemo(() => {
    const labels = new Set<string>();
    for (const c of COUNTRIES) for (const r of g5[c] ?? []) labels.add(r.label);
    const arr = [...labels];
    const valueForSort = (label: string) => {
      switch (sort) {
        case "us_abs":
          return Math.abs(g5.US?.find((r) => r.label === label)?.b ?? 0);
        case "disagreement": {
          const vals = COUNTRIES.map((c) => g5[c]?.find((r) => r.label === label)?.b ?? 0);
          return Math.max(...vals) - Math.min(...vals);
        }
        case "sig": {
          const s = g5.US?.find((r) => r.label === label)?.sig ?? "ns";
          const order = { "***": 4, "**": 3, "*": 2, ns: 1 };
          return (order as Record<string, number>)[s] ?? 0;
        }
        case "alpha":
          return label;
      }
    };
    if (sort === "alpha") {
      arr.sort((a, b) => String(valueForSort(a)).localeCompare(String(valueForSort(b))));
    } else {
      arr.sort((a, b) => (valueForSort(b) as number) - (valueForSort(a) as number));
    }
    if (onlySig) {
      return arr.filter((l) => (g5.US?.find((r) => r.label === l)?.sig ?? "ns") !== "ns");
    }
    return arr;
  }, [g5, sort, onlySig]);

  const heatmapData = useMemo(() => {
    const out: Record<Country, CoefRow[]> = { US: [], DE: [], IN: [] };
    for (const c of COUNTRIES) {
      out[c] = orderedLabels
        .map((l) => g5[c]?.find((r) => r.label === l))
        .filter((r): r is CoefRow => Boolean(r));
    }
    return out;
  }, [g5, orderedLabels]);

  // Drilldown: cluster data for the active label
  const drilldown = useMemo(() => {
    if (!activeLabel) return null;
    const clusterRow = clusterFreq.find((c) => normalizeLabel(c.label) === normalizeLabel(activeLabel));
    // Best-effort match cluster key from CLUSTER_KEYS
    const clusterKeyIdx = CLUSTER_LABELS.findIndex((l) => normalizeLabel(l) === normalizeLabel(activeLabel));
    const clusterKey = clusterKeyIdx >= 0 ? CLUSTER_KEYS[clusterKeyIdx] : null;
    const bit = clusterKeyIdx >= 0 ? 1 << clusterKeyIdx : 0;

    const perCountryCoef = COUNTRIES.map((c) => ({
      country: c,
      row: g5[c]?.find((r) => r.label === activeLabel) ?? null,
    }));

    // Row-level stats within current slicer
    let withCluster: CompactRow[] = [];
    let withoutCluster: CompactRow[] = [];
    if (bit && filtered.length > 0) {
      for (const r of filtered) {
        if ((r.cl & bit) !== 0) withCluster.push(r);
        else withoutCluster.push(r);
      }
    }

    const tierMix = (rows: CompactRow[]) => {
      const total = rows.filter((r) => r.t !== null).length;
      if (total === 0) return null;
      const counts: [number, number, number] = [0, 0, 0];
      for (const r of rows) if (r.t !== null) counts[r.t] += 1;
      return {
        total,
        mix: AI_TIER_ORDER.map((t, i) => ({ tier: t, pct: counts[i] / total, n: counts[i] })),
      };
    };

    const withMix = tierMix(withCluster);
    const withoutMix = tierMix(withoutCluster);

    // AI share lift
    const aiShare = (arr: CompactRow[]) => {
      const total = arr.filter((r) => r.t !== null).length;
      const ai = arr.filter((r) => r.t !== null && r.t >= 1).length;
      return total === 0 ? { p: 0, n: 0, ci: { lo: 0, hi: 0 } } : { p: ai / total, n: total, ci: wilsonCi(ai, total) };
    };
    const aiWith = aiShare(withCluster);
    const aiWithout = aiShare(withoutCluster);

    return {
      label: activeLabel,
      clusterKey,
      clusterRow,
      perCountryCoef,
      withCluster,
      withoutCluster,
      withMix,
      withoutMix,
      aiWith,
      aiWithout,
    };
  }, [activeLabel, clusterFreq, filtered, g5]);

  return (
    <div className="mx-auto max-w-[1500px] space-y-6">
      <PageHeader
        eyebrow="Section 5.3"
        title="Skill clusters → AI requirement"
        description="Cross-country logit AMEs plus real row-level drill-down. Click a cell to see how postings with that cluster compare to those without, within whatever slice you've built."
      />

      <div className="grid gap-6 lg:grid-cols-[280px_minmax(0,1fr)_340px]">
        <div>
          <SlicerBar handle={slicer} rows={rows ?? []} />
        </div>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <CardTitle className="text-base">Cross-country AME heatmap</CardTitle>
                <CardDescription className="text-xs">
                  Values in percentage points. Click a cell to drill down.
                </CardDescription>
              </div>
              <div className="flex flex-wrap gap-3">
                <div className="flex flex-col gap-1">
                  <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                    Sort
                  </span>
                  <select
                    value={sort}
                    onChange={(e) => setSort(e.target.value as SortMode)}
                    className="h-8 rounded-md border border-input bg-background px-2 text-xs"
                  >
                    {SORT_MODES.map((m) => (
                      <option key={m} value={m}>{SORT_LABEL[m]}</option>
                    ))}
                  </select>
                </div>
                <label className="flex items-end gap-1.5 pb-1 text-xs">
                  <input
                    type="checkbox"
                    className="accent-primary"
                    checked={onlySig}
                    onChange={(e) => setOnlySig(e.target.checked || null)}
                  />
                  Only significant
                </label>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <ClusterHeatmap
              data={heatmapData}
              activeLabel={activeLabel}
              onCellClick={(label) => setActiveLabel(label)}
            />
          </CardContent>
        </Card>

        <aside>
          {drilldown ? (
            <Card className="sticky top-4 max-h-[calc(100svh-2rem)] overflow-y-auto">
              <CardHeader>
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <CardDescription>Cluster detail</CardDescription>
                    <CardTitle className="text-base">{drilldown.label}</CardTitle>
                  </div>
                  <Button variant="ghost" size="icon" onClick={() => setActiveLabel(null)} aria-label="Close">
                    <X className="size-4" />
                  </Button>
                </div>
                {filtered.length !== (rows?.length ?? 0) && (
                  <Badge variant="outline" className="mt-2 w-fit text-[10px]">
                    Within slicer · {formatNumber(filtered.length)} postings
                  </Badge>
                )}
              </CardHeader>
              <CardContent className="space-y-4 text-sm">
                <div>
                  <div className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    AME by country (Model C)
                  </div>
                  <div className="space-y-1">
                    {drilldown.perCountryCoef.map(({ country, row }) => (
                      <div key={country} className="flex items-center justify-between">
                        <span className="text-muted-foreground">
                          {COUNTRY_FLAGS[country]} {COUNTRY_LABELS[country]}
                        </span>
                        {row && row.b !== null ? (
                          <span className={sigColor(row.sig) + " tabular-nums"}>
                            {row.b > 0 ? "+" : ""}{(row.b * 100).toFixed(1)} pp · {row.sig}
                          </span>
                        ) : (
                          <span className="text-xs text-muted-foreground">—</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {drilldown.withCluster.length > 0 || drilldown.withoutCluster.length > 0 ? (
                  <div>
                    <div className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Row-level lift in this slice
                    </div>
                    <div className="space-y-1 text-xs">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">With cluster</span>
                        <span className="tabular-nums">
                          {formatPct(drilldown.aiWith.p * 100)} AI · n = {formatNumber(drilldown.aiWith.n)}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="font-medium">Without cluster</span>
                        <span className="tabular-nums">
                          {formatPct(drilldown.aiWithout.p * 100)} AI · n = {formatNumber(drilldown.aiWithout.n)}
                        </span>
                      </div>
                      <div className="flex items-center justify-between pt-1">
                        <span className="font-semibold">Observed lift</span>
                        <span className={
                          "font-semibold tabular-nums " +
                          ((drilldown.aiWith.p - drilldown.aiWithout.p) * 100 > 0 ? "text-sky-700" : "text-amber-700")
                        }>
                          {((drilldown.aiWith.p - drilldown.aiWithout.p) * 100) > 0 ? "+" : ""}
                          {((drilldown.aiWith.p - drilldown.aiWithout.p) * 100).toFixed(1)} pp
                        </span>
                      </div>
                    </div>

                    {drilldown.withMix && drilldown.withoutMix && (
                      <div className="mt-3 space-y-2">
                        <MixBar label="With" mix={drilldown.withMix.mix} total={drilldown.withMix.total} />
                        <MixBar label="Without" mix={drilldown.withoutMix.mix} total={drilldown.withoutMix.total} />
                      </div>
                    )}
                  </div>
                ) : null}

                {drilldown.clusterRow && (
                  <div>
                    <div className="mb-2 flex items-baseline justify-between">
                      <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                        Defining skills
                      </span>
                      <span className="text-[10px] tabular-nums text-muted-foreground">
                        {formatNumber(drilldown.clusterRow.frequency)} postings total ({formatPct(drilldown.clusterRow.pct)})
                      </span>
                    </div>
                    <ul className="space-y-0.5 text-xs">
                      {drilldown.clusterRow.top_skills.slice(0, 12).map((s) => (
                        <li key={s.skill} className="flex items-center justify-between">
                          <span className="truncate pr-2 font-medium">{s.skill}</span>
                          <span className="shrink-0 tabular-nums text-muted-foreground">{s.pct.toFixed(1)}%</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {drilldown.withCluster.length > 0 && (
                  <div>
                    <div className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Example postings (3 of {formatNumber(drilldown.withCluster.length)})
                    </div>
                    <ul className="space-y-1 text-xs">
                      {drilldown.withCluster.slice(0, 3).map((r, i) => (
                        <li key={i} className="rounded border px-2 py-1">
                          <div className="font-medium">{r.jt ?? "—"}</div>
                          <div className="text-[10px] text-muted-foreground">
                            {COUNTRY_FLAGS[r.co]} · {r.jf ?? "—"} · {r.cp ?? "—"}
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Click a cell</CardTitle>
                <CardDescription className="text-xs">
                  Drilldown will show per-country AME, the observed AI-share lift inside your current slice, and example postings that actually include the cluster.
                </CardDescription>
              </CardHeader>
            </Card>
          )}
        </aside>
      </div>
    </div>
  );
}

function MixBar({
  label,
  mix,
  total,
}: {
  label: string;
  mix: { tier: typeof AI_TIER_ORDER[number]; pct: number; n: number }[];
  total: number;
}) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[10px] text-muted-foreground">
        <span className="font-medium">{label}</span>
        <span>N = {formatNumber(total)}</span>
      </div>
      <div className="flex h-4 overflow-hidden rounded">
        {mix.map((t) => (
          <div
            key={t.tier}
            title={`${t.tier}: ${(t.pct * 100).toFixed(1)}% (n=${t.n})`}
            style={{ backgroundColor: TIER_COLORS[t.tier], width: `${t.pct * 100}%` }}
          />
        ))}
      </div>
    </div>
  );
}

