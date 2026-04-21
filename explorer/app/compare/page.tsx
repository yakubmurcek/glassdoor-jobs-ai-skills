"use client";

import { Suspense, useMemo } from "react";
import Link from "next/link";
import { AlertTriangle, ArrowRight, Loader2 } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SlicerBar } from "@/components/slicer/slicer-bar";
import { useSlicer } from "@/lib/state/slicer";
import {
  CLUSTER_KEYS,
  CLUSTER_LABELS,
  filterRows,
  summarizeSlice,
  useRows,
  type CompactRow,
  type Slicer,
} from "@/lib/data/rows-store";
import { powerLevel } from "@/lib/stats";
import { AI_TIER_ORDER, TIER_COLORS } from "@/lib/constants";
import { formatNumber, formatPct } from "@/lib/utils";

export default function ComparePage() {
  return (
    <Suspense fallback={<Loading />}>
      <CompareContent />
    </Suspense>
  );
}

function Loading() {
  return (
    <div className="flex min-h-[400px] items-center justify-center gap-2 text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin" />
      Loading dataset…
    </div>
  );
}

function CompareContent() {
  const { rows, loading, error } = useRows();
  const sliceA = useSlicer("a");
  const sliceB = useSlicer("b");

  const rowsA = useMemo(() => (rows ? filterRows(rows, sliceA.value) : []), [rows, sliceA.value]);
  const rowsB = useMemo(() => (rows ? filterRows(rows, sliceB.value) : []), [rows, sliceB.value]);
  const total = rows?.length ?? 0;
  const sumA = useMemo(() => summarizeSlice(rowsA, total), [rowsA, total]);
  const sumB = useMemo(() => summarizeSlice(rowsB, total), [rowsB, total]);
  const clusterDiff = useMemo(() => buildClusterDiff(rowsA, rowsB), [rowsA, rowsB]);

  if (loading) return <Loading />;
  if (error) {
    return (
      <div className="mx-auto max-w-3xl rounded-md border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
        Could not load <span className="font-mono">rows.json</span> — {error}
      </div>
    );
  }

  const underA = powerLevel(rowsA.length);
  const underB = powerLevel(rowsB.length);

  return (
    <div className="mx-auto max-w-[1500px] space-y-6">
      <PageHeader
        eyebrow="Workbench"
        title="Compare"
        description="Define two slices and contrast every metric side by side. Useful for answering questions like 'does AI presence in senior software engineering differ between US and India?' — build the two slices, read the delta."
      />

      <section className="grid gap-4 md:grid-cols-2">
        <Card className="border-sky-500/40">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Slice A</CardTitle>
            <CardDescription>
              {rowsA.length === 0
                ? "Build a slice on the left."
                : `${formatNumber(rowsA.length)} postings.`}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <SlicerBar handle={sliceA} rows={rows ?? []} dense title="Slice A" />
          </CardContent>
        </Card>
        <Card className="border-amber-500/40">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Slice B</CardTitle>
            <CardDescription>
              {rowsB.length === 0
                ? "Build a slice on the right."
                : `${formatNumber(rowsB.length)} postings.`}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <SlicerBar handle={sliceB} rows={rows ?? []} dense title="Slice B" />
          </CardContent>
        </Card>
      </section>

      {(underA === "tiny" || underB === "tiny") && (
        <div className="flex items-center gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <AlertTriangle className="size-4" />
          At least one slice has fewer than 5 postings. Relax filters to get meaningful comparison.
        </div>
      )}

      <section className="grid gap-4 md:grid-cols-3">
        <DiffCard
          label="Postings N"
          a={sumA.n}
          b={sumB.n}
          format={(v) => formatNumber(v)}
          showDeltaPct={false}
        />
        <DiffCard
          label="AI share"
          a={sumA.ai_share * 100}
          b={sumB.ai_share * 100}
          format={(v) => `${v.toFixed(1)}%`}
          suffixA={`95% CI [${formatPct(sumA.ai_share_ci.lo * 100)}–${formatPct(sumA.ai_share_ci.hi * 100)}]`}
          suffixB={`95% CI [${formatPct(sumB.ai_share_ci.lo * 100)}–${formatPct(sumB.ai_share_ci.hi * 100)}]`}
        />
        <DiffCard
          label="Applied/Core share"
          a={sumA.applied_share * 100}
          b={sumB.applied_share * 100}
          format={(v) => `${v.toFixed(1)}%`}
          suffixA={`95% CI [${formatPct(sumA.applied_share_ci.lo * 100)}–${formatPct(sumA.applied_share_ci.hi * 100)}]`}
          suffixB={`95% CI [${formatPct(sumB.applied_share_ci.lo * 100)}–${formatPct(sumB.applied_share_ci.hi * 100)}]`}
        />
      </section>

      <section className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Tier mix</CardTitle>
            <CardDescription className="text-xs">Stack compares composition at a glance.</CardDescription>
          </CardHeader>
          <CardContent>
            <TierMixStack label="A" mix={sumA.tier_mix} n={sumA.n} />
            <div className="h-2" />
            <TierMixStack label="B" mix={sumB.tier_mix} n={sumB.n} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Mean salary (disclosed)</CardTitle>
            <CardDescription className="text-xs">Per currency, with 95% CI.</CardDescription>
          </CardHeader>
          <CardContent>
            <SalarySection label="A" sum={sumA} />
            <div className="my-2 border-t border-dashed" />
            <SalarySection label="B" sum={sumB} />
          </CardContent>
        </Card>
      </section>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Top cluster lift differences</CardTitle>
          <CardDescription className="text-xs">
            For each skill cluster, share of postings in A vs B that include it. Largest |Δ| first.
          </CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full min-w-[600px] text-xs">
            <thead className="text-muted-foreground">
              <tr>
                <th className="px-2 py-1.5 text-left font-medium">Cluster</th>
                <th className="px-2 py-1.5 text-right font-medium">A</th>
                <th className="px-2 py-1.5 text-right font-medium">B</th>
                <th className="px-2 py-1.5 text-right font-medium">Δ (pp)</th>
              </tr>
            </thead>
            <tbody>
              {clusterDiff.slice(0, 10).map((r) => (
                <tr key={r.key} className="border-t">
                  <td className="px-2 py-1.5 font-medium">{r.label}</td>
                  <td className="px-2 py-1.5 text-right tabular-nums">{(r.pctA * 100).toFixed(1)}%</td>
                  <td className="px-2 py-1.5 text-right tabular-nums">{(r.pctB * 100).toFixed(1)}%</td>
                  <td className={
                    "px-2 py-1.5 text-right font-semibold tabular-nums " +
                    (Math.abs(r.deltaPct) < 1 ? "text-muted-foreground" : r.deltaPct > 0 ? "text-sky-700" : "text-amber-700")
                  }>
                    {r.deltaPct > 0 ? "+" : ""}{r.deltaPct.toFixed(1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Salary distribution overlay</CardTitle>
          <CardDescription className="text-xs">
            Histogram of disclosed salaries (dominant currency per slice). Normalised as share within each slice.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <SalaryHistogram rowsA={rowsA} rowsB={rowsB} />
        </CardContent>
      </Card>

      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <span>Drill into rows:</span>
        <Link
          href={linkForSlice("/explorer", sliceA.value)}
          className="inline-flex items-center gap-1 rounded-md border border-sky-500/40 px-2 py-1 font-medium text-sky-700 hover:bg-sky-500/10"
        >
          Open Slice A in Dataset <ArrowRight className="size-3" />
        </Link>
        <Link
          href={linkForSlice("/explorer", sliceB.value)}
          className="inline-flex items-center gap-1 rounded-md border border-amber-500/40 px-2 py-1 font-medium text-amber-700 hover:bg-amber-500/10"
        >
          Open Slice B in Dataset <ArrowRight className="size-3" />
        </Link>
      </div>
    </div>
  );
}

function DiffCard({
  label,
  a,
  b,
  format,
  suffixA,
  suffixB,
  showDeltaPct = true,
}: {
  label: string;
  a: number;
  b: number;
  format: (v: number) => string;
  suffixA?: string;
  suffixB?: string;
  showDeltaPct?: boolean;
}) {
  const delta = a - b;
  const deltaPct = b === 0 ? null : ((a - b) / Math.abs(b)) * 100;
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 pt-0 text-sm">
        <div className="grid grid-cols-2 gap-4">
          <div className="border-l-2 border-sky-500 pl-2">
            <div className="text-[10px] font-medium uppercase tracking-wider text-sky-700">A</div>
            <div className="text-xl font-semibold tabular-nums">{format(a)}</div>
            {suffixA && <div className="text-[10px] text-muted-foreground">{suffixA}</div>}
          </div>
          <div className="border-l-2 border-amber-500 pl-2">
            <div className="text-[10px] font-medium uppercase tracking-wider text-amber-700">B</div>
            <div className="text-xl font-semibold tabular-nums">{format(b)}</div>
            {suffixB && <div className="text-[10px] text-muted-foreground">{suffixB}</div>}
          </div>
        </div>
        <div className="pt-1 text-xs">
          <span className="text-muted-foreground">Δ A − B: </span>
          <span className={"font-semibold tabular-nums " + (delta >= 0 ? "text-sky-700" : "text-amber-700")}>
            {delta > 0 ? "+" : ""}{format(delta)}
          </span>
          {showDeltaPct && deltaPct !== null && (
            <span className="ml-2 text-muted-foreground">({deltaPct > 0 ? "+" : ""}{deltaPct.toFixed(1)}% rel.)</span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function TierMixStack({
  label,
  mix,
  n,
}: {
  label: string;
  mix: { tier: typeof AI_TIER_ORDER[number]; pct: number; n: number }[];
  n: number;
}) {
  if (n === 0) return <div className="text-xs text-muted-foreground">Slice {label}: —</div>;
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="font-medium">Slice {label}</span>
        <span className="text-muted-foreground">N = {formatNumber(n)}</span>
      </div>
      <div className="flex h-6 overflow-hidden rounded-md">
        {mix.map((t) => (
          <div
            key={t.tier}
            title={`${t.tier}: ${(t.pct * 100).toFixed(1)}% (n=${t.n})`}
            style={{ backgroundColor: TIER_COLORS[t.tier], width: `${t.pct * 100}%` }}
          />
        ))}
      </div>
      <div className="mt-1 flex flex-wrap gap-3 text-[10px] text-muted-foreground">
        {mix.map((t) => (
          <span key={t.tier} className="flex items-center gap-1">
            <span className="inline-block size-2 rounded-sm" style={{ backgroundColor: TIER_COLORS[t.tier] }} />
            {t.tier} {(t.pct * 100).toFixed(1)}%
          </span>
        ))}
      </div>
    </div>
  );
}

function SalarySection({ label, sum }: { label: string; sum: ReturnType<typeof summarizeSlice> }) {
  if (sum.mean_salary_by_currency.length === 0) {
    return <div className="text-xs text-muted-foreground">Slice {label}: no disclosed salaries.</div>;
  }
  return (
    <div>
      <div className="mb-1 text-xs font-medium">Slice {label}</div>
      <ul className="space-y-0.5 text-xs">
        {sum.mean_salary_by_currency.map((s) => (
          <li key={s.currency} className="flex items-center justify-between">
            <span className="tabular-nums">
              {currencySymbol(s.currency)}
              {Math.round(s.mean).toLocaleString()}
            </span>
            <span className="text-muted-foreground">
              n = {formatNumber(s.n)}
              {s.ci ? ` · 95% CI ±${Math.round((s.ci.hi - s.ci.lo) / 2).toLocaleString()}` : ""}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

interface ClusterDiffRow {
  key: string;
  label: string;
  pctA: number;
  pctB: number;
  deltaPct: number; // percentage points
}

function buildClusterDiff(rowsA: CompactRow[], rowsB: CompactRow[]): ClusterDiffRow[] {
  const out: ClusterDiffRow[] = [];
  for (let i = 0; i < CLUSTER_KEYS.length; i += 1) {
    const bit = 1 << i;
    let aK = 0;
    let bK = 0;
    for (const r of rowsA) if (r.cl & bit) aK += 1;
    for (const r of rowsB) if (r.cl & bit) bK += 1;
    const pA = rowsA.length ? aK / rowsA.length : 0;
    const pB = rowsB.length ? bK / rowsB.length : 0;
    out.push({
      key: CLUSTER_KEYS[i],
      label: CLUSTER_LABELS[i],
      pctA: pA,
      pctB: pB,
      deltaPct: (pA - pB) * 100,
    });
  }
  out.sort((a, b) => Math.abs(b.deltaPct) - Math.abs(a.deltaPct));
  return out;
}

function SalaryHistogram({ rowsA, rowsB }: { rowsA: CompactRow[]; rowsB: CompactRow[] }) {
  const pickDominant = (rows: CompactRow[]) => {
    const bucket = new Map<string, number[]>();
    for (const r of rows) {
      if (r.sm !== null && r.cur) {
        let arr = bucket.get(r.cur);
        if (!arr) { arr = []; bucket.set(r.cur, arr); }
        arr.push(r.sm);
      }
    }
    let best: string | null = null;
    let bestArr: number[] = [];
    for (const [cur, arr] of bucket.entries()) {
      if (arr.length > bestArr.length) { best = cur; bestArr = arr; }
    }
    return { currency: best, values: bestArr };
  };
  const a = pickDominant(rowsA);
  const b = pickDominant(rowsB);
  if (a.values.length === 0 && b.values.length === 0) {
    return <div className="p-6 text-center text-sm text-muted-foreground">No disclosed salaries in either slice.</div>;
  }
  // Use combined range for x-axis
  const all = [...a.values, ...b.values];
  const min = Math.min(...all);
  const max = Math.max(...all);
  const bins = 18;
  const step = (max - min) / bins;
  const bucketize = (arr: number[]) => {
    const out = Array.from({ length: bins }, () => 0);
    for (const v of arr) {
      let idx = Math.floor((v - min) / step);
      if (idx === bins) idx = bins - 1;
      out[idx] += 1;
    }
    const total = arr.length || 1;
    return out.map((n) => n / total);
  };
  const aBuckets = bucketize(a.values);
  const bBuckets = bucketize(b.values);
  const maxPct = Math.max(...aBuckets, ...bBuckets, 0.01);
  return (
    <div>
      <div className="mb-1 flex items-center gap-3 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1"><span className="inline-block size-2 rounded-sm bg-sky-500/60" />A {a.currency ? `(${a.currency}, n=${a.values.length})` : ""}</span>
        <span className="flex items-center gap-1"><span className="inline-block size-2 rounded-sm bg-amber-500/60" />B {b.currency ? `(${b.currency}, n=${b.values.length})` : ""}</span>
      </div>
      <div className="flex h-40 items-end gap-0.5">
        {Array.from({ length: bins }, (_, i) => {
          const x0 = min + i * step;
          const x1 = min + (i + 1) * step;
          return (
            <div key={i} className="relative flex flex-1 flex-col items-center justify-end" title={`${Math.round(x0).toLocaleString()}–${Math.round(x1).toLocaleString()}`}>
              <div
                className="w-full rounded-t bg-sky-500/50"
                style={{ height: `${(aBuckets[i] / maxPct) * 100}%`, minHeight: aBuckets[i] > 0 ? 2 : 0 }}
              />
              <div
                className="w-full rounded-t bg-amber-500/50"
                style={{ height: `${(bBuckets[i] / maxPct) * 100}%`, minHeight: bBuckets[i] > 0 ? 2 : 0 }}
              />
            </div>
          );
        })}
      </div>
      <div className="mt-1 flex justify-between text-[10px] tabular-nums text-muted-foreground">
        <span>{Math.round(min).toLocaleString()}</span>
        <span>{Math.round(max).toLocaleString()}</span>
      </div>
    </div>
  );
}

function linkForSlice(base: string, s: Slicer): string {
  const sp = new URLSearchParams();
  if (s.countries.length) sp.set("co", s.countries.join(","));
  if (s.tiers.length) sp.set("t", s.tiers.join(","));
  if (s.jobFamilies.length) sp.set("jf", s.jobFamilies.join(","));
  if (s.seniority.length) sp.set("sen", s.seniority.join(","));
  if (s.edu.length) sp.set("ed", s.edu.join(","));
  if (s.sizeBands.length) sp.set("sz", s.sizeBands.join(","));
  if (s.industries.length) sp.set("ind", s.industries.join(","));
  if (s.states.length) sp.set("state", s.states.join(","));
  if (s.clustersAny.length) sp.set("cany", s.clustersAny.join(","));
  if (s.clustersNone.length) sp.set("cnone", s.clustersNone.join(","));
  if (s.salaryDisclosedOnly) sp.set("sal", "true");
  if (s.search) sp.set("q", s.search);
  if (s.experienceMin != null) sp.set("expMin", String(s.experienceMin));
  if (s.experienceMax != null) sp.set("expMax", String(s.experienceMax));
  const qs = sp.toString();
  return qs ? `${base}?${qs}` : base;
}

function currencySymbol(cur: string | null): string {
  switch (cur) {
    case "USD": return "$";
    case "EUR": return "€";
    case "INR": return "₹";
    default: return cur ? cur + " " : "";
  }
}
