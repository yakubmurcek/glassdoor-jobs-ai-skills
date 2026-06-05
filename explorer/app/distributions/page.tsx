"use client";

import { Suspense, useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { DensityChart } from "@/components/charts/density-chart";
import { useRows, type CompactRow } from "@/lib/data/rows-store";
import {
  AI_TIER_ORDER,
  COUNTRIES,
  COUNTRY_COLORS,
  COUNTRY_FLAGS,
  COUNTRY_LABELS,
  EDU_LABELS,
  TIER_COLORS,
  type AITier,
  type Country,
} from "@/lib/constants";
import { formatNumber, formatPct } from "@/lib/utils";

type Lens = "country" | "tier";
type Currency = "USD" | "EUR" | "INR";

const CURRENCY_OF: Record<Country, Currency> = {
  US: "USD",
  DE: "EUR",
  IN: "INR",
};

function fmtCurrency(v: number, cur: Currency): string {
  if (cur === "INR") {
    if (v >= 1_000_000) return `₹${(v / 100_000).toFixed(0)}L`;
    if (v >= 100_000) return `₹${(v / 100_000).toFixed(1)}L`;
    return `₹${(v / 1000).toFixed(0)}k`;
  }
  if (cur === "USD") return `$${(v / 1000).toFixed(0)}k`;
  if (cur === "EUR") return `€${(v / 1000).toFixed(0)}k`;
  return v.toFixed(0);
}

export default function DistributionsPage() {
  return (
    <Suspense fallback={<LoadingShell />}>
      <DistributionsContent />
    </Suspense>
  );
}

function LoadingShell() {
  return (
    <div className="flex min-h-[400px] items-center justify-center gap-2 text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin" />
      Loading distributions…
    </div>
  );
}

function DistributionsContent() {
  const { rows, loading, error } = useRows();
  const [salaryCountry, setSalaryCountry] = useState<Country>("US");
  const [expLens, setExpLens] = useState<Lens>("tier");
  const [expCountry, setExpCountry] = useState<Country>("US");

  // Salary distributions
  const salaryByTier = useMemo(() => {
    if (!rows) return null;
    const cur = CURRENCY_OF[salaryCountry];
    const out: Record<AITier, number[]> = {
      None: [],
      "AI Integration": [],
      "Applied/Core AI": [],
    };
    for (const r of rows) {
      if (r.co !== salaryCountry) continue;
      if (r.sm === null || r.cur !== cur) continue;
      if (r.t === null) continue;
      const tier = AI_TIER_ORDER[r.t];
      out[tier].push(r.sm);
    }
    return out;
  }, [rows, salaryCountry]);

  const salaryClip = useMemo<[number, number]>(() => {
    const cur = CURRENCY_OF[salaryCountry];
    if (cur === "USD") return [20000, 350000];
    if (cur === "EUR") return [20000, 200000];
    return [200000, 4500000]; // INR
  }, [salaryCountry]);

  const expSeries = useMemo(() => {
    if (!rows) return [];
    if (expLens === "tier") {
      // tiers within country
      return AI_TIER_ORDER.map((tier) => {
        const tierIdx = AI_TIER_ORDER.indexOf(tier);
        const vals: number[] = [];
        for (const r of rows) {
          if (r.co !== expCountry) continue;
          if (r.t !== tierIdx) continue;
          if (r.ex === null) continue;
          if (r.ex < 0 || r.ex > 25) continue;
          vals.push(r.ex);
        }
        return { label: tier, color: TIER_COLORS[tier], values: vals };
      });
    } else {
      // countries
      return COUNTRIES.map((c) => {
        const vals: number[] = [];
        for (const r of rows) {
          if (r.co !== c) continue;
          if (r.ex === null) continue;
          if (r.ex < 0 || r.ex > 25) continue;
          vals.push(r.ex);
        }
        return {
          label: `${COUNTRY_FLAGS[c]} ${COUNTRY_LABELS[c]}`,
          color: COUNTRY_COLORS[c],
          values: vals,
        };
      });
    }
  }, [rows, expLens, expCountry]);

  // Education breakdown
  const eduBreakdown = useMemo(() => {
    if (!rows) return [];
    const eduOrder = ["no_degree", "high_school", "associate", "bachelor", "master"];
    return COUNTRIES.map((c) => {
      const tierByEdu: Record<string, [number, number, number]> = {};
      for (const r of rows) {
        if (r.co !== c) continue;
        if (!r.ed || r.t === null) continue;
        if (!tierByEdu[r.ed]) tierByEdu[r.ed] = [0, 0, 0];
        tierByEdu[r.ed][r.t] += 1;
      }
      const rowsOut = eduOrder
        .filter((e) => tierByEdu[e])
        .map((e) => {
          const counts = tierByEdu[e];
          const tot = counts[0] + counts[1] + counts[2];
          return {
            edu: EDU_LABELS[e] ?? e,
            n: tot,
            none: tot ? counts[0] / tot : 0,
            integration: tot ? counts[1] / tot : 0,
            applied: tot ? counts[2] / tot : 0,
          };
        });
      return { country: c, rows: rowsOut };
    });
  }, [rows]);

  // AI share by experience bucket
  const aiByExp = useMemo(() => {
    if (!rows) return [];
    const buckets = [
      { label: "0-1y", min: 0, max: 1 },
      { label: "2-3y", min: 2, max: 3 },
      { label: "4-5y", min: 4, max: 5 },
      { label: "6-9y", min: 6, max: 9 },
      { label: "10+y", min: 10, max: 99 },
    ];
    return COUNTRIES.map((c) => ({
      country: c,
      buckets: buckets.map((b) => {
        let n = 0;
        let ai = 0;
        for (const r of rows) {
          if (r.co !== c) continue;
          if (r.ex === null || r.t === null) continue;
          if (r.ex < b.min || r.ex > b.max) continue;
          n += 1;
          if (r.t >= 1) ai += 1;
        }
        return { ...b, n, ai_share: n === 0 ? 0 : ai / n };
      }),
    }));
  }, [rows]);

  if (loading || !rows) return <LoadingShell />;
  if (error) {
    return (
      <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
        Failed to load dataset: {error}
      </div>
    );
  }

  const cur = CURRENCY_OF[salaryCountry];

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <PageHeader
        eyebrow="Distributions"
        title="Underneath the means"
        description="Aggregates hide variance. These overlapping density estimates show where the bulk of postings actually sit — and how AI-tier separation looks across the wage and experience axes."
      />

      {/* Salary density */}
      <Card>
        <CardHeader className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div>
            <CardTitle>Salary distribution by AI tier</CardTitle>
            <CardDescription>
              Smoothed kernel-style histogram over disclosed salaries. Notch on the baseline = median.
            </CardDescription>
          </div>
          <ToggleGroup
            type="single"
            value={salaryCountry}
            onValueChange={(v: string) => v && setSalaryCountry(v as Country)}
          >
            {COUNTRIES.map((c) => (
              <ToggleGroupItem key={c} value={c}>
                {COUNTRY_FLAGS[c]} {c}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </CardHeader>
        <CardContent>
          {salaryByTier ? (
            <div className="grid gap-4 md:grid-cols-3">
              {AI_TIER_ORDER.map((tier) => (
                <div key={tier} className="rounded-lg border bg-card p-3">
                  <div className="mb-1 flex items-center justify-between">
                    <span className="text-xs font-semibold" style={{ color: TIER_COLORS[tier] }}>
                      {tier}
                    </span>
                    <Badge variant="outline" className="text-[10px]">
                      n={formatNumber(salaryByTier[tier].length)}
                    </Badge>
                  </div>
                  <DensityChart
                    series={[
                      {
                        label: tier,
                        color: TIER_COLORS[tier],
                        values: salaryByTier[tier],
                      },
                    ]}
                    height={180}
                    bins={28}
                    clip={salaryClip}
                    formatX={(v) => fmtCurrency(v, cur)}
                  />
                </div>
              ))}
              <div className="rounded-lg border bg-card p-3 md:col-span-3">
                <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Overlay · {COUNTRY_LABELS[salaryCountry]}
                </div>
                <DensityChart
                  series={AI_TIER_ORDER.map((tier) => ({
                    label: tier,
                    color: TIER_COLORS[tier],
                    values: salaryByTier[tier],
                  }))}
                  height={260}
                  bins={48}
                  clip={salaryClip}
                  formatX={(v) => fmtCurrency(v, cur)}
                />
              </div>
            </div>
          ) : null}
          <p className="mt-3 text-[11px] text-muted-foreground">
            Domain auto-trimmed to the 2nd–98th percentile to suppress data-entry outliers.
            Salary disclosure rate in {COUNTRY_LABELS[salaryCountry]}: roughly{" "}
            {formatPct(
              (salaryByTier
                ? Object.values(salaryByTier).reduce((s, v) => s + v.length, 0)
                : 0) /
                rows.filter((r) => r.co === salaryCountry).length *
                100,
              1,
            )}.
          </p>
        </CardContent>
      </Card>

      {/* Experience distributions */}
      <Card>
        <CardHeader className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div>
            <CardTitle>Years of experience required</CardTitle>
            <CardDescription>
              Ground-truth from the LLM&rsquo;s extracted minimum-experience field. Clipped at 25 y.
            </CardDescription>
          </div>
          <div className="flex flex-wrap gap-3">
            <div className="flex flex-col gap-1">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Compare</span>
              <ToggleGroup
                type="single"
                value={expLens}
                onValueChange={(v: string) => v && setExpLens(v as Lens)}
              >
                <ToggleGroupItem value="tier">By tier · single country</ToggleGroupItem>
                <ToggleGroupItem value="country">By country · pooled</ToggleGroupItem>
              </ToggleGroup>
            </div>
            {expLens === "tier" ? (
              <div className="flex flex-col gap-1">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Country</span>
                <ToggleGroup
                  type="single"
                  value={expCountry}
                  onValueChange={(v: string) => v && setExpCountry(v as Country)}
                >
                  {COUNTRIES.map((c) => (
                    <ToggleGroupItem key={c} value={c}>
                      {COUNTRY_FLAGS[c]}
                    </ToggleGroupItem>
                  ))}
                </ToggleGroup>
              </div>
            ) : null}
          </div>
        </CardHeader>
        <CardContent>
          <DensityChart
            series={expSeries}
            height={260}
            bins={26}
            clip={[0, 25]}
            formatX={(v) => `${v.toFixed(0)}y`}
          />
        </CardContent>
      </Card>

      {/* AI by experience bucket */}
      <Card>
        <CardHeader>
          <CardTitle>AI mention rate by experience bucket</CardTitle>
          <CardDescription>
            Is AI hiring concentrated at the senior end? In all three countries, AI share rises with required experience —
            but the curves differ in slope and ceiling.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-3">
            {aiByExp.map((cBucket) => (
              <div key={cBucket.country} className="rounded-lg border p-3">
                <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
                  <span>{COUNTRY_FLAGS[cBucket.country]}</span>
                  <span>{COUNTRY_LABELS[cBucket.country]}</span>
                </div>
                <ul className="space-y-1.5">
                  {cBucket.buckets.map((b) => {
                    const max = Math.max(
                      ...cBucket.buckets.map((x) => x.ai_share),
                    );
                    const pct = max === 0 ? 0 : (b.ai_share / max) * 100;
                    return (
                      <li key={b.label} className="grid grid-cols-[60px_1fr_50px] items-center gap-2 text-xs">
                        <span className="font-mono text-[11px] text-muted-foreground">{b.label}</span>
                        <div className="relative h-3 overflow-hidden rounded-sm bg-muted/60">
                          <div
                            className="h-full"
                            style={{
                              width: `${pct}%`,
                              background: COUNTRY_COLORS[cBucket.country],
                              opacity: b.n < 50 ? 0.4 : 1,
                            }}
                          />
                        </div>
                        <span className="text-right tabular-nums font-medium">
                          {formatPct(b.ai_share * 100, 1)}
                        </span>
                      </li>
                    );
                  })}
                </ul>
                <div className="mt-2 text-[10px] text-muted-foreground">
                  Bars are width-scaled to the country max. Faded = n &lt; 50.
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Education x tier */}
      <Card>
        <CardHeader>
          <CardTitle>Education requirements × AI tier</CardTitle>
          <CardDescription>
            Tier composition stratified by required education. The Applied/Core column tilts heavily toward Master+ in all
            three markets — but not uniformly.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            {eduBreakdown.map(({ country, rows: eduRows }) => (
              <div key={country} className="rounded-lg border p-3">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-sm font-semibold">
                    {COUNTRY_FLAGS[country]} {COUNTRY_LABELS[country]}
                  </span>
                </div>
                {eduRows.length === 0 ? (
                  <div className="py-8 text-center text-xs text-muted-foreground">
                    No education data
                  </div>
                ) : (
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-[10px] uppercase tracking-wider text-muted-foreground">
                        <th className="text-left font-medium pb-1">Edu</th>
                        <th className="text-right font-medium pb-1">n</th>
                        <th className="text-right font-medium pb-1">Applied/Core</th>
                        <th className="text-right font-medium pb-1">Integration</th>
                      </tr>
                    </thead>
                    <tbody>
                      {eduRows.map((r) => (
                        <tr key={r.edu} className="border-t border-border/50">
                          <td className="py-1.5">{r.edu}</td>
                          <td className="text-right tabular-nums text-muted-foreground">
                            {formatNumber(r.n)}
                          </td>
                          <td className="text-right tabular-nums" style={{ color: TIER_COLORS["Applied/Core AI"] }}>
                            {formatPct(r.applied * 100, 1)}
                          </td>
                          <td className="text-right tabular-nums" style={{ color: TIER_COLORS["AI Integration"] }}>
                            {formatPct(r.integration * 100, 1)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            ))}
          </div>
          <p className="mt-3 text-[11px] text-muted-foreground">
            <strong>Read this:</strong> The Applied/Core column climbs steeply with required education in every country, but
            the slope is steepest in the US. Bachelor-level postings rarely demand Applied/Core AI — that&rsquo;s a graduate-degree
            labour market.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
