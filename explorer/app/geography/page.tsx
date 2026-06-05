"use client";

import { Suspense, useMemo, useState } from "react";
import { Loader2, MapPin } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useRows, type CompactRow } from "@/lib/data/rows-store";
import {
  COUNTRIES,
  COUNTRY_COLORS,
  COUNTRY_FLAGS,
  COUNTRY_LABELS,
  type Country,
} from "@/lib/constants";
import { formatNumber, formatPct } from "@/lib/utils";
import { wilsonCi } from "@/lib/stats";

type Field = "st" | "ct";
type Metric = "ai_share" | "applied_share" | "count";

interface GeoRow {
  key: string;
  n: number;
  ai_share: number;
  ai_lo: number;
  ai_hi: number;
  applied_share: number;
  ai_count: number;
}

function aggregate(rows: readonly CompactRow[], country: Country, field: Field, minN: number): GeoRow[] {
  const buckets = new Map<string, CompactRow[]>();
  for (const r of rows) {
    if (r.co !== country) continue;
    const key = field === "st" ? r.st : r.ct;
    if (!key) continue;
    let arr = buckets.get(key);
    if (!arr) {
      arr = [];
      buckets.set(key, arr);
    }
    arr.push(r);
  }
  const out: GeoRow[] = [];
  for (const [key, arr] of buckets.entries()) {
    if (arr.length < minN) continue;
    let n = 0;
    let ai = 0;
    let applied = 0;
    for (const r of arr) {
      if (r.t === null) continue;
      n += 1;
      if (r.t >= 1) ai += 1;
      if (r.t === 2) applied += 1;
    }
    if (n === 0) continue;
    const ci = wilsonCi(ai, n);
    out.push({
      key,
      n,
      ai_share: ai / n,
      ai_lo: ci.lo,
      ai_hi: ci.hi,
      applied_share: applied / n,
      ai_count: ai,
    });
  }
  return out;
}

const SORT_LABEL: Record<Metric, string> = {
  ai_share: "AI mention rate",
  applied_share: "Applied/Core share",
  count: "Posting volume",
};

const FIELD_LABEL: Record<Field, string> = {
  st: "State / region",
  ct: "City",
};

const STATE_DISPLAY: Record<string, string> = {
  CA: "California",
  TX: "Texas",
  NY: "New York",
  WA: "Washington",
  MA: "Massachusetts",
  IL: "Illinois",
  VA: "Virginia",
  NJ: "New Jersey",
  GA: "Georgia",
  PA: "Pennsylvania",
  CO: "Colorado",
  FL: "Florida",
  NC: "North Carolina",
  MD: "Maryland",
  AZ: "Arizona",
  OH: "Ohio",
  MI: "Michigan",
  MN: "Minnesota",
  CT: "Connecticut",
  OR: "Oregon",
  TN: "Tennessee",
  IN: "Indiana",
  WI: "Wisconsin",
  MO: "Missouri",
  UT: "Utah",
  DC: "Washington DC",
  NV: "Nevada",
};

export default function GeographyPage() {
  return (
    <Suspense fallback={<LoadingShell />}>
      <GeographyContent />
    </Suspense>
  );
}

function LoadingShell() {
  return (
    <div className="flex min-h-[400px] items-center justify-center gap-2 text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin" />
      Aggregating geography…
    </div>
  );
}

function GeographyContent() {
  const { rows, loading, error } = useRows();
  const [country, setCountry] = useState<Country>("US");
  const [field, setField] = useState<Field>("st");
  const [metric, setMetric] = useState<Metric>("ai_share");
  const [topN, setTopN] = useState<number>(15);

  const data = useMemo(() => {
    if (!rows) return [];
    const minN = field === "st" ? 30 : 60;
    const agg = aggregate(rows, country, field, minN);
    agg.sort((a, b) => {
      if (metric === "count") return b.n - a.n;
      if (metric === "applied_share") return b.applied_share - a.applied_share;
      return b.ai_share - a.ai_share;
    });
    return agg;
  }, [rows, country, field, metric]);

  const display = useMemo(() => data.slice(0, topN), [data, topN]);

  const baselineAi = useMemo(() => {
    if (!rows) return 0;
    let n = 0;
    let ai = 0;
    for (const r of rows) {
      if (r.co !== country) continue;
      if (r.t === null) continue;
      n += 1;
      if (r.t >= 1) ai += 1;
    }
    return n === 0 ? 0 : ai / n;
  }, [rows, country]);

  const maxValue = useMemo(() => {
    if (display.length === 0) return 1;
    if (metric === "count") return Math.max(...display.map((r) => r.n));
    if (metric === "applied_share") return Math.max(...display.map((r) => r.applied_share));
    return Math.max(...display.map((r) => r.ai_share));
  }, [display, metric]);

  if (loading || !rows) return <LoadingShell />;
  if (error) {
    return (
      <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
        Failed: {error}
      </div>
    );
  }

  const accent = COUNTRY_COLORS[country];

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <PageHeader
        eyebrow="Geography"
        title="Where AI hiring concentrates"
        description="Which states (US) or cities (DE / IN) drive the AI hiring signal? Each row is a Wilson 95% CI on the AI mention rate within that geography. Reference line = country average."
      />

      <Card>
        <CardHeader className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div className="flex flex-wrap gap-3">
            <ControlGroup label="Country">
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
            </ControlGroup>
            <ControlGroup label="Granularity">
              <ToggleGroup
                type="single"
                value={field}
                onValueChange={(v: string) => v && setField(v as Field)}
              >
                <ToggleGroupItem value="st">State / region</ToggleGroupItem>
                <ToggleGroupItem value="ct">City</ToggleGroupItem>
              </ToggleGroup>
            </ControlGroup>
            <ControlGroup label="Sort by">
              <ToggleGroup
                type="single"
                value={metric}
                onValueChange={(v: string) => v && setMetric(v as Metric)}
              >
                <ToggleGroupItem value="ai_share">AI rate</ToggleGroupItem>
                <ToggleGroupItem value="applied_share">Applied/Core</ToggleGroupItem>
                <ToggleGroupItem value="count">Volume</ToggleGroupItem>
              </ToggleGroup>
            </ControlGroup>
            <ControlGroup label="Show">
              <ToggleGroup
                type="single"
                value={String(topN)}
                onValueChange={(v: string) => v && setTopN(Number(v))}
              >
                <ToggleGroupItem value="10">Top 10</ToggleGroupItem>
                <ToggleGroupItem value="15">Top 15</ToggleGroupItem>
                <ToggleGroupItem value="25">Top 25</ToggleGroupItem>
              </ToggleGroup>
            </ControlGroup>
          </div>
          <Badge variant="outline">
            {data.length} {field === "st" ? "regions" : "cities"} pass min n
          </Badge>
        </CardHeader>
        <CardContent>
          {display.length === 0 ? (
            <div className="rounded-lg border bg-muted/30 p-8 text-center text-sm text-muted-foreground">
              Not enough rows in {COUNTRY_LABELS[country]} {field === "st" ? "by state" : "by city"} to display.
            </div>
          ) : (
            <div className="space-y-1.5">
              {/* Header — hidden on narrow widths where each row stacks */}
              <div className="hidden md:grid md:grid-cols-[160px_1fr_64px_64px] lg:grid-cols-[180px_1fr_72px_72px] items-center gap-3 px-2 pb-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                <div>{FIELD_LABEL[field]}</div>
                <div>{SORT_LABEL[metric]}</div>
                <div className="text-right">N</div>
                <div className="text-right">App/Core</div>
              </div>
              {/* Rows */}
              {display.map((r) => {
                const isAiMetric = metric !== "count";
                const value = metric === "count" ? r.n : metric === "applied_share" ? r.applied_share : r.ai_share;
                const widthPct = (value / maxValue) * 100;
                const ciLoPct = (r.ai_lo / maxValue) * 100;
                const ciHiPct = (r.ai_hi / maxValue) * 100;
                const baselinePct = isAiMetric ? (baselineAi / maxValue) * 100 : null;
                const labelDisplay = field === "st" && country === "US"
                  ? `${r.key}${STATE_DISPLAY[r.key] ? ` · ${STATE_DISPLAY[r.key]}` : ""}`
                  : r.key;
                return (
                  <div
                    key={r.key}
                    className="group flex flex-col gap-1.5 rounded-md border-b border-border/40 px-2 py-2 hover:bg-muted/40 md:grid md:grid-cols-[160px_1fr_64px_64px] md:items-center md:gap-3 md:py-1.5 lg:grid-cols-[180px_1fr_72px_72px]"
                  >
                    <div className="flex items-baseline justify-between gap-2 md:block">
                      <div className="truncate text-xs font-medium" title={labelDisplay}>
                        {labelDisplay}
                      </div>
                      <span className="md:hidden text-[10px] tabular-nums text-muted-foreground">
                        n={formatNumber(r.n)} · App/Core {formatPct(r.applied_share * 100, 1)}
                      </span>
                    </div>
                    <div className="relative h-5 rounded-sm bg-muted/40">
                      {baselinePct !== null && baselinePct > 0 && baselinePct < 100 ? (
                        <div
                          className="absolute inset-y-0 w-px bg-foreground/40"
                          style={{ left: `${baselinePct}%` }}
                          title={`Country average ${formatPct(baselineAi * 100, 1)}`}
                        />
                      ) : null}
                      <div
                        className="absolute inset-y-0 left-0 rounded-sm"
                        style={{
                          width: `${widthPct}%`,
                          background: accent,
                          opacity: r.n < 80 ? 0.55 : 1,
                        }}
                      />
                      {isAiMetric ? (
                        <div
                          className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full bg-foreground/30"
                          style={{
                            left: `${ciLoPct}%`,
                            width: `${Math.max(0.5, ciHiPct - ciLoPct)}%`,
                          }}
                          title={`95% Wilson CI: ${formatPct(r.ai_lo * 100, 1)} – ${formatPct(r.ai_hi * 100, 1)}`}
                        />
                      ) : null}
                      <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] font-semibold tabular-nums text-foreground/80 mix-blend-difference">
                        {metric === "count"
                          ? formatNumber(r.n)
                          : formatPct(value * 100, 1)}
                      </span>
                    </div>
                    <div className="hidden md:block text-right text-xs tabular-nums text-muted-foreground">
                      {formatNumber(r.n)}
                    </div>
                    <div className="hidden md:block text-right text-xs tabular-nums" style={{ color: accent }}>
                      {formatPct(r.applied_share * 100, 1)}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          <div className="mt-4 flex flex-wrap items-center gap-3 text-[11px] text-muted-foreground">
            <div className="flex items-center gap-1.5">
              <span className="inline-block h-3 w-3 rounded-sm" style={{ background: accent }} />
              {SORT_LABEL[metric]}
            </div>
            {metric !== "count" ? (
              <div className="flex items-center gap-1.5">
                <span className="inline-block h-1 w-6 rounded-full bg-foreground/30" />
                95% Wilson CI
              </div>
            ) : null}
            {metric !== "count" ? (
              <div className="flex items-center gap-1.5">
                <span className="inline-block h-3 w-px bg-foreground/40" />
                {COUNTRY_LABELS[country]} avg = {formatPct(baselineAi * 100, 1)}
              </div>
            ) : null}
            <div className="opacity-70">Faded bars = small sample (n &lt; 80). Min n = {field === "st" ? 30 : 60}.</div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <MapPin className="size-4 text-primary" />
            Reading the geography
          </CardTitle>
          <CardDescription>How to interpret what you&rsquo;re seeing</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 text-sm md:grid-cols-2">
          <Tip
            title="The CA / WA / MA effect"
            body="In the US, AI hiring concentrates in California, Washington and Massachusetts. The state-level rates can clear 30%, but the geography does most of the explanatory work — controlling for state in Model C shrinks the AI premium toward (but not to) zero."
          />
          <Tip
            title="Bangalore vs national average"
            body="Switch to India / city granularity. Bangalore and Hyderabad sit well above India&rsquo;s 6.3% mention rate, but the biggest gap by absolute AI postings is Bangalore. The city tells the story of India&rsquo;s AI hiring — the country average undersells the cluster."
          />
          <Tip
            title="Country average reference"
            body="The vertical line on each row is the national mean. A bar that pushes well past the line is hiring AI more than the national average; a bar that ends short is below average. Confidence intervals tell you whether the gap is statistically real."
          />
          <Tip
            title="Why min sample size?"
            body="At very low n the Wilson CI is wide enough that it dominates the visual. We hide states/cities with fewer than 30 (state) or 60 (city) postings to avoid drawing attention to noise."
          />
        </CardContent>
      </Card>
    </div>
  );
}

function ControlGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{label}</span>
      {children}
    </div>
  );
}

function Tip({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-lg border bg-card p-3">
      <div className="text-xs font-semibold">{title}</div>
      <div className="mt-1 text-xs leading-relaxed text-muted-foreground">{body}</div>
    </div>
  );
}
