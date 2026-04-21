"use client";

import { Suspense, useMemo } from "react";
import { useQueryState, parseAsStringLiteral } from "nuqs";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AlertTriangle, Loader2 } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SlicerBar } from "@/components/slicer/slicer-bar";
import { useSlicer } from "@/lib/state/slicer";
import {
  filterRows,
  groupBy,
  summarizeSlice,
  topClusterLifts,
  useRows,
  type GroupByDim,
} from "@/lib/data/rows-store";
import { powerLevel } from "@/lib/stats";
import {
  AI_TIER_ORDER,
  COUNTRY_COLORS,
  COUNTRY_FLAGS,
  COUNTRY_LABELS,
  TIER_COLORS,
  type Country,
} from "@/lib/constants";
import { formatNumber, formatPct } from "@/lib/utils";

const GROUP_BY_OPTS = [
  "country",
  "job_family",
  "seniority",
  "edu",
  "industry",
  "state",
  "size",
  "tier",
] as const;
type GroupByOpt = (typeof GROUP_BY_OPTS)[number];
const groupByParser = parseAsStringLiteral(GROUP_BY_OPTS).withDefault("country");

const METRICS = ["ai_share", "applied_share", "mean_salary", "count", "tier_mix"] as const;
type Metric = (typeof METRICS)[number];
const metricParser = parseAsStringLiteral(METRICS).withDefault("ai_share");

const GROUP_LABEL: Record<GroupByOpt, string> = {
  country: "Country",
  job_family: "Job family",
  seniority: "Seniority",
  edu: "Education",
  industry: "Industry",
  state: "State",
  size: "Firm size",
  tier: "AI tier",
};

const METRIC_LABEL: Record<Metric, string> = {
  ai_share: "AI share (%)",
  applied_share: "Applied/Core share (%)",
  mean_salary: "Mean salary",
  count: "Posting count",
  tier_mix: "Tier mix (%)",
};

export default function AnalyzePage() {
  return (
    <Suspense fallback={<LoadingShell />}>
      <AnalyzeContent />
    </Suspense>
  );
}

function LoadingShell() {
  return (
    <div className="flex min-h-[400px] items-center justify-center gap-2 text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin" />
      Loading analytical dataset…
    </div>
  );
}

function AnalyzeContent() {
  const { rows, loading, error } = useRows();
  const slicer = useSlicer();
  const [dimRaw, setDim] = useQueryState("group", groupByParser);
  const [metricRaw, setMetric] = useQueryState("metric", metricParser);
  const dim = dimRaw as GroupByDim;
  const metric = metricRaw as Metric;

  const filtered = useMemo(() => (rows ? filterRows(rows, slicer.value) : []), [rows, slicer.value]);
  const baseline = rows ?? [];
  const summary = useMemo(
    () => summarizeSlice(filtered, baseline.length),
    [filtered, baseline.length],
  );
  const groups = useMemo(() => groupBy(filtered, dim), [filtered, dim]);
  const lifts = useMemo(
    () => (filtered.length >= 20 ? topClusterLifts(filtered, baseline, 5) : []),
    [filtered, baseline],
  );

  // Sort groups in a natural order for the selected dimension
  const groupsSorted = useMemo(() => {
    const arr = [...groups];
    if (dim === "seniority") {
      const order = ["Junior", "Mid", "Senior", "Lead+"];
      arr.sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key));
    } else if (dim === "tier") {
      arr.sort((a, b) => AI_TIER_ORDER.indexOf(a.key as typeof AI_TIER_ORDER[number]) - AI_TIER_ORDER.indexOf(b.key as typeof AI_TIER_ORDER[number]));
    } else if (dim === "size") {
      const order = ["1-200", "201-1000", "1001-10000", "10000+"];
      arr.sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key));
    } else {
      // sort by N desc, keep top 20 if too many
      arr.sort((a, b) => b.n - a.n);
    }
    return arr.slice(0, 20);
  }, [groups, dim]);

  const chartData = useMemo(() => {
    if (metric === "tier_mix") {
      return groupsSorted.map((g) => {
        const tm: Record<string, number> = {};
        for (const t of g.tier_mix) tm[t.tier] = t.pct * 100;
        return { label: g.key, n: g.n, ...tm };
      });
    }
    return groupsSorted.map((g) => {
      let v = 0;
      let errLo = 0;
      let errHi = 0;
      if (metric === "ai_share") {
        v = g.ai_share * 100;
        errLo = (g.ai_share - g.ai_share_ci.lo) * 100;
        errHi = (g.ai_share_ci.hi - g.ai_share) * 100;
      } else if (metric === "applied_share") {
        v = g.applied_share * 100;
        errLo = (g.applied_share - g.applied_share_ci.lo) * 100;
        errHi = (g.applied_share_ci.hi - g.applied_share) * 100;
      } else if (metric === "mean_salary") {
        v = g.mean_salary ?? 0;
      } else if (metric === "count") {
        v = g.n;
      }
      return {
        label: g.key,
        n: g.n,
        value: v,
        err: Math.max(errLo, errHi),
        currency: g.salary_currency,
      };
    });
  }, [groupsSorted, metric]);

  const power = powerLevel(filtered.length);

  if (loading) return <LoadingShell />;
  if (error) {
    return (
      <div className="mx-auto max-w-3xl rounded-md border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
        Could not load <span className="font-mono">rows.json</span> — {error}
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1400px] space-y-6">
      <PageHeader
        eyebrow="Workbench"
        title="Analyze"
        description="The full 44 832-posting dataset, sliced live. Combine any filters, pick a dimension to group by and a metric to plot. Numbers are computed from row-level data, not pre-aggregates — every chart reflects exactly the slice you've built."
      />

      <div className="grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)_280px]">
        {/* Left: slicer */}
        <div className="space-y-4">
          <SlicerBar handle={slicer} rows={baseline} />
        </div>

        {/* Center: chart */}
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <div className="flex flex-wrap items-end justify-between gap-3">
                <div>
                  <CardTitle className="text-base">
                    {METRIC_LABEL[metric]} by {GROUP_LABEL[dim]}
                  </CardTitle>
                  <CardDescription className="text-xs">
                    {filtered.length === 0
                      ? "No postings match the current slice."
                      : `${formatNumber(filtered.length)} postings, grouped into ${groupsSorted.length} bucket${groupsSorted.length === 1 ? "" : "s"}.`}
                  </CardDescription>
                </div>
                <div className="flex flex-wrap gap-3">
                  <Picker
                    label="Group by"
                    value={dim}
                    onChange={(v) => setDim(v as GroupByOpt)}
                    options={GROUP_BY_OPTS.map((x) => ({ value: x, label: GROUP_LABEL[x] }))}
                  />
                  <Picker
                    label="Metric"
                    value={metric}
                    onChange={(v) => setMetric(v as Metric)}
                    options={METRICS.map((x) => ({ value: x, label: METRIC_LABEL[x] }))}
                  />
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {power === "tiny" ? (
                <div className="rounded-md border border-destructive/30 bg-destructive/5 p-6 text-center text-sm text-destructive">
                  <AlertTriangle className="mx-auto mb-2 size-5" />
                  Slice has only {filtered.length} postings — too few for meaningful aggregation. Loosen some filters.
                </div>
              ) : groupsSorted.length === 0 ? (
                <div className="p-10 text-center text-sm text-muted-foreground">
                  No groups to display. Relax your filters.
                </div>
              ) : metric === "tier_mix" ? (
                <TierMixChart data={chartData as TierMixDatum[]} dim={dim} />
              ) : (
                <MetricBarChart data={chartData as MetricDatum[]} metric={metric} dim={dim} />
              )}

              {power === "low" && (
                <div className="mt-3 flex items-center gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-800 dark:text-amber-300">
                  <AlertTriangle className="size-3.5" />
                  Under-powered: {filtered.length} postings. Wilson 95% CIs are wide; treat differences with caution.
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Detail table</CardTitle>
              <CardDescription className="text-xs">
                Every group with its N, AI share (95% CI), Applied/Core share, and mean salary.
              </CardDescription>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              <table className="w-full min-w-[600px] text-xs">
                <thead className="text-muted-foreground">
                  <tr>
                    <th className="px-2 py-1.5 text-left font-medium">{GROUP_LABEL[dim]}</th>
                    <th className="px-2 py-1.5 text-right font-medium">N</th>
                    <th className="px-2 py-1.5 text-right font-medium">AI share</th>
                    <th className="px-2 py-1.5 text-right font-medium">Applied/Core</th>
                    <th className="px-2 py-1.5 text-right font-medium">Mean salary</th>
                  </tr>
                </thead>
                <tbody>
                  {groupsSorted.map((g) => (
                    <tr key={g.key} className="border-t">
                      <td className="px-2 py-1.5 font-medium">
                        {dim === "country" ? `${COUNTRY_FLAGS[g.key as Country] ?? ""} ${COUNTRY_LABELS[g.key as Country] ?? g.key}` : g.key}
                      </td>
                      <td className="px-2 py-1.5 text-right tabular-nums">{formatNumber(g.n)}</td>
                      <td className="px-2 py-1.5 text-right tabular-nums">
                        {formatPct(g.ai_share * 100)}{" "}
                        <span className="text-muted-foreground">
                          [{formatPct(g.ai_share_ci.lo * 100)}–{formatPct(g.ai_share_ci.hi * 100)}]
                        </span>
                      </td>
                      <td className="px-2 py-1.5 text-right tabular-nums">
                        {formatPct(g.applied_share * 100)}
                      </td>
                      <td className="px-2 py-1.5 text-right tabular-nums">
                        {g.mean_salary !== null
                          ? `${currencySymbol(g.salary_currency)}${Math.round(g.mean_salary).toLocaleString()} (n=${g.salary_n})`
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </div>

        {/* Right: stats strip */}
        <aside className="space-y-4">
          <Card className="sticky top-4">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Slice summary</CardTitle>
              <CardDescription className="text-xs">
                Updated from row data as you slice.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <Stat
                label="Postings in slice"
                value={
                  <>
                    <span className="tabular-nums">{formatNumber(summary.n)}</span>
                    <span className="ml-1 text-xs text-muted-foreground">
                      ({(summary.share_of_total * 100).toFixed(1)}% of total)
                    </span>
                  </>
                }
              />
              <Stat
                label="AI share"
                value={
                  summary.n === 0 ? (
                    "—"
                  ) : (
                    <>
                      <span className="tabular-nums">{formatPct(summary.ai_share * 100)}</span>
                      <span className="ml-1 text-xs text-muted-foreground">
                        [{formatPct(summary.ai_share_ci.lo * 100)}–{formatPct(summary.ai_share_ci.hi * 100)}]
                      </span>
                    </>
                  )
                }
              />
              <Stat
                label="Applied/Core share"
                value={
                  summary.n === 0 ? (
                    "—"
                  ) : (
                    <>
                      <span className="tabular-nums">{formatPct(summary.applied_share * 100)}</span>
                      <span className="ml-1 text-xs text-muted-foreground">
                        [{formatPct(summary.applied_share_ci.lo * 100)}–{formatPct(summary.applied_share_ci.hi * 100)}]
                      </span>
                    </>
                  )
                }
              />
              <div>
                <div className="mb-1 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Mean salary (disclosed)
                </div>
                {summary.mean_salary_by_currency.length === 0 ? (
                  <div className="text-xs text-muted-foreground">—</div>
                ) : (
                  <ul className="space-y-0.5 text-xs">
                    {summary.mean_salary_by_currency.map((s) => (
                      <li key={s.currency} className="flex items-center justify-between">
                        <span>
                          {currencySymbol(s.currency)}
                          {Math.round(s.mean).toLocaleString()}
                        </span>
                        <span className="text-muted-foreground">
                          n = {formatNumber(s.n)}{" "}
                          {s.ci
                            ? `· ±${((s.ci.hi - s.ci.lo) / 2).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
                            : ""}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              {lifts.length > 0 && (
                <div>
                  <div className="mb-1 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Top cluster lifts
                  </div>
                  <ul className="space-y-0.5 text-xs">
                    {lifts.map((l) => (
                      <li key={l.cluster_key} className="flex items-center justify-between">
                        <span className="truncate pr-2 font-medium">{l.cluster_label}</span>
                        <span className="tabular-nums text-muted-foreground">
                          {l.lift_pp > 0 ? "+" : ""}
                          {l.lift_pp.toFixed(1)} pp
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </CardContent>
          </Card>
        </aside>
      </div>
    </div>
  );
}

interface MetricDatum {
  label: string;
  n: number;
  value: number;
  err: number;
  currency: string | null;
}

interface TierMixDatum {
  label: string;
  n: number;
  None: number;
  "AI Integration": number;
  "Applied/Core AI": number;
}

function MetricBarChart({ data, metric, dim }: { data: MetricDatum[]; metric: Metric; dim: GroupByDim }) {
  const isPercent = metric === "ai_share" || metric === "applied_share";
  return (
    <ResponsiveContainer width="100%" height={Math.max(260, data.length * 32 + 60)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 40, bottom: 4, left: 120 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.3} />
        <XAxis
          type="number"
          tick={{ fontSize: 11 }}
          tickFormatter={
            isPercent
              ? (v) => `${v}%`
              : metric === "mean_salary"
                ? (v) => Math.round(v / 1000) + "k"
                : (v) => String(v)
          }
        />
        <YAxis
          dataKey="label"
          type="category"
          tick={{ fontSize: 11 }}
          width={120}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          cursor={{ fill: "rgba(0,0,0,0.04)" }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const p = payload[0].payload as MetricDatum;
            return (
              <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
                <div className="mb-1 font-semibold">{p.label}</div>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-muted-foreground">{METRIC_LABEL[metric]}</span>
                  <span className="tabular-nums">
                    {isPercent
                      ? `${p.value.toFixed(1)}%`
                      : metric === "mean_salary"
                        ? `${currencySymbol(p.currency)}${Math.round(p.value).toLocaleString()}`
                        : formatNumber(p.value)}
                  </span>
                </div>
                {isPercent && (
                  <div className="text-xs text-muted-foreground">±{p.err.toFixed(1)} pp (95% Wilson)</div>
                )}
                <div className="mt-1 text-xs text-muted-foreground">N = {formatNumber(p.n)}</div>
              </div>
            );
          }}
        />
        <Bar dataKey="value" isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={dim === "country" ? (COUNTRY_COLORS[d.label as Country] ?? "#3c6ea8") : "#3c6ea8"} />
          ))}
          {isPercent && <ErrorBar dataKey="err" width={4} strokeWidth={1} stroke="#333" />}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function TierMixChart({ data, dim }: { data: TierMixDatum[]; dim: GroupByDim }) {
  return (
    <ResponsiveContainer width="100%" height={Math.max(260, data.length * 34 + 60)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 20, bottom: 4, left: 120 }} stackOffset="expand">
        <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.3} />
        <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => `${Math.round(v * 100)}%`} domain={[0, 1]} />
        <YAxis dataKey="label" type="category" tick={{ fontSize: 11 }} width={120} axisLine={false} tickLine={false} />
        <Tooltip
          cursor={{ fill: "rgba(0,0,0,0.04)" }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const p = payload[0].payload as TierMixDatum;
            return (
              <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
                <div className="mb-1 font-semibold">{p.label}</div>
                {AI_TIER_ORDER.map((t) => (
                  <div key={t} className="flex items-center gap-2">
                    <span className="inline-block size-2 rounded-sm" style={{ backgroundColor: TIER_COLORS[t] }} />
                    <span className="min-w-28">{t}</span>
                    <span className="tabular-nums">{p[t].toFixed(1)}%</span>
                  </div>
                ))}
                <div className="mt-1 text-xs text-muted-foreground">N = {formatNumber(p.n)}</div>
              </div>
            );
          }}
        />
        {AI_TIER_ORDER.map((t) => (
          <Bar key={t} dataKey={t} stackId="a" fill={TIER_COLORS[t]} isAnimationActive={false} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

function Picker({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-8 rounded-md border border-input bg-background px-2 text-xs"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="mt-0.5">{value}</div>
    </div>
  );
}

function currencySymbol(cur: string | null): string {
  switch (cur) {
    case "USD":
      return "$";
    case "EUR":
      return "€";
    case "INR":
      return "₹";
    default:
      return cur ? cur + " " : "";
  }
}
