"use client";

import { Suspense, useMemo } from "react";
import { useQueryState, parseAsStringLiteral } from "nuqs";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { getG6, getG7 } from "@/lib/data/loaders";
import {
  AI_TIER_ORDER,
  COUNTRIES,
  COUNTRY_COLORS,
  COUNTRY_FLAGS,
  COUNTRY_LABELS,
  TIER_COLORS,
  type AITier,
  type Country,
} from "@/lib/constants";
import { sigColor } from "@/lib/utils";

const MODELS = ["A", "B", "C"] as const;
type Model = (typeof MODELS)[number];
const modelParser = parseAsStringLiteral(MODELS).withDefault("C");

const MODEL_DESCRIPTIONS: Record<Model, { title: string; subtitle: string; controls: string }> = {
  A: {
    title: "Model A · Baseline",
    subtitle: "Raw AI premium",
    controls: "Country fixed effects only. No controls for job family, seniority, or education.",
  },
  B: {
    title: "Model B · + Structural controls",
    subtitle: "Add job family, seniority, education",
    controls: "Adds job family, seniority band, education, state/industry fixed effects.",
  },
  C: {
    title: "Model C · Full specification",
    subtitle: "+ Skill-cluster controls",
    controls: "Full controls from Model B plus 21 skill-cluster dummies. This is the preferred spec in the thesis.",
  },
};

export default function PremiumPage() {
  return (
    <Suspense fallback={<div className="mx-auto max-w-7xl" />}>
      <PremiumContent />
    </Suspense>
  );
}

function PremiumContent() {
  const g6 = getG6();
  const g7 = getG7();
  const [model, setModel] = useQueryState("model", modelParser);

  // For the A/B/C chart, turn coefficients into percentage premium with SE
  const decompositionData = useMemo(() => {
    return MODELS.map((m) => {
      const rows = g6[m];
      const integ = rows.find((r) => r.tier === "AI Integration");
      const applied = rows.find((r) => r.tier === "Applied/Core AI");
      return {
        model: m,
        modelLabel: MODEL_DESCRIPTIONS[m].subtitle,
        Integration: integ?.b != null ? integ.b * 100 : 0,
        IntegrationSE: integ?.se != null ? integ.se * 100 : 0,
        IntegrationSig: integ?.sig ?? "ns",
        Applied: applied?.b != null ? applied.b * 100 : 0,
        AppliedSE: applied?.se != null ? applied.se * 100 : 0,
        AppliedSig: applied?.sig ?? "ns",
      };
    });
  }, [g6]);

  // Cross-country data for g7
  const crossCountryData = useMemo(() => {
    return COUNTRIES.map((c) => {
      const rows = g7[c];
      const integ = rows.find((r) => r.tier === "AI Integration");
      const applied = rows.find((r) => r.tier === "Applied/Core AI");
      const integHalf = integ?.ci_high != null && integ.b != null ? (integ.ci_high - integ.b) * 100 : 0;
      const appliedHalf = applied?.ci_high != null && applied.b != null ? (applied.ci_high - applied.b) * 100 : 0;
      return {
        country: c,
        label: `${COUNTRY_FLAGS[c]} ${COUNTRY_LABELS[c]}`,
        Integration: integ?.b != null ? integ.b * 100 : 0,
        IntegrationErr: integHalf,
        IntegrationSig: integ?.sig ?? "ns",
        Applied: applied?.b != null ? applied.b * 100 : 0,
        AppliedErr: appliedHalf,
        AppliedSig: applied?.sig ?? "ns",
      };
    });
  }, [g7]);

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <PageHeader
        eyebrow="Section 5.4"
        title="Salary premium decomposition"
        description="How much higher is the posted log-wage when the role requires AI Integration or Applied/Core AI? Toggle through Models A → B → C to watch controls eat the raw premium — what's left is the structural AI wage effect."
      />

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <CardTitle>A → B → C · How the premium survives controls</CardTitle>
              <CardDescription>
                US only. Reference category = No AI. Bars are % premium over the reference (exp(β) − 1). Whiskers = ±1 SE.
              </CardDescription>
            </div>
            <Badge variant="outline">Thesis Table 5</Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <Tabs value={model} onValueChange={(v) => setModel(v as Model)}>
            <TabsList>
              {MODELS.map((m) => (
                <TabsTrigger key={m} value={m}>
                  Model {m}
                </TabsTrigger>
              ))}
            </TabsList>
            {MODELS.map((m) => {
              const d = decompositionData.find((x) => x.model === m)!;
              return (
                <TabsContent key={m} value={m} className="space-y-4 pt-4">
                  <div className="rounded-md border bg-muted/30 p-3 text-sm">
                    <div className="font-semibold">{MODEL_DESCRIPTIONS[m].title}</div>
                    <div className="text-muted-foreground">{MODEL_DESCRIPTIONS[m].controls}</div>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <TierStat
                      tier="AI Integration"
                      value={d.Integration}
                      se={d.IntegrationSE}
                      sig={d.IntegrationSig}
                    />
                    <TierStat
                      tier="Applied/Core AI"
                      value={d.Applied}
                      se={d.AppliedSE}
                      sig={d.AppliedSig}
                    />
                  </div>
                </TabsContent>
              );
            })}
          </Tabs>

          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={decompositionData} margin={{ top: 20, right: 20, bottom: 4, left: 4 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.35} />
              <XAxis
                dataKey="modelLabel"
                tick={{ fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                interval={0}
              />
              <YAxis
                tickFormatter={(v) => `${v}%`}
                tick={{ fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                cursor={{ fill: "rgba(0,0,0,0.04)" }}
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null;
                  return (
                    <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
                      <div className="mb-1 font-semibold">{label}</div>
                      {payload.map((p) => (
                        <div key={String(p.dataKey)} className="flex items-center gap-2">
                          <span
                            className="inline-block size-2 rounded-sm"
                            style={{ backgroundColor: p.color as string }}
                          />
                          <span>{String(p.dataKey)}</span>
                          <span className="tabular-nums">
                            {(p.value as number).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  );
                }}
              />
              <ReferenceLine y={0} stroke="#999" />
              <Bar dataKey="Integration" fill={TIER_COLORS["AI Integration"]} isAnimationActive={false}>
                <ErrorBar dataKey="IntegrationSE" strokeWidth={1} width={6} stroke="#333" />
                <LabelList dataKey="Integration" position="top" formatter={(v: number) => `${v.toFixed(1)}%`} style={{ fontSize: 10 }} />
              </Bar>
              <Bar dataKey="Applied" fill={TIER_COLORS["Applied/Core AI"]} isAnimationActive={false}>
                <ErrorBar dataKey="AppliedSE" strokeWidth={1} width={6} stroke="#333" />
                <LabelList dataKey="Applied" position="top" formatter={(v: number) => `${v.toFixed(1)}%`} style={{ fontSize: 10 }} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <CardTitle>Cross-country salary premium (Model C)</CardTitle>
              <CardDescription>
                Per country full-spec OLS on log(salary). Whiskers = 95% CI. DE has low salary disclosure (8%) so CIs are wide.
              </CardDescription>
            </div>
            <Badge variant="outline">Thesis Figure 7</Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            {crossCountryData.map((d) => (
              <Card key={d.country} className="border-none bg-muted/20 shadow-none">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">
                    {COUNTRY_FLAGS[d.country as Country]} {COUNTRY_LABELS[d.country as Country]}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 pt-0 text-sm">
                  <div className="flex items-center justify-between">
                    <span
                      className="inline-flex items-center gap-2"
                      style={{ color: TIER_COLORS["AI Integration"] }}
                    >
                      <span
                        className="inline-block size-2.5 rounded-sm"
                        style={{ backgroundColor: TIER_COLORS["AI Integration"] }}
                      />
                      AI Integration
                    </span>
                    <span className={sigColor(d.IntegrationSig) + " tabular-nums"}>
                      {d.Integration > 0 ? "+" : ""}
                      {d.Integration.toFixed(1)}% · {d.IntegrationSig}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span
                      className="inline-flex items-center gap-2"
                      style={{ color: TIER_COLORS["Applied/Core AI"] }}
                    >
                      <span
                        className="inline-block size-2.5 rounded-sm"
                        style={{ backgroundColor: TIER_COLORS["Applied/Core AI"] }}
                      />
                      Applied/Core AI
                    </span>
                    <span className={sigColor(d.AppliedSig) + " tabular-nums"}>
                      {d.Applied > 0 ? "+" : ""}
                      {d.Applied.toFixed(1)}% · {d.AppliedSig}
                    </span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={crossCountryData}
              margin={{ top: 20, right: 12, bottom: 4, left: 4 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.35} />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis
                tickFormatter={(v) => `${v}%`}
                tick={{ fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                cursor={{ fill: "rgba(0,0,0,0.04)" }}
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null;
                  return (
                    <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
                      <div className="mb-1 font-semibold">{label}</div>
                      {payload.map((p) => (
                        <div key={String(p.dataKey)} className="flex items-center gap-2">
                          <span
                            className="inline-block size-2 rounded-sm"
                            style={{ backgroundColor: p.color as string }}
                          />
                          <span>{String(p.dataKey)}</span>
                          <span className="tabular-nums">
                            {(p.value as number).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  );
                }}
              />
              <ReferenceLine y={0} stroke="#999" />
              <Bar dataKey="Integration" isAnimationActive={false}>
                {crossCountryData.map((_, i) => (
                  <Cell key={i} fill={TIER_COLORS["AI Integration"]} />
                ))}
                <ErrorBar dataKey="IntegrationErr" strokeWidth={1} width={6} stroke="#333" />
                <LabelList dataKey="Integration" position="top" formatter={(v: number) => `${v.toFixed(1)}%`} style={{ fontSize: 10 }} />
              </Bar>
              <Bar dataKey="Applied" isAnimationActive={false}>
                {crossCountryData.map((_, i) => (
                  <Cell key={i} fill={TIER_COLORS["Applied/Core AI"]} />
                ))}
                <ErrorBar dataKey="AppliedErr" strokeWidth={1} width={6} stroke="#333" />
                <LabelList dataKey="Applied" position="top" formatter={(v: number) => `${v.toFixed(1)}%`} style={{ fontSize: 10 }} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <section className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Shrinkage walk</CardTitle>
            <CardDescription>How much of the raw premium is structural selection vs. a pure AI effect.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <ShrinkageRow
              label="AI Integration"
              a={decompositionData[0].Integration}
              b={decompositionData[1].Integration}
              c={decompositionData[2].Integration}
              tier="AI Integration"
            />
            <ShrinkageRow
              label="Applied/Core AI"
              a={decompositionData[0].Applied}
              b={decompositionData[1].Applied}
              c={decompositionData[2].Applied}
              tier="Applied/Core AI"
            />
            <p className="pt-2 text-xs text-muted-foreground">
              Each row shows the premium in Models A → B → C. The gap between A and C is the composition effect — the part of the raw premium that was really about seniority, education and clusters rather than AI itself.
            </p>
          </CardContent>
        </Card>

        <Card className="border-amber-500/30">
          <CardHeader>
            <CardTitle className="text-base">Predict a salary (deferred)</CardTitle>
            <CardDescription>
              This feature is parked pending a one-off Stata export.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>
              The plan is a posting-spec → predicted log-salary calculator with 95% prediction interval and per-variable marginal effects (country × tier × job family × seniority × education × cluster set). Client-side math, no backend.
            </p>
            <p>
              What&apos;s missing: the full Model C coefficient vector + vcov matrix. Today only the 2 tier dummies are exported (what you see above). The other 50-ish coefficients (job-family FE, seniority bands, 21 cluster dummies, state/industry FE, intercept, σ) live in <span className="font-mono text-xs">ai_skills_thesis_final.log</span>.
            </p>
            <p>
              Unblocks via ~5 lines of <span className="font-mono text-xs">esttab</span> in <span className="font-mono text-xs">analysis/stata/main.do</span>, followed by one Stata re-run. See <span className="font-mono">/about</span> for the exact snippet.
            </p>
          </CardContent>
        </Card>
      </section>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">What the shrinkage tells us</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>
            The raw US Applied/Core premium ({decompositionData[0].Applied.toFixed(1)}%) drops to ~{decompositionData[2].Applied.toFixed(1)}% once we control for job family, seniority, education and 21 skill clusters. That <span className="font-semibold text-foreground">~{(decompositionData[0].Applied - decompositionData[2].Applied).toFixed(0)} pp shrinkage</span> is the portion of the raw AI premium that is actually a selection effect — AI roles happen to also be more senior, better educated and cluster-rich.
          </p>
          <p>
            What&apos;s left in Model C is the cleanest estimate: roughly <span className="font-semibold text-foreground">{decompositionData[2].Applied.toFixed(1)}%</span> for Applied/Core AI and <span className="font-semibold text-foreground">{decompositionData[2].Integration.toFixed(1)}%</span> for AI Integration, both with p &lt; 0.001 in the US. Germany&apos;s coefficient is comparable in sign but the CI crosses zero; India&apos;s is smaller but significant.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function ShrinkageRow({
  label,
  a,
  b,
  c,
  tier,
}: {
  label: string;
  a: number;
  b: number;
  c: number;
  tier: AITier;
}) {
  const maxAbs = Math.max(Math.abs(a), Math.abs(b), Math.abs(c), 1);
  const bar = (v: number) => Math.round((Math.abs(v) / maxAbs) * 100);
  const color = TIER_COLORS[tier];
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="font-semibold">{label}</span>
        <span className="text-muted-foreground">
          Δ A→C: <span className="font-semibold tabular-nums text-foreground">{(a - c).toFixed(1)} pp</span>
        </span>
      </div>
      {[
        { lab: "A · raw", v: a },
        { lab: "B · + structural", v: b },
        { lab: "C · + clusters", v: c },
      ].map((s) => (
        <div key={s.lab} className="mb-1 flex items-center gap-2 text-xs">
          <span className="w-28 shrink-0 text-muted-foreground">{s.lab}</span>
          <div className="relative h-4 flex-1 rounded bg-muted/40">
            <div
              className="absolute left-0 top-0 h-full rounded"
              style={{ width: `${bar(s.v)}%`, backgroundColor: color, opacity: 0.85 }}
            />
          </div>
          <span className="w-14 text-right font-semibold tabular-nums">+{s.v.toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

function TierStat({
  tier,
  value,
  se,
  sig,
}: {
  tier: AITier;
  value: number;
  se: number;
  sig: string;
}) {
  return (
    <div className="rounded-lg border p-3">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span
          className="inline-block size-2.5 rounded-sm"
          style={{ backgroundColor: TIER_COLORS[tier] }}
        />
        {tier}
      </div>
      <div className="mt-1 flex items-baseline gap-2">
        <span className="text-2xl font-semibold tabular-nums">
          {value > 0 ? "+" : ""}
          {value.toFixed(1)}%
        </span>
        <span className={sigColor(sig) + " text-xs font-medium"}>{sig}</span>
      </div>
      <div className="mt-1 text-xs text-muted-foreground">
        SE ±{se.toFixed(1)} pp
      </div>
    </div>
  );
}
