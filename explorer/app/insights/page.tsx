"use client";

import Link from "next/link";
import { useMemo } from "react";
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
import { ArrowRight, ChevronDown, Sparkles } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { FindingCard } from "@/components/insights/finding-card";
import { BigNumber } from "@/components/insights/big-number";
import { InsightsToc } from "@/components/insights/toc";
import { CountryBars, countryDatum } from "@/components/charts/country-bars";
import { TierStackedBar } from "@/components/charts/tier-stacked-bar";
import {
  getG1,
  getG2,
  getG7,
  getKpi,
  getMetadata,
  getClusters,
} from "@/lib/data/loaders";
import {
  COUNTRIES,
  COUNTRY_COLORS,
  COUNTRY_FLAGS,
  TIER_COLORS,
} from "@/lib/constants";
import { formatCoef, formatNumber, formatPct, sigColor } from "@/lib/utils";

export default function InsightsPage() {
  const kpi = getKpi();
  const g1 = getG1();
  const g2 = getG2();
  const g7 = getG7();
  const clusters = getClusters();
  const meta = getMetadata();

  const us = kpi.countries.find((c) => c.country === "US")!;
  const de = kpi.countries.find((c) => c.country === "DE")!;
  const ind = kpi.countries.find((c) => c.country === "IN")!;

  const aiShareData = COUNTRIES.map((c) => {
    const k = kpi.countries.find((r) => r.country === c)!;
    return countryDatum(c, k.ai_share, `${formatNumber(k.total_jobs)} postings`);
  });

  const appliedShareData = COUNTRIES.map((c) => {
    const k = kpi.countries.find((r) => r.country === c)!;
    return countryDatum(c, k.tier_applied_pct, `Applied/Core only`);
  });

  const familyChart = useMemo(() => {
    const families = Array.from(new Set(g2.map((r) => r.job_family))).filter(Boolean);
    return families.map((fam) => {
      const us = g2.find((r) => r.country === "US" && r.job_family === fam);
      const de = g2.find((r) => r.country === "DE" && r.job_family === fam);
      const ind = g2.find((r) => r.country === "IN" && r.job_family === fam);
      return {
        family: fam,
        US: us ? us.ai_share : 0,
        DE: de ? de.ai_share : 0,
        IN: ind ? ind.ai_share : 0,
        n_us: us?.n ?? 0,
        n_de: de?.n ?? 0,
        n_in: ind?.n ?? 0,
      };
    }).sort((a, b) => (b.US + b.DE + b.IN) - (a.US + a.DE + a.IN));
  }, [g2]);

  const topClusters = useMemo(() => clusters.slice(0, 8), [clusters]);

  // Premium chart data
  const premiumData = COUNTRIES.map((c) => {
    const row = g7[c]?.find((r) => r.coef === "applied_core");
    const k = kpi.countries.find((kc) => kc.country === c)!;
    return {
      country: c,
      label: `${COUNTRY_FLAGS[c]} ${c}`,
      premium: k.premium_applied_pct ?? 0,
      ci_lo: row?.ci_low ? row.ci_low * 100 : 0,
      ci_hi: row?.ci_high ? row.ci_high * 100 : 0,
      sig: k.premium_applied_sig,
    };
  });

  const tocEntries = [
    { id: "f-01", label: "The headline gap", short: "Headline gap" },
    { id: "f-02", label: "Two flavors of AI", short: "Two flavors" },
    { id: "f-03", label: "Where AI demand concentrates", short: "Where it concentrates" },
    { id: "f-04", label: "The wage premium", short: "Wage premium" },
    { id: "f-05", label: "Generative AI dominates", short: "Generative AI" },
    { id: "f-06", label: "Germany — the structural anomaly", short: "Germany anomaly" },
    { id: "f-07", label: "Why postings matter", short: "Why postings matter" },
  ];

  return (
    <div className="mx-auto max-w-6xl space-y-10">
      {/* Hero */}
      <section className="relative overflow-hidden rounded-3xl border bg-gradient-to-br from-primary/10 via-primary/5 to-transparent p-8 md:p-12">
        <div className="absolute -right-20 -top-20 size-72 rounded-full bg-primary/10 blur-3xl" />
        <div className="absolute -bottom-32 -left-20 size-96 rounded-full bg-amber-500/10 blur-3xl" />
        <div className="relative">
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border bg-background/80 px-3 py-1 text-xs font-medium backdrop-blur">
            <Sparkles className="size-3.5 text-primary" />
            Seven findings · the thesis in 5 minutes
          </div>
          <h1 className="max-w-3xl text-3xl font-semibold leading-[1.1] tracking-tight md:text-5xl">
            How <span className="text-primary">AI is reshaping</span> IT hiring across three economies.
          </h1>
          <p className="mt-4 max-w-2xl text-base text-muted-foreground md:text-lg">
            A guided tour through the empirical evidence: {formatNumber(meta.counts.rows_total ?? 0)} Glassdoor postings,
            three labour markets, twenty-one skill clusters, and one consistent story —
            <span className="text-foreground"> what employers ask for is the most reliable signal we have left</span>.
          </p>

          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            <BigNumber
              value={`${formatPct(us.ai_share, 1)}`}
              label="US · AI mention rate"
              sublabel={`${formatNumber(us.total_jobs)} postings · 2024-25`}
              accent={COUNTRY_COLORS.US}
              size="lg"
            />
            <BigNumber
              value={`${formatPct(de.ai_share, 1)}`}
              label="Germany · AI mention rate"
              sublabel={`${formatNumber(de.total_jobs)} postings · 2024-25`}
              accent={COUNTRY_COLORS.DE}
              size="lg"
            />
            <BigNumber
              value={`${formatPct(ind.ai_share, 1)}`}
              label="India · AI mention rate"
              sublabel={`${formatNumber(ind.total_jobs)} postings · 2024-25`}
              accent={COUNTRY_COLORS.IN}
              size="lg"
            />
          </div>

          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Button asChild>
              <Link href="#f-01">
                Start the story
                <ChevronDown />
              </Link>
            </Button>
            <Button asChild variant="outline">
              <Link href="/analyze">
                Skip to workbench
                <ArrowRight />
              </Link>
            </Button>
            <span className="text-xs text-muted-foreground">Press space to scroll · or ⌘ K to navigate</span>
          </div>
        </div>
      </section>

      {/* Findings stream + TOC */}
      <div className="flex gap-6 xl:-mr-48">
      <div className="flex-1 space-y-10 min-w-0">
      {/* Finding 1: The headline gap */}
        <FindingCard
          index={1}
          eyebrow="The headline gap"
          accent={COUNTRY_COLORS.US}
          headline={
            <>
              US postings ask for AI{" "}
              <span style={{ color: COUNTRY_COLORS.US }}>3.3×</span> more often than Indian ones.
            </>
          }
          takeaway={
            <>
              The most consequential finding in the thesis is also the simplest. Once we read every job description and
              flag any explicit AI requirement, US firms stand out as the heaviest adopter — even though the salary disclosure rates,
              labour-market structure, and dominant industries differ wildly between the three countries.
            </>
          }
          chart={<CountryBars data={aiShareData} />}
          evidence={
            <>
              <strong>Methodology:</strong> hybrid extractor — deterministic dictionary match (≈300 AI-related skills) ∪
              GPT-4o classifier on full description. <strong>N total = {formatNumber(meta.counts.rows_total ?? 0)}</strong>.
              All gaps significant at p &lt; 0.001 in country-level Wilson confidence intervals.
            </>
          }
        />

      {/* Finding 2: Two flavours of AI */}
      <FindingCard
        index={2}
        side="right"
        accent={TIER_COLORS["Applied/Core AI"]}
        eyebrow="Two flavors of AI"
        headline={
          <>
            <span style={{ color: TIER_COLORS["AI Integration"] }}>Integration</span> ≠{" "}
            <span style={{ color: TIER_COLORS["Applied/Core AI"] }}>Applied AI</span>.
          </>
        }
        takeaway={
          <>
            Beyond a binary AI / no-AI split, postings cleave into two functionally distinct tiers. AI Integration jobs use
            pre-built AI (LLM APIs, AI-powered features) while Applied/Core AI roles build it (modelling, ML infra, research).
            Germany flips the ratio: Applied/Core ({formatPct(de.tier_applied_pct)}) beats Integration ({formatPct(de.tier_integration_pct)})
            — the only country where it does.
          </>
        }
        chart={<TierStackedBar data={g1} />}
        evidence={
          <>
            From Stata Table 1. Tier classification is rule-based on the LLM&rsquo;s &quot;{`desc_tier_llm`}&quot; field;
            agreement with the dictionary-only fallback is &gt;94 % on a stratified hand-checked sample of 100 postings.
          </>
        }
      />

      {/* Finding 3: Where AI is asked for */}
      <FindingCard
        index={3}
        eyebrow="Where the demand concentrates"
        headline={
          <>
            Data &amp; AI roles are saturated.{" "}
            <span className="text-primary">DevOps &amp; Cloud</span> is catching up.
          </>
        }
        takeaway={
          <>
            Within IT, AI hiring is not uniform. Data &amp; AI specialists almost universally need AI skills (&gt; 70 %),
            but the more striking story is in adjacent families — DevOps/Cloud and Software Engineering — where AI is
            quietly becoming a baseline expectation rather than a specialist requirement.
          </>
        }
        chart={
          <div style={{ width: "100%", height: 300 }}>
            <ResponsiveContainer>
              <BarChart data={familyChart} margin={{ top: 8, right: 12, left: 0, bottom: 30 }}>
                <CartesianGrid stroke="#e5e7eb" strokeDasharray="2 2" vertical={false} />
                <XAxis
                  dataKey="family"
                  tick={{ fontSize: 11 }}
                  angle={-22}
                  textAnchor="end"
                  height={60}
                  interval={0}
                />
                <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => `${v}%`} width={42} />
                <Tooltip
                  contentStyle={{
                    borderRadius: 8,
                    fontSize: 12,
                    padding: "6px 10px",
                    border: "1px solid #e5e7eb",
                  }}
                  formatter={(v: number) => `${v.toFixed(1)}%`}
                />
                <Bar dataKey="US" fill={COUNTRY_COLORS.US} radius={[3, 3, 0, 0]} />
                <Bar dataKey="DE" fill={COUNTRY_COLORS.DE} radius={[3, 3, 0, 0]} />
                <Bar dataKey="IN" fill={COUNTRY_COLORS.IN} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        }
        evidence={
          <>
            Job-family taxonomy from <code className="rounded bg-muted px-1 font-mono text-[10px]">job_family</code>{" "}
            (rule-based on title patterns, see <Link href="/about" className="underline">methodology</Link>). Bars are AI mention rates within each family,
            country-stratified.
          </>
        }
      />

      {/* Finding 4: The wage premium */}
      <FindingCard
        index={4}
        side="right"
        accent={COUNTRY_COLORS.US}
        eyebrow="The wage premium"
        headline={
          <>
            US Applied/Core AI roles pay{" "}
            <span style={{ color: COUNTRY_COLORS.US }}>+{formatCoef(us.premium_applied_pct ?? 0, 1)}%</span> more.
          </>
        }
        takeaway={
          <>
            After full controls (job family, seniority, education, state, industry, firm size, plus presence of all 21
            skill clusters), the AI tier still carries a robust log-wage premium in the US. The premium is{" "}
            <span className={sigColor(us.premium_applied_sig)}>{us.premium_applied_sig}</span> in the saturated US labour
            market and <span className={sigColor(ind.premium_applied_sig)}>{ind.premium_applied_sig}</span> in India —
            and statistically invisible in Germany only because salary disclosure there is &lt; 8 %.
          </>
        }
        chart={
          <div style={{ width: "100%", height: 280 }}>
            <ResponsiveContainer>
              <BarChart data={premiumData} margin={{ top: 16, right: 16, left: 0, bottom: 12 }}>
                <CartesianGrid stroke="#e5e7eb" strokeDasharray="2 2" vertical={false} />
                <XAxis dataKey="label" tick={{ fontSize: 12 }} tickLine={false} />
                <YAxis
                  tick={{ fontSize: 11 }}
                  tickFormatter={(v) => `${v}%`}
                  width={42}
                  domain={[-5, 25]}
                />
                <Tooltip
                  contentStyle={{
                    borderRadius: 8,
                    fontSize: 12,
                    padding: "6px 10px",
                    border: "1px solid #e5e7eb",
                  }}
                  formatter={(v: number) => `${v.toFixed(1)}%`}
                />
                <Bar dataKey="premium" radius={[6, 6, 0, 0]}>
                  {premiumData.map((d) => (
                    <Cell key={d.country} fill={COUNTRY_COLORS[d.country]} fillOpacity={d.sig === "ns" ? 0.45 : 1} />
                  ))}
                  <ErrorBar dataKey={(d: typeof premiumData[number]) => [d.premium - d.ci_lo, d.ci_hi - d.premium]} stroke="#0008" strokeWidth={1.4} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        }
        evidence={
          <>
            From thesis Model C: <code className="rounded bg-muted px-1 font-mono text-[10px]">log(salary_mid) ~ tier + family + seniority + edu + state + industry + size + clusters</code>{" "}
            with HC1 robust SEs. Reference category: tier &quot;None&quot;. Error bars = 95 % CI. NS bars are translucent.
          </>
        }
      />

      {/* Finding 5: Generative AI */}
      <FindingCard
        index={5}
        eyebrow="The cluster crowning everything"
        headline={
          <>
            Generative AI is the <span className="text-primary">single most demanded</span> skill cluster in 2024–25.
          </>
        }
        takeaway={
          <>
            Across 21 skill clusters extracted from the dataset, Generative AI is mentioned in nearly every AI-tagged posting
            and shows the highest cross-country lift. This was not the case in pre-2023 LinkedIn / Glassdoor data — the
            cluster effectively didn&rsquo;t exist as a hiring signal until ChatGPT.
          </>
        }
        chart={
          <div style={{ width: "100%", height: 320 }}>
            <ResponsiveContainer>
              <BarChart
                data={topClusters.map((c) => ({ label: c.label, pct: c.pct, n: c.frequency }))}
                layout="vertical"
                margin={{ top: 6, right: 30, left: 110, bottom: 6 }}
              >
                <CartesianGrid stroke="#e5e7eb" strokeDasharray="2 2" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => `${v}%`} />
                <YAxis
                  dataKey="label"
                  type="category"
                  tick={{ fontSize: 11 }}
                  width={108}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  contentStyle={{
                    borderRadius: 8,
                    fontSize: 12,
                    padding: "6px 10px",
                    border: "1px solid #e5e7eb",
                  }}
                  formatter={(v: number, _name, p) => [`${v.toFixed(1)}% (${formatNumber(p?.payload?.n ?? 0)} postings)`, "Mention rate"]}
                />
                <Bar dataKey="pct" fill="#3c6ea8" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        }
        evidence={
          <>
            Cluster frequencies on AI-tagged US postings (k-means over hard-skill embeddings, k=21 chosen by silhouette).{" "}
            See <Link href="/clusters" className="underline">/clusters</Link> for the full cross-country heatmap and{" "}
            <Link href="/network" className="underline">/network</Link> for which clusters appear together.
          </>
        }
      />

      {/* Finding 6: Germany — the outlier */}
      <FindingCard
        index={6}
        side="right"
        accent={COUNTRY_COLORS.DE}
        eyebrow="The German anomaly"
        headline={
          <>
            Germany hires Applied/Core AI{" "}
            <span style={{ color: COUNTRY_COLORS.DE }}>more than the US does</span> — the salary signal can&rsquo;t see it.
          </>
        }
        takeaway={
          <>
            <strong>{formatPct(de.tier_applied_pct)}</strong> of German IT postings are Applied/Core AI vs{" "}
            <strong>{formatPct(us.tier_applied_pct)}</strong> in the US. But because German firms post salary on only{" "}
            ~ 8 % of listings — vs ~ 35 % in the US — Model C&rsquo;s power to detect a wage premium in Germany is
            extremely limited. The structural signal is louder than the disclosed-wage signal.
          </>
        }
        chart={<CountryBars data={appliedShareData} />}
        evidence={
          <>
            Disclosure rate per country computed in-browser from the row-level dataset. The contrast motivates the thesis&rsquo;
            decision to use posting volume × tier mix — rather than salary alone — as the primary outcome.
          </>
        }
      />

      {/* Finding 7: Implications */}
      <FindingCard
        index={7}
        accent="#3c8a6a"
        eyebrow="Why this matters"
        headline={
          <>
            Job postings are the{" "}
            <span style={{ color: "#3c8a6a" }}>highest-frequency, lowest-latency</span> labour-market signal we have.
          </>
        }
        takeaway={
          <>
            Salary surveys arrive with a one-to-three-year lag and shrink as transparency declines. Postings update
            daily, capture detailed skill stacks, and let us see <em>what firms are buying</em> rather than
            <em> what they paid yesterday</em>. For AI specifically — a fast-moving, mostly bottom-up labour shock —
            postings are the best window we have. The dataset behind every chart in this thesis is fully reusable; the{" "}
            <Link href="/analyze" className="underline">workbench</Link> lets you re-derive any number on your own slice.
          </>
        }
        chart={
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
            <BigNumber
              size="md"
              value={formatNumber(meta.counts.rows_total ?? 0)}
              label="Postings analysed"
              sublabel="3 countries · 2024–25"
            />
            <BigNumber
              size="md"
              value={meta.counts.clusters}
              label="Skill clusters"
              sublabel="k-means on embeddings"
            />
            <BigNumber
              size="md"
              value={meta.counts.job_families}
              label="Job families"
              sublabel="rule-based taxonomy"
            />
            <BigNumber
              size="md"
              value="3"
              label="OLS specifications"
              sublabel="A → B → C shrinkage"
            />
            <BigNumber
              size="md"
              value="9"
              label="Logit regressions"
              sublabel="3 countries × 3 outcomes"
            />
            <BigNumber
              size="md"
              value="100%"
              label="Reproducible"
              sublabel="Stata + Python · MIT-licensed"
            />
          </div>
        }
        evidence={
          <>
            Want to verify a number? Open <Link href="/analyze" className="underline">/analyze</Link> and rebuild any slice.
            Want the underlying methodology? See <Link href="/about" className="underline">/about</Link>. Every chart is
            re-derived from the same {formatNumber(meta.counts.rows_total ?? 0)}-row JSON shipped to your browser.
          </>
        }
      />
      </div>
      <InsightsToc entries={tocEntries} />
      </div>

      {/* CTA */}
      <section className="rounded-2xl border bg-gradient-to-br from-primary/5 to-transparent p-6 md:p-10">
        <div className="grid gap-6 md:grid-cols-3 md:items-center">
          <div className="md:col-span-2">
            <Badge variant="outline" className="mb-2">For researchers</Badge>
            <h2 className="text-2xl font-semibold tracking-tight">Now go cut the data yourself.</h2>
            <p className="mt-2 text-sm text-muted-foreground md:text-base">
              These seven findings are the headline. The workbench at <Link href="/analyze" className="underline">/analyze</Link>{" "}
              re-aggregates the same {formatNumber(meta.counts.rows_total ?? 0)} postings under any filter combination — country, family,
              seniority, education, industry, firm size, salary disclosure, even cluster presence. Every number recomputes with N
              and 95% Wilson CI so you can tell signal from noise.
            </p>
          </div>
          <div className="flex flex-col gap-2">
            <Button asChild>
              <Link href="/analyze">Open workbench<ArrowRight /></Link>
            </Button>
            <Button asChild variant="outline">
              <Link href="/network">Explore skill network<ArrowRight /></Link>
            </Button>
            <Button asChild variant="ghost">
              <Link href="/about">Read methodology</Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}
