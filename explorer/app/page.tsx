import Link from "next/link";
import {
  ArrowRight,
  GitCompareArrows,
  Globe2,
  Layers,
  Network,
  Presentation,
  SlidersHorizontal,
  Sparkles,
  TrendingUp,
} from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { TierStackedBar } from "@/components/charts/tier-stacked-bar";
import { SeniorityFamilyHeatmap } from "@/components/charts/seniority-family-heatmap";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { getG1, getKpi, getMetadata } from "@/lib/data/loaders";
import { COUNTRY_COLORS, COUNTRY_FLAGS, type Country } from "@/lib/constants";
import { formatCoef, formatNumber, formatPct, sigColor } from "@/lib/utils";

export default function HomePage() {
  const g1 = getG1();
  const kpi = getKpi();
  const meta = getMetadata();

  const us = kpi.countries.find((c) => c.country === "US")!;
  const de = kpi.countries.find((c) => c.country === "DE")!;
  const ind = kpi.countries.find((c) => c.country === "IN")!;

  return (
    <div className="mx-auto max-w-7xl space-y-10">
      {/* HERO */}
      <section className="relative overflow-hidden rounded-3xl border bg-gradient-to-br from-primary/10 via-primary/5 to-transparent p-6 md:p-10">
        <div className="absolute -right-24 -top-24 size-96 rounded-full bg-primary/15 blur-3xl" />
        <div className="absolute -bottom-32 -left-24 size-96 rounded-full bg-amber-500/10 blur-3xl" />
        <div className="relative">
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border bg-background/80 px-3 py-1 text-xs font-medium backdrop-blur">
            <Sparkles className="size-3.5 text-primary" />
            v2 · {formatNumber(meta.counts.rows_total ?? 0)} postings · 3 countries · 21 skill clusters
          </div>
          <h1 className="max-w-4xl text-3xl font-semibold leading-[1.1] tracking-tight md:text-5xl">
            AI skill requirements in IT job postings — an interactive companion to the master&rsquo;s thesis.
          </h1>
          <p className="mt-4 max-w-3xl text-base text-muted-foreground md:text-lg">
            Every chart in the thesis re-derived in your browser. Slice the dataset by country, family, seniority,
            education, industry, firm size, salary disclosure, even cluster presence — every number recomputes with N
            and a 95% Wilson confidence interval.
          </p>

          <div className="mt-6 grid gap-3 sm:grid-cols-3">
            {kpi.countries.map((c) => (
              <div
                key={c.country}
                className="rounded-2xl border bg-background/80 p-4 shadow-sm backdrop-blur transition-all hover:shadow-md"
              >
                <div className="flex items-center justify-between">
                  <div className="text-xs uppercase tracking-wider text-muted-foreground">
                    {COUNTRY_FLAGS[c.country as Country]} {c.country_label}
                  </div>
                  <Badge variant="outline" className="text-[10px]">
                    {formatNumber(c.total_jobs)}
                  </Badge>
                </div>
                <div
                  className="mt-2 text-4xl font-semibold tabular-nums tracking-tight md:text-5xl"
                  style={{ color: COUNTRY_COLORS[c.country as Country] }}
                >
                  {formatPct(c.ai_share, 1)}
                </div>
                <div className="mt-1 text-xs text-muted-foreground">
                  AI · Applied/Core {formatPct(c.tier_applied_pct, 1)} · Premium{" "}
                  <span className={sigColor(c.premium_applied_sig)}>
                    {c.premium_applied_pct === null ? "—" : `+${formatCoef(c.premium_applied_pct, 1)}%`}{" "}
                    {c.premium_applied_sig !== "ns" ? c.premium_applied_sig : ""}
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6 flex flex-wrap gap-2">
            <Button asChild>
              <Link href="/insights">
                <Sparkles />
                Read the story
                <ArrowRight />
              </Link>
            </Button>
            <Button asChild variant="outline">
              <Link href="/analyze">
                <SlidersHorizontal />
                Open workbench
              </Link>
            </Button>
            <Button asChild variant="outline">
              <Link href="/network">
                <Network />
                Skill network
              </Link>
            </Button>
            <Button asChild variant="ghost">
              <Link href="/present">
                <Presentation />
                Defense mode
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* TWO-COLUMN: tier composition + headline findings */}
      <section className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex items-start justify-between gap-4">
              <div>
                <CardTitle>AI tier composition by country</CardTitle>
                <CardDescription>
                  The structural baseline. Everything on /analyze re-aggregates from the same {formatNumber(meta.counts.rows_total ?? 0)} postings.
                </CardDescription>
              </div>
              <Badge variant="outline">Thesis Table 1</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <TierStackedBar data={g1} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Headline findings</CardTitle>
            <CardDescription>The four numbers that matter most.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            <FindingRow
              title="AI adoption gap"
              body={
                <>
                  US firms ask for AI in{" "}
                  <span className="font-semibold tabular-nums">{formatPct(us.ai_share)}</span> of postings —
                  <span className="font-semibold"> {formatCoef(us.ai_share / ind.ai_share, 1)}×</span> India ({formatPct(ind.ai_share)}).
                </>
              }
            />
            <FindingRow
              title="US salary premium"
              body={
                <>
                  Applied/Core AI carries a{" "}
                  <span className={sigColor(us.premium_applied_sig)}>
                    +{formatCoef(us.premium_applied_pct, 1)}% ({us.premium_applied_sig})
                  </span>{" "}
                  log-wage premium under full controls.
                </>
              }
            />
            <FindingRow
              title="Germany — structural"
              body={
                <>
                  Applied/Core share in DE ({formatPct(de.tier_applied_pct)}) beats US ({formatPct(us.tier_applied_pct)}); wages
                  too sparse (~8% disclosure) to detect a premium.
                </>
              }
            />
            <FindingRow
              title="India — smaller but real"
              body={
                <>
                  Just {formatPct(ind.ai_share)} of IN postings ask for AI, yet Applied/Core pays{" "}
                  <span className={sigColor(ind.premium_applied_sig)}>
                    +{formatCoef(ind.premium_applied_pct, 1)}% ({ind.premium_applied_sig})
                  </span>.
                </>
              }
            />
            <Button asChild variant="ghost" size="sm" className="ml-auto block w-fit">
              <Link href="/insights">
                See all 7 findings
                <ArrowRight />
              </Link>
            </Button>
          </CardContent>
        </Card>
      </section>

      {/* SENIORITY × FAMILY HEATMAP */}
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between gap-4">
            <div>
              <CardTitle>AI mention rate by family × seniority</CardTitle>
              <CardDescription>
                A 2D cut that no single chart in the thesis shows. Hover any cell for the underlying N and rates.
              </CardDescription>
            </div>
            <Badge variant="outline">v2 · new</Badge>
          </div>
        </CardHeader>
        <CardContent>
          <SeniorityFamilyHeatmap />
        </CardContent>
      </Card>

      {/* WHAT'S NEW IN v2 */}
      <section className="rounded-2xl border bg-gradient-to-br from-primary/5 to-transparent p-6 md:p-8">
        <div className="mb-2 inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-primary">
          <Sparkles className="size-3.5" /> What&rsquo;s new in v2
        </div>
        <h2 className="text-2xl font-semibold tracking-tight">Five new ways to read the data</h2>
        <p className="mt-2 max-w-3xl text-sm text-muted-foreground md:text-base">
          v1 shipped pre-aggregated charts. v2 ships the row-level dataset, an interactive workbench, and four new analytical views built on top of it.
        </p>
        <div className="mt-5 grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          <NewView
            href="/insights"
            icon={Sparkles}
            title="Insights — the story"
            body="A guided seven-finding tour through the empirical evidence. Built for readers who want the result without reading the thesis."
          />
          <NewView
            href="/network"
            icon={Network}
            title="Skill network"
            body="Force-style co-occurrence graph over 21 clusters. Hover a node to see which skills travel together — and how the pattern flips by country."
          />
          <NewView
            href="/distributions"
            icon={TrendingUp}
            title="Distributions"
            body="Smoothed density estimates for salary, experience and education by AI tier. Where aggregate means hide bimodal tails."
          />
          <NewView
            href="/geography"
            icon={Globe2}
            title="Geography"
            body="State and city-level AI hiring rates with Wilson 95% CIs, sorted by your chosen metric. CA, WA, MA dominate; Bangalore tells India's story."
          />
          <NewView
            href="/present"
            icon={Presentation}
            title="Defense mode"
            body="Nine full-screen slides with big numbers — designed for thesis defense or a lightning talk. Arrow keys to advance."
          />
          <NewView
            href="/clusters"
            icon={Layers}
            title="Skill clusters · drill-down"
            body="(carried from v1) 21 clusters × 3 countries with logit AMEs. Click any cell for the underlying postings."
          />
        </div>
      </section>

      {/* WORKBENCH PROMOS */}
      <section className="grid gap-4 md:grid-cols-2">
        <Card className="bg-gradient-to-br from-card to-primary/5">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <SlidersHorizontal className="size-4 text-primary" />
              Build your own slice
            </CardTitle>
            <CardDescription>
              On <code className="rounded bg-muted px-1 font-mono text-[10px]">/analyze</code>, combine any filters and pick a metric — AI share,
              Applied/Core, mean salary, count, or tier mix. Every bar shows N + 95% CI.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild variant="default">
              <Link href="/analyze">Open workbench<ArrowRight /></Link>
            </Button>
          </CardContent>
        </Card>
        <Card className="bg-gradient-to-br from-card to-amber-500/5">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <GitCompareArrows className="size-4 text-amber-700" />
              Compare two slices
            </CardTitle>
            <CardDescription>
              On <code className="rounded bg-muted px-1 font-mono text-[10px]">/compare</code>, define two arbitrary slices side by side. Useful for
              questions like &ldquo;is senior Data &amp; AI more similar between US/IN or US/DE?&rdquo;
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild variant="outline">
              <Link href="/compare">Compare slices<ArrowRight /></Link>
            </Button>
          </CardContent>
        </Card>
      </section>

      {/* META */}
      <section className="rounded-xl border bg-muted/30 p-4 text-xs text-muted-foreground">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="outline" className="text-[10px]">Snapshot</Badge>
          <span>
            Generated {new Date(meta.generated_at).toLocaleString()} from Stata run{" "}
            <code className="font-mono">{meta.run_dir}</code>.
          </span>
        </div>
        <p className="mt-2 leading-relaxed">
          Every chart on every page is computed in your browser from a single compact dataset shipped as{" "}
          <code className="font-mono">rows.json</code>. No server calls, no rate limits — slice freely.
        </p>
      </section>
    </div>
  );
}

function FindingRow({ title, body }: { title: string; body: React.ReactNode }) {
  return (
    <div className="border-l-2 border-primary/60 pl-3">
      <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        {title}
      </div>
      <div className="mt-1 text-sm leading-relaxed">{body}</div>
    </div>
  );
}

function NewView({
  href,
  icon: Icon,
  title,
  body,
}: {
  href: string;
  icon: typeof Sparkles;
  title: string;
  body: string;
}) {
  return (
    <Link
      href={href}
      className="group flex h-full flex-col gap-2 rounded-xl border bg-card p-4 transition-all hover:border-primary/50 hover:shadow-md"
    >
      <div className="flex items-start justify-between">
        <div className="flex size-8 items-center justify-center rounded-md bg-primary/10">
          <Icon className="size-4 text-primary" />
        </div>
        <ArrowRight className="size-4 -translate-x-1 opacity-0 transition-all group-hover:translate-x-0 group-hover:opacity-100" />
      </div>
      <div className="text-sm font-semibold">{title}</div>
      <div className="text-xs leading-relaxed text-muted-foreground">{body}</div>
    </Link>
  );
}
