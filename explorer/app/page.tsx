import Link from "next/link";
import { ArrowRight, GitCompareArrows, SlidersHorizontal } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { KpiCard } from "@/components/kpi/kpi-card";
import { TierStackedBar } from "@/components/charts/tier-stacked-bar";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { getG1, getKpi, getMetadata } from "@/lib/data/loaders";
import { COUNTRY_FLAGS, COUNTRY_COLORS, type Country } from "@/lib/constants";
import { formatCoef, formatNumber, formatPct, sigColor } from "@/lib/utils";

export default function HomePage() {
  const g1 = getG1();
  const kpi = getKpi();
  const meta = getMetadata();

  const us = kpi.countries.find((c) => c.country === "US")!;
  const de = kpi.countries.find((c) => c.country === "DE")!;
  const ind = kpi.countries.find((c) => c.country === "IN")!;

  return (
    <div className="mx-auto max-w-7xl space-y-8">
      <PageHeader
        eyebrow="Thesis companion"
        title="AI skill requirements in IT job postings"
        description="An interactive research companion to the master's thesis comparing Glassdoor IT postings across the US, Germany and India. Start with the headline numbers below, then open Analyze to slice the full dataset yourself."
        actions={
          <div className="flex flex-wrap gap-2">
            <Button asChild>
              <Link href="/analyze">
                <SlidersHorizontal />
                Open workbench
                <ArrowRight />
              </Link>
            </Button>
            <Button asChild variant="outline">
              <Link href="/compare">
                <GitCompareArrows />
                Compare slices
              </Link>
            </Button>
          </div>
        }
      />

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {kpi.countries.map((c) => (
          <KpiCard
            key={c.country}
            label={`${COUNTRY_FLAGS[c.country as Country]} ${c.country_label} — AI share`}
            value={formatPct(c.ai_share)}
            sublabel={`${formatNumber(c.total_jobs)} postings · Applied/Core ${formatPct(c.tier_applied_pct)}`}
            accent={COUNTRY_COLORS[c.country as Country]}
          />
        ))}
      </section>

      <section className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex items-start justify-between gap-4">
              <div>
                <CardTitle>AI tier composition by country</CardTitle>
                <CardDescription>
                  The structural baseline. Everything on /analyze re-aggregates from the same 44 832 postings.
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
            <CardDescription>Most important numbers from the thesis.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            <FindingRow
              title="AI adoption gap"
              body={
                <>
                  Postings requiring any AI skill in the US (
                  <span className="font-semibold tabular-nums">{formatPct(us.ai_share)}</span>
                  ) are <span className="font-semibold">{formatCoef(us.ai_share / ind.ai_share, 1)}×</span> higher than in India ({formatPct(ind.ai_share)}).
                </>
              }
            />
            <FindingRow
              title="US salary premium holds"
              body={
                <>
                  Applied/Core AI carries a{" "}
                  <span className={sigColor(us.premium_applied_sig)}>
                    {formatCoef(us.premium_applied_pct, 1)}% ({us.premium_applied_sig})
                  </span>{" "}
                  log-wage premium after full controls (thesis Model C).
                </>
              }
            />
            <FindingRow
              title="Germany: structural, not wage-driven"
              body={
                <>
                  Applied/Core AI share in DE ({formatPct(de.tier_applied_pct)}) beats the US ({formatPct(us.tier_applied_pct)}), yet the DE premium is only{" "}
                  <span className={sigColor(de.premium_applied_sig)}>
                    {formatCoef(de.premium_applied_pct, 1)}% ({de.premium_applied_sig})
                  </span>{" "}
                  — salary disclosure is too sparse (8%) to detect more.
                </>
              }
            />
            <FindingRow
              title="India: smaller but real"
              body={
                <>
                  Only {formatPct(ind.ai_share)} of IN postings mention AI, but Applied/Core AI still pays{" "}
                  <span className={sigColor(ind.premium_applied_sig)}>
                    {formatCoef(ind.premium_applied_pct, 1)}% ({ind.premium_applied_sig})
                  </span>.
                </>
              }
            />
          </CardContent>
        </Card>
      </section>

      <section className="rounded-xl border bg-gradient-to-br from-primary/5 to-transparent p-6">
        <div className="mb-2 text-xs font-medium uppercase tracking-wider text-primary">
          What can I do here?
        </div>
        <h2 className="text-xl font-semibold">Explore the same data from your own angle</h2>
        <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
          The dataset behind every chart in the thesis is loaded in your browser (~{formatNumber(meta.counts.rows_total ?? 0)} postings). No pre-aggregated view is forced on you — every number recomputes from the slice you build.
        </p>
        <ul className="mt-4 grid gap-3 text-sm md:grid-cols-2">
          <UseCase
            title="Build a slice, watch it re-aggregate"
            body="On /analyze, combine country × job family × seniority × education × cluster presence and pick a metric (AI share, Applied/Core share, mean salary, count). Every bar shows N and 95% CI so you can tell signal from noise."
            href="/analyze"
          />
          <UseCase
            title="Compare two slices side by side"
            body="On /compare, define two arbitrary slices and get a diff table: tier mix, mean salary, cluster lift. Useful for questions like 'does senior Data & AI differ more between US and India, or between US and Germany?'"
            href="/compare"
          />
          <UseCase
            title="Click a cluster, see the actual postings"
            body="On /clusters the heatmap is just the entry point. Click any cell for per-country AME plus the observed AI-share lift from real postings with and without that cluster in your current slice."
            href="/clusters"
          />
          <UseCase
            title="Pull out an exact posting"
            body="/explorer shows the full ~45k dataset with the same slicer. Search, sort, export the filtered set as CSV for offline use."
            href="/explorer"
          />
        </ul>
      </section>

      <section className="pt-2">
        <p className="text-xs text-muted-foreground">
          Data snapshot generated {new Date(meta.generated_at).toLocaleString()} from Stata run{" "}
          <span className="font-mono">{meta.run_dir}</span>. All aggregates and slices you see are computed in your browser from a ~{formatNumber(meta.counts.rows_total ?? 0)}-row compact dataset shipped as <span className="font-mono">rows.json</span>.
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

function UseCase({ title, body, href }: { title: string; body: string; href: string }) {
  return (
    <li>
      <Link
        href={href}
        className="group flex h-full flex-col rounded-lg border bg-card p-4 transition-all hover:border-primary/50 hover:shadow-md"
      >
        <div className="flex items-center gap-2 text-sm font-semibold">
          {title}
          <ArrowRight className="size-3.5 opacity-0 transition-all group-hover:translate-x-0.5 group-hover:opacity-100" />
        </div>
        <p className="mt-1 text-xs text-muted-foreground">{body}</p>
      </Link>
    </li>
  );
}
