import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getMetadata, getKpi } from "@/lib/data/loaders";
import { COUNTRIES, COUNTRY_FLAGS, COUNTRY_LABELS, type Country } from "@/lib/constants";
import { formatNumber, formatPct } from "@/lib/utils";

export default function AboutPage() {
  const meta = getMetadata();
  const kpi = getKpi();

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <PageHeader
        eyebrow="Methodology"
        title="About this companion"
        description="How the thesis data flows from scraped Glassdoor postings through the extraction pipeline into the charts and numbers you see here."
      />

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Research question</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm leading-relaxed">
          <p>
            The master&apos;s thesis asks: <span className="font-semibold">how prevalent are AI skill requirements in IT job postings, and is there a measurable salary premium for them?</span> It compares three labour markets with very different AI adoption speeds — the United States, Germany, and India.
          </p>
          <p className="text-muted-foreground">
            The argument proceeds in three steps: (1) classify each posting into an AI tier based on the job description, (2) model the probability of each tier as a function of skill clusters and structural controls, and (3) regress log-salary on tier membership to isolate the wage premium net of composition.
          </p>
        </CardContent>
      </Card>

      <section className="grid gap-4 md:grid-cols-3">
        {kpi.countries.map((c) => (
          <Card key={c.country}>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">
                {COUNTRY_FLAGS[c.country as Country]} {c.country_label}
              </CardTitle>
              <CardDescription className="text-xs">
                {formatNumber(c.total_jobs)} IT postings · {formatPct(c.ai_share)} AI share
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-1 pt-0 text-xs text-muted-foreground">
              <div>None — {formatPct(c.tier_none_pct)}</div>
              <div>AI Integration — {formatPct(c.tier_integration_pct)}</div>
              <div>Applied/Core AI — {formatPct(c.tier_applied_pct)}</div>
            </CardContent>
          </Card>
        ))}
      </section>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Data pipeline</CardTitle>
          <CardDescription>From raw scrapes to the JSONs you&apos;re looking at now.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <Step n={1} title="Scraping">
            Glassdoor IT postings for each country were scraped during Q4 2025 / Q1 2026. Each record carries a job title, company, city/state, description, disclosed salary range (where present), and company metadata.
          </Step>
          <Step n={2} title="AI-skill extraction">
            Every posting runs through a hybrid extractor: a deterministic dictionary of ~1 400 hard-skill terms grouped into 21 clusters, plus an OpenAI LLM pass that reads the description and assigns an <span className="font-mono text-xs">ai_level</span> — <span className="font-semibold">None</span>, <span className="font-semibold">AI Integration</span>, or <span className="font-semibold">Applied/Core AI</span>. The union of both paths forms the final skill / tier fields.
          </Step>
          <Step n={3} title="Econometric analysis (Stata)">
            Three model families: (i) a binary logit of Pr(AI required) for each country, (ii) a multinomial logit of Pr(tier) on the US sub-sample, and (iii) OLS on log(mid-point salary) with tier dummies as the regressor of interest. Models A / B / C progressively add controls (country FE → + structural controls → + 21 cluster dummies).
          </Step>
          <Step n={4} title="Charts & this app">
            The Stata run exports <span className="font-mono text-xs">charts_data/g1–g7.csv</span> with pre-aggregated coefficients and shares. A Python script (<span className="font-mono text-xs">explorer/scripts/build_data.py</span>) reshapes those CSVs into the JSONs served statically by this Next.js app.
          </Step>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Caveats</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>
            <span className="font-semibold text-foreground">Salary disclosure varies.</span> Roughly 73% of US postings include a salary range; the figure drops to ~8% in Germany and ~35% in India. That limits the statistical power of the DE premium estimate in particular.
          </p>
          <p>
            <span className="font-semibold text-foreground">Sample ≠ population.</span> Glassdoor tilts toward listed, mid-to-large employers and English-language postings; generalisation to the full IT labour market should be cautious.
          </p>
          <p>
            <span className="font-semibold text-foreground">LLM classification isn&apos;t perfect.</span> A random sample of 500 postings was hand-checked against the model&apos;s tier assignment; agreement was ≈94%. See the thesis for the confusion matrix.
          </p>
          <p>
            <span className="font-semibold text-foreground">Wilson CIs assume independence.</span> The row-level CIs reported on /analyze and /compare treat postings as i.i.d.; intra-company correlation isn&apos;t modelled there. The regression coefficients on /clusters and /premium are Huber–White robust, as specified in Stata.
          </p>
        </CardContent>
      </Card>

      <Card className="border-amber-500/30">
        <CardHeader>
          <CardTitle className="text-base">Planned / deferred features</CardTitle>
          <CardDescription>
            These require a new Stata export pass that hasn&apos;t been run yet.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-muted-foreground">
          <div className="rounded-md border bg-amber-500/5 px-3 py-2">
            <div className="font-semibold text-foreground">Salary predictor (counterfactual calculator)</div>
            <p className="mt-1">
              Given a posting spec (country × tier × job family × seniority × education × cluster set), predict log-salary + 95% prediction interval + per-variable marginal effects. Requires the full Model C coefficient vector. Currently only the 2 tier coefficients are exported; the 21 cluster dummies, education, seniority, state/industry fixed effects, intercept and σ live in the Stata <span className="font-mono text-xs">.log</span>.
            </p>
          </div>
          <div className="rounded-md border bg-amber-500/5 px-3 py-2">
            <div className="font-semibold text-foreground">Per-regression fit stats</div>
            <p className="mt-1">
              N, pseudo-R², log-likelihood for each regression — surfaced in every tooltip that reports a coefficient. Needs <span className="font-mono text-xs">eststo</span> + <span className="font-mono text-xs">esttab</span> pass with <span className="font-mono text-xs">stats(N r2_p ll)</span>.
            </p>
          </div>
          <div className="rounded-md border bg-amber-500/5 px-3 py-2">
            <div className="font-semibold text-foreground">Bootstrapped CIs on AMEs</div>
            <p className="mt-1">
              Alternative to Wald CIs on the /clusters heatmap. Needs <span className="font-mono text-xs">margins, dydx(cluster_*) post vce(bootstrap, reps(500))</span>.
            </p>
          </div>
          <p className="pt-1">
            Workflow: add the exports to <span className="font-mono text-xs">analysis/stata/main.do</span>, re-run once, then re-run <span className="font-mono text-xs">pnpm run build-data</span>. No app code changes required beyond the stubbed pages picking up the new JSONs.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Data snapshot</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3 text-sm md:grid-cols-2">
          <Row k="Stata run" v={<span className="font-mono text-xs">{meta.run_dir}</span>} />
          <Row k="Generated at" v={new Date(meta.generated_at).toLocaleString()} />
          <Row k="Countries" v={meta.counts.countries} />
          <Row k="Clusters" v={meta.counts.clusters} />
          <Row k="Job families" v={meta.counts.job_families} />
          <Row k="Sampled rows" v={meta.counts.jobs_sample} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Where this lives</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>
            Full thesis and LaTeX source are tracked in the same repository as the app. The Stata analysis code is in <span className="font-mono text-xs">analysis/stata/</span>, the Python extraction pipeline in <span className="font-mono text-xs">ai_skills/</span>, and the source of this companion app in <span className="font-mono text-xs">explorer/</span>.
          </p>
          <div className="flex flex-wrap gap-2 pt-2">
            {COUNTRIES.map((c) => (
              <Badge key={c} variant="outline">
                {COUNTRY_FLAGS[c]} {COUNTRY_LABELS[c]} — n = {formatNumber(kpi.countries.find((x) => x.country === c)?.total_jobs ?? 0)}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Jump back into the data</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-2 md:grid-cols-2">
          <AboutLink href="/tiers" label="AI tier composition" />
          <AboutLink href="/job-families" label="AI share by job family" />
          <AboutLink href="/clusters" label="Skill-cluster heatmap" />
          <AboutLink href="/premium" label="Salary premium (Models A/B/C)" />
          <AboutLink href="/explorer" label="Browse the sampled postings" />
        </CardContent>
      </Card>
    </div>
  );
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
        {n}
      </div>
      <div>
        <div className="font-semibold">{title}</div>
        <div className="mt-0.5 text-muted-foreground">{children}</div>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-dashed py-1.5">
      <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
        {k}
      </span>
      <span className="tabular-nums">{v}</span>
    </div>
  );
}

function AboutLink({ href, label }: { href: string; label: string }) {
  return (
    <Link
      href={href}
      className="flex items-center justify-between rounded-md border px-3 py-2 text-sm transition-colors hover:border-primary/40 hover:bg-primary/5"
    >
      <span>{label}</span>
      <ArrowUpRight className="size-4 text-muted-foreground" />
    </Link>
  );
}
