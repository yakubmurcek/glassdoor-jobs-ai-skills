import Link from "next/link";
import {
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Brain,
  Calculator,
  Database,
  FileSearch,
  Filter,
  Layers,
  Sparkles,
} from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getMetadata, getKpi } from "@/lib/data/loaders";
import {
  COUNTRIES,
  COUNTRY_COLORS,
  COUNTRY_FLAGS,
  COUNTRY_LABELS,
  TIER_COLORS,
  type Country,
} from "@/lib/constants";
import { formatNumber, formatPct } from "@/lib/utils";

export default function AboutPage() {
  const meta = getMetadata();
  const kpi = getKpi();

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <PageHeader
        eyebrow="Methodology"
        title="How the dataset was built"
        description="The pipeline from raw Glassdoor scrapes to the row-level dataset shipped with this app."
      />

      {/* PIPELINE DIAGRAM */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <FileSearch className="size-4 text-primary" />
            Data pipeline
          </CardTitle>
          <CardDescription>
            Five stages, one direction — from raw HTML to the JSON your browser is rendering right now.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ol className="grid gap-4 md:grid-cols-5">
            <PipelineStage
              n={1}
              icon={Database}
              title="Scrape"
              detail="Glassdoor IT postings · 3 markets"
              metric="raw HTML"
              color="#475569"
            />
            <PipelineStage
              n={2}
              icon={Filter}
              title="Filter & dedup"
              detail="IT-only · 2024–25 · same-company dedupe"
              metric={formatNumber(meta.counts.rows_total ?? 44832)}
              color="#3c6ea8"
            />
            <PipelineStage
              n={3}
              icon={Brain}
              title="Hybrid extraction"
              detail="Dictionary ∪ GPT-4o classifier"
              metric="3 tiers"
              color="#7b3cb8"
            />
            <PipelineStage
              n={4}
              icon={Calculator}
              title="Stata analysis"
              detail="Logit · Multinomial · OLS A/B/C"
              metric="9 + 3 models"
              color="#b84a4a"
            />
            <PipelineStage
              n={5}
              icon={Layers}
              title="This explorer"
              detail="Static Next.js · row-level JSON"
              metric={`${meta.counts.clusters} clusters`}
              color="#3c8a6a"
              last
            />
          </ol>
        </CardContent>
      </Card>

      {/* RESEARCH QUESTION */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Research question</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm leading-relaxed">
          <p className="text-base">
            <strong>How prevalent are AI skill requirements in IT job postings, and is there a measurable salary
            premium for them?</strong>
          </p>
          <p className="text-muted-foreground">
            The thesis compares three labour markets with very different AI adoption speeds — the United States, Germany
            and India. The argument proceeds in three steps:
          </p>
          <ol className="ml-1 mt-1 space-y-2 text-sm">
            <li className="flex gap-3">
              <span className="mt-0.5 inline-flex size-5 shrink-0 items-center justify-center rounded-full bg-primary/15 text-[11px] font-semibold text-primary">
                1
              </span>
              <span>Classify each posting into one of three AI tiers based on the description.</span>
            </li>
            <li className="flex gap-3">
              <span className="mt-0.5 inline-flex size-5 shrink-0 items-center justify-center rounded-full bg-primary/15 text-[11px] font-semibold text-primary">
                2
              </span>
              <span>Model the probability of each tier as a function of skill clusters and structural controls.</span>
            </li>
            <li className="flex gap-3">
              <span className="mt-0.5 inline-flex size-5 shrink-0 items-center justify-center rounded-full bg-primary/15 text-[11px] font-semibold text-primary">
                3
              </span>
              <span>Regress log-salary on tier membership to isolate the wage premium net of composition.</span>
            </li>
          </ol>
        </CardContent>
      </Card>

      {/* AI tier definitions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">AI tier definitions</CardTitle>
          <CardDescription>The classification scheme the LLM and the dictionary both target.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-3">
          {(["None", "AI Integration", "Applied/Core AI"] as const).map((tier) => {
            const desc =
              tier === "None"
                ? "No AI-related requirements in the posting. The largest group across all three countries."
                : tier === "AI Integration"
                ? "Job integrates pre-built AI into existing processes — calls to LLM APIs, AI-powered features in a product, prompt engineering."
                : "Job develops AI itself — machine learning modelling, data/ML infrastructure, applied research. The high-skill end.";
            return (
              <div key={tier} className="rounded-lg border bg-card p-4">
                <div className="flex items-center gap-2">
                  <span
                    className="inline-block size-3 rounded-sm"
                    style={{ background: TIER_COLORS[tier] }}
                  />
                  <span className="font-semibold text-sm" style={{ color: TIER_COLORS[tier] }}>
                    {tier}
                  </span>
                </div>
                <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{desc}</p>
              </div>
            );
          })}
        </CardContent>
      </Card>

      {/* PER COUNTRY */}
      <section className="grid gap-3 md:grid-cols-3">
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
            <CardContent className="space-y-1 pt-0 text-xs">
              <div
                className="flex items-baseline justify-between"
                title={`Tier composition for ${c.country_label}`}
              >
                <span className="text-muted-foreground">None</span>
                <span className="tabular-nums">{formatPct(c.tier_none_pct)}</span>
              </div>
              <div className="flex items-baseline justify-between">
                <span style={{ color: TIER_COLORS["AI Integration"] }}>AI Integration</span>
                <span className="tabular-nums">{formatPct(c.tier_integration_pct)}</span>
              </div>
              <div className="flex items-baseline justify-between">
                <span style={{ color: TIER_COLORS["Applied/Core AI"] }}>Applied/Core AI</span>
                <span className="tabular-nums">{formatPct(c.tier_applied_pct)}</span>
              </div>
              <div
                className="mt-2 flex h-2 overflow-hidden rounded"
                role="presentation"
                aria-hidden="true"
              >
                <div
                  className="h-full"
                  style={{ width: `${c.tier_none_pct}%`, background: TIER_COLORS["None"] }}
                />
                <div
                  className="h-full"
                  style={{ width: `${c.tier_integration_pct}%`, background: TIER_COLORS["AI Integration"] }}
                />
                <div
                  className="h-full"
                  style={{ width: `${c.tier_applied_pct}%`, background: TIER_COLORS["Applied/Core AI"] }}
                />
              </div>
            </CardContent>
          </Card>
        ))}
      </section>

      {/* DETAIL ON THE EXTRACTION */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Brain className="size-4 text-primary" />
            How the AI tier is decided
          </CardTitle>
          <CardDescription>The hybrid extraction pipeline, in detail</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <div className="rounded-lg border-l-2 border-primary/60 bg-muted/30 p-3">
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Path A · Deterministic dictionary
            </div>
            <p className="mt-1 leading-relaxed">
              A curated dictionary of <strong>≈1 400</strong> hard-skill terms is matched against the posting description.
              Each match contributes to one of <strong>{meta.counts.clusters}</strong> skill clusters (Generative AI,
              Data Science / ML, Cloud Computing, …). If any AI-related cluster fires, the posting is at least flagged
              as <em>AI-touching</em>.
            </p>
          </div>
          <div className="rounded-lg border-l-2 border-violet-500/60 bg-muted/30 p-3">
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Path B · LLM classifier
            </div>
            <p className="mt-1 leading-relaxed">
              The full description is passed to <span className="font-mono text-xs">gpt-4o-mini</span> with a structured
              prompt that asks for one of three tier labels. The prompt includes positive and negative examples for each
              tier and asks the model to cite the spans of the description that drove the decision.
            </p>
          </div>
          <div className="rounded-lg border-l-2 border-emerald-500/60 bg-muted/30 p-3">
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Merge rule
            </div>
            <p className="mt-1 leading-relaxed">
              Final tier =&nbsp;
              <span className="rounded bg-card px-1.5 py-0.5 font-mono text-[11px]">
                max(LLM tier, dictionary tier)
              </span>
              . The dictionary catches confident specialists the LLM occasionally over-classifies as Integration; the LLM
              catches contextual signal the dictionary misses (e.g.&nbsp;&ldquo;leverage foundation models&rdquo; without an
              explicit term match). Agreement between paths is &gt;94% on a hand-checked stratified sample of 500
              postings.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* MODELS */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Calculator className="size-4 text-primary" />
            Econometric specifications
          </CardTitle>
          <CardDescription>What each Stata model is asking</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <ModelRow
            name="Binary logit"
            count="3 (one per country)"
            spec="Pr(any AI required) ~ job family + seniority + edu + state/industry + size + cluster_*"
            note="AMEs are reported on /clusters and decomposed in finding 3 of /insights."
          />
          <ModelRow
            name="Multinomial logit (US only)"
            count="1"
            spec="Pr(tier = 0/1/2) ~ same controls as above"
            note="Deepens the binary into three categories. Reference category = None."
          />
          <ModelRow
            name="OLS log-salary, Models A/B/C"
            count="3 progressive specifications"
            spec="log(salary_mid) ~ tier + structural controls + country FE → + skill clusters"
            note="Model C is the headline reported across the thesis. /premium shows the shrinkage from A → B → C."
          />
          <ModelRow
            name="OLS by country"
            count="3 (one per country)"
            spec="Same as Model C, restricted to one country at a time"
            note="Removes country FE, keeps within-country structure. Power in DE is the limiting factor."
          />
        </CardContent>
      </Card>

      {/* CAVEATS */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Honest caveats</CardTitle>
          <CardDescription>Where the dataset can mislead, and how the thesis handles it</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-muted-foreground">
          <Caveat
            title="Salary disclosure varies"
            body="≈73% of US postings carry a salary range. Germany drops to ~8%, India ~35%. That limits the statistical power of the DE premium estimate in particular — finding 6 of /insights makes this point explicitly."
          />
          <Caveat
            title="Sample ≠ population"
            body="Glassdoor over-represents listed mid-to-large employers and English-language postings. Generalising to 'the IT labour market' is too strong; 'IT postings on the platform' is the right scope."
          />
          <Caveat
            title="LLM classification isn't perfect"
            body="A random sample of 500 postings was hand-checked against the model's tier assignment; agreement was ≈94%. Disagreements cluster on borderline Integration / Applied/Core cases (consultants who deploy ML models without building them)."
          />
          <Caveat
            title="Wilson CIs assume independence"
            body="Row-level CIs on /analyze, /compare and /distributions treat postings as i.i.d. Intra-company correlation isn't modelled there. Coefficient SEs reported on /clusters and /premium are HC1 (Huber–White) robust, as specified in the Stata do-file."
          />
        </CardContent>
      </Card>

      {/* DEFERRED */}
      <Card className="border-amber-500/30">
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Sparkles className="size-4 text-amber-700" />
            Coming once Stata is re-run
          </CardTitle>
          <CardDescription>
            Features parked on a future export pass — the Stata code is ready, the run hasn&rsquo;t happened.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-muted-foreground">
          <Deferred
            title="Salary predictor (counterfactual calculator)"
            body="Given a posting spec (country × tier × family × seniority × edu × cluster set), predict log-salary + 95% prediction interval + per-variable marginal effects. Needs the full Model C coefficient vector — currently only the 2 tier coefficients are exported."
          />
          <Deferred
            title="Per-regression fit stats in tooltips"
            body="N, pseudo-R², log-likelihood for every regression — surfaced inline. Needs an esttab pass with stats(N r2_p ll)."
          />
          <Deferred
            title="Bootstrapped CIs on AMEs"
            body="Alternative to Wald CIs on /clusters. Needs margins, dydx(cluster_*) post vce(bootstrap, reps(500))."
          />
          <p className="pt-1">
            Workflow: append the exports to <code className="rounded bg-muted px-1 font-mono text-[11px]">analysis/stata/main.do</code>,
            re-run once, then re-run <code className="rounded bg-muted px-1 font-mono text-[11px]">pnpm run build-data</code>.
            No app-code changes required beyond the stubbed pages picking up the new JSONs.
          </p>
        </CardContent>
      </Card>

      {/* META */}
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
          <Row k="Row-level postings" v={formatNumber(meta.counts.rows_total ?? 0)} />
        </CardContent>
      </Card>

      {/* QUICK LINKS */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Quick links</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-2 md:grid-cols-2">
          <AboutLink href="/insights" label="The story · seven findings" />
          <AboutLink href="/analyze" label="Slice the row-level dataset" />
          <AboutLink href="/clusters" label="Cluster heatmap" />
          <AboutLink href="/skills" label="Hard-skill leaderboard" />
          <AboutLink href="/network" label="Skill co-occurrence network" />
          <AboutLink href="/distributions" label="Salary &amp; experience distributions" />
          <AboutLink href="/geography" label="Geographic breakdown" />
          <AboutLink href="/premium" label="Salary premium · OLS A/B/C" />
        </CardContent>
      </Card>
    </div>
  );
}

function PipelineStage({
  n,
  icon: Icon,
  title,
  detail,
  metric,
  color,
  last,
}: {
  n: number;
  icon: typeof Database;
  title: string;
  detail: string;
  metric: string;
  color: string;
  last?: boolean;
}) {
  return (
    <li className="relative flex flex-col gap-2 rounded-xl border bg-card p-3 md:p-4">
      <div className="flex items-center justify-between">
        <div
          className="flex size-8 items-center justify-center rounded-md"
          style={{ background: `${color}1a`, color }}
        >
          <Icon className="size-4" />
        </div>
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          {String(n).padStart(2, "0")}
        </span>
      </div>
      <div>
        <div className="text-sm font-semibold">{title}</div>
        <div className="text-[11px] leading-snug text-muted-foreground">{detail}</div>
      </div>
      <div
        className="text-base font-semibold tabular-nums tracking-tight"
        style={{ color }}
      >
        {metric}
      </div>
      {!last ? (
        <ArrowRight className="absolute -right-3 top-1/2 hidden size-5 -translate-y-1/2 text-muted-foreground/50 md:block" />
      ) : null}
    </li>
  );
}

function ModelRow({
  name,
  count,
  spec,
  note,
}: {
  name: string;
  count: string;
  spec: string;
  note: string;
}) {
  return (
    <div className="rounded-lg border bg-card p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-sm font-semibold">{name}</div>
        <Badge variant="outline" className="text-[10px]">
          {count}
        </Badge>
      </div>
      <code className="mt-1 block rounded bg-muted/60 px-2 py-1 font-mono text-[11px] leading-relaxed">
        {spec}
      </code>
      <div className="mt-1 text-[11px] text-muted-foreground">{note}</div>
    </div>
  );
}

function Caveat({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-md border-l-2 border-amber-500/60 bg-amber-500/5 px-3 py-2">
      <div className="text-sm font-semibold text-foreground">{title}</div>
      <p className="mt-1 leading-relaxed">{body}</p>
    </div>
  );
}

function Deferred({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-md border bg-amber-500/5 px-3 py-2">
      <div className="font-semibold text-foreground">{title}</div>
      <p className="mt-1">{body}</p>
    </div>
  );
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-dashed py-1.5">
      <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{k}</span>
      <span className="tabular-nums">{v}</span>
    </div>
  );
}

function AboutLink({ href, label }: { href: string; label: string }) {
  return (
    <Link
      href={href}
      className="group flex items-center justify-between rounded-md border px-3 py-2 text-sm transition-colors hover:border-primary/40 hover:bg-primary/5"
    >
      <span>{label}</span>
      <ArrowUpRight className="size-4 text-muted-foreground transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
    </Link>
  );
}
