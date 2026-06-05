"use client";

import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, ArrowRight, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { getKpi, getMetadata, getG1, getClusters, getJobFamilies } from "@/lib/data/loaders";
import {
  COUNTRIES,
  COUNTRY_COLORS,
  COUNTRY_FLAGS,
  COUNTRY_LABELS,
  TIER_COLORS,
  type Country,
} from "@/lib/constants";
import { formatCoef, formatNumber, formatPct } from "@/lib/utils";
import { cn } from "@/lib/utils";

interface Slide {
  title: string;
  content: React.ReactNode;
}

export default function PresentPage() {
  const kpi = getKpi();
  const meta = getMetadata();
  const g1 = getG1();
  const clusters = getClusters();
  const families = getJobFamilies();

  const us = kpi.countries.find((c) => c.country === "US")!;
  const de = kpi.countries.find((c) => c.country === "DE")!;
  const ind = kpi.countries.find((c) => c.country === "IN")!;

  const [slide, setSlide] = useState(0);

  const slides: Slide[] = [
    {
      title: "AI skill requirements in IT job postings",
      content: (
        <div className="flex flex-col items-center gap-6 text-center">
          <div className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
            Master&rsquo;s thesis · 2026
          </div>
          <div className="text-5xl font-semibold leading-tight tracking-tight md:text-7xl">
            How AI is reshaping
            <br />
            <span className="text-primary">IT hiring</span>{" "}
            across three economies
          </div>
          <div className="max-w-2xl text-base text-muted-foreground md:text-lg">
            {formatNumber(meta.counts.rows_total ?? 0)} Glassdoor postings · United States · Germany · India
          </div>
          <div className="mt-4 flex flex-wrap items-center justify-center gap-2 text-xs text-muted-foreground">
            <kbd className="rounded border bg-card px-2 py-1 font-mono text-[10px]">→</kbd>
            <span>or</span>
            <kbd className="rounded border bg-card px-2 py-1 font-mono text-[10px]">space</kbd>
            <span>to advance</span>
          </div>
        </div>
      ),
    },
    {
      title: "The headline gap",
      content: (
        <div className="flex flex-col gap-10">
          <div className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
            Finding 1 of 7
          </div>
          <div className="grid gap-8 md:grid-cols-3">
            {COUNTRIES.map((c) => {
              const k = kpi.countries.find((kc) => kc.country === c)!;
              return (
                <div key={c} className="flex flex-col gap-3">
                  <div className="flex items-center gap-2 text-xl">
                    <span>{COUNTRY_FLAGS[c]}</span>
                    <span className="font-semibold">{COUNTRY_LABELS[c]}</span>
                  </div>
                  <div
                    className="text-7xl font-semibold tabular-nums leading-none md:text-8xl"
                    style={{ color: COUNTRY_COLORS[c] }}
                  >
                    {formatPct(k.ai_share, 1)}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    of {formatNumber(k.total_jobs)} postings ask for AI
                  </div>
                </div>
              );
            })}
          </div>
          <div className="text-xl text-muted-foreground md:text-2xl">
            US firms ask for AI{" "}
            <strong style={{ color: COUNTRY_COLORS.US }}>
              {formatCoef(us.ai_share / ind.ai_share, 1)}×
            </strong>{" "}
            more often than Indian firms.
          </div>
        </div>
      ),
    },
    {
      title: "Two flavours of AI",
      content: (
        <div className="flex flex-col gap-10">
          <div className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
            Finding 2 of 7
          </div>
          <div className="text-3xl font-semibold leading-tight md:text-5xl">
            <span style={{ color: TIER_COLORS["AI Integration"] }}>Integration</span>{" "}
            ≠{" "}
            <span style={{ color: TIER_COLORS["Applied/Core AI"] }}>Applied AI</span>.
          </div>
          <div className="space-y-5">
            {COUNTRIES.map((c) => {
              const k = kpi.countries.find((kc) => kc.country === c)!;
              const total = k.tier_integration_pct + k.tier_applied_pct;
              const intPct = (k.tier_integration_pct / Math.max(0.01, total)) * 100;
              const appPct = (k.tier_applied_pct / Math.max(0.01, total)) * 100;
              return (
                <div key={c}>
                  <div className="mb-1 flex items-center justify-between text-sm">
                    <span className="font-medium">
                      {COUNTRY_FLAGS[c]} {COUNTRY_LABELS[c]}
                    </span>
                    <span className="text-xs text-muted-foreground tabular-nums">
                      Integration {formatPct(k.tier_integration_pct, 1)} · Applied/Core {formatPct(k.tier_applied_pct, 1)}
                    </span>
                  </div>
                  <div className="flex h-8 overflow-hidden rounded">
                    <div
                      className="flex items-center justify-center text-[10px] font-bold text-white"
                      style={{ width: `${intPct}%`, background: TIER_COLORS["AI Integration"] }}
                    >
                      {intPct > 12 ? `${intPct.toFixed(0)}%` : ""}
                    </div>
                    <div
                      className="flex items-center justify-center text-[10px] font-bold text-white"
                      style={{ width: `${appPct}%`, background: TIER_COLORS["Applied/Core AI"] }}
                    >
                      {appPct > 12 ? `${appPct.toFixed(0)}%` : ""}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="text-lg text-muted-foreground md:text-xl">
            Germany is the only country where Applied/Core dominates Integration.
          </div>
        </div>
      ),
    },
    {
      title: "The wage premium",
      content: (
        <div className="flex flex-col gap-10">
          <div className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
            Finding 3 of 7
          </div>
          <div className="text-2xl font-semibold leading-tight md:text-4xl">
            US Applied/Core AI roles pay
          </div>
          <div className="text-7xl font-semibold tabular-nums leading-none md:text-9xl" style={{ color: COUNTRY_COLORS.US }}>
            +{formatCoef(us.premium_applied_pct ?? 0, 1)}%
          </div>
          <div className="text-lg text-muted-foreground md:text-xl">
            after full controls (job family, seniority, education, state, industry, firm size, all 21 skill clusters).
            Significant at <strong>{us.premium_applied_sig}</strong> level.
          </div>
          <div className="grid gap-3 text-sm md:grid-cols-3">
            {COUNTRIES.map((c) => {
              const k = kpi.countries.find((kc) => kc.country === c)!;
              return (
                <div
                  key={c}
                  className="rounded-xl border bg-card p-4 shadow-sm"
                >
                  <div className="text-xs uppercase tracking-wider text-muted-foreground">
                    {COUNTRY_FLAGS[c]} {COUNTRY_LABELS[c]}
                  </div>
                  <div className="mt-1 text-3xl font-semibold tabular-nums" style={{ color: COUNTRY_COLORS[c] }}>
                    +{formatCoef(k.premium_applied_pct ?? 0, 1)}%
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Applied/Core · {k.premium_applied_sig}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ),
    },
    {
      title: "Generative AI is the headline cluster",
      content: (
        <div className="flex flex-col gap-10">
          <div className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
            Finding 4 of 7
          </div>
          <div className="text-3xl font-semibold md:text-5xl">
            Generative AI is the <span className="text-primary">single most-demanded</span> skill cluster.
          </div>
          <div className="space-y-2">
            {clusters.slice(0, 6).map((c, i) => {
              const max = clusters[0].pct;
              const w = (c.pct / max) * 100;
              return (
                <div key={c.key} className="grid grid-cols-[120px_1fr_60px] items-center gap-3">
                  <span className="text-sm font-medium">{c.label}</span>
                  <div className="relative h-6 rounded-sm bg-muted/50">
                    <div
                      className="absolute inset-y-0 left-0 rounded-sm"
                      style={{
                        width: `${w}%`,
                        background: i === 0 ? "#7b3cb8" : "#3c6ea8",
                      }}
                    />
                  </div>
                  <span className="text-right text-sm tabular-nums">{formatPct(c.pct, 1)}</span>
                </div>
              );
            })}
          </div>
          <div className="text-base text-muted-foreground md:text-lg">
            Effectively didn&rsquo;t exist as a hiring signal pre-ChatGPT (Nov 2022). Now appears in the majority of US Applied/Core AI postings.
          </div>
        </div>
      ),
    },
    {
      title: "Germany — the structural anomaly",
      content: (
        <div className="flex flex-col gap-10">
          <div className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
            Finding 5 of 7
          </div>
          <div className="text-3xl font-semibold leading-tight md:text-5xl">
            Germany hires Applied/Core AI{" "}
            <span style={{ color: COUNTRY_COLORS.DE }}>more than the US</span>.
            <br />
            <span className="text-muted-foreground">The salary signal can&rsquo;t see it.</span>
          </div>
          <div className="grid gap-6 md:grid-cols-2">
            <div className="rounded-xl border bg-card p-6">
              <div className="text-xs uppercase tracking-wider text-muted-foreground">Applied/Core share</div>
              <div className="mt-2 flex items-baseline gap-3">
                <div className="text-5xl font-semibold tabular-nums" style={{ color: COUNTRY_COLORS.DE }}>
                  {formatPct(de.tier_applied_pct, 1)}
                </div>
                <div className="text-sm text-muted-foreground">vs US {formatPct(us.tier_applied_pct, 1)}</div>
              </div>
            </div>
            <div className="rounded-xl border bg-card p-6">
              <div className="text-xs uppercase tracking-wider text-muted-foreground">Salary disclosure</div>
              <div className="mt-2 flex items-baseline gap-3">
                <div className="text-5xl font-semibold tabular-nums" style={{ color: COUNTRY_COLORS.DE }}>
                  ~8%
                </div>
                <div className="text-sm text-muted-foreground">vs US ~35%</div>
              </div>
            </div>
          </div>
          <div className="text-base text-muted-foreground md:text-lg">
            Why posting volume + tier mix is the right outcome to track in Germany — not disclosed wages.
          </div>
        </div>
      ),
    },
    {
      title: "Implications",
      content: (
        <div className="flex flex-col gap-10">
          <div className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
            Finding 6 of 7
          </div>
          <div className="text-3xl font-semibold leading-tight md:text-5xl">
            Job postings are the <span className="text-primary">highest-frequency, lowest-latency</span> labour-market signal we have.
          </div>
          <ul className="space-y-3 text-base md:text-lg">
            <li className="border-l-2 border-primary/60 pl-4">
              <strong>Salary surveys</strong> arrive with a 1–3 year lag and shrink as transparency declines.
            </li>
            <li className="border-l-2 border-primary/60 pl-4">
              <strong>Job postings</strong> update daily, capture detailed skill stacks, and reveal what firms are buying — not what they paid yesterday.
            </li>
            <li className="border-l-2 border-primary/60 pl-4">
              <strong>For AI specifically</strong> — a fast, mostly bottom-up labour shock — postings are the best window we have.
            </li>
          </ul>
        </div>
      ),
    },
    {
      title: "The dataset",
      content: (
        <div className="flex flex-col gap-10">
          <div className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
            Finding 7 of 7 · Reproducibility
          </div>
          <div className="text-3xl font-semibold leading-tight md:text-5xl">
            All numbers are <span className="text-primary">re-derivable</span>.
          </div>
          <div className="grid grid-cols-2 gap-6 md:grid-cols-4">
            <Tile value={formatNumber(meta.counts.rows_total ?? 0)} label="Postings analysed" />
            <Tile value={meta.counts.clusters} label="Skill clusters" />
            <Tile value={meta.counts.job_families} label="Job families" />
            <Tile value="3" label="Countries" />
            <Tile value="9" label="Logit regressions" />
            <Tile value="3" label="OLS specifications" />
            <Tile value="100%" label="Reproducible" />
            <Tile value="MIT" label="Licence" />
          </div>
          <div className="mt-2 text-base text-muted-foreground md:text-lg">
            Workbench at <code className="rounded bg-muted px-2 py-0.5 font-mono text-sm">/analyze</code> · methodology at{" "}
            <code className="rounded bg-muted px-2 py-0.5 font-mono text-sm">/about</code>
          </div>
        </div>
      ),
    },
    {
      title: "Thank you",
      content: (
        <div className="flex flex-col items-center gap-6 text-center">
          <Sparkles className="size-10 text-primary" />
          <div className="text-5xl font-semibold leading-tight md:text-7xl">Thank you.</div>
          <div className="max-w-xl text-base text-muted-foreground md:text-lg">
            Questions? The companion explorer is at the URL on the title slide. Slice the data however you want — every
            number recomputes from the row-level dataset shipped to your browser.
          </div>
        </div>
      ),
    },
  ];

  const next = useCallback(() => setSlide((s) => Math.min(slides.length - 1, s + 1)), [slides.length]);
  const prev = useCallback(() => setSlide((s) => Math.max(0, s - 1)), []);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      // Skip when a dialog (e.g. command palette) is open or an input is focused
      if (document.querySelector('[role="dialog"][data-state="open"]')) return;
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
        return;
      }
      if (e.key === "ArrowRight" || e.key === " " || e.key === "PageDown") {
        e.preventDefault();
        next();
      } else if (e.key === "ArrowLeft" || e.key === "PageUp") {
        e.preventDefault();
        prev();
      } else if (/^[1-9]$/.test(e.key)) {
        const idx = Number(e.key) - 1;
        if (idx < slides.length) setSlide(idx);
      } else if (e.key === "Home") {
        setSlide(0);
      } else if (e.key === "End") {
        setSlide(slides.length - 1);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [next, prev, slides.length]);

  const current = slides[slide];

  return (
    <div className="-mx-4 -my-6 flex min-h-[calc(100svh-3.5rem)] flex-col bg-gradient-to-br from-background via-background to-primary/5 md:-mx-8 md:-my-8">
      <div className="flex items-center justify-between border-b bg-background/40 px-6 py-3 backdrop-blur md:px-10">
        <div className="flex items-center gap-3 text-xs">
          <Badge variant="outline">Defense mode</Badge>
          <span className="font-mono uppercase tracking-wider text-muted-foreground">
            {String(slide + 1).padStart(2, "0")} / {String(slides.length).padStart(2, "0")}
          </span>
          <span className="hidden text-muted-foreground md:inline">{current.title}</span>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="ghost" onClick={prev} disabled={slide === 0}>
            <ArrowLeft />
          </Button>
          <Button size="sm" variant="ghost" onClick={next} disabled={slide === slides.length - 1}>
            <ArrowRight />
          </Button>
        </div>
      </div>

      <div className="flex flex-1 items-center justify-center px-8 py-12 md:px-20 md:py-20">
        <div className="w-full max-w-5xl animate-in fade-in slide-in-from-bottom-2 duration-500">
          {current.content}
        </div>
      </div>

      <div className="border-t bg-background/40 px-6 py-3 backdrop-blur md:px-10">
        <div className="flex items-center justify-between text-[11px] text-muted-foreground">
          <div className="flex items-center gap-3">
            <span>Master&rsquo;s thesis · 2026 · AI skill requirements in IT</span>
          </div>
          <div className="flex items-center gap-1">
            {slides.map((_, i) => (
              <button
                key={i}
                onClick={() => setSlide(i)}
                aria-label={`Slide ${i + 1}`}
                className={cn(
                  "h-1.5 rounded-full transition-all",
                  i === slide ? "w-6 bg-primary" : "w-1.5 bg-muted-foreground/30 hover:bg-muted-foreground/60",
                )}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Tile({ value, label }: { value: React.ReactNode; label: string }) {
  return (
    <div className="rounded-xl border bg-card p-5 shadow-sm">
      <div className="text-3xl font-semibold tabular-nums tracking-tight md:text-4xl">{value}</div>
      <div className="mt-1 text-xs uppercase tracking-wider text-muted-foreground">{label}</div>
    </div>
  );
}
