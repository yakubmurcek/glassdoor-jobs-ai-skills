"use client";

import { Suspense, useDeferredValue, useMemo, useState } from "react";
import Link from "next/link";
import { ArrowUpDown, Loader2, Search, Sparkles, X } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useRows, type CompactRow } from "@/lib/data/rows-store";
import {
  COUNTRIES,
  COUNTRY_COLORS,
  COUNTRY_FLAGS,
  COUNTRY_LABELS,
  TIER_COLORS,
  type Country,
} from "@/lib/constants";
import { formatNumber, formatPct } from "@/lib/utils";
import { cn } from "@/lib/utils";

type SortBy = "freq" | "ai_share" | "ai_lift" | "applied_share" | "alpha";

interface SkillStat {
  skill: string;
  freq: number;
  with_tier: number;
  ai_count: number;
  applied_count: number;
  ai_share: number;
  applied_share: number;
  ai_lift: number; // ai_share / baseline
}

const STOPWORDS = new Set(["", "n/a", "none", "various"]);

function buildSkillStats(rows: readonly CompactRow[], country: Country | "ALL"): SkillStat[] {
  // Compute baseline AI share over the same subset
  let baselineN = 0;
  let baselineAi = 0;
  for (const r of rows) {
    if (country !== "ALL" && r.co !== country) continue;
    if (r.t === null) continue;
    baselineN += 1;
    if (r.t >= 1) baselineAi += 1;
  }
  const baseline = baselineN === 0 ? 0 : baselineAi / baselineN;

  const map = new Map<string, SkillStat>();
  for (const r of rows) {
    if (country !== "ALL" && r.co !== country) continue;
    if (!r.hs) continue;
    const tokens = r.hs.split(/[,;]/).map((s) => s.trim().toLowerCase()).filter(Boolean);
    const seen = new Set<string>();
    for (const t of tokens) {
      if (seen.has(t)) continue;
      if (STOPWORDS.has(t)) continue;
      if (t.length > 60) continue;
      seen.add(t);
      let s = map.get(t);
      if (!s) {
        s = {
          skill: t,
          freq: 0,
          with_tier: 0,
          ai_count: 0,
          applied_count: 0,
          ai_share: 0,
          applied_share: 0,
          ai_lift: 0,
        };
        map.set(t, s);
      }
      s.freq += 1;
      if (r.t !== null) {
        s.with_tier += 1;
        if (r.t >= 1) s.ai_count += 1;
        if (r.t === 2) s.applied_count += 1;
      }
    }
  }
  const out: SkillStat[] = [];
  for (const s of map.values()) {
    if (s.with_tier === 0) continue;
    s.ai_share = s.ai_count / s.with_tier;
    s.applied_share = s.applied_count / s.with_tier;
    s.ai_lift = baseline > 0 ? s.ai_share / baseline : 0;
    out.push(s);
  }
  return out;
}

function titleCase(s: string): string {
  // Words that should be uppercased (treated case-insensitively)
  const upper = new Set([
    "aws", "gcp", "sql", "ci/cd", "ml", "nlp", "ai", "api", "ux", "ui", "ios",
    "tcp", "udp", "grpc", "rest", "json", "xml", "yaml", "html", "css", "dns",
    "vpn", "saas", "iaas", "paas", "sso", "oauth", "ssl", "tls", "vcs",
    "nosql", "rdbms", "etl", "elt", "jvm", "vm", "ide", "qa", "k8s", "llm",
    "rag", "gpu", "cpu", "ml/ai", "mlops",
  ]);
  // Words/suffixes that should keep their existing casing exactly
  const keep = new Set(["js", "ts", "py", "io", "ts/js", "next.js", "node.js"]);
  if (upper.has(s)) return s.toUpperCase();
  if (keep.has(s)) return s;
  return s
    .split(/(\s|\.|-|\/)/)
    .map((w) => {
      const lw = w.toLowerCase();
      if (upper.has(lw)) return w.toUpperCase();
      if (keep.has(lw)) return lw;
      if (!w || /^\W$/.test(w)) return w;
      return w[0].toUpperCase() + w.slice(1);
    })
    .join("");
}

export default function SkillsPage() {
  return (
    <Suspense fallback={<LoadingShell />}>
      <SkillsContent />
    </Suspense>
  );
}

function LoadingShell() {
  return (
    <div className="flex min-h-[400px] items-center justify-center gap-2 text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin" />
      Tallying hard skills…
    </div>
  );
}

function SkillsContent() {
  const { rows, loading, error } = useRows();
  const [country, setCountry] = useState<Country | "ALL">("ALL");
  const [sortBy, setSortBy] = useState<SortBy>("freq");
  const [search, setSearch] = useState("");
  const [topN, setTopN] = useState<number>(40);
  const [minN, setMinN] = useState<number>(50);

  const deferredSearch = useDeferredValue(search);

  const skillStats = useMemo(() => {
    if (!rows) return null;
    return buildSkillStats(rows, country);
  }, [rows, country]);

  const filtered = useMemo(() => {
    if (!skillStats) return [];
    const q = deferredSearch.trim().toLowerCase();
    let arr = skillStats.filter((s) => s.with_tier >= minN);
    if (q) arr = arr.filter((s) => s.skill.includes(q));
    arr = arr.slice();
    if (sortBy === "freq") arr.sort((a, b) => b.freq - a.freq);
    else if (sortBy === "ai_share") arr.sort((a, b) => b.ai_share - a.ai_share);
    else if (sortBy === "ai_lift") arr.sort((a, b) => b.ai_lift - a.ai_lift);
    else if (sortBy === "applied_share") arr.sort((a, b) => b.applied_share - a.applied_share);
    else if (sortBy === "alpha") arr.sort((a, b) => a.skill.localeCompare(b.skill));
    return arr.slice(0, topN);
  }, [skillStats, deferredSearch, sortBy, topN, minN]);

  const total = skillStats?.length ?? 0;
  const filteredTotal = useMemo(() => {
    if (!skillStats) return 0;
    const q = deferredSearch.trim().toLowerCase();
    return skillStats.filter((s) => s.with_tier >= minN && (!q || s.skill.includes(q))).length;
  }, [skillStats, deferredSearch, minN]);

  const baseline = useMemo(() => {
    if (!rows) return 0;
    let n = 0;
    let ai = 0;
    for (const r of rows) {
      if (country !== "ALL" && r.co !== country) continue;
      if (r.t === null) continue;
      n += 1;
      if (r.t >= 1) ai += 1;
    }
    return n === 0 ? 0 : ai / n;
  }, [rows, country]);

  if (loading || !rows || !skillStats) return <LoadingShell />;
  if (error) {
    return <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">Failed: {error}</div>;
  }

  const accent = country === "ALL" ? "#3c6ea8" : COUNTRY_COLORS[country];
  const maxFreq = filtered.length > 0 ? filtered[0].freq : 1;

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <PageHeader
        eyebrow="Hard skills"
        title="Which skills employers actually ask for"
        description={`${formatNumber(total)} distinct hard-skill terms extracted from posting descriptions, computed live in your browser. Filter, sort, search — every column reflects the row-level dataset.`}
      />

      <Card>
        <CardHeader className="flex flex-col gap-4">
          <div className="flex flex-wrap items-end gap-3">
            <ControlGroup label="Country">
              <ToggleGroup
                type="single"
                value={country}
                onValueChange={(v: string) => v && setCountry(v as Country | "ALL")}
              >
                <ToggleGroupItem value="ALL">All</ToggleGroupItem>
                {COUNTRIES.map((c) => (
                  <ToggleGroupItem key={c} value={c}>
                    {COUNTRY_FLAGS[c]} {c}
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
            </ControlGroup>
            <ControlGroup label="Sort by">
              <ToggleGroup
                type="single"
                value={sortBy}
                onValueChange={(v: string) => v && setSortBy(v as SortBy)}
              >
                <ToggleGroupItem value="freq">Frequency</ToggleGroupItem>
                <ToggleGroupItem value="ai_share">AI share</ToggleGroupItem>
                <ToggleGroupItem value="ai_lift">AI lift ×</ToggleGroupItem>
                <ToggleGroupItem value="applied_share">Applied/Core</ToggleGroupItem>
                <ToggleGroupItem value="alpha">A–Z</ToggleGroupItem>
              </ToggleGroup>
            </ControlGroup>
            <ControlGroup label="Min n">
              <ToggleGroup
                type="single"
                value={String(minN)}
                onValueChange={(v: string) => v && setMinN(Number(v))}
              >
                <ToggleGroupItem value="20">20</ToggleGroupItem>
                <ToggleGroupItem value="50">50</ToggleGroupItem>
                <ToggleGroupItem value="200">200</ToggleGroupItem>
                <ToggleGroupItem value="1000">1k</ToggleGroupItem>
              </ToggleGroup>
            </ControlGroup>
            <ControlGroup label="Show">
              <ToggleGroup
                type="single"
                value={String(topN)}
                onValueChange={(v: string) => v && setTopN(Number(v))}
              >
                <ToggleGroupItem value="20">20</ToggleGroupItem>
                <ToggleGroupItem value="40">40</ToggleGroupItem>
                <ToggleGroupItem value="100">100</ToggleGroupItem>
                <ToggleGroupItem value="500">500</ToggleGroupItem>
              </ToggleGroup>
            </ControlGroup>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <div className="relative flex-1 min-w-64">
              <Search className="absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Filter skills (e.g. python, kubernetes, llm, rag, transformer)…"
                className="pl-8"
              />
              {search ? (
                <button
                  type="button"
                  onClick={() => setSearch("")}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  aria-label="Clear"
                >
                  <X className="size-4" />
                </button>
              ) : null}
            </div>
            <div className="text-xs text-muted-foreground">
              Showing <strong>{filtered.length}</strong> of <strong>{filteredTotal}</strong>{" "}
              {filteredTotal === 1 ? "skill" : "skills"} that pass the filters · baseline AI share for this slice ={" "}
              <strong>{formatPct(baseline * 100, 1)}</strong>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {filtered.length === 0 ? (
            <div className="rounded-lg border bg-muted/30 p-8 text-center text-sm text-muted-foreground">
              No skills match those filters.
            </div>
          ) : (
            <div className="overflow-hidden rounded-lg border">
              <table className="w-full text-xs">
                <thead className="bg-muted/40">
                  <tr className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    <th className="text-left font-semibold px-3 py-2 w-8">#</th>
                    <th className="text-left font-semibold px-3 py-2">Skill</th>
                    <th className="text-left font-semibold px-3 py-2">Frequency (n)</th>
                    <th className="text-left font-semibold px-3 py-2">AI tier mix</th>
                    <th className="text-right font-semibold px-3 py-2">AI share</th>
                    <th className="text-right font-semibold px-3 py-2">Lift ×</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((s, i) => {
                    const wPct = (s.freq / maxFreq) * 100;
                    const intPct = (s.ai_count - s.applied_count) / Math.max(1, s.with_tier);
                    const appPct = s.applied_count / Math.max(1, s.with_tier);
                    const nonePct = 1 - intPct - appPct;
                    return (
                      <tr key={s.skill} className="border-t border-border/40 hover:bg-muted/30">
                        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{i + 1}</td>
                        <td className="px-3 py-2 font-medium">{titleCase(s.skill)}</td>
                        <td className="px-3 py-2">
                          <div className="flex items-center gap-2">
                            <div className="relative h-3 w-24 rounded bg-muted/60">
                              <div
                                className="absolute inset-y-0 left-0 rounded"
                                style={{ width: `${wPct}%`, background: accent }}
                              />
                            </div>
                            <span className="tabular-nums text-[11px] text-muted-foreground">
                              {formatNumber(s.freq)}
                            </span>
                          </div>
                        </td>
                        <td className="px-3 py-2">
                          <div className="flex h-3 w-32 overflow-hidden rounded">
                            <div
                              className="h-full"
                              style={{ width: `${nonePct * 100}%`, background: TIER_COLORS["None"] }}
                              title={`None ${formatPct(nonePct * 100, 1)}`}
                            />
                            <div
                              className="h-full"
                              style={{ width: `${intPct * 100}%`, background: TIER_COLORS["AI Integration"] }}
                              title={`Integration ${formatPct(intPct * 100, 1)}`}
                            />
                            <div
                              className="h-full"
                              style={{ width: `${appPct * 100}%`, background: TIER_COLORS["Applied/Core AI"] }}
                              title={`Applied/Core ${formatPct(appPct * 100, 1)}`}
                            />
                          </div>
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {formatPct(s.ai_share * 100, 1)}
                        </td>
                        <td
                          className={cn(
                            "px-3 py-2 text-right tabular-nums font-semibold",
                            s.ai_lift >= 2
                              ? "text-violet-700"
                              : s.ai_lift >= 1.2
                              ? "text-emerald-700"
                              : s.ai_lift < 0.8
                              ? "text-amber-700"
                              : "text-muted-foreground",
                          )}
                        >
                          ×{s.ai_lift.toFixed(2)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          <div className="mt-4 flex flex-wrap items-center gap-4 text-[11px] text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span
                className="inline-block h-2.5 w-2.5 rounded-sm"
                style={{ background: TIER_COLORS["None"] }}
              />
              None
            </span>
            <span className="flex items-center gap-1.5">
              <span
                className="inline-block h-2.5 w-2.5 rounded-sm"
                style={{ background: TIER_COLORS["AI Integration"] }}
              />
              AI Integration
            </span>
            <span className="flex items-center gap-1.5">
              <span
                className="inline-block h-2.5 w-2.5 rounded-sm"
                style={{ background: TIER_COLORS["Applied/Core AI"] }}
              />
              Applied/Core AI
            </span>
            <span className="ml-auto">
              <strong>Lift ×</strong> = P(AI | skill) ÷ P(AI). Above 1 = the skill predicts AI hiring.
            </span>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Sparkles className="size-4 text-primary" />
            Pre-built lenses
          </CardTitle>
          <CardDescription>One-click presets that map to common research questions</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          <PresetButton
            title="Top AI predictors"
            body="Skills with the highest lift over the country baseline. The strongest signals that a posting is AI-related."
            onClick={() => {
              setSortBy("ai_lift");
              setMinN(200);
              setTopN(20);
              setSearch("");
            }}
          />
          <PresetButton
            title="LLM / generative-AI vocabulary"
            body="Filter to the new vocabulary that emerged post-ChatGPT — RAG, transformers, embeddings, vector DBs, prompt engineering."
            onClick={() => {
              setSortBy("freq");
              setMinN(20);
              setTopN(40);
              setSearch("rag");
            }}
          />
          <PresetButton
            title="Volume-only top 50"
            body="The bread-and-butter skills — Python, SQL, AWS, Java… without any AI weighting. The IT-jobs commodity layer."
            onClick={() => {
              setSortBy("freq");
              setMinN(1000);
              setTopN(50);
              setSearch("");
            }}
          />
        </CardContent>
      </Card>

      <p className="text-[11px] text-muted-foreground">
        <strong>Methodology note.</strong> Hard skills are extracted by the deterministic dictionary pass (≈1 400 known IT/CS terms) per posting.
        Each row in the table is one normalised skill term; the same posting can contribute to many rows. AI tier mix is computed conditional on
        the posting having a tier classification (some rows lack one — those are excluded from the &ldquo;with tier&rdquo; denominator). See{" "}
        <Link href="/about" className="underline">/about</Link> for the full pipeline.
      </p>
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

function PresetButton({ title, body, onClick }: { title: string; body: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="group flex h-full flex-col items-start gap-1.5 rounded-lg border bg-card p-3 text-left transition-all hover:border-primary/40 hover:shadow-sm"
    >
      <div className="text-sm font-semibold">{title}</div>
      <div className="text-xs leading-relaxed text-muted-foreground">{body}</div>
      <div className="mt-1 inline-flex items-center gap-1 text-[10px] font-semibold text-primary opacity-0 transition-opacity group-hover:opacity-100">
        <ArrowUpDown className="size-3" /> Apply preset
      </div>
    </button>
  );
}
