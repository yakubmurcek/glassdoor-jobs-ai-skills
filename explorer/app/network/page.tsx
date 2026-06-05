"use client";

import { Suspense, useMemo, useState } from "react";
import Link from "next/link";
import { Loader2, Network as NetworkIcon } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  ToggleGroup,
  ToggleGroupItem,
} from "@/components/ui/toggle-group";
import {
  CLUSTER_KEYS,
  CLUSTER_LABELS,
  useRows,
  type CompactRow,
} from "@/lib/data/rows-store";
import {
  COUNTRIES,
  COUNTRY_COLORS,
  COUNTRY_FLAGS,
  COUNTRY_LABELS,
  type Country,
} from "@/lib/constants";
import { formatNumber, formatPct } from "@/lib/utils";

type Mode = "ALL" | Country;
type EdgeMetric = "lift" | "joint" | "conditional";

interface CoocStats {
  marginals: number[];
  joint: number[][]; // joint[i][j] = count of postings with both
  total: number;
}

function computeCooc(rows: readonly CompactRow[]): CoocStats {
  const n = CLUSTER_KEYS.length;
  const marginals = new Array<number>(n).fill(0);
  const joint: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  let total = 0;
  for (const r of rows) {
    if (!r.cl) continue;
    total += 1;
    const m = r.cl;
    for (let i = 0; i < n; i += 1) {
      if (!((m >> i) & 1)) continue;
      marginals[i] += 1;
      // upper triangle
      for (let j = i + 1; j < n; j += 1) {
        if ((m >> j) & 1) {
          joint[i][j] += 1;
          joint[j][i] += 1;
        }
      }
    }
  }
  return { marginals, joint, total };
}

interface Edge {
  i: number;
  j: number;
  weight: number; // 0..1 or beyond — depending on metric
  joint: number;
}

function computeEdges(stats: CoocStats, metric: EdgeMetric, threshold: number): Edge[] {
  const out: Edge[] = [];
  const n = CLUSTER_KEYS.length;
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const ij = stats.joint[i][j];
      if (ij < 30) continue; // need at least 30 co-occurrences
      const pi = stats.marginals[i] / stats.total;
      const pj = stats.marginals[j] / stats.total;
      const pij = ij / stats.total;
      let w = 0;
      if (metric === "lift") {
        w = pi > 0 && pj > 0 ? pij / (pi * pj) : 0;
      } else if (metric === "joint") {
        w = pij;
      } else {
        // conditional: P(j|i) (asymmetric — take max for undirected)
        const cij = pi > 0 ? pij / pi : 0;
        const cji = pj > 0 ? pij / pj : 0;
        w = Math.max(cij, cji);
      }
      if (w >= threshold) {
        out.push({ i, j, weight: w, joint: ij });
      }
    }
  }
  return out;
}

export default function NetworkPage() {
  return (
    <Suspense fallback={<LoadingShell />}>
      <NetworkContent />
    </Suspense>
  );
}

function LoadingShell() {
  return (
    <div className="flex min-h-[400px] items-center justify-center gap-2 text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin" />
      Loading row-level dataset…
    </div>
  );
}

function NetworkContent() {
  const { rows, loading, error } = useRows();
  const [mode, setMode] = useState<Mode>("ALL");
  const [metric, setMetric] = useState<EdgeMetric>("lift");
  const [aiOnly, setAiOnly] = useState<boolean>(true);
  const [hovered, setHovered] = useState<number | null>(null);
  const [pinned, setPinned] = useState<number | null>(null);

  const filtered = useMemo(() => {
    if (!rows) return null;
    return rows.filter((r) => {
      if (mode !== "ALL" && r.co !== mode) return false;
      if (aiOnly && (r.t === null || r.t === 0)) return false;
      return true;
    });
  }, [rows, mode, aiOnly]);

  const stats = useMemo(() => (filtered ? computeCooc(filtered) : null), [filtered]);

  const threshold = metric === "lift" ? 1.2 : metric === "conditional" ? 0.4 : 0.04;
  const allEdges = useMemo(
    () => (stats ? computeEdges(stats, metric, threshold) : []),
    [stats, metric, threshold],
  );

  const focused = pinned ?? hovered;

  // Layout: circular
  const N = CLUSTER_KEYS.length;
  const W = 720;
  const H = 720;
  const cx = W / 2;
  const cy = H / 2;
  const R = Math.min(W, H) * 0.36;
  const NODE_R = 14;

  // Hide clusters with no presence in the slice; sort by total marginal (heaviest first)
  const order = useMemo(() => {
    if (!stats) return Array.from({ length: N }, (_, i) => i);
    const minMarginal = Math.max(5, Math.floor(stats.total * 0.001));
    return Array.from({ length: N }, (_, i) => i)
      .filter((i) => stats.marginals[i] >= minMarginal)
      .sort((a, b) => stats.marginals[b] - stats.marginals[a]);
  }, [stats, N]);
  const visibleN = order.length;
  const visibleSet = useMemo(() => new Set(order), [order]);
  const edges = useMemo(
    () => allEdges.filter((e) => visibleSet.has(e.i) && visibleSet.has(e.j)),
    [allEdges, visibleSet],
  );

  const positions = useMemo(() => {
    const denom = Math.max(1, visibleN);
    return order.map((idx, slot) => {
      const angle = (slot / denom) * Math.PI * 2 - Math.PI / 2;
      return {
        idx,
        slot,
        x: cx + Math.cos(angle) * R,
        y: cy + Math.sin(angle) * R,
        angle,
      };
    });
  }, [order, visibleN, cx, cy, R]);

  const posByIdx = useMemo(() => {
    const map = new Map<number, (typeof positions)[number]>();
    for (const p of positions) map.set(p.idx, p);
    return map;
  }, [positions]);

  const wMin = useMemo(() => {
    let m = Infinity;
    for (const e of edges) m = Math.min(m, e.weight);
    return Number.isFinite(m) ? m : 0;
  }, [edges]);
  const wMax = useMemo(() => {
    let m = -Infinity;
    for (const e of edges) m = Math.max(m, e.weight);
    return Number.isFinite(m) ? m : 1;
  }, [edges]);

  function edgeOpacity(w: number, isFocusEdge: boolean): number {
    const base = wMax === wMin ? 0.5 : (w - wMin) / (wMax - wMin);
    if (focused === null) return 0.15 + base * 0.55;
    return isFocusEdge ? 0.25 + base * 0.7 : 0.04;
  }

  function edgeColor(w: number): string {
    // gradient from gray → primary as w increases
    if (wMax === wMin) return "#3c6ea8";
    const t = (w - wMin) / (wMax - wMin);
    if (t < 0.33) return "#9ca3af";
    if (t < 0.66) return "#3c6ea8";
    return "#7b3cb8";
  }

  function nodeOpacity(idx: number): number {
    if (focused === null) return 1;
    if (focused === idx) return 1;
    // is connected?
    for (const e of edges) {
      if ((e.i === focused && e.j === idx) || (e.j === focused && e.i === idx)) return 1;
    }
    return 0.25;
  }

  // Top neighbours of focused node
  const topNeighbours = useMemo(() => {
    if (focused === null || !stats) return [];
    const n = CLUSTER_KEYS.length;
    const arr: { idx: number; lift: number; joint: number; cond: number }[] = [];
    const pi = stats.marginals[focused] / stats.total;
    for (let j = 0; j < n; j += 1) {
      if (j === focused) continue;
      const ij = stats.joint[focused][j];
      if (ij < 5) continue;
      const pj = stats.marginals[j] / stats.total;
      const pij = ij / stats.total;
      const lift = pi > 0 && pj > 0 ? pij / (pi * pj) : 0;
      const cond = pi > 0 ? pij / pi : 0;
      arr.push({ idx: j, lift, joint: ij, cond });
    }
    arr.sort((a, b) => b.lift - a.lift);
    return arr.slice(0, 8);
  }, [focused, stats]);

  if (loading || !rows || !stats) {
    return <LoadingShell />;
  }
  if (error) {
    return (
      <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
        Failed to load dataset: {error}
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <PageHeader
        eyebrow="Skill network"
        title="Which skill clusters travel together?"
        description={`A circular co-occurrence graph over ${formatNumber(stats.total)} postings (after filters). Edges = co-occurrence above chance — thicker bonds = stronger pairing. Click a node to pin it.`}
      />

      <Card>
        <CardHeader className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div className="flex flex-wrap items-center gap-3">
            <ControlGroup label="Country">
              <ToggleGroup
                type="single"
                value={mode}
                onValueChange={(v: string) => v && setMode(v as Mode)}
              >
                <ToggleGroupItem value="ALL">All</ToggleGroupItem>
                {COUNTRIES.map((c) => (
                  <ToggleGroupItem key={c} value={c}>
                    {COUNTRY_FLAGS[c]} {c}
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
            </ControlGroup>
            <ControlGroup label="Edge metric">
              <ToggleGroup
                type="single"
                value={metric}
                onValueChange={(v: string) => v && setMetric(v as EdgeMetric)}
              >
                <ToggleGroupItem value="lift">Lift</ToggleGroupItem>
                <ToggleGroupItem value="joint">Joint share</ToggleGroupItem>
                <ToggleGroupItem value="conditional">Conditional</ToggleGroupItem>
              </ToggleGroup>
            </ControlGroup>
            <ControlGroup label="Subset">
              <ToggleGroup
                type="single"
                value={aiOnly ? "ai" : "all"}
                onValueChange={(v: string) => v && setAiOnly(v === "ai")}
              >
                <ToggleGroupItem value="ai">AI postings only</ToggleGroupItem>
                <ToggleGroupItem value="all">All postings</ToggleGroupItem>
              </ToggleGroup>
            </ControlGroup>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <Badge variant="outline">{edges.length} edges shown</Badge>
            <Badge variant="outline">N = {formatNumber(stats.total)}</Badge>
            <Button variant="ghost" size="sm" onClick={() => setPinned(null)}>Reset</Button>
          </div>
        </CardHeader>
        <CardContent className="grid gap-6 md:grid-cols-3">
          <div className="md:col-span-2 overflow-hidden rounded-xl border bg-gradient-to-br from-slate-50 to-white">
            <svg
              width="100%"
              viewBox={`0 0 ${W} ${H}`}
              preserveAspectRatio="xMidYMid meet"
              onMouseLeave={() => setHovered(null)}
              role="img"
              aria-label="Skill cluster co-occurrence network"
            >
              {/* edges */}
              <g>
                {edges.map((e) => {
                  const a = posByIdx.get(e.i)!;
                  const b = posByIdx.get(e.j)!;
                  const isFocusEdge = focused !== null && (e.i === focused || e.j === focused);
                  const op = edgeOpacity(e.weight, isFocusEdge);
                  const stroke = edgeColor(e.weight);
                  // Quadratic curve through center, slightly bent
                  const mx = (a.x + b.x) / 2;
                  const my = (a.y + b.y) / 2;
                  const dx = mx - cx;
                  const dy = my - cy;
                  const k = 0.6;
                  const cqx = cx + dx * k;
                  const cqy = cy + dy * k;
                  const sw = wMax === wMin ? 1.2 : 0.6 + ((e.weight - wMin) / (wMax - wMin)) * 2.4;
                  return (
                    <path
                      key={`${e.i}-${e.j}`}
                      d={`M ${a.x} ${a.y} Q ${cqx} ${cqy} ${b.x} ${b.y}`}
                      stroke={stroke}
                      strokeOpacity={op}
                      strokeWidth={isFocusEdge ? sw + 1 : sw}
                      fill="none"
                    />
                  );
                })}
              </g>
              {/* nodes */}
              <g>
                {positions.map((p) => {
                  const margin = stats.marginals[p.idx];
                  const share = margin / stats.total;
                  const r = NODE_R + Math.sqrt(share) * 32;
                  const op = nodeOpacity(p.idx);
                  const isFocus = focused === p.idx;
                  const labelDistance = r + 12;
                  const ax = cx + Math.cos(p.angle) * (R + labelDistance);
                  const ay = cy + Math.sin(p.angle) * (R + labelDistance);
                  const anchor = Math.cos(p.angle) > 0.05 ? "start" : Math.cos(p.angle) < -0.05 ? "end" : "middle";
                  return (
                    <g
                      key={p.idx}
                      onMouseEnter={() => setHovered(p.idx)}
                      onClick={() => setPinned((cur) => (cur === p.idx ? null : p.idx))}
                      style={{ cursor: "pointer" }}
                      opacity={op}
                    >
                      <circle
                        cx={p.x}
                        cy={p.y}
                        r={r}
                        fill={isFocus ? "#3c6ea8" : "white"}
                        stroke={isFocus ? "#243f63" : "#3c6ea8"}
                        strokeWidth={isFocus ? 2.5 : 1.5}
                      />
                      <text
                        x={p.x}
                        y={p.y + 4}
                        textAnchor="middle"
                        fontSize={11}
                        fontWeight={600}
                        fill={isFocus ? "white" : "#243f63"}
                      >
                        {formatPct(share * 100, 0)}
                      </text>
                      <text
                        x={ax}
                        y={ay}
                        textAnchor={anchor}
                        fontSize={11}
                        fontWeight={isFocus ? 700 : 500}
                        fill="#1f2937"
                      >
                        {CLUSTER_LABELS[p.idx]}
                      </text>
                    </g>
                  );
                })}
              </g>
            </svg>
          </div>

          {/* Side panel */}
          <div className="space-y-4">
            <div className="rounded-lg border bg-card p-4">
              <div className="text-xs uppercase tracking-wider text-muted-foreground">
                {pinned !== null ? "Pinned cluster" : focused !== null ? "Hovered cluster" : "Pick a cluster"}
              </div>
              <div className="mt-1 text-base font-semibold">
                {focused !== null ? CLUSTER_LABELS[focused] : "Hover or click any node"}
              </div>
              {focused !== null ? (
                <div className="mt-2 space-y-1 text-xs text-muted-foreground">
                  <div>
                    <strong className="text-foreground">{formatPct((stats.marginals[focused] / stats.total) * 100, 1)}</strong>{" "}
                    of {formatNumber(stats.total)} postings ({formatNumber(stats.marginals[focused])} mentions).
                  </div>
                  <div>Slot #{posByIdx.get(focused)?.slot} on the ring.</div>
                </div>
              ) : null}
            </div>

            {focused !== null ? (
              <div className="rounded-lg border bg-card p-4">
                <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Top co-occurring clusters
                </div>
                <ul className="space-y-1.5 text-xs">
                  {topNeighbours.map((n) => (
                    <li
                      key={n.idx}
                      className="flex items-center justify-between gap-3 rounded-md border-l-2 border-primary/40 bg-muted/40 px-2 py-1.5"
                      onMouseEnter={() => setHovered(n.idx)}
                    >
                      <div className="flex-1 truncate font-medium">{CLUSTER_LABELS[n.idx]}</div>
                      <div className="flex shrink-0 items-center gap-2 text-[10px] tabular-nums">
                        <span className="text-primary">×{n.lift.toFixed(2)}</span>
                        <span className="text-muted-foreground">{(n.cond * 100).toFixed(0)}%</span>
                      </div>
                    </li>
                  ))}
                </ul>
                <div className="mt-2 text-[10px] text-muted-foreground">
                  Lift × · Conditional %  P(other | this).
                </div>
              </div>
            ) : null}

            <div className="rounded-lg border bg-muted/30 p-4 text-xs">
              <div className="mb-1 font-semibold">How to read</div>
              <p className="leading-relaxed text-muted-foreground">
                Each <strong>node</strong> is a skill cluster, sized by mention frequency. Each <strong>edge</strong> connects
                clusters that co-occur more often than chance ({metric === "lift" ? "lift > 1.2" : metric === "conditional" ? "P(j|i) > 0.4" : "joint share > 4%"}).
                Hover or click a node to surface its tightest neighbours.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Interpretation tips</CardTitle>
          <CardDescription>What patterns to look for in the network</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 text-sm md:grid-cols-3">
          <Tip
            title="The ML/Data triangle"
            body="Generative AI ↔ Data Science / ML ↔ Data Engineering form an unusually tight triangle in US AI postings. Switch country to IN — the triangle weakens, suggesting Indian Applied/Core AI roles are more siloed."
          />
          <Tip
            title="Cloud as connective tissue"
            body="Cloud Computing tends to bridge clusters that don't otherwise co-occur (Frontend × Data Engineering, for example). It's the second-largest hub by degree centrality."
          />
          <Tip
            title="Switch to 'All postings'"
            body="With AI-only off, Generative AI shrinks dramatically (mentioned in only ~3% of all IT postings). The visual collapse is exactly the AI-tier gap from the Insights page."
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
