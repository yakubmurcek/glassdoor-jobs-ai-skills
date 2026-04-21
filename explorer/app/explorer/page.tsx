"use client";

import { Suspense, useMemo, useState } from "react";
import { Download, Loader2 } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { SlicerBar } from "@/components/slicer/slicer-bar";
import { useSlicer } from "@/lib/state/slicer";
import { filterRows, useRows, type CompactRow } from "@/lib/data/rows-store";
import { AI_TIER_ORDER, COUNTRY_FLAGS, TIER_COLORS, type AITier, type Country } from "@/lib/constants";
import { formatNumber } from "@/lib/utils";

export default function ExplorerPage() {
  return (
    <Suspense fallback={<Loading />}>
      <ExplorerContent />
    </Suspense>
  );
}

function Loading() {
  return (
    <div className="flex min-h-[400px] items-center justify-center gap-2 text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin" />
      Loading dataset…
    </div>
  );
}

const PAGE_SIZE = 40;

function ExplorerContent() {
  const { rows, loading, error } = useRows();
  const slicer = useSlicer();
  const [page, setPage] = useState(0);
  const [downloading, setDownloading] = useState(false);

  const filtered = useMemo(() => (rows ? filterRows(rows, slicer.value) : []), [rows, slicer.value]);

  const pageRows = useMemo(
    () => filtered.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE),
    [filtered, page],
  );

  // Reset pagination when slice shrinks below current page
  useMemo(() => {
    if (page * PAGE_SIZE >= filtered.length) setPage(0);
  }, [filtered.length, page]);

  if (loading) return <Loading />;
  if (error) {
    return (
      <div className="mx-auto max-w-3xl rounded-md border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
        Could not load <span className="font-mono">rows.json</span> — {error}
      </div>
    );
  }

  function downloadCsv() {
    setDownloading(true);
    const cols: (keyof CompactRow | string)[] = [
      "id", "co", "jt", "cp", "ct", "st", "jf", "sc",
      "tier", "ai",
      "sn", "sm", "sx", "cur",
      "ed", "ex", "sen",
      "in", "se", "sz",
      "hs",
    ];
    const header: Record<string, string> = {
      id: "id", co: "country", jt: "job_title", cp: "company",
      ct: "city", st: "state", jf: "job_family", sc: "skill_cluster",
      tier: "ai_tier", ai: "is_real_ai",
      sn: "salary_min", sm: "salary_mid", sx: "salary_max", cur: "pay_currency",
      ed: "edu_level", ex: "experience_min", sen: "seniority_band",
      in: "industry", se: "sector", sz: "size_band",
      hs: "hardskills",
    };
    const escape = (v: unknown) => {
      if (v == null) return "";
      const s = String(v);
      if (s.includes(";") || s.includes("\"") || s.includes("\n")) return `"${s.replace(/"/g, '""')}"`;
      return s;
    };
    const headerLine = cols.map((c) => header[c as string]).join(";");
    const body = filtered
      .map((r) => cols
        .map((c) => {
          if (c === "tier") return r.t === null ? "" : AI_TIER_ORDER[r.t];
          return escape((r as unknown as Record<string, unknown>)[c as string]);
        })
        .join(";"))
      .join("\n");
    const csv = `${headerLine}\n${body}`;
    const blob = new Blob(["\ufeff", csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `glassdoor_ai_${filtered.length}rows.csv`;
    a.click();
    URL.revokeObjectURL(url);
    setTimeout(() => setDownloading(false), 400);
  }

  return (
    <div className="mx-auto max-w-[1500px] space-y-6">
      <PageHeader
        eyebrow="Section 5.5"
        title="Dataset browser"
        description={`All ${formatNumber(rows?.length ?? 0)} postings, live-filtered by the same slicer used on /analyze and /compare. Sort by any column, page through results, export the filtered set as CSV.`}
        actions={
          <Button onClick={downloadCsv} disabled={filtered.length === 0 || downloading}>
            <Download className="size-4" />
            Export {formatNumber(filtered.length)} rows
          </Button>
        }
      />

      <div className="grid gap-6 lg:grid-cols-[300px_minmax(0,1fr)]">
        <div>
          <SlicerBar handle={slicer} rows={rows ?? []} />
        </div>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <CardTitle className="text-sm">
                {formatNumber(filtered.length)} postings
              </CardTitle>
              <CardDescription className="text-xs">
                Showing {filtered.length === 0 ? 0 : page * PAGE_SIZE + 1}–{Math.min(filtered.length, (page + 1) * PAGE_SIZE)} · hard skills truncated to 5 tokens.
              </CardDescription>
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full text-xs">
                <thead className="bg-muted/40">
                  <tr>
                    <Th>Country</Th>
                    <Th>Job title</Th>
                    <Th>Company</Th>
                    <Th>Family</Th>
                    <Th>Tier</Th>
                    <Th className="text-right">Salary (mid)</Th>
                    <Th>Seniority</Th>
                    <Th>Education</Th>
                    <Th>Top skills</Th>
                  </tr>
                </thead>
                <tbody>
                  {pageRows.length === 0 ? (
                    <tr>
                      <td colSpan={9} className="px-3 py-6 text-center text-sm text-muted-foreground">
                        No rows match the current slicer.
                      </td>
                    </tr>
                  ) : pageRows.map((r, i) => (
                    <tr key={`${r.id}-${i}`} className="border-t hover:bg-muted/30">
                      <Td>
                        <span className="inline-flex items-center gap-1">
                          <span>{COUNTRY_FLAGS[r.co as Country]}</span>
                          <span className="font-mono text-[10px] text-muted-foreground">{r.co}</span>
                        </span>
                      </Td>
                      <Td>
                        <span className="block max-w-[260px] truncate font-medium" title={r.jt ?? ""}>
                          {r.jt ?? "—"}
                        </span>
                      </Td>
                      <Td>
                        <span className="block max-w-[140px] truncate text-muted-foreground" title={r.cp ?? ""}>
                          {r.cp ?? "—"}
                        </span>
                      </Td>
                      <Td>{r.jf ?? "—"}</Td>
                      <Td>
                        {r.t === null ? (
                          <span className="text-muted-foreground">—</span>
                        ) : (
                          <span
                            className="inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[10px]"
                            style={{ borderColor: TIER_COLORS[AI_TIER_ORDER[r.t] as AITier] + "66", color: TIER_COLORS[AI_TIER_ORDER[r.t] as AITier] }}
                          >
                            <span className="inline-block size-1.5 rounded-sm" style={{ backgroundColor: TIER_COLORS[AI_TIER_ORDER[r.t] as AITier] }} />
                            {AI_TIER_ORDER[r.t]}
                          </span>
                        )}
                      </Td>
                      <Td className="text-right tabular-nums">
                        {r.sm === null ? "—" : `${currencySymbol(r.cur)}${Math.round(r.sm).toLocaleString()}`}
                      </Td>
                      <Td>{r.sen ?? "—"}</Td>
                      <Td className="capitalize">{(r.ed ?? "—").replace(/_/g, " ")}</Td>
                      <Td>
                        <div className="flex flex-wrap gap-0.5 max-w-[240px]">
                          {(r.hs ?? "").split(",").map((s) => s.trim()).filter(Boolean).slice(0, 5).map((s) => (
                            <span key={s} className="rounded border px-1 py-0.5 text-[10px]">
                              {s}
                            </span>
                          ))}
                        </div>
                      </Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
              <span>
                Page {page + 1} / {Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))}
              </span>
              <div className="flex items-center gap-1">
                <Button variant="outline" size="sm" disabled={page === 0} onClick={() => setPage(page - 1)}>
                  Prev
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={(page + 1) * PAGE_SIZE >= filtered.length}
                  onClick={() => setPage(page + 1)}
                >
                  Next
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function Th({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <th className={"px-2 py-1.5 text-left text-[10px] font-medium uppercase tracking-wider text-muted-foreground " + (className ?? "")}>
      {children}
    </th>
  );
}

function Td({ children, className }: { children: React.ReactNode; className?: string }) {
  return <td className={"px-2 py-1.5 align-top " + (className ?? "")}>{children}</td>;
}

function currencySymbol(cur: string | null): string {
  switch (cur) {
    case "USD": return "$";
    case "EUR": return "€";
    case "INR": return "₹";
    default: return cur ? cur + " " : "";
  }
}
