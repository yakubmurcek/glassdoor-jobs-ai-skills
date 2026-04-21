"use client";

import { useMemo, useState } from "react";
import {
  ColumnDef,
  ColumnFiltersState,
  SortingState,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { ArrowDown, ArrowUp, ArrowUpDown, ChevronLeft, ChevronRight } from "lucide-react";
import type { JobRow } from "@/lib/data/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  COUNTRY_FLAGS,
  TIER_COLORS,
  type AITier,
  type Country,
} from "@/lib/constants";
import { cn } from "@/lib/utils";

const TIER_LABEL: Record<string, AITier> = {
  none: "None",
  integration: "AI Integration",
  applied: "Applied/Core AI",
};

interface Props {
  data: JobRow[];
  globalFilter: string;
  country?: Country | null;
  tier?: string | null;
}

export function JobsTable({ data, globalFilter, country, tier }: Props) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);

  const filtered = useMemo(() => {
    return data.filter((r) => {
      if (country && r.country !== country) return false;
      if (tier && r.desc_tier_llm !== tier) return false;
      return true;
    });
  }, [data, country, tier]);

  const columns = useMemo<ColumnDef<JobRow>[]>(
    () => [
      {
        accessorKey: "country",
        header: "Country",
        cell: (info) => {
          const c = info.getValue() as Country;
          return (
            <span className="inline-flex items-center gap-1 text-sm">
              <span>{COUNTRY_FLAGS[c]}</span>
              <span className="font-mono text-xs text-muted-foreground">{c}</span>
            </span>
          );
        },
        size: 80,
      },
      {
        accessorKey: "job_title",
        header: "Job title",
        cell: (info) => (
          <span className="block max-w-[280px] truncate font-medium" title={info.getValue() as string}>
            {info.getValue() as string}
          </span>
        ),
        size: 280,
      },
      {
        accessorKey: "company",
        header: "Company",
        cell: (info) => (
          <span className="block max-w-[160px] truncate text-sm text-muted-foreground" title={String(info.getValue() ?? "")}>
            {(info.getValue() as string) ?? "—"}
          </span>
        ),
        size: 160,
      },
      {
        accessorKey: "job_family",
        header: "Family",
        cell: (info) => <span className="text-xs">{(info.getValue() as string) ?? "—"}</span>,
        size: 140,
      },
      {
        accessorKey: "desc_tier_llm",
        header: "AI tier",
        cell: (info) => {
          const raw = info.getValue() as string | null;
          if (!raw) return <span className="text-xs text-muted-foreground">—</span>;
          const tier = TIER_LABEL[raw] ?? raw;
          const color = TIER_COLORS[tier as AITier] ?? "#94a3b8";
          return (
            <span
              className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium"
              style={{ borderColor: color + "66", color }}
            >
              <span className="inline-block size-2 rounded-sm" style={{ backgroundColor: color }} />
              {tier}
            </span>
          );
        },
        size: 140,
      },
      {
        accessorKey: "salary_mid",
        header: "Salary (mid)",
        cell: (info) => {
          const v = info.getValue() as number | null;
          if (v == null) return <span className="text-xs text-muted-foreground">—</span>;
          const cur = (info.row.original.pay_currency ?? "USD") as string;
          return (
            <span className="tabular-nums">
              {cur === "EUR" ? "€" : cur === "INR" ? "₹" : "$"}
              {Math.round(v).toLocaleString("en-US")}
            </span>
          );
        },
        sortingFn: (a, b) => {
          const av = a.original.salary_mid ?? -1;
          const bv = b.original.salary_mid ?? -1;
          return av - bv;
        },
        size: 120,
      },
      {
        accessorKey: "edu_level_det",
        header: "Education",
        cell: (info) => <span className="text-xs capitalize">{((info.getValue() as string) ?? "—").replace(/_/g, " ")}</span>,
        size: 120,
      },
      {
        accessorKey: "experience_min_llm",
        header: "Exp (y)",
        cell: (info) => {
          const v = info.getValue() as number | null;
          return <span className="text-xs tabular-nums">{v == null ? "—" : v}</span>;
        },
        size: 70,
      },
      {
        accessorKey: "hardskills",
        header: "Top hard skills",
        cell: (info) => {
          const v = (info.getValue() as string) ?? "";
          const tokens = v.split(",").map((s) => s.trim()).filter(Boolean).slice(0, 6);
          if (tokens.length === 0) return <span className="text-xs text-muted-foreground">—</span>;
          return (
            <div className="flex flex-wrap gap-1">
              {tokens.map((t) => (
                <Badge key={t} variant="outline" className="text-[10px] font-normal">
                  {t}
                </Badge>
              ))}
            </div>
          );
        },
        size: 300,
        enableSorting: false,
      },
    ],
    [],
  );

  const table = useReactTable({
    data: filtered,
    columns,
    state: { sorting, columnFilters, globalFilter },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    globalFilterFn: (row, _colId, filterValue) => {
      if (!filterValue) return true;
      const needle = String(filterValue).toLowerCase();
      const values = [
        row.original.job_title,
        row.original.company,
        row.original.job_family,
        row.original.city,
        row.original.state,
        row.original.industry,
        row.original.hardskills,
        row.original.softskills,
      ];
      return values.some((v) => (v ?? "").toString().toLowerCase().includes(needle));
    },
    initialState: {
      pagination: { pageSize: 25 },
    },
  });

  const rows = table.getRowModel().rows;
  const totalVisible = table.getFilteredRowModel().rows.length;
  const pageIndex = table.getState().pagination.pageIndex;
  const pageCount = table.getPageCount();

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto rounded-lg border">
        <table className="w-full text-sm">
          <thead className="bg-muted/40">
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((h) => {
                  const canSort = h.column.getCanSort();
                  const sort = h.column.getIsSorted();
                  return (
                    <th
                      key={h.id}
                      className={cn(
                        "px-3 py-2 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground",
                        canSort && "cursor-pointer select-none hover:text-foreground",
                      )}
                      onClick={canSort ? h.column.getToggleSortingHandler() : undefined}
                      style={{ width: h.getSize() }}
                    >
                      <span className="inline-flex items-center gap-1">
                        {flexRender(h.column.columnDef.header, h.getContext())}
                        {canSort && (
                          sort === "asc"
                            ? <ArrowUp className="size-3" />
                            : sort === "desc"
                              ? <ArrowDown className="size-3" />
                              : <ArrowUpDown className="size-3 opacity-30" />
                        )}
                      </span>
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="px-3 py-8 text-center text-sm text-muted-foreground">
                  No rows match the current filters.
                </td>
              </tr>
            ) : (
              rows.map((r) => (
                <tr key={r.id} className="border-t transition-colors hover:bg-muted/30">
                  {r.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="px-3 py-2 align-top">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
        <span>
          Showing {rows.length ? pageIndex * 25 + 1 : 0}–{pageIndex * 25 + rows.length} of {totalVisible.toLocaleString()} rows
        </span>
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            <ChevronLeft className="size-3.5" />
            Prev
          </Button>
          <span className="px-2 tabular-nums">
            {pageIndex + 1} / {Math.max(pageCount, 1)}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            Next
            <ChevronRight className="size-3.5" />
          </Button>
        </div>
      </div>
    </div>
  );
}
