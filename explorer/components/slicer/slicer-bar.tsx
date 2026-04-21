"use client";

import { useMemo, useState } from "react";
import { Check, Copy, Link2, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  AI_TIERS,
  COUNTRIES,
  COUNTRY_FLAGS,
  COUNTRY_LABELS,
  TIER_COLORS,
  type AITier,
  type Country,
} from "@/lib/constants";
import {
  CLUSTER_KEYS,
  CLUSTER_LABELS,
  SENIORITY_ORDER,
  SIZE_ORDER,
  activeSlicerCount,
  uniqueSorted,
  type CompactRow,
  type Slicer,
} from "@/lib/data/rows-store";
import type { SlicerHandle } from "@/lib/state/slicer";

interface Props {
  handle: SlicerHandle;
  rows: readonly CompactRow[];
  /** When true, renders in-line instead of full sticky column (used on /compare). */
  dense?: boolean;
  title?: string;
  onCopyLink?: () => void;
}

export function SlicerBar({ handle, rows, dense = false, title = "Slicer", onCopyLink }: Props) {
  const s = handle.value;
  const [copied, setCopied] = useState(false);

  const families = useMemo(() => uniqueSorted(rows, "jf"), [rows]);
  const industries = useMemo(() => uniqueSorted(rows, "in", 12), [rows]);
  const states = useMemo(() => uniqueSorted(rows, "st", 30), [rows]);
  const educations = useMemo(() => {
    const set = new Set<string>();
    for (const r of rows) if (r.ed) set.add(r.ed);
    return [...set].sort();
  }, [rows]);

  function copyLink() {
    const url = typeof window !== "undefined" ? window.location.href : "";
    navigator.clipboard?.writeText(url).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
    onCopyLink?.();
  }

  const activeCount = activeSlicerCount(s);

  return (
    <div className={cn("space-y-4", dense ? "" : "rounded-lg border bg-card p-4")}>
      <div className="flex items-baseline justify-between gap-2">
        <div>
          <div className="text-sm font-semibold">{title}</div>
          <div className="text-[11px] text-muted-foreground">
            {activeCount === 0 ? "No filters — showing all postings" : `${activeCount} active filter${activeCount === 1 ? "" : "s"}`}
          </div>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            onClick={copyLink}
            title="Copy current analytical link"
            disabled={typeof window === "undefined"}
          >
            {copied ? <Check className="size-3.5" /> : <Link2 className="size-3.5" />}
            {copied ? "Copied" : "Copy link"}
          </Button>
          <Button variant="ghost" size="sm" onClick={() => handle.reset()} disabled={activeCount === 0}>
            Reset
          </Button>
        </div>
      </div>

      <DimensionSection label="Country">
        <div className="flex flex-wrap gap-1.5">
          {COUNTRIES.map((c) => (
            <Chip
              key={c}
              active={s.countries.includes(c)}
              onClick={() => handle.set({ countries: toggle(s.countries, c) })}
            >
              <span className="mr-1">{COUNTRY_FLAGS[c]}</span>
              {COUNTRY_LABELS[c]}
            </Chip>
          ))}
        </div>
      </DimensionSection>

      <DimensionSection label="AI tier">
        <div className="flex flex-wrap gap-1.5">
          {AI_TIERS.map((t) => (
            <Chip
              key={t}
              active={s.tiers.includes(t)}
              onClick={() => handle.set({ tiers: toggle(s.tiers as readonly AITier[], t) as AITier[] })}
            >
              <span
                className="mr-1 inline-block size-2 rounded-sm align-middle"
                style={{ backgroundColor: TIER_COLORS[t] }}
              />
              {t}
            </Chip>
          ))}
        </div>
      </DimensionSection>

      <DimensionSection label="Job family">
        <ChipWrap
          items={families}
          selected={s.jobFamilies}
          onToggle={(v) => handle.set({ jobFamilies: toggle(s.jobFamilies, v) })}
        />
      </DimensionSection>

      <DimensionSection label="Seniority">
        <div className="flex flex-wrap gap-1.5">
          {SENIORITY_ORDER.map((x) => (
            x ? <Chip
              key={x}
              active={s.seniority.includes(x)}
              onClick={() => handle.set({ seniority: toggle(s.seniority, x) as Slicer["seniority"] })}
            >
              {x}
            </Chip> : null
          ))}
        </div>
      </DimensionSection>

      <DimensionSection label="Education">
        <ChipWrap
          items={educations}
          selected={s.edu}
          onToggle={(v) => handle.set({ edu: toggle(s.edu, v) })}
        />
      </DimensionSection>

      <DimensionSection label="Firm size">
        <div className="flex flex-wrap gap-1.5">
          {SIZE_ORDER.map((x) => (
            x ? <Chip
              key={x}
              active={s.sizeBands.includes(x)}
              onClick={() => handle.set({ sizeBands: toggle(s.sizeBands, x) as Slicer["sizeBands"] })}
            >
              {x}
            </Chip> : null
          ))}
        </div>
      </DimensionSection>

      {industries.length > 0 && (
        <DimensionSection label={`Industry · top ${industries.length}`}>
          <ChipWrap
            items={industries}
            selected={s.industries}
            onToggle={(v) => handle.set({ industries: toggle(s.industries, v) })}
          />
        </DimensionSection>
      )}

      <DimensionSection label="Salary">
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            className="accent-primary"
            checked={s.salaryDisclosedOnly}
            onChange={(e) => handle.set({ salaryDisclosedOnly: e.target.checked })}
          />
          Only postings that disclose salary
        </label>
      </DimensionSection>

      <DimensionSection label="Must include any of (clusters)">
        <ClusterPicker
          selected={s.clustersAny}
          onToggle={(v) => handle.set({ clustersAny: toggle(s.clustersAny, v) })}
        />
      </DimensionSection>

      <DimensionSection label="Must exclude all of (clusters)">
        <ClusterPicker
          selected={s.clustersNone}
          onToggle={(v) => handle.set({ clustersNone: toggle(s.clustersNone, v) })}
          muted
        />
      </DimensionSection>

      {states.length > 0 && (
        <DimensionSection label={`State / region · top ${states.length}`}>
          <ChipWrap
            items={states}
            selected={s.states}
            onToggle={(v) => handle.set({ states: toggle(s.states, v) })}
          />
        </DimensionSection>
      )}

      <DimensionSection label="Search">
        <Input
          value={s.search}
          onChange={(e) => handle.set({ search: e.target.value })}
          placeholder="Title, company, skills…"
          className="h-8 text-xs"
        />
      </DimensionSection>
    </div>
  );
}

function DimensionSection({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      {children}
    </div>
  );
}

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "inline-flex h-7 items-center rounded-full border px-2.5 text-xs transition-colors",
        active
          ? "border-primary bg-primary/10 text-foreground"
          : "border-input text-muted-foreground hover:text-foreground",
      )}
    >
      {children}
      {active && <X className="ml-1 size-3" />}
    </button>
  );
}

function ChipWrap({
  items,
  selected,
  onToggle,
}: {
  items: string[];
  selected: string[];
  onToggle: (v: string) => void;
}) {
  if (items.length === 0) return <span className="text-xs text-muted-foreground">—</span>;
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((v) => (
        <Chip key={v} active={selected.includes(v)} onClick={() => onToggle(v)}>
          {v}
        </Chip>
      ))}
    </div>
  );
}

function ClusterPicker({
  selected,
  onToggle,
  muted = false,
}: {
  selected: string[];
  onToggle: (key: string) => void;
  muted?: boolean;
}) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {CLUSTER_KEYS.map((key, i) => {
        const label = CLUSTER_LABELS[i];
        const active = selected.includes(key);
        return (
          <button
            key={key}
            type="button"
            onClick={() => onToggle(key)}
            className={cn(
              "inline-flex h-7 items-center rounded-full border px-2.5 text-xs transition-colors",
              active
                ? muted
                  ? "border-destructive/50 bg-destructive/10 text-foreground"
                  : "border-primary bg-primary/10 text-foreground"
                : "border-input text-muted-foreground hover:text-foreground",
            )}
          >
            {label}
            {active && <X className="ml-1 size-3" />}
          </button>
        );
      })}
    </div>
  );
}

function toggle<T>(arr: readonly T[], v: T): T[] {
  if (arr.includes(v)) return arr.filter((x) => x !== v);
  return [...arr, v];
}

export function CopyLinkButton() {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={() => {
        const url = typeof window !== "undefined" ? window.location.href : "";
        navigator.clipboard?.writeText(url).catch(() => {});
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
      }}
    >
      {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
      {copied ? "Copied" : "Copy link"}
    </Button>
  );
}
