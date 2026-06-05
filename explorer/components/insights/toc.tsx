"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

export interface TocEntry {
  id: string;
  label: string;
  short?: string;
}

interface Props {
  entries: TocEntry[];
}

export function InsightsToc({ entries }: Props) {
  const [active, setActive] = useState<string | null>(entries[0]?.id ?? null);

  useEffect(() => {
    const els = entries
      .map((e) => document.getElementById(e.id))
      .filter((el): el is HTMLElement => el !== null);
    if (els.length === 0) return;

    const observer = new IntersectionObserver(
      (records) => {
        // Of the entries currently inside the active band (top 15%–45% of the
        // viewport, set via rootMargin below), pick the one whose top edge is
        // furthest down. That's the entry the reader most recently scrolled
        // *into*, which feels like the natural "active" item — TOC advances
        // as soon as a new finding's heading enters the band.
        let best: { id: string; top: number } | null = null;
        for (const rec of records) {
          if (!rec.isIntersecting) continue;
          const top = rec.boundingClientRect.top;
          if (!best || top > best.top) {
            best = { id: rec.target.id, top };
          }
        }
        if (best) setActive(best.id);
      },
      { rootMargin: "-15% 0px -55% 0px", threshold: [0, 0.25, 0.5, 0.75, 1] },
    );
    for (const el of els) observer.observe(el);
    return () => observer.disconnect();
  }, [entries]);

  return (
    <aside className="sticky top-6 hidden w-44 self-start xl:block">
      <div className="rounded-xl border bg-card/60 p-3 backdrop-blur">
        <div className="mb-2 px-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
          Findings
        </div>
        <ol className="space-y-0.5">
          {entries.map((e, i) => {
            const isActive = active === e.id;
            return (
              <li key={e.id}>
                <a
                  href={`#${e.id}`}
                  className={cn(
                    "group flex items-start gap-2 rounded-md px-2 py-1.5 text-xs transition-colors",
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-muted/60 hover:text-foreground",
                  )}
                >
                  <span
                    className={cn(
                      "mt-0.5 inline-flex size-4 shrink-0 items-center justify-center rounded-full font-mono text-[9px] font-semibold tabular-nums",
                      isActive
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted text-muted-foreground group-hover:bg-foreground/10",
                    )}
                  >
                    {i + 1}
                  </span>
                  <span className="leading-snug">{e.short ?? e.label}</span>
                </a>
              </li>
            );
          })}
        </ol>
      </div>
    </aside>
  );
}
