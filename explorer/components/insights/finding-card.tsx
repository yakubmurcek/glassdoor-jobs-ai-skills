"use client";

import { useState } from "react";
import { Link2, Check } from "lucide-react";
import { cn } from "@/lib/utils";

interface FindingCardProps {
  index: number;
  id?: string;
  eyebrow: string;
  headline: React.ReactNode;
  takeaway: React.ReactNode;
  chart: React.ReactNode;
  /** N, model name, etc. — small print under the chart */
  evidence?: React.ReactNode;
  side?: "left" | "right";
  accent?: string;
}

export function FindingCard({
  index,
  id,
  eyebrow,
  headline,
  takeaway,
  chart,
  evidence,
  side = "left",
  accent,
}: FindingCardProps) {
  const [copied, setCopied] = useState(false);
  const anchorId = id ?? `f-${String(index).padStart(2, "0")}`;

  async function copyLink() {
    if (typeof window === "undefined") return;
    const url = `${window.location.origin}${window.location.pathname}#${anchorId}`;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(url);
        setCopied(true);
        setTimeout(() => setCopied(false), 1600);
      }
    } catch {
      // Clipboard write failed (insecure origin or permissions). No-op.
    }
  }

  return (
    <article
      id={anchorId}
      className="group relative grid gap-6 rounded-2xl border bg-card p-6 shadow-sm scroll-mt-24 md:grid-cols-12 md:p-8"
    >
      <div className="absolute left-6 top-6 flex items-center gap-2 md:left-8 md:top-8">
        <span className="text-[10px] font-mono font-semibold uppercase tracking-[0.2em] text-muted-foreground/70">
          F-{String(index).padStart(2, "0")}
        </span>
        <button
          type="button"
          onClick={copyLink}
          aria-label="Copy direct link to this finding"
          className="inline-flex items-center gap-1 rounded-md border bg-background/80 px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground opacity-0 transition-opacity hover:bg-muted hover:text-foreground group-hover:opacity-100 focus:opacity-100"
        >
          {copied ? (
            <>
              <Check className="size-3 text-emerald-600" />
              copied
            </>
          ) : (
            <>
              <Link2 className="size-3" />
              link
            </>
          )}
        </button>
      </div>
      <div
        className={cn(
          "md:col-span-5 flex flex-col justify-center gap-3",
          side === "right" && "md:order-2",
        )}
      >
        <div
          className="text-xs font-semibold uppercase tracking-wider"
          style={{ color: accent ?? "var(--primary)" }}
        >
          {eyebrow}
        </div>
        <h3 className="text-2xl font-semibold leading-tight tracking-tight md:text-3xl">
          {headline}
        </h3>
        <p className="text-sm text-muted-foreground md:text-base">{takeaway}</p>
      </div>
      <div
        className={cn(
          "md:col-span-7 flex flex-col gap-3",
          side === "right" && "md:order-1",
        )}
      >
        <div className="rounded-xl border bg-background/50 p-3">{chart}</div>
        {evidence ? (
          <div className="text-[11px] leading-relaxed text-muted-foreground">{evidence}</div>
        ) : null}
      </div>
    </article>
  );
}
