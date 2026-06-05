"use client";

import { cn } from "@/lib/utils";

interface BigNumberProps {
  value: React.ReactNode;
  label: React.ReactNode;
  sublabel?: React.ReactNode;
  accent?: string;
  className?: string;
  size?: "sm" | "md" | "lg" | "xl";
}

export function BigNumber({
  value,
  label,
  sublabel,
  accent,
  className,
  size = "md",
}: BigNumberProps) {
  const sizeClass = {
    sm: "text-3xl",
    md: "text-5xl",
    lg: "text-6xl",
    xl: "text-7xl md:text-8xl",
  }[size];

  return (
    <div className={cn("flex flex-col gap-1", className)}>
      <div
        className={cn("font-semibold tabular-nums tracking-tight leading-none", sizeClass)}
        style={accent ? { color: accent } : undefined}
      >
        {value}
      </div>
      <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      {sublabel ? (
        <div className="text-xs text-muted-foreground">{sublabel}</div>
      ) : null}
    </div>
  );
}
