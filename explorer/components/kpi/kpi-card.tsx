import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";

interface KpiCardProps {
  label: string;
  value: string | number;
  sublabel?: string;
  trend?: string;
  trendPositive?: boolean | null;
  accent?: string;
  className?: string;
}

export function KpiCard({
  label,
  value,
  sublabel,
  trend,
  trendPositive,
  accent,
  className,
}: KpiCardProps) {
  return (
    <Card className={cn("relative overflow-hidden", className)}>
      {accent ? (
        <div
          className="absolute inset-x-0 top-0 h-1"
          style={{ backgroundColor: accent }}
        />
      ) : null}
      <div className="p-5">
        <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          {label}
        </div>
        <div className="mt-2 flex items-baseline gap-2">
          <span className="text-3xl font-semibold tabular-nums">{value}</span>
          {trend ? (
            <span
              className={cn(
                "text-xs font-medium",
                trendPositive === true && "text-emerald-600 dark:text-emerald-400",
                trendPositive === false && "text-rose-600 dark:text-rose-400",
                trendPositive == null && "text-muted-foreground",
              )}
            >
              {trend}
            </span>
          ) : null}
        </div>
        {sublabel ? (
          <div className="mt-1 text-xs text-muted-foreground">{sublabel}</div>
        ) : null}
      </div>
    </Card>
  );
}
