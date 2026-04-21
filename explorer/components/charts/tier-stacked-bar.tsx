"use client";

import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { G1CountryRow } from "@/lib/data/types";
import { AI_TIER_ORDER, TIER_COLORS } from "@/lib/constants";
import { formatPct } from "@/lib/utils";

interface Props {
  data: G1CountryRow[];
  selectedCountries?: string[];
  onSelectCountry?: (country: string) => void;
  compact?: boolean;
}

export function TierStackedBar({
  data,
  selectedCountries,
  onSelectCountry,
  compact,
}: Props) {
  const chartData = useMemo(
    () =>
      data.map((row) => ({
        country: row.country,
        country_label: row.country_label,
        total: row.total,
        None: row.None.pct,
        "AI Integration": row["AI Integration"].pct,
        "Applied/Core AI": row["Applied/Core AI"].pct,
      })),
    [data],
  );

  const height = compact ? 220 : 340;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={chartData}
        margin={{ top: 12, right: 12, bottom: 8, left: 4 }}
      >
        <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.35} />
        <XAxis
          dataKey="country_label"
          tick={{ fontSize: 12 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tick={{ fontSize: 11 }}
          tickFormatter={(v) => `${v}%`}
          domain={[0, 100]}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          cursor={{ fill: "rgba(0,0,0,0.04)" }}
          content={({ active, payload, label }) => {
            if (!active || !payload?.length) return null;
            const total = payload[0]?.payload?.total ?? 0;
            return (
              <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
                <div className="mb-1 font-semibold">{label}</div>
                <div className="text-muted-foreground">
                  N = {total.toLocaleString()}
                </div>
                <div className="mt-1.5 space-y-0.5">
                  {AI_TIER_ORDER.map((tier) => {
                    const value = payload.find((p) => p.dataKey === tier)?.value as
                      | number
                      | undefined;
                    return (
                      <div key={tier} className="flex items-center gap-2">
                        <span
                          className="inline-block size-2 rounded-sm"
                          style={{ backgroundColor: TIER_COLORS[tier] }}
                        />
                        <span className="min-w-28 text-foreground">{tier}</span>
                        <span className="tabular-nums text-muted-foreground">
                          {formatPct(value)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          }}
        />
        <Legend
          wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
          iconType="square"
        />
        {AI_TIER_ORDER.map((tier, idx) => (
          <Bar
            key={tier}
            dataKey={tier}
            stackId="tiers"
            fill={TIER_COLORS[tier]}
            isAnimationActive={false}
            onClick={(data) => onSelectCountry?.(data.country as string)}
            cursor={onSelectCountry ? "pointer" : undefined}
          >
            {chartData.map((entry, i) => {
              const isDim =
                selectedCountries &&
                selectedCountries.length > 0 &&
                !selectedCountries.includes(entry.country);
              return (
                <Cell
                  key={`${tier}-${i}`}
                  fillOpacity={isDim ? 0.35 : 1}
                  stroke="#ffffff"
                  strokeWidth={1}
                />
              );
            })}
            {idx === AI_TIER_ORDER.length - 1 ? (
              <LabelList
                dataKey="total"
                position="top"
                formatter={(v: number | string) => `N=${Number(v).toLocaleString()}`}
                style={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
              />
            ) : null}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
