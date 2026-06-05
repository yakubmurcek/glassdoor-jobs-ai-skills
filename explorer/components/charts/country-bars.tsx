"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { COUNTRY_COLORS, COUNTRY_FLAGS, type Country } from "@/lib/constants";

interface Datum {
  country: Country;
  label: string;
  value: number;
  sublabel?: string;
}

interface Props {
  data: Datum[];
  unit?: string;
  height?: number;
  domain?: [number, number];
  formatter?: (v: number) => string;
}

export function CountryBars({
  data,
  unit = "%",
  height = 220,
  domain,
  formatter,
}: Props) {
  const fmt = formatter ?? ((v: number) => `${v.toFixed(1)}${unit}`);
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 16, right: 16, left: 0, bottom: 12 }}>
          <CartesianGrid stroke="#e5e7eb" strokeDasharray="2 2" vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 12 }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            domain={domain}
            tickFormatter={(v) => `${v}${unit}`}
            width={42}
          />
          <Tooltip
            cursor={{ fill: "#0001" }}
            contentStyle={{
              borderRadius: 8,
              fontSize: 12,
              padding: "6px 10px",
              border: "1px solid #e5e7eb",
            }}
            formatter={(value: number, _name, payload) => [fmt(value), payload?.payload?.sublabel ?? ""]}
          />
          <Bar dataKey="value" radius={[6, 6, 0, 0]}>
            {data.map((d) => (
              <Cell key={d.country} fill={COUNTRY_COLORS[d.country]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function countryDatum(
  country: Country,
  value: number,
  sublabel?: string,
): Datum {
  return {
    country,
    label: `${COUNTRY_FLAGS[country]} ${country}`,
    value,
    sublabel,
  };
}
