"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  BookOpen,
  DollarSign,
  GitCompareArrows,
  Home,
  Layers,
  SlidersHorizontal,
  Table2,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Overview", icon: Home, summary: "Headline findings" },
  { href: "/analyze", label: "Analyze", icon: SlidersHorizontal, summary: "Slice & group the full dataset" },
  { href: "/compare", label: "Compare", icon: GitCompareArrows, summary: "Slice A vs slice B" },
  { href: "/clusters", label: "Skill clusters", icon: Layers, summary: "Cross-country AME + drill-down" },
  { href: "/premium", label: "Salary premium", icon: DollarSign, summary: "Shrinkage & counterfactuals" },
  { href: "/explorer", label: "Dataset", icon: Table2, summary: "Browse all 44k postings" },
  { href: "/about", label: "About & method", icon: BookOpen, summary: "Methodology & deferred work" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sticky top-0 hidden h-svh w-64 shrink-0 flex-col border-r bg-sidebar md:flex">
      <div className="flex h-14 items-center gap-2 border-b px-5">
        <div className="size-7 rounded-md bg-primary/10 flex items-center justify-center">
          <BarChart3 className="size-4 text-primary" />
        </div>
        <div className="flex flex-col">
          <span className="text-sm font-semibold leading-tight">AI Skills Explorer</span>
          <span className="text-[11px] text-muted-foreground leading-tight">
            Thesis companion
          </span>
        </div>
      </div>
      <nav className="flex-1 overflow-y-auto px-2 py-3">
        <ul className="space-y-1">
          {NAV.map((item) => {
            const active = pathname === item.href;
            const Icon = item.icon;
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className={cn(
                    "group flex items-start gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
                    active
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "hover:bg-accent hover:text-accent-foreground",
                  )}
                >
                  <Icon className="mt-0.5 size-4 shrink-0" />
                  <div className="flex flex-col leading-tight">
                    <span className="font-medium">{item.label}</span>
                    <span
                      className={cn(
                        "text-[11px]",
                        active ? "text-primary-foreground/80" : "text-muted-foreground",
                      )}
                    >
                      {item.summary}
                    </span>
                  </div>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
      <div className="border-t px-4 py-3 text-[11px] leading-relaxed text-muted-foreground">
        <div className="font-medium text-foreground">Master&apos;s thesis</div>
        <div>AI skill requirements in IT job postings, US vs DE vs IN.</div>
      </div>
    </aside>
  );
}
