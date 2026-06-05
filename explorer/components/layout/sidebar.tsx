"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  BookOpen,
  DollarSign,
  GitCompareArrows,
  Globe2,
  Hash,
  Home,
  Layers,
  Network,
  Presentation,
  SlidersHorizontal,
  Sparkles,
  Table2,
  TrendingUp,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface NavItem {
  href: string;
  label: string;
  icon: typeof Home;
  summary: string;
}

interface NavGroup {
  title: string;
  items: NavItem[];
}

const NAV_GROUPS: NavGroup[] = [
  {
    title: "Story",
    items: [
      { href: "/", label: "Overview", icon: Home, summary: "Headline findings" },
      { href: "/insights", label: "Insights", icon: Sparkles, summary: "Narrative · 7 key findings" },
      { href: "/present", label: "Defense mode", icon: Presentation, summary: "Big-number slides" },
    ],
  },
  {
    title: "Workbench",
    items: [
      { href: "/analyze", label: "Analyze", icon: SlidersHorizontal, summary: "Slice & re-aggregate" },
      { href: "/compare", label: "Compare", icon: GitCompareArrows, summary: "Slice A vs slice B" },
    ],
  },
  {
    title: "Deep dives",
    items: [
      { href: "/clusters", label: "Skill clusters", icon: Layers, summary: "21-cluster heatmap" },
      { href: "/skills", label: "Hard skills", icon: Hash, summary: "Top-N skill leaderboard" },
      { href: "/network", label: "Skill network", icon: Network, summary: "Co-occurrence graph" },
      { href: "/distributions", label: "Distributions", icon: TrendingUp, summary: "Salary · experience · edu" },
      { href: "/geography", label: "Geography", icon: Globe2, summary: "State & city breakdown" },
      { href: "/premium", label: "Salary premium", icon: DollarSign, summary: "OLS shrinkage A→B→C" },
    ],
  },
  {
    title: "Data",
    items: [
      { href: "/explorer", label: "Dataset", icon: Table2, summary: "Browse all 44k postings" },
      { href: "/about", label: "About & method", icon: BookOpen, summary: "Methodology · caveats" },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sticky top-0 hidden h-svh w-64 shrink-0 flex-col border-r bg-sidebar md:flex">
      <div className="flex h-14 items-center gap-2 border-b px-5">
        <div className="size-7 rounded-md bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center">
          <BarChart3 className="size-4 text-primary-foreground" />
        </div>
        <div className="flex flex-col">
          <span className="text-sm font-semibold leading-tight">AI Skills Explorer</span>
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground leading-tight">
            v2 · thesis companion
          </span>
        </div>
      </div>
      <nav className="flex-1 overflow-y-auto px-2 py-3">
        {NAV_GROUPS.map((group) => (
          <div key={group.title} className="mb-3">
            <div className="px-3 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground/80">
              {group.title}
            </div>
            <ul className="space-y-0.5">
              {group.items.map((item) => {
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
          </div>
        ))}
      </nav>
      <div className="border-t px-4 py-3 text-[11px] leading-relaxed text-muted-foreground">
        <div className="font-medium text-foreground">Master&apos;s thesis · 2026</div>
        <div>AI skill requirements in IT job postings · US vs DE vs IN</div>
      </div>
    </aside>
  );
}
