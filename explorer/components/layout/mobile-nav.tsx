"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Globe2,
  Hash,
  Home,
  Layers,
  Network,
  SlidersHorizontal,
  Sparkles,
  Table2,
  TrendingUp,
} from "lucide-react";
import { cn } from "@/lib/utils";

const ITEMS = [
  { href: "/", label: "Home", icon: Home },
  { href: "/insights", label: "Insights", icon: Sparkles },
  { href: "/analyze", label: "Analyze", icon: SlidersHorizontal },
  { href: "/skills", label: "Skills", icon: Hash },
  { href: "/network", label: "Network", icon: Network },
  { href: "/distributions", label: "Dist.", icon: TrendingUp },
  { href: "/geography", label: "Geo", icon: Globe2 },
  { href: "/clusters", label: "Clust.", icon: Layers },
  { href: "/explorer", label: "Data", icon: Table2 },
];

// Routes that should render full-bleed without the bottom navigation
// (defense mode is the only one for now).
const HIDDEN_ON: ReadonlyArray<string> = ["/present"];

export function MobileNav() {
  const pathname = usePathname();
  if (HIDDEN_ON.includes(pathname)) return null;
  return (
    <nav className="sticky bottom-0 z-40 flex items-stretch overflow-x-auto border-t bg-background/95 backdrop-blur md:hidden">
      {ITEMS.map((item) => {
        const active = pathname === item.href;
        const Icon = item.icon;
        return (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              "flex min-w-[60px] flex-1 flex-col items-center justify-center gap-0.5 py-2 text-[10px] font-medium",
              active ? "text-primary" : "text-muted-foreground",
            )}
          >
            <Icon className="size-4" />
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
