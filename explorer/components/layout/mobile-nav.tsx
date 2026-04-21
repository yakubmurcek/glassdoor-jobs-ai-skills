"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BookOpen,
  DollarSign,
  GitCompareArrows,
  Home,
  Layers,
  SlidersHorizontal,
  Table2,
} from "lucide-react";
import { cn } from "@/lib/utils";

const ITEMS = [
  { href: "/", label: "Home", icon: Home },
  { href: "/analyze", label: "Analyze", icon: SlidersHorizontal },
  { href: "/compare", label: "Compare", icon: GitCompareArrows },
  { href: "/clusters", label: "Skills", icon: Layers },
  { href: "/premium", label: "Wages", icon: DollarSign },
  { href: "/explorer", label: "Data", icon: Table2 },
  { href: "/about", label: "Info", icon: BookOpen },
];

export function MobileNav() {
  const pathname = usePathname();
  return (
    <nav className="sticky bottom-0 z-40 flex items-stretch border-t bg-background/95 backdrop-blur md:hidden">
      {ITEMS.map((item) => {
        const active = pathname === item.href;
        const Icon = item.icon;
        return (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              "flex flex-1 flex-col items-center justify-center gap-0.5 py-2 text-[10px] font-medium",
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
