"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import * as Dialog from "@radix-ui/react-dialog";
import {
  ArrowRight,
  BookOpen,
  DollarSign,
  GitCompareArrows,
  Globe2,
  Hash,
  Home,
  Layers,
  Network,
  Presentation,
  Search,
  SlidersHorizontal,
  Sparkles,
  Table2,
  TrendingUp,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface Cmd {
  href?: string;
  hash?: string;
  label: string;
  icon: typeof Home;
  group: string;
  hint?: string;
  keywords?: string;
}

const COMMANDS: Cmd[] = [
  // Story
  { href: "/", label: "Overview", icon: Home, group: "Story", hint: "Headline KPIs", keywords: "home" },
  { href: "/insights", label: "Insights — the story", icon: Sparkles, group: "Story", hint: "Seven findings", keywords: "narrative" },
  { href: "/present", label: "Defense mode", icon: Presentation, group: "Story", hint: "Big-number slides", keywords: "presentation slides" },
  // Workbench
  { href: "/analyze", label: "Analyze workbench", icon: SlidersHorizontal, group: "Workbench", hint: "Slice & re-aggregate" },
  { href: "/compare", label: "Compare two slices", icon: GitCompareArrows, group: "Workbench", hint: "Side-by-side diff" },
  // Deep dives
  { href: "/clusters", label: "Skill clusters · heatmap", icon: Layers, group: "Deep dives", hint: "21-cluster AME map" },
  { href: "/skills", label: "Hard skills · leaderboard", icon: Hash, group: "Deep dives", hint: "Top-N skills with AI lift" },
  { href: "/network", label: "Skill network", icon: Network, group: "Deep dives", hint: "Co-occurrence graph" },
  { href: "/distributions", label: "Distributions", icon: TrendingUp, group: "Deep dives", hint: "Salary · experience density" },
  { href: "/geography", label: "Geography", icon: Globe2, group: "Deep dives", hint: "State / city breakdown" },
  { href: "/premium", label: "Salary premium · OLS", icon: DollarSign, group: "Deep dives", hint: "Models A → B → C" },
  // Data
  { href: "/explorer", label: "Browse all 44k postings", icon: Table2, group: "Data" },
  { href: "/about", label: "Methodology & caveats", icon: BookOpen, group: "Data", hint: "Pipeline · models · limits" },
  // Insight jumps
  { href: "/insights", hash: "f-01", label: "Finding 1 · Headline gap", icon: ArrowRight, group: "Jump to finding" },
  { href: "/insights", hash: "f-02", label: "Finding 2 · Two flavors of AI", icon: ArrowRight, group: "Jump to finding" },
  { href: "/insights", hash: "f-03", label: "Finding 3 · Where demand concentrates", icon: ArrowRight, group: "Jump to finding" },
  { href: "/insights", hash: "f-04", label: "Finding 4 · Wage premium", icon: ArrowRight, group: "Jump to finding" },
  { href: "/insights", hash: "f-05", label: "Finding 5 · Generative AI dominates", icon: ArrowRight, group: "Jump to finding" },
  { href: "/insights", hash: "f-06", label: "Finding 6 · Germany anomaly", icon: ArrowRight, group: "Jump to finding" },
  { href: "/insights", hash: "f-07", label: "Finding 7 · Why postings matter", icon: ArrowRight, group: "Jump to finding" },
];

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const router = useRouter();

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      } else if (e.key === "/" && !open && !isInputElement(e.target)) {
        e.preventDefault();
        setOpen(true);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  useEffect(() => {
    if (open) {
      setQuery("");
      setActive(0);
    }
  }, [open]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return COMMANDS;
    return COMMANDS.filter((c) =>
      `${c.label} ${c.hint ?? ""} ${c.keywords ?? ""} ${c.group}`.toLowerCase().includes(q),
    );
  }, [query]);

  const groups = useMemo(() => {
    const map = new Map<string, Cmd[]>();
    for (const c of filtered) {
      let arr = map.get(c.group);
      if (!arr) {
        arr = [];
        map.set(c.group, arr);
      }
      arr.push(c);
    }
    return [...map.entries()];
  }, [filtered]);

  function go(c: Cmd) {
    setOpen(false);
    if (!c.href) return;
    const url = c.hash ? `${c.href}#${c.hash}` : c.href;
    router.push(url);
  }

  function onListKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      e.stopPropagation();
      setActive((i) => Math.min(filtered.length - 1, i + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      e.stopPropagation();
      setActive((i) => Math.max(0, i - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      e.stopPropagation();
      const target = filtered[active];
      if (target) go(target);
    } else if (e.key === " " || e.key === "PageDown" || e.key === "PageUp" || /^[1-9]$/.test(e.key)) {
      // Stop these from bubbling to /present's slide handler when palette is open
      e.stopPropagation();
    }
  }

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=closed]:animate-out data-[state=closed]:fade-out-0" />
        <Dialog.Content
          className="fixed left-1/2 top-1/4 z-50 w-[min(92vw,640px)] -translate-x-1/2 rounded-2xl border bg-card shadow-xl outline-none data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0"
          onKeyDown={onListKeyDown}
        >
          <Dialog.Title className="sr-only">Command palette</Dialog.Title>
          <Dialog.Description className="sr-only">
            Jump anywhere in the AI Skills Explorer.
          </Dialog.Description>
          <div className="flex items-center gap-2 border-b px-4 py-3">
            <Search className="size-4 text-muted-foreground" />
            <input
              autoFocus
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setActive(0);
              }}
              placeholder="Jump to a page or finding…"
              className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
            />
            <kbd className="rounded border bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
              esc
            </kbd>
          </div>
          <div className="max-h-[60vh] overflow-y-auto p-2">
            {groups.length === 0 ? (
              <div className="px-3 py-8 text-center text-sm text-muted-foreground">
                No matches.
              </div>
            ) : (
              groups.map(([group, items]) => (
                <div key={group} className="mb-2">
                  <div className="px-2 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    {group}
                  </div>
                  <ul>
                    {items.map((c) => {
                      const isActive = filtered[active] === c;
                      const Icon = c.icon;
                      return (
                        <li key={`${c.label}-${c.hash ?? ""}`}>
                          <button
                            type="button"
                            onMouseEnter={() => setActive(filtered.indexOf(c))}
                            onClick={() => go(c)}
                            className={cn(
                              "flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-left text-sm transition-colors",
                              isActive ? "bg-primary/10 text-foreground" : "hover:bg-muted/60",
                            )}
                          >
                            <Icon className={cn("size-4 shrink-0", isActive ? "text-primary" : "text-muted-foreground")} />
                            <span className="flex-1 truncate">
                              <span className="font-medium">{c.label}</span>
                              {c.hint ? (
                                <span className="ml-2 text-xs text-muted-foreground">{c.hint}</span>
                              ) : null}
                            </span>
                            {isActive ? (
                              <ArrowRight className="size-3.5 text-primary" />
                            ) : null}
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ))
            )}
          </div>
          <div className="flex items-center justify-between border-t px-4 py-2 text-[10px] text-muted-foreground">
            <div className="flex items-center gap-3">
              <span className="inline-flex items-center gap-1">
                <kbd className="rounded border bg-muted px-1.5 py-0.5 font-mono">↑↓</kbd>
                navigate
              </span>
              <span className="inline-flex items-center gap-1">
                <kbd className="rounded border bg-muted px-1.5 py-0.5 font-mono">↵</kbd>
                go
              </span>
            </div>
            <span className="font-mono">⌘K · /</span>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function isInputElement(target: EventTarget | null): boolean {
  if (!target || !(target instanceof HTMLElement)) return false;
  return (
    target.tagName === "INPUT" ||
    target.tagName === "TEXTAREA" ||
    target.isContentEditable
  );
}
