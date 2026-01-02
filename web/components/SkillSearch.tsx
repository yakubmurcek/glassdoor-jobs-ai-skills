"use client";

import { useState } from "react";
import { Search } from "lucide-react";
import { searchSkills } from "@/lib/api";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useDebounce } from "@/hooks/use-debounce";
import { SearchResult } from "@/types/api";
import { useEffect } from "react";

export default function SkillSearch() {
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebounce(query, 500);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!debouncedQuery.trim()) {
      setResults([]);
      return;
    }

    const fetch = async () => {
      setLoading(true);
      try {
        const data = await searchSkills(debouncedQuery);
        setResults(data);
      } catch (e) {
        console.error("Search failed", e);
      } finally {
        setLoading(false);
      }
    };

    fetch();
  }, [debouncedQuery]);

  return (
    <div className="w-full max-w-2xl mx-auto space-y-4">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4" />
        <Input
          className="pl-10 bg-slate-900 border-slate-700 text-slate-100 placeholder:text-slate-500"
          placeholder="Search for skills (e.g. 'building llm agents')..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      {results.length > 0 && (
        <div className="grid gap-2">
          {results.map((res) => (
            <Card
              key={res.skill}
              className="bg-slate-900/50 border-slate-800 hover:bg-slate-800/50 transition-colors"
            >
              <CardContent className="p-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-slate-200">
                    {res.skill}
                  </span>
                  <Badge
                    variant={res.type === "hard" ? "default" : "secondary"}
                    className="text-[10px] h-5"
                  >
                    {res.type}
                  </Badge>
                </div>
                <div className="text-xs text-slate-500 font-mono">
                  {(res.score * 100).toFixed(1)}% Match
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
