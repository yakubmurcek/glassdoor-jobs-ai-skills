"use client";

import SkillGalaxy from "@/components/SkillGalaxy";
import SkillSearch from "@/components/SkillSearch";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";

import { useRef } from "react";

export default function Home() {
  const skillGalaxyRef = useRef<{ resetView: () => void }>(null);
  return (
    <main className="flex min-h-screen flex-col bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950/50 backdrop-blur sticky top-0 z-50">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
              AI Skills Universe
            </span>
            <Badge
              variant="outline"
              className="text-slate-400 border-slate-700"
            >
              Thesis v1.0
            </Badge>
          </div>

          <nav className="flex items-center gap-4">
            <Link
              href="/"
              className="text-sm font-medium hover:text-blue-400 transition-colors"
            >
              Explorer
            </Link>
            <Link
              href="/analyzer"
              className="text-sm font-medium text-slate-400 hover:text-blue-400 transition-colors"
            >
              Job Analyzer
            </Link>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 container mx-auto px-4 py-8">
        {/* Intro Section */}
        <div className="mb-8">
          <h1 className="text-4xl font-extrabold tracking-tight mb-2">
            The Geometry of <span className="text-blue-400">Labor Demand</span>
          </h1>
          <p className="text-slate-400 max-w-2xl text-lg">
            Visualizing the semantic relationships between thousands of skills
            extracted from US job descriptions. Explore the distinction between
            &quot;AI-Native&quot; engineering and &quot;AI-Adopter&quot; usage.
          </p>
        </div>

        {/* Galaxy Visualization */}
        <div className="grid gap-6">
          <Card className="bg-slate-900 border-slate-800 shadow-2xl overflow-hidden">
            <CardHeader>
              <CardTitle className="flex justify-between items-center text-slate-100">
                <span>Skill Embedding Space (t-SNE)</span>
                <div className="space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="bg-slate-950 border-slate-700 text-slate-300 hover:bg-slate-800"
                    onClick={() => skillGalaxyRef.current?.resetView()}
                  >
                    Reset View
                  </Button>
                </div>
              </CardTitle>
              <CardDescription className="text-slate-500">
                2D Projection of 384-dimensional embeddings
                (SentenceTransformers)
              </CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <SkillGalaxy ref={skillGalaxyRef} />
            </CardContent>
          </Card>

          {/* Semantic Search Demo */}
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="bg-slate-900 border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100">
                  Semantic Skill Search
                </CardTitle>
                <CardDescription className="text-slate-500">
                  Test the embedding model. Type query like &quot;generative
                  ai&quot; to see related skills.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <SkillSearch />
              </CardContent>
            </Card>

            <Card className="bg-slate-900 border-slate-800">
              <CardHeader>
                <CardTitle className="text-slate-100">
                  About the Model
                </CardTitle>
                <CardDescription className="text-slate-500">
                  Architecture details
                </CardDescription>
              </CardHeader>
              <CardContent className="text-slate-400 space-y-2 text-sm">
                <p>
                  <strong>Model:</strong> all-MiniLM-L6-v2
                  (SentenceTransformers)
                </p>
                <p>
                  <strong>Dimensions:</strong> 384
                </p>
                <p>
                  <strong>Vector Store:</strong> ChromaDB / In-Memory
                </p>
                <p>
                  This lightweight model runs locally and powers both the
                  visualization and the fuzzy matching logic used in the thesis.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </main>
  );
}
