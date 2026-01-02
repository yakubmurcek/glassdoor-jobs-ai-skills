"use client";

import {
  useEffect,
  useState,
  useRef,
  useMemo,
  forwardRef,
  useImperativeHandle,
} from "react";
import dynamic from "next/dynamic";
import { getSkills } from "@/lib/api";
import { SkillPoint } from "@/types/api";
import { Card } from "@/components/ui/card";
import { Loader2 } from "lucide-react";

// Dynamic import for client-side only rendering of the graph
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full text-muted-foreground">
      <Loader2 className="animate-spin mr-2" /> Loading Graph Engine...
    </div>
  ),
});

const HARD_SKILL_COLOR = "#1f78b4";
const SOFT_SKILL_COLOR = "#33a02c";

interface SkillGalaxyProps {
  className?: string;
}

const SkillGalaxy = forwardRef<{ resetView: () => void }, SkillGalaxyProps>(
  ({ className }, ref) => {
    const [skills, setSkills] = useState<SkillPoint[]>([]);
    const [loading, setLoading] = useState(true);
    const graphRef = useRef<any>(null);

    useImperativeHandle(ref, () => ({
      resetView: () => {
        // Reduced padding since we have structural padding via layout anchor
        graphRef.current?.zoomToFit(400, 50);
      },
    }));

    useEffect(() => {
      const fetchData = async () => {
        try {
          const data = await getSkills();
          setSkills(data);
        } catch (err) {
          console.error("Failed to fetch skills", err);
        } finally {
          setLoading(false);
        }
      };
      fetchData();
    }, []);

    // Compute graph data
    const graphData = useMemo(() => {
      if (!skills.length) return { nodes: [], links: [] };

      // Calculate bounds to add a layout anchor
      const yValues = skills.map((s) => s.y);
      const minY = Math.min(...yValues);
      const maxY = Math.max(...yValues);
      const height = maxY - minY;

      // Create standard nodes
      const nodes = skills.map((s) => ({
        id: s.id,
        label: s.label,
        group: s.type,
        val: 1, // size
        frequency: s.frequency, // data for scaling
        // Use PCA projection from backend as fixed positions
        fx: s.x,
        fy: s.y,
        x: s.x, // ForceGraph consumes these
        y: s.y,
      }));

      // Add an invisible anchor node to the layout to effectively shift the view.
      // We want the content (cluster) to visually move DOWN.
      // This means we need the camera to look HIGHER.
      // To force the center HIGHER, we add an anchor ABOVE the cluster.
      // Assuming Cartesian coordinates (Y increases UP), MaxY is Top.
      // So we add to MaxY.
      nodes.push({
        id: "layout-anchor-helper",
        label: "",
        group: "layout-helper",
        val: 0,
        frequency: 0,
        fx: 0, // Centered horizontally
        fy: maxY + height * 0.5,
        x: 0,
        y: maxY + height * 0.5,
      });

      return {
        nodes,
        links: [], // No links needed if we have fixed positions
      };
    }, [skills]);

    // If we have no links, nodes will fly away. We need centripetal force.
    // We can configure d3 forces via graphRef.

    return (
      <div
        className={`relative w-full h-[600px] border rounded-xl overflow-hidden bg-slate-950 text-slate-100 ${
          className || ""
        }`}
      >
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center z-10 bg-slate-950/80">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
            <span className="ml-2">Loading Universe...</span>
          </div>
        )}

        {!loading && (
          // @ts-ignore
          <ForceGraph2D
            ref={graphRef}
            graphData={graphData}
            // Removed nodeAutoColorBy to ensure custom renderer is the source of truth
            backgroundColor="#020617" // slate-950
            enableNodeDrag={false}
            // Custom rendering to show labels permanently
            nodeCanvasObject={(node: any, ctx, globalScale) => {
              if (node.group === "layout-helper") return;

              const label = node.label;
              const fontSize = 12 / globalScale;
              ctx.font = `${fontSize}px Sans-Serif`;

              // Draw Node
              // Scale size by frequency (min 3, max 10)
              const baseSize = 3;
              const freq = node.frequency || 1;
              // frequency range is likely 1-20 based on dict variants
              const size = Math.min(12, baseSize + Math.log2(freq) * 2);

              ctx.beginPath();
              ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
              // Explicit color assignment matching the legend
              ctx.fillStyle =
                node.group === "hard" ? HARD_SKILL_COLOR : SOFT_SKILL_COLOR;
              ctx.fill();

              // Draw Text
              ctx.textAlign = "center";
              ctx.textBaseline = "middle";
              ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
              ctx.fillText(label, node.x, node.y + size + fontSize); // Draw below the node
            }}
            nodeCanvasObjectMode={() => "replace"} // We carry out the full render
            d3VelocityDecay={0.1}
            cooldownTicks={0}
            onEngineStop={() => {
              // Only auto-zoom once on initial load
              if (!graphRef.current?.hasZoomed) {
                // Reduced padding since we have structural padding now
                graphRef.current?.zoomToFit(400, 50);
                graphRef.current.hasZoomed = true;
              }
            }}
          />
        )}

        <div className="absolute bottom-4 left-4 pointer-events-none">
          <Card className="p-4 bg-slate-900/80 border-slate-800 backdrop-blur-sm pointer-events-auto text-slate-100">
            <h3 className="font-bold text-slate-100 mb-2">Legend</h3>
            <div className="flex gap-2">
              <div className="flex items-center text-xs">
                <span
                  className="w-3 h-3 rounded-full mr-2"
                  style={{ backgroundColor: HARD_SKILL_COLOR }}
                ></span>{" "}
                Hard Skills
              </div>
              <div className="flex items-center text-xs">
                <span
                  className="w-3 h-3 rounded-full mr-2"
                  style={{ backgroundColor: SOFT_SKILL_COLOR }}
                ></span>{" "}
                Soft Skills
              </div>
            </div>
            <div className="mt-2 text-xs text-slate-400">
              {skills.length} skills loaded
            </div>
          </Card>
        </div>
      </div>
    );
  }
);

SkillGalaxy.displayName = "SkillGalaxy";
export default SkillGalaxy;
