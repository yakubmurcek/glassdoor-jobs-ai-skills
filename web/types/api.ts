export interface SkillPoint {
  id: string;
  label: string;
  type: "hard" | "soft";
  embedding: number[];
  x: number;
  y: number;
  frequency: number;
}

export interface SearchResult {
  skill: string;
  type: "hard" | "soft";
  score: number;
}

export interface SearchResponse {
  results: SearchResult[];
}
