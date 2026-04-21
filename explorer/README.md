# AI Skills Explorer — Thesis Companion

Interactive web app that visualizes findings from the master's thesis on **AI skill requirements in IT job postings**, comparing the United States, Germany, and India.

Built as a static-friendly Next.js 15 app that reads pre-aggregated JSON snapshots of the Stata analysis. Everything lives under `explorer/` and is independent of the main thesis codebase.

---

## Stack

- **Next.js 15 · React 19 · TypeScript**
- **Tailwind v4** + shadcn/ui primitives (new-york style)
- **Recharts** for bar / stacked / error-bar charts, custom SVG for the cluster heatmap
- **TanStack Table v8** for the dataset browser
- **nuqs** for URL-synced filter state — every view is a copy-pasteable link
- **Python / pandas** for the offline data build step (`scripts/build_data.py`)

## Sections

| Route | What it shows |
|---|---|
| `/` | Overview — 6 KPI cards, stacked tier composition, headline findings |
| `/tiers` | AI tier composition per country, filtered by country × tier |
| `/job-families` | AI share per job family — small multiples with 5 sort modes |
| `/clusters` | 21 × 3 cross-country heatmap of logit AMEs with drill-down side panel (top hard skills per cluster) |
| `/premium` | Salary premium OLS decomposition (Model A → B → C) plus the cross-country comparison |
| `/explorer` | Sampled dataset browser (500 stratified rows), searchable, sortable, CSV export |
| `/about` | Methodology, caveats, data snapshot metadata |

All filters sync to the URL via `nuqs` so any slice can be shared.

---

## Local development

```bash
# from the explorer/ directory
pnpm install
pnpm run build-data     # regenerate JSON snapshots from Stata output
pnpm dev                # http://localhost:3000
```

The `build-data` script calls `uv run python scripts/build_data.py` and expects:

- Stata run output at `../analysis/stata/output/thesis_final_run_*/charts_data/*.csv`
- Raw datasets at `../data/outputs/{us,de,in}_relevant_ai_stata.csv`
- Cluster → skills mapping at `../data/outputs/us_relevant_ai_stata_cluster_skills.csv`

Outputs land in `public/data/` as typed JSON (validated via the `lib/data/types.ts` interfaces).

---

## Production build

```bash
pnpm build              # prebuild hook runs build-data automatically
pnpm start              # node server
```

Build produces fully static pages — ~106 kB shared JS, each route 2–20 kB on top. Deploy target is Vercel by default; any static host works because the app has no server routes.

---

## Repository layout

```
explorer/
├── app/                  # Next.js 15 App Router pages
├── components/
│   ├── charts/           # tier-stacked-bar, cluster-heatmap
│   ├── dataset/          # TanStack-based jobs table
│   ├── filters/          # country / tier chip filters
│   ├── kpi/              # dashboard KPI cards
│   ├── layout/           # sidebar, mobile nav, page header
│   └── ui/               # shadcn primitives
├── lib/
│   ├── constants.ts      # country / tier / color palette (mirrors analysis/charts/build_charts.py)
│   ├── data/             # typed loaders + TypeScript interfaces
│   ├── state/            # zustand store (unused in current pages; reserved for cross-filter work)
│   └── utils.ts          # cn(), formatters, sig-colour helper
├── public/data/          # generated JSON snapshots (product of build_data.py)
├── scripts/
│   └── build_data.py     # Stata CSVs → JSON pipeline
└── README.md
```

## Caveats

- DE = Germany (not Czechia), confirmed against the thesis `docs/prakticka_cast_3.md`.
- The sampled 500-row dataset is a qualitative sanity check, not the full 69k-posting dataset — those are kept in `data/outputs/` and not shipped to the browser.
- Cluster keys occasionally differ between the Stata export (`g5_*.csv`) and the dictionary (`cluster_skills.csv`) due to separator-escaping — the `/clusters` page joins on a normalised label, not on the raw key.

## License

Part of the master's thesis project. Use for review / reproduction within that scope.
