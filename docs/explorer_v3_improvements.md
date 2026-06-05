# Explorer v3 — Improvement Backlog

Status as of 2026-05-09. Captures the next set of upgrades for the thesis-companion app at `/explorer/`. Numbered, with rationale; prioritised into bands so we can ship in order without scope creep.

The product today (v2) ships:

- 13 routes, row-level dataset (~45k postings) shipped to the browser.
- Storytelling (`/insights`), workbench (`/analyze`), comparison (`/compare`), 5 deep-dive views (clusters, skills, network, distributions, geography), salary OLS (`/premium`), defense mode (`/present`), dataset browser, command palette (⌘K).
- Sticky TOC + scroll-spy on /insights, copy-link buttons per finding.

What follows is what would matter most to a reader (the advisor) and a self-serve researcher (other thesis readers, future reviewers, the user).

---

## Band A — Highest leverage (ship first)

### A1. Stata re-export + salary calculator (`/premium/calculator`)
**Why.** Already memoised in `project_explorer_v2_deferred_stata.md`. The single biggest "wow" feature still missing. Given a posting spec (country × tier × family × seniority × edu × cluster set) → predicted log-salary + 95% PI + per-variable marginal effects.

**What it needs.**
- Append `eststo ols_C_full` + `esttab cells("b se ci_l ci_u p")` for the full Model C coefficient vector.
- Save `e(V)` as CSV so we can compute SE on prediction.
- New `lib/calculator.ts` — pure function `predict(spec) → {logSalary, salary, pi95, marginals[]}`.

**Acceptance.** Tooltip on every premium number cites the model's N + R²; the calculator page returns a real number with PI for any reasonable spec.

### A2. Time dimension (`/timeline`)
**Why.** Postings carry `date_posted` but it's not surfaced anywhere. The thesis story is about an *AI shock* — it's natural to show 2024-Q4 → 2025-Q1 → … movement. Currently we treat the dataset as cross-section.

**What it needs.**
- Add `date_posted` (or quarter-bucket) to compact rows.
- Time-series chart of AI mention rate per country, tier, top cluster.
- Annotation marker on ChatGPT release (Nov 2022) and major model releases (GPT-4, Claude 3, etc.).

**Acceptance.** A reader can see the AI hiring slope per country and ask "is the gap widening or narrowing?"

### A3. Czechia replacement for Germany
**Why.** Thesis title says **US vs Czechia**. The dataset uses DE because CZ scrape coverage was thin. If CZ data is ready or can be re-scraped, swap DE → CZ. Otherwise, surface this caveat explicitly on every country chart (currently only on `/about`).

**What it needs.** Either a CZ scrape pass or a prominent on-chart disclaimer reading "Germany used as European-market proxy — see /about".

**Acceptance.** A reader who lands on `/insights` directly knows Germany is a proxy; or, ideally, the data is actually CZ.

### A4. Reproducibility manifest export
**Why.** Researchers want to cite specific slices. Today the URL serialises filters but there's no canonical "this is the calculation that produced 16.15%" handoff.

**What it needs.**
- "Export this view" button on every page → JSON containing: filters, N, point estimates, 95% CIs, model name, snapshot SHA.
- A short BibTeX snippet ("Murček 2026, accessed via explorer SHA xyz") for thesis citations.

**Acceptance.** Reader can paste the JSON into a footnote and another reader can replay the slice exactly.

### A5. Cluster definition viewer
**Why.** "Generative AI" cluster sounds intuitive but the reader has no way to know it includes "rag", "embeddings", "vector db", etc. and excludes "computer vision". The full mapping exists in `data/outputs/us_relevant_ai_stata_cluster_skills.csv` but isn't surfaced.

**What it needs.** Click any cluster on `/clusters` or `/network` → side panel listing the dictionary terms that compose it, with their individual frequencies. Already partially done on `/clusters` — extend to `/network` and `/skills`.

**Acceptance.** Every cluster name on the site is one click from its full term list.

---

## Band B — Researcher polish (ship second)

### B1. CI overlap badge on `/compare`
Two slices side by side, but CIs aren't compared. Add an explicit "CIs overlap" / "CIs distinct" indicator with the gap in pp.

### B2. "Show calculation" expando on every aggregate
For each KPI / bar / cell, a tiny "▾" that opens a panel: numerator/denominator counts, formula, sample size, source model.

### B3. Statistical test runner
On `/compare`, a "Run test" button — chi-squared on tier mix, Welch t-test on mean salary, Mann-Whitney on experience. Output: statistic, p-value, effect size, footnote about assumptions.

### B4. Power analysis side-tool
Given a slice's N, baseline rate, and desired alpha = 0.05, what minimum-detectable-effect would we have 80% power for? Useful for explaining why DE / small-state slices look "noisy".

### B5. Saved views (localStorage)
"Bookmark this slice" → list at `/analyze`. URL already supports it; this is just a per-user pin list.

### B6. Annotations on charts
Researcher writes a note ("⚠ this includes contractor postings"); note is stored in localStorage keyed by the chart's URL fragment. Toggle visibility.

### B7. Citation export per finding
Each `/insights` finding carries a "Cite this" button → BibTeX entry referencing the thesis paragraph + the live URL.

---

## Band C — Visualization upgrades

### C1. Choropleth on `/geography`
Real US map with state shading, plus DE/CZ/IN city scatter. Use `topojson` + a static `us-atlas` import (no live geocoding). Mobile-friendly because vector.

### C2. Animated chart transitions
When filters change, bars/cells should tween rather than snap. Recharts has it; SVG paths in `/network` and `/distributions` need explicit `<animate>` or a state-diff system.

### C3. Story overlay on `/insights`
Optional "guided tour" mode that pans the page, highlights a chart, narrates the next finding (just text, no audio). Press → to advance.

### C4. Dark mode
CSS variables already exist. Add a toggle in the sidebar; respect `prefers-color-scheme`.

### C5. Print stylesheet
Every page should produce a clean A4/letter print. Useful for thesis appendix and for reviewers who annotate on paper.

### C6. Cross-page slicer pinning
Filters set on `/analyze` should optionally carry over to `/distributions` and `/geography`. Today URL-state is per-page.

### C7. Highlight-on-hover across panels
Hover a "Data & AI" family on `/analyze` → fades all other panels' bars except Data & AI. Coordinated highlighting is a common dataviz upgrade.

### C8. OG-image / shareable card per slice
When you share a URL, the preview should render the slice's headline number ("Senior Data & AI · US · 23.4% salary premium · n=412") not the generic site card.

---

## Band D — Performance & scale

### D1. Compress / columnarise `rows.json`
Currently 26 MB JSON. Switch to a columnar binary format (e.g. Apache Arrow IPC, or just typed arrays in a single binary) to drop transfer size 3–5×.

### D2. Web Worker for aggregations
`buildSkillStats`, `computeCooc`, `groupBy` block the main thread for 50–150 ms each. Offload via Worker so UI never freezes during a country-toggle.

### D3. Long-cache + revalidation on JSON
Set `Cache-Control: public, max-age=31536000, immutable` on `/data/rows.json` (filename already has run-dir-derivable hash if we want). Today every cold visit re-downloads.

### D4. Code-split per-finding charts
`/insights` first-load = 236 kB. Each `<FindingCard>` chart could lazy-load. Trade off: scroll might trigger jank.

---

## Band E — Data quality / authoring

### E1. Surface skill normalisation mappings
The dictionary collapses "py" → "python", "k8s" → "kubernetes", etc. That mapping is invisible. A `/about/normalisation` page listing every alias.

### E2. Mistake / disagreement viewer
500 hand-checked postings exist (per /about). Show them: which postings did the LLM and dictionary disagree on? Useful for the methodology section.

### E3. Cluster re-labelling tool
Let the researcher rename a cluster ("Generative AI" → "Foundation Models") and see all charts update. Stored locally, exportable as a patch.

### E4. Per-cluster confusion matrix
For the multinomial logit on US, show predicted-vs-actual tier confusion. Surfaces where the model is uncertain.

---

## Band F — Accessibility & screen-reader

### F1. ARIA-described charts
Every chart needs an `aria-label` summarising the headline numerically — "AI mention rate: US 20.6%, Germany 18.3%, India 6.3%."

### F2. Keyboard heatmap navigation
Cells in `/clusters` and the family×seniority heatmap should be tabbable, with arrow-key navigation between adjacent cells.

### F3. Higher-contrast palette
Audit current colour pairs against WCAG AA (4.5:1 for body, 3:1 for large text). The DE purple on cream may fail.

### F4. Reduced-motion fallback
`prefers-reduced-motion` should disable the hero blur animations and the slide transitions on `/present`.

---

## Band G — Long shots / nice-to-haves

### G1. Multi-thesis comparison mode
Slot in another study's dataset (Bell & Co 2024, Acemoglu & Restrepo benchmarks) and compare side by side.

### G2. Job-posting full-text viewer
On `/explorer`, click a row → open the full description with skill mentions highlighted in-context. Today only `jobs_sample.json` ships sample text.

### G3. Embed widgets for blog/thesis
Generate `<iframe>` snippets for any chart so the thesis or a blog post can embed live numbers.

### G4. Telemetry (privacy-respecting)
Plausible / Fathom — see which findings get the most engagement. No PII.

### G5. Multi-lingual UI (CS / EN)
Toggle so the Czech advisor reads the explorer in Czech. Most thesis readers will read the thesis in CS but the explorer in EN; either is fine, but offering both is polite.

---

## What I would actually ship next, ranked

If we have one weekend:

1. A1 (calculator) — highest "wow" + already designed.
2. A5 (cluster definition viewer) — small, makes every cluster citation honest.
3. A4 (reproducibility manifest) — small, makes the explorer citable.
4. C5 (print stylesheet) — small, helps the advisor mark up on paper.
5. F1 (ARIA chart labels) — small, lifts professional polish.

If we have a week:

6. A2 (time dimension) — the most-asked question reviewers will have ("is the gap widening?").
7. C1 (choropleth) — visual upgrade matched by a real analytical purpose.
8. B2 ("show calculation" expando) — every number self-documenting.
9. D2 (web worker) — pre-empts complaints when the dataset grows past 100k.

If we have a month:

10. A3 (Czechia data swap) — gold-standard since the title is US-vs-CZ.
11. Everything in B (researcher polish) — turns the explorer from "interactive thesis" into "thesis defence platform".

---

## Definitely *not* doing

- **Authentication / accounts.** This is a research artefact, not a SaaS.
- **Server-side database.** Static JSON + browser compute is the right architecture; do not migrate to a backend until the dataset crosses ~1M postings.
- **AI-generated commentary.** Every number must trace to a model and an N. No LLM glosses on top of charts.
- **Polling-based realtime updates.** The dataset is a research snapshot, not a live feed.

---

*Maintained by Yakub Murček. Add an entry by appending to the relevant band, prefix the next available number, and date the change in this footer.*

*Last touched: 2026-05-09 — explorer v2 shipped, audit pass complete, this backlog seeded.*
