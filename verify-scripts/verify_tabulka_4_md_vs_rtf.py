"""
Verify that numeric values in prakticka_cast_3.md Table 4
match the Tabulka_4_Mlogit_AI_Tier.rtf output.

The MD table cites AMEs in percentage-points (AME*100).
The RTF has AMEs in decimal form, rounded to 3 decimals.
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path("/Users/yakub/Projects/glassdoor-jobs-ai-skills")
RTF = ROOT / "analysis/stata/output/thesis_final_run_21_Apr_2026_23-18-05/Tabulka_4_Mlogit_AI_Tier.rtf"
MD = ROOT / "docs/prakticka_cast_3.md"

# ----------------------------- Parse RTF -----------------------------
# Schema: one "variable row" (AME) followed by one "SE row" (in parens).
# Columns: USA-None, USA-Integ, USA-Applied, DE-None, DE-Integ, DE-Applied, IN-None, IN-Integ, IN-Applied

rtf_text = RTF.read_text()
lines = rtf_text.split("\n")


def extract_cells(line: str) -> list[str]:
    """Extract cell text values from an RTF table row."""
    # Find all {....} groups after \intbl\qc
    return re.findall(r"\\pard\\intbl\\qc\s*\{([^}]*)\}", line)


# Map: variable display-name -> (line with AMEs, line with SEs)
rtf_rows: dict[str, tuple[list[str], list[str]]] = {}

i = 0
while i < len(lines):
    line = lines[i]
    # Identify a label line: first cell is "\ql {Variable Name}" with non-empty text
    m = re.search(r"\\pard\\intbl\\ql\s*\{([^}]+)\}", line)
    if m:
        label = m.group(1).strip()
        if label and not label.startswith("\\i "):
            cells = extract_cells(line)
            if len(cells) == 9:
                # next line should be SE row (empty first cell, 9 paren cells)
                if i + 1 < len(lines):
                    se_cells = extract_cells(lines[i + 1])
                    if len(se_cells) == 9 and all(c.startswith("(") for c in se_cells):
                        rtf_rows[label] = (cells, se_cells)
                        i += 2
                        continue
                # Some rows have no SE row (e.g. Ano / Ano / Ano, N, Pseudo)
                rtf_rows[label] = (cells, [])
    i += 1


# Print summary
print("=" * 70)
print("RTF ROWS PARSED:")
print("=" * 70)
for label, (ame, se) in rtf_rows.items():
    print(f"  {label}: AME={len(ame)} SE={len(se)}")
print(f"\nTotal rows: {len(rtf_rows)}")


# ----------------------------- Parse MD -----------------------------
md_text = MD.read_text()

# Extract Tabulka 4 block
m = re.search(r"\*\*Tabulka 4.*?\*\*(.*?)_AI Int\. = AI Integration.*?_", md_text, re.DOTALL)
if m is None:
    raise RuntimeError("Could not locate Tabulka 4 block in MD")
md_block = m.group(1)

# Parse markdown table rows
md_rows: dict[str, list[str]] = {}
for ln in md_block.split("\n"):
    ln = ln.strip()
    if not ln.startswith("|"):
        continue
    parts = [p.strip() for p in ln.strip("|").split("|")]
    if len(parts) < 7:
        continue
    # Skip header/separator
    if parts[0].startswith("---") or parts[0].startswith("Skill") or parts[0] == "":
        continue
    # parts: [label, USA AI Int, USA App, DE AI Int, DE App, IN AI Int, IN App]
    md_rows[parts[0]] = parts[1:7]


print("\n" + "=" * 70)
print("MD ROWS PARSED:")
print("=" * 70)
for k, v in md_rows.items():
    print(f"  {k}: {v}")


# ----------------------------- Compare -----------------------------
# MD columns order: USA Int, USA App, DE Int, DE App, IN Int, IN App
# RTF columns order (for "Integ"/"Applied" sub): cells[1]=USA Integ, cells[2]=USA Applied,
# cells[4]=DE Integ, cells[5]=DE Applied, cells[7]=IN Integ, cells[8]=IN Applied
RTF_COL_FOR_MD = {
    "USA AI Int.": 1,
    "USA App.": 2,
    "Německo AI Int.": 4,
    "Německo App.": 5,
    "Indie AI Int.": 7,
    "Indie App.": 8,
}
MD_COL_ORDER = list(RTF_COL_FOR_MD.keys())

# MD -> RTF label mapping  (map MD variable name to RTF label)
LABEL_MAP = {
    "Generative AI": "Generative AI",
    "Data Science / ML": "Data Science / ML",
    "Dynamic Web": "Dynamic Web",
    "Cloud Computing": "Cloud Computing",
    "Data Engineering": "Data Engineering",
    "BI & Analytics": "BI & Analytics",
    "Frontend Development": "Frontend Development",
    "Enterprise Platforms": "Enterprise Platforms",
    "DevOps & Containers": "DevOps & Containers",
    "Backend Development": "Backend Development",
    "Systems Programming": "Systems Programming",
    "Architecture & Methods": "Architecture & Methods",
    "Security & Identity": "Security & Identity",
    "Mobile & Desktop": "Mobile & Desktop",
    "Testing / QA & Debugging": "Testing / QA & Debugging",
    "Databases & Storage": "Databases & Storage",
    "Networking": "Networking",
    "OS & Embedded": "OS & Embedded",
    "Certifications": "Certifications",
    "Scripting / Shell": "Scripting / Shell",
    "Enterprise / Managed": "Enterprise / Managed",
}

NUM_STARS_PATTERN = re.compile(r"([+\-−]?\s*[0-9]+[.,][0-9]+)\s*((?:\\?\*)+|ns)?", re.IGNORECASE)


def parse_md_cell(s: str) -> tuple[float, str]:
    """Parse a markdown cell like '+23,3 \\*\\*\\*' or '−1,4 \\*' or '+0,8 ns' -> (val, stars)."""
    s_clean = s.replace("\\*", "*").replace(" ", "").replace(",", ".").replace("−", "-")
    m = re.match(r"([+\-]?[0-9]+\.[0-9]+)(\*+|ns)?", s_clean)
    if not m:
        raise ValueError(f"Cannot parse MD cell: {s!r}")
    val = float(m.group(1))
    star = m.group(2) or ""
    if star == "ns":
        star = ""
    return val, star


def parse_rtf_cell(s: str) -> tuple[float, str]:
    """Parse RTF cell like '-0.332{\\super ***}' or '0.011' -> (val, stars)."""
    # remove super markers
    s2 = s.replace("{\\super ", "").replace("}", "").replace(" ", "")
    # pull out stars from trailing
    m = re.match(r"(-?[0-9]+\.[0-9]+)(\**)$", s2)
    if not m:
        raise ValueError(f"Cannot parse RTF cell: {s!r} -> {s2!r}")
    return float(m.group(1)), m.group(2)


# Perform comparison
print("\n" + "=" * 70)
print("COMPARISON: MD citations (p.p.) vs RTF AMEs (decimal × 100)")
print("=" * 70)

mismatches = []
total_checked = 0

for md_label, rtf_label in LABEL_MAP.items():
    if md_label not in md_rows:
        print(f"  [MISSING IN MD] {md_label}")
        continue
    if rtf_label not in rtf_rows:
        print(f"  [MISSING IN RTF] {rtf_label}")
        continue
    md_cells = md_rows[md_label]
    rtf_ame, _ = rtf_rows[rtf_label]

    for col_idx, md_col_label in enumerate(MD_COL_ORDER):
        rtf_idx = RTF_COL_FOR_MD[md_col_label]
        md_val_str = md_cells[col_idx]
        rtf_cell = rtf_ame[rtf_idx]

        try:
            md_val, md_stars = parse_md_cell(md_val_str)
            rtf_val, rtf_stars = parse_rtf_cell(rtf_cell)
        except ValueError as e:
            print(f"  PARSE-ERROR {md_label} / {md_col_label}: {e}")
            continue

        total_checked += 1

        # Compare: MD is in p.p., RTF is decimal. rtf_val * 100 == md_val
        rtf_as_pp = rtf_val * 100
        # Tolerance: 0.05 (one last-digit unit)
        val_ok = abs(rtf_as_pp - md_val) <= 0.06
        # Sign
        sign_ok = (rtf_as_pp >= 0) == (md_val >= 0) or abs(rtf_as_pp) < 0.005
        # Star significance check
        star_ok = md_stars == rtf_stars

        if not (val_ok and star_ok):
            mismatches.append(
                (md_label, md_col_label, md_val_str, rtf_cell, rtf_as_pp, md_stars, rtf_stars)
            )

print(f"\nTotal MD cells checked: {total_checked}")
print(f"Mismatches: {len(mismatches)}")
if mismatches:
    print("\nDetailed mismatches:")
    for row in mismatches:
        print(f"  {row}")
else:
    print("\nALL CELLS MATCH")


# ----------------------------- Verify N and Pseudo R² -----------------------------
print("\n" + "=" * 70)
print("FOOTER VERIFICATION")
print("=" * 70)

if "N" in rtf_rows:
    ns = rtf_rows["N"][0]
    print(f"RTF N: {ns}")
if "Pseudo R\\u178?" in rtf_rows:
    r2 = rtf_rows["Pseudo R\\u178?"][0]
    print(f"RTF Pseudo R²: {r2}")

# MD expected:
# N: 17 848 / 17 848 / 6 402 / 6 402 / 14 186 / 14 186
# Pseudo R²: 0,323 / 0,323 / 0,361 / 0,361 / 0,518 / 0,518
print("MD citations for N:",
      md_rows.get("N"))
print("MD citations for Pseudo R²:",
      md_rows.get("Pseudo R²"))

# Also verify Stata log values
print("\nLog ground truth:")
print("  N: US=17,848 DE=6,402 IN=14,186")
print("  Pseudo R²: US=0.3226→0.323, DE=0.3608→0.361, IN=0.5182→0.518")
