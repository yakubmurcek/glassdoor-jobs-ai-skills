#!/usr/bin/env python3
"""Generates an RTF appendix listing the definitions of job_family and skill_cluster.

Source of truth:
  - ai_skills/cli.py  -> _JOB_FAMILY_PATTERNS  (job_family)
  - ai_skills/skills_dictionary.py  -> _CATEGORIES / SKILL_TO_FAMILY  (skill_cluster)
  - docs/navrh_agregace_kategorii.md  (aggregation rules applied before modelling)

Output: docs/Priloha_Kategorie_JobFamily_SkillClusters.rtf
"""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

from ai_skills.skills_dictionary import SKILL_TO_FAMILY, _CATEGORIES


def _load_job_family_patterns() -> list[tuple[str, str]]:
    """Extract _JOB_FAMILY_PATTERNS from ai_skills/cli.py via AST (it lives inside a function)."""
    src = (Path(__file__).resolve().parent.parent / "ai_skills" / "cli.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_JOB_FAMILY_PATTERNS"
        ):
            return [
                (ast.literal_eval(el.elts[0]), ast.literal_eval(el.elts[1]))
                for el in node.value.elts  # type: ignore[attr-defined]
            ]
    raise RuntimeError("_JOB_FAMILY_PATTERNS not found in ai_skills/cli.py")


_JOB_FAMILY_PATTERNS = _load_job_family_patterns()


OUT = Path(__file__).resolve().parent.parent / "docs" / "Priloha_Kategorie_JobFamily_SkillClusters.rtf"


def rtf_escape(text: str) -> str:
    """Escape a Python str for RTF body content (codepage 1250 doc, non-ASCII -> \\uN?)."""
    buf = []
    for ch in text:
        code = ord(ch)
        if ch == "\\":
            buf.append("\\\\")
        elif ch == "{":
            buf.append("\\{")
        elif ch == "}":
            buf.append("\\}")
        elif code < 128:
            buf.append(ch)
        else:
            # RTF unicode escape: signed 16-bit int followed by '?'
            signed = code if code < 32768 else code - 65536
            buf.append(f"\\u{signed}?")
    return "".join(buf)


def safe_col(fam: str) -> str:
    s = fam.lower().replace(" ", "_").replace("&", "_").replace(",", "")
    return "cluster_" + s.replace("__", "_").strip("_")  # replicates cli.py behaviour; note: single pass


def actual_col(fam: str) -> str:
    """Replicates ai_skills/cli.py:488-491 exactly (single pass of __ -> _)."""
    s = fam.lower().replace(" ", "_").replace("&", "_").replace(",", "")
    s = s.replace("__", "_").strip("_")
    return f"cluster_{s}"


# ---------- Build the RTF ----------
HEADER = (
    r"{\rtf1\ansi\ansicpg1250\uc1\deff0"
    r"{\fonttbl{\f0\froman Times New Roman;}{\f1\fmodern Courier New;}{\f2\fswiss Arial;}}"
    r"{\colortbl;\red0\green0\blue0;\red80\green80\blue80;}"
    r"\paperw11906\paperh16838\margl1134\margr1134\margt1134\margb1134"
    r"\fs22\f0\lang1029"
)
FOOTER = r"}"


def h1(text: str) -> str:
    return r"\pard\sb240\sa120\keepn\f0\b\fs32 " + rtf_escape(text) + r"\b0\fs22\par" + "\n"


def h2(text: str) -> str:
    return r"\pard\sb180\sa80\keepn\f0\b\fs26 " + rtf_escape(text) + r"\b0\fs22\par" + "\n"


def h3(text: str) -> str:
    return r"\pard\sb140\sa60\keepn\f0\b\fs24 " + rtf_escape(text) + r"\b0\fs22\par" + "\n"


def p(text: str, *, italic: bool = False) -> str:
    style = r"\i " if italic else ""
    end = r"\i0" if italic else ""
    return r"\pard\sa100\ql " + style + rtf_escape(text) + end + r"\par" + "\n"


def code_inline(text: str) -> str:
    return r"{\f1 " + rtf_escape(text) + r"}"


def para_with_code(chunks: list[tuple[str, str]]) -> str:
    """chunks: list of (kind, text) where kind in {'t','c','b'} (text / code / bold)."""
    out = [r"\pard\sa100\ql "]
    for kind, t in chunks:
        if kind == "c":
            out.append(r"{\f1 " + rtf_escape(t) + r"}")
        elif kind == "b":
            out.append(r"{\b " + rtf_escape(t) + r"}")
        else:
            out.append(rtf_escape(t))
    out.append(r"\par" + "\n")
    return "".join(out)


def bullet(text_chunks: list[tuple[str, str]]) -> str:
    out = [r"\pard\fi-283\li283\sa60\ql {\f2\'95}\tab "]
    for kind, t in text_chunks:
        if kind == "c":
            out.append(r"{\f1 " + rtf_escape(t) + r"}")
        elif kind == "b":
            out.append(r"{\b " + rtf_escape(t) + r"}")
        else:
            out.append(rtf_escape(t))
    out.append(r"\par" + "\n")
    return "".join(out)


def skills_block(skills: list[str]) -> str:
    # Monospace, justified, small, wrapped - one comma-separated line
    body = ", ".join(skills)
    return r"\pard\sa120\ql\f1\fs20 " + rtf_escape(body) + r"\f0\fs22\par" + "\n"


# ---------- Data assembly ----------
by_fam: dict[str, list[str]] = defaultdict(list)
for skill, fam in SKILL_TO_FAMILY.items():
    by_fam[fam].append(skill)

ordered_families = [name for _, name in _CATEGORIES]

# Families dropped from modelling per docs/navrh_agregace_kategorii.md (section F)
DROPPED_CLUSTERS = {
    "cluster_legacy__mainframe",
    "cluster_data_analysis__stats",
    "cluster_tools__editors",
}

# Job families merged into Other (section C of the aggregation doc)
MERGED_JOB_FAMILIES = {"Frontend & Design", "QA & Testing", "Security", "Systems & Embedded"}


# ---------- Compose document ----------
parts: list[str] = [HEADER]

parts.append(h1("Příloha: Definice kategorií job_family a skill_cluster"))
parts.append(
    p(
        "Tato příloha dokumentuje, jak byly v datasetu vytvořeny kategoriální "
        "proměnné job_family (rodina pozice) a skill_cluster (rodina technické "
        "dovednosti), použité v regresních modelech (Tabulky 2 a 3). "
        "Výčet je strojově vygenerován přímo ze zdrojového kódu – zachycuje "
        "stav odpovídající finálnímu běhu Stata analýzy.",
    )
)
parts.append(
    p(
        "Zdroje definic: ai_skills/cli.py (job_family, generování dummy "
        "cluster_*), ai_skills/skills_dictionary.py (mapa SKILL_TO_FAMILY), "
        "docs/navrh_agregace_kategorii.md (následné agregace pro splnění "
        "podmínky n \u2265 50 v nejmenší podskupině Applied/Core AI).",
        italic=True,
    )
)

# =========================================================================
# 1. JOB FAMILY
# =========================================================================
parts.append(h1("1. job_family – 10 rodin pozic + Other"))
parts.append(
    p(
        "Rodina pozice je přiřazena deterministicky na základě názvu inzerátu "
        "(sloupec job_title). Aplikuje se uspořádaný seznam regulárních výrazů "
        "– vyhrává první shoda, zbytek inzerátů spadne do kategorie Other. "
        "Vzory pokrývají anglické i německé klíčové výrazy, aby byla zajištěna "
        "kompatibilita s DE subdatasetem."
    )
)
parts.append(
    para_with_code(
        [
            ("t", "Zdroj: "),
            ("c", "ai_skills/cli.py"),
            ("t", ", konstanta "),
            ("c", "_JOB_FAMILY_PATTERNS"),
            ("t", " (ř. 419–430)."),
        ]
    )
)

parts.append(h2("1.1 Pořadí a definice regulárních vzorů"))
for idx, (name, pattern) in enumerate(_JOB_FAMILY_PATTERNS, start=1):
    parts.append(h3(f"{idx}. {name}"))
    parts.append(
        para_with_code(
            [
                ("b", "Regex: "),
                ("c", pattern),
            ]
        )
    )

parts.append(h3(f"{len(_JOB_FAMILY_PATTERNS) + 1}. Other"))
parts.append(
    p(
        "Záchytná kategorie – všechny inzeráty, které neodpovídají žádnému "
        "z výše uvedených vzorů."
    )
)

# =========================================================================
# 2. SKILL CLUSTER
# =========================================================================
parts.append(h1("2. skill_cluster – 24 rodin technických dovedností"))
parts.append(
    p(
        "Každé kanonické hard-skill slovo je prostřednictvím slovníku "
        "SKILL_TO_FAMILY přiřazeno právě do jedné rodiny. Pro každou rodinu "
        "je v datasetu vytvořena binární indikátorová proměnná cluster_<nazev> "
        "(hodnota 1, pokud byla v inzerátu extrahována alespoň jedna dovednost "
        "z dané rodiny, jinak 0). Rodiny jsou definovány jako množiny "
        "(CAT_*) a procházejí se v pořadí uvedeném v konstantě _CATEGORIES; "
        "pokud by se tatáž dovednost objevila ve více množinách, vyhrává "
        "přiřazení z pozdější kategorie v tomto seznamu."
    )
)
parts.append(
    para_with_code(
        [
            ("t", "Zdroje: "),
            ("c", "ai_skills/skills_dictionary.py"),
            ("t", " (množiny CAT_* a "),
            ("c", "_CATEGORIES"),
            ("t", ", ř. 1180–1537); "),
            ("c", "ai_skills/cli.py"),
            ("t", " (generování dummy sloupců cluster_*, ř. 477–499)."),
        ]
    )
)

parts.append(h2("2.1 Výčet rodin a jejich obsahu"))
for idx, fam in enumerate(ordered_families, start=1):
    skills = sorted(by_fam.get(fam, []))
    col = actual_col(fam)
    marker = "  [vyřazeno z modelů – viz sekce 3]" if col in DROPPED_CLUSTERS else ""
    parts.append(h3(f"{idx}. {fam}{marker}"))
    parts.append(
        para_with_code(
            [
                ("b", "Stata sloupec: "),
                ("c", col),
                ("t", "    |    "),
                ("b", "Počet dovedností: "),
                ("t", str(len(skills))),
            ]
        )
    )
    parts.append(skills_block(skills))

# =========================================================================
# 3. AGGREGATIONS
# =========================================================================
parts.append(h1("3. Úpravy kategorií pro finální modely"))
parts.append(
    p(
        "Kvůli splnění podmínky alespoň 50 pozorování ve všech podkategoriích "
        "nejmenší skupiny závislé proměnné ai_level (Applied/Core AI, n = 1 313) "
        "byly před odhadem multinomiálních logistických modelů provedeny dvě "
        "agregační úpravy. Celý postup je odůvodněn v dokumentu "
        "docs/navrh_agregace_kategorii.md (sekce C a F)."
    )
)

parts.append(h2("3.1 job_family – sloučení řídce obsazených rodin do Other"))
parts.append(
    p(
        "Rodiny pozic s nízkým výskytem v podskupině Applied/Core AI byly "
        "sloučeny do kategorie Other. Samostatně zůstaly nosné rodiny (Software "
        "Engineer/Developer, Data & AI, DevOps & Cloud, Management atd.)."
    )
)
for fam in sorted(MERGED_JOB_FAMILIES):
    parts.append(
        bullet(
            [
                ("c", fam),
                ("t", "  ->  sloučeno do Other"),
            ]
        )
    )

parts.append(h2("3.2 skill_cluster – vyřazení tří rodin z regresních modelů"))
parts.append(
    p(
        "Následující tři rodiny měly v podskupině Applied/Core AI (respektive "
        "v celém datasetu) zanedbatelný výskyt a byly proto z modelů kompletně "
        "vyřazeny, aby se předešlo problému tzv. perfect prediction. Vlastní "
        "sloupce však v surových datech zůstávají pro účely deskriptivní "
        "statistiky."
    )
)
DROP_REASONS = [
    ("cluster_legacy__mainframe", "pouze 49 inzerátů v celém datasetu 18 000 pozic"),
    ("cluster_data_analysis__stats", "29 pozorování v Applied/Core AI"),
    ("cluster_tools__editors", "47 pozorování v Applied/Core AI"),
]
for col, reason in DROP_REASONS:
    parts.append(
        bullet(
            [
                ("c", col),
                ("t", " – " + reason),
            ]
        )
    )

parts.append(FOOTER)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("".join(parts), encoding="cp1250", errors="xmlcharrefreplace")
print(f"Wrote: {OUT}  ({OUT.stat().st_size} bytes)")
