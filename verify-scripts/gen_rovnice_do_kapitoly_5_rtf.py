#!/usr/bin/env python3
"""Generates a standalone RTF with three ready-to-paste insertion blocks for the
existing methodology subsections §5.2, §5.3 and §5.4 of docs/prakticka_cast_3.md.

Follows the supervisor's requested style: long descriptive paragraph introducing
named vectors of covariates (analogous to the Mincer example with PT, PE, FB, JE,
LO), followed by the formal equation in vector form.

Per the supervisor's explicit recommendation, the multinomial logit block is kept
minimal — just a short sentence stating that mlogit extends the binary logit and
is used as a refinement of the results, without a separate formal equation.

Each block is prefaced with a bold instruction ("Vložit za paragraf končící
slovy …") so the user can locate the insertion point in the Word document.

Output: docs/Rovnice_do_kapitoly_5.rtf
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "Rovnice_do_kapitoly_5.rtf"


# ---------------------------------------------------------------------------
# RTF helpers (same conventions as gen_sekce_4_4_ekonometricke_metody_rtf.py)
# ---------------------------------------------------------------------------
def rtf_escape(text: str) -> str:
    buf: list[str] = []
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
            signed = code if code < 32768 else code - 65536
            buf.append(f"\\u{signed}?")
    return "".join(buf)


def render_inline(text: str) -> str:
    replacements = [
        ("[[i]]", r"{\i "), ("[[/i]]", r"}"),
        ("[[b]]", r"{\b "), ("[[/b]]", r"}"),
        ("[[sub]]", r"{\sub "), ("[[/sub]]", r"}"),
        ("[[sup]]", r"{\super "), ("[[/sup]]", r"}"),
        ("[[c]]", r"{\f1 "), ("[[/c]]", r"}"),
    ]
    escaped = rtf_escape(text)
    for src, dst in replacements:
        escaped = escaped.replace(rtf_escape(src), dst)
    return escaped


def h2(text: str) -> str:
    return r"\pard\sb280\sa140\keepn\f0\b\fs28 " + render_inline(text) + r"\b0\fs22\par" + "\n"


def instr(text: str) -> str:
    """Instruction line (italic, grey) telling the user where to paste."""
    return (
        r"\pard\sb240\sa80\keepn\f0\i\cf2 "
        + render_inline(text)
        + r"\i0\cf1\par" + "\n"
    )


def p(text: str) -> str:
    return r"\pard\sa120\sl300\slmult1\ql " + render_inline(text) + r"\par" + "\n"


def eq(text: str) -> str:
    return r"\pard\sb120\sa160\sl280\slmult1\qc " + render_inline(text) + r"\par" + "\n"


HEADER = (
    r"{\rtf1\ansi\ansicpg1250\uc1\deff0"
    r"{\fonttbl"
    r"{\f0\froman Times New Roman;}"
    r"{\f1\fmodern Courier New;}"
    r"{\f2\fswiss Arial;}"
    r"}"
    r"{\colortbl;\red0\green0\blue0;\red100\green100\blue100;}"
    r"\paperw11906\paperh16838\margl1418\margr1418\margt1418\margb1418"
    r"\fs22\f0\lang1029"
)
FOOTER = r"}"


parts: list[str] = [HEADER]

parts.append(
    r"\pard\sb0\sa280\qc\f0\b\fs32 "
    + render_inline("Rovnice k doplnění do kapitoly 5")
    + r"\b0\fs22\par" + "\n"
)

parts.append(
    p(
        "Tento soubor obsahuje tři krátké bloky, které stačí zkopírovat do "
        "stávajícího Word dokumentu (kapitola 5) na uvedená místa. Styl "
        "formálního zápisu modelů odpovídá doporučení vedoucího (pojmenované "
        "vektory regresorů, analogicky Mincerovu příkladu s vektory PT, PE, "
        "FB, JE, LO). Pro multinomický logit je dle vedoucího uvedena pouze "
        "krátká zmínka bez samostatné formální rovnice."
    )
)


# ===========================================================================
# Block 1 — §5.2 Binary logit
# ===========================================================================
parts.append(h2("§5.2 — binární logit"))

parts.append(
    instr(
        "Vložit za paragraf končící slovy \u201e\u2026 pro USA, Německo a Indii.\u201c "
        "(hned za první odstavec sekce \u201e5.2 Determinanty AI požadavku (binární logit)\u201c)."
    )
)

parts.append(
    p(
        "Pro modelování pravděpodobnosti AI požadavku odhadujeme binární "
        "logistický model"
    )
)

parts.append(
    eq(
        "[[i]]P[[/i]]\u2009([[c]]has_ai[[/c]][[sub]]i[[/sub]] = 1 | "
        "[[i]]JF[[/i]][[sub]]i[[/sub]], [[i]]HC[[/i]][[sub]]i[[/sub]], "
        "[[i]]FC[[/i]][[sub]]i[[/sub]], [[i]]SE[[/i]][[sub]]i[[/sub]], "
        "[[i]]WA[[/i]][[sub]]i[[/sub]]) = [[i]]F[[/i]]\u2009("
        "[[i]]\u03b1[[/i]] + "
        "[[i]]\u03b2[[/i]]\u00b7[[i]]JF[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b3[[/i]]\u00b7[[i]]HC[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b4[[/i]]\u00b7[[i]]FC[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b6[[/i]]\u00b7[[i]]SE[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b7[[/i]]\u00b7[[i]]WA[[/i]][[sub]]i[[/sub]])"
    )
)

parts.append(
    p(
        "objasňující pravděpodobnost AI požadavku v pracovním inzerátu, "
        "přičemž závislá proměnná [[c]]has_ai[[/c]][[sub]]i[[/sub]] "
        "nabývá hodnoty 1 u inzerátů klasifikovaných jako AI Integration nebo "
        "Applied/Core AI a hodnoty 0 u inzerátů bez AI požadavku. Tento výsledek "
        "je predikován vektorem vysvětlujících proměnných pro každý inzerát "
        "([[i]]i[[/i]]), rozloženým do tematických skupin: vektoru profesní "
        "skupiny [[i]]JF[[/i]][[sub]]i[[/sub]] ([[c]]job_family[[/c]] podle "
        "Tabulky 2); vektoru lidského kapitálu [[i]]HC[[/i]][[sub]]i[[/sub]] "
        "(vzdělání [[c]]edu_logit[[/c]] s referenční kategorií Bachelor+ "
        "a kategorie zkušeností [[c]]exp_category[[/c]] s referenční kategorií "
        "Mid 3\u20135 let); vektoru firemních charakteristik "
        "[[i]]FC[[/i]][[sub]]i[[/sub]] (typ organizace a velikost firmy); "
        "vektoru sektorových indikátorů [[i]]SE[[/i]][[sub]]i[[/sub]] (NACE Rev. 2 "
        "na úrovni 1-digit); a indikátoru pracovního uspořádání "
        "[[i]]WA[[/i]][[sub]]i[[/sub]] (dummy remote práce). V alternativní "
        "specifikaci (Tabulka 3) je vektor [[i]]JF[[/i]][[sub]]i[[/sub]] nahrazen "
        "vektorem [[i]]SC[[/i]][[sub]]i[[/sub]] obsahujícím 21 binárních "
        "indikátorů skill clusterů. [[i]]F[[/i]] značí kumulativní standardní "
        "logistickou distribuční funkci a [[i]]\u03b2[[/i]], [[i]]\u03b3[[/i]], "
        "[[i]]\u03b4[[/i]], [[i]]\u03b6[[/i]], [[i]]\u03b7[[/i]] jsou vektory "
        "odhadovaných parametrů. Parametry jsou odhadovány metodou maximální "
        "věrohodnosti se standardními chybami klastrovanými na úrovni firmy."
    )
)


# ===========================================================================
# Block 2 — §5.3 Multinomial logit (short mention per supervisor)
# ===========================================================================
parts.append(h2("§5.3 — multinomický logit (krátká zmínka)"))

parts.append(
    instr(
        "Vložit za paragraf končící slovy \u201e\u2026 velikost firmy, remote práce).\u201c "
        "(za druhý odstavec sekce \u201e5.3 Používání vs. vývoj AI\u201c)."
    )
)

parts.append(
    p(
        "Multinomický logit rozšiřuje binární logit z §5.2 na tříhodnotovou "
        "závislou proměnnou [[c]]ai_level[[/c]][[sub]]i[[/sub]] \u2208 "
        "\u007bNone, AI Integration, Applied/Core AI\u007d (kategorie None "
        "slouží jako referenční) a v této práci je využit jako upřesnění "
        "výsledků binárního logitu. Formální zápis modelu je strukturálně "
        "analogický rovnici binárního logitu z §5.2 \u2014 pro každou "
        "z nereferenčních kategorií se modeluje logaritmická šance proti "
        "kategorii None jako lineární kombinace téže sady regresorů (viz "
        "specifikaci v předchozím odstavci) \u2014 a není zde proto samostatně "
        "uváděn. Analytický přínos této specifikace oproti binárnímu logitu "
        "spočívá v tom, že dovednostní determinanty povrchové integrace AI "
        "(AI Integration) a hluboké AI expertízy (Applied/Core AI) lze zachytit "
        "odděleně, nikoli jako jejich sloučený efekt."
    )
)


# ===========================================================================
# Block 3 — §5.4 OLS Mincer wage regression
# ===========================================================================
parts.append(h2("§5.4 \u2014 OLS mzdová regrese (Mincer)"))

parts.append(
    instr(
        "Vložit za paragraf končící slovy \u201e\u2026 ověřena v Příloze D.\u201c "
        "(za první odstavec sekce \u201e5.4 Mzdová prémie za AI (OLS)\u201c)."
    )
)

parts.append(
    p(
        "Pro odhad mzdových efektů AI dovedností je použita mzdová rovnice "
        "Mincerova typu (Lemieux, 2006; Mincer, 1974), kde ln\u2009"
        "[[i]]w[[/i]][[sub]]i[[/sub]] označuje přirozený logaritmus inzerované "
        "roční mzdy u [[i]]i[[/i]]-tého inzerátu (v USD; konverze měn u "
        "německých a indických inzerátů je popsána v §5.1). "
        "Vysvětlující proměnné zahrnují vektor úrovní AI "
        "[[i]]AI[[/i]][[sub]]i[[/sub]] (binární indikátory pro "
        "[[c]]ai_level[[/c]] = AI Integration a [[c]]ai_level[[/c]] = "
        "Applied/Core AI, s referenční kategorií None); vektor "
        "lidského kapitálu [[i]]HC[[/i]][[sub]]i[[/sub]] (kategorické vzdělání "
        "[[c]]edu_ols[[/c]] s 5 úrovněmi a kategorie zkušeností "
        "[[c]]exp_category[[/c]] se 4 úrovněmi); vektor dovednostních klastrů "
        "[[i]]SC[[/i]][[sub]]i[[/sub]] (19 binárních indikátorů skill clusterů, "
        "tj. všech 21 technických rodin dovedností kromě Generative AI "
        "a Data Science\u2009/\u2009ML, které jsou vyřazeny kvůli cirkularitě "
        "s klasifikací [[c]]ai_level[[/c]]); vektor firemních charakteristik "
        "[[i]]FC[[/i]][[sub]]i[[/sub]] (typ organizace a velikost firmy); vektor "
        "sektorových indikátorů [[i]]SE[[/i]][[sub]]i[[/sub]] (NACE Rev. 2 "
        "na úrovni 1-digit); a indikátor pracovního uspořádání "
        "[[i]]WA[[/i]][[sub]]i[[/sub]] (dummy remote práce). Odhadovaný model má "
        "tvar:"
    )
)

parts.append(
    eq(
        "ln\u2009[[i]]w[[/i]][[sub]]i[[/sub]] = [[i]]\u03b1[[/i]] + "
        "[[i]]\u03b3[[/i]]\u00b7[[i]]AI[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b2[[/i]][[sub]]1[[/sub]]\u00b7[[i]]HC[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b2[[/i]][[sub]]2[[/sub]]\u00b7[[i]]SC[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b2[[/i]][[sub]]3[[/sub]]\u00b7[[i]]FC[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b2[[/i]][[sub]]4[[/sub]]\u00b7[[i]]SE[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b2[[/i]][[sub]]5[[/sub]]\u00b7[[i]]WA[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b5[[/i]][[sub]]i[[/sub]]"
    )
)

parts.append(
    p(
        "kde [[i]]\u03b3[[/i]] a [[i]]\u03b2[[/i]][[sub]]1[[/sub]] až "
        "[[i]]\u03b2[[/i]][[sub]]5[[/sub]] jsou vektory regresních koeficientů "
        "přiřazené příslušným skupinám regresorů a "
        "[[i]]\u03b5[[/i]][[sub]]i[[/sub]] je náhodná chyba. Hlavním zájmovým "
        "vektorem je [[i]]\u03b3[[/i]] = ([[i]]\u03b3[[/i]][[sub]]1[[/sub]], "
        "[[i]]\u03b3[[/i]][[sub]]2[[/sub]])\u2032; jeho složky "
        "[[i]]\u03b3[[/i]][[sub]]k[[/sub]] ([[i]]k[[/i]]\u2009=\u20091, 2) "
        "odpovídají při log-lineární specifikaci přibližně semielasticitám "
        "mzdy vůči příslušné úrovni AI, přičemž relativní mzdová prémie je "
        "přesně rovna 100\u00b7[exp([[i]]\u03b3[[/i]][[sub]]k[[/sub]])"
        "\u2009\u2212\u20091]\u2009% (při malých hodnotách "
        "[[i]]\u03b3[[/i]][[sub]]k[[/sub]] přibližně 100\u00b7"
        "[[i]]\u03b3[[/i]][[sub]]k[[/sub]]\u2009%). "
        "Model je odhadnut samostatně pro každou zemi (USA, Německo, Indie) se "
        "standardními chybami klastrovanými na úrovni firmy."
    )
)


parts.append(FOOTER)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("".join(parts), encoding="cp1250", errors="xmlcharrefreplace")
print(f"Wrote: {OUT}  ({OUT.stat().st_size} bytes)")
