#!/usr/bin/env python3
"""Generates a standalone RTF file with section 4.4 "Ekonometrické metody"
extended by formal mathematical notation of all three econometric models
(binary logit, multinomial logit, OLS Mincer wage equation).

Goal: paste-ready RTF for direct insertion into the thesis Word document.

Output: docs/Sekce_4_4_Ekonometricke_metody.rtf
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "Sekce_4_4_Ekonometricke_metody.rtf"


# ---------------------------------------------------------------------------
# RTF helpers (codepage 1250 document -> non-ASCII chars via \u escapes)
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


# Markers for inline formatting inside paragraphs.
# The renderer recognises the following mini-markup:
#   [[i]]...[[/i]]   italic
#   [[b]]...[[/b]]   bold
#   [[sub]]...[[/sub]]   subscript
#   [[sup]]...[[/sup]]   superscript
#   [[c]]...[[/c]]   inline code (Courier New)

def render_inline(text: str) -> str:
    replacements = [
        ("[[i]]", r"{\i "), ("[[/i]]", r"}"),
        ("[[b]]", r"{\b "), ("[[/b]]", r"}"),
        ("[[sub]]", r"{\sub "), ("[[/sub]]", r"}"),
        ("[[sup]]", r"{\super "), ("[[/sup]]", r"}"),
        ("[[c]]", r"{\f1 "), ("[[/c]]", r"}"),
    ]
    # Escape first, then swap marker *escapes* back to raw RTF control words.
    escaped = rtf_escape(text)
    for src, dst in replacements:
        escaped_src = rtf_escape(src)
        escaped = escaped.replace(escaped_src, dst)
    return escaped


def h1(text: str) -> str:
    return r"\pard\sb360\sa160\keepn\f0\b\fs36 " + render_inline(text) + r"\b0\fs22\par" + "\n"


def h2(text: str) -> str:
    return r"\pard\sb280\sa140\keepn\f0\b\fs28 " + render_inline(text) + r"\b0\fs22\par" + "\n"


def p(text: str) -> str:
    return r"\pard\sa120\sl300\slmult1\ql " + render_inline(text) + r"\par" + "\n"


def eq(text: str) -> str:
    """A centered display equation, slightly smaller line spacing."""
    return (
        r"\pard\sb120\sa160\sl280\slmult1\qc "
        + render_inline(text)
        + r"\par" + "\n"
    )


def eq_label(text: str) -> str:
    """Right-aligned equation number / label below an equation."""
    return (
        r"\pard\sb0\sa160\qr\i "
        + render_inline(text)
        + r"\i0\par" + "\n"
    )


def bullet(text: str) -> str:
    return (
        r"\pard\fi-283\li360\sa60\ql {\f2\'95}\tab "
        + render_inline(text)
        + r"\par" + "\n"
    )


# ---------------------------------------------------------------------------
# Document skeleton
# ---------------------------------------------------------------------------

HEADER = (
    r"{\rtf1\ansi\ansicpg1250\uc1\deff0"
    r"{\fonttbl"
    r"{\f0\froman Times New Roman;}"
    r"{\f1\fmodern Courier New;}"
    r"{\f2\fswiss Arial;}"
    r"}"
    r"{\colortbl;\red0\green0\blue0;\red80\green80\blue80;}"
    r"\paperw11906\paperh16838\margl1418\margr1418\margt1418\margb1418"
    r"\fs22\f0\lang1029"
)
FOOTER = r"}"


# ---------------------------------------------------------------------------
# Greek & math primitives (kept literal in source so text stays readable)
# ---------------------------------------------------------------------------
# α 945  β 946  γ 947  ε 949  λ 955  Λ 923  Σ 931  Π 928
# ∂ 8706  ∈ 8712  ≈ 8776  · 183  ² 178  ³ 179
# Italicised variables use [[i]]...[[/i]], subscripts [[sub]]i[[/sub]].


parts: list[str] = [HEADER]


# -------------------- 4.4 heading + intro --------------------
parts.append(h1("4.4 Ekonometrické metody"))

parts.append(
    p(
        "Samotná analýza stojí na třech hlavních typech modelů. "
        "Binární logistická regrese modeluje pravděpodobnost, zda inzerát "
        "vyžaduje AI dovednosti, a jejím cílem je zjistit, jaký typ pozice "
        "a technologický profil se s AI požadavky pojí. Výsledky jsou reportovány "
        "jako průměrné marginální efekty (AME), tedy hodnoty, které lze rovnou "
        "číst jako změnu pravděpodobnosti v procentních bodech při změně nezávislé "
        "proměnné o jednotku. Multinomická logistická regrese rozlišuje dva typy "
        "AI pozic, AI Integration a Applied/Core AI, proti referenční kategorii "
        "None, a umožňuje tak zachytit, v čem se liší pozice, které AI pouze "
        "používají, od těch, kde se AI reálně vyvíjí. OLS mzdová regrese "
        "kvantifikuje mzdovou prémii za AI dovednosti. Závislou proměnnou je "
        "[[c]]ln_salary[[/c]]; specifikace vychází z Mincerovy mzdové rovnice "
        "(Mincer, 1974) rozšířené o úrovně AI a kontrolu dovedností, pozice "
        "a firmy. Hlavní specifikace je odhadována zvlášť pro každou zemi."
    )
)

parts.append(
    p(
        "V následujících pododdílech je uveden formální zápis každého modelu. "
        "Ve všech případech označuje [[i]]i[[/i]] pořadové číslo inzerátu "
        "([[i]]i[[/i]] = 1, \u2026, [[i]]N[[/i]]), [[i]]x[[/i]][[sub]]i[[/sub]] "
        "vektor nezávislých proměnných a [[i]]\u03b2[[/i]] (případně "
        "[[i]]\u03b3[[/i]], [[i]]\u03b1[[/i]]) vektory odhadovaných parametrů."
    )
)


# -------------------- Binary logit --------------------
parts.append(h2("4.4.1 Binární logistická regrese"))

parts.append(
    p(
        "Závislou proměnnou je binární indikátor [[c]]has_ai[[/c]][[sub]]i[[/sub]] "
        "\u2208 \u007b0, 1\u007d, který nabývá hodnoty 1 pro inzeráty "
        "klasifikované jako AI Integration nebo Applied/Core AI a hodnoty 0 "
        "pro inzeráty bez AI požadavku. Pravděpodobnost AI požadavku podmíněná "
        "vektorem nezávislých proměnných [[i]]x[[/i]][[sub]]i[[/sub]] je "
        "modelována logistickou distribuční funkcí \u039b(\u00b7):"
    )
)

parts.append(
    eq(
        "[[i]]P[[/i]]\u2009([[c]]has_ai[[/c]][[sub]]i[[/sub]] = 1 | "
        "[[i]]x[[/i]][[sub]]i[[/sub]]) = \u039b([[i]]x[[/i]][[sub]]i[[/sub]]\u2032 "
        "[[i]]\u03b2[[/i]]) = exp([[i]]x[[/i]][[sub]]i[[/sub]]\u2032 "
        "[[i]]\u03b2[[/i]])\u2009/\u2009[1 + exp([[i]]x[[/i]][[sub]]i[[/sub]]\u2032 "
        "[[i]]\u03b2[[/i]])]"
    )
)
parts.append(eq_label("(4.1)"))

parts.append(p("ekvivalentní zápis prostřednictvím logitové transformace:"))

parts.append(
    eq(
        "logit\u2009[[i]]P[[/i]][[sub]]i[[/sub]] = ln "
        "\u007b[[i]]P[[/i]][[sub]]i[[/sub]] / (1 \u2212 "
        "[[i]]P[[/i]][[sub]]i[[/sub]])\u007d = "
        "[[i]]\u03b2[[/i]][[sub]]0[[/sub]] + [[i]]\u03b2[[/i]][[sub]]1[[/sub]]"
        "[[i]]x[[/i]][[sub]]1[[i]]i[[/i]][[/sub]] + \u2026 + "
        "[[i]]\u03b2[[/i]][[sub]]K[[/sub]][[i]]x[[/i]][[sub]]K[[i]]i[[/i]][[/sub]]"
    )
)
parts.append(eq_label("(4.2)"))

parts.append(
    p(
        "Vektor [[i]]x[[/i]][[sub]]i[[/sub]] obsahuje profesní skupinu "
        "([[c]]job_family[[/c]], Tabulka 2), vzdělání ([[c]]edu_logit[[/c]], "
        "3 úrovně s referenční kategorií Bachelor+), zkušenosti "
        "([[c]]exp_category[[/c]], 4 úrovně s referenční kategorií Mid 3\u20135 "
        "let), NACE sektor, typ a velikost firmy a indikátor remote práce. "
        "V alternativní specifikaci (Tabulka 3) je [[c]]job_family[[/c]] "
        "nahrazena 21 binárními indikátory skill clusterů při zachování "
        "ostatních kontrol."
    )
)

parts.append(
    p(
        "Pro přehlednost lze regresory rozdělit do tematických vektorů: "
        "profesní skupina [[i]]JF[[/i]][[sub]]i[[/sub]] ([[c]]job_family[[/c]]), "
        "lidský kapitál [[i]]HC[[/i]][[sub]]i[[/sub]] (vzdělání a zkušenost), "
        "firemní charakteristiky [[i]]FC[[/i]][[sub]]i[[/sub]] (typ a velikost "
        "firmy), sektor [[i]]SE[[/i]][[sub]]i[[/sub]] (NACE 1-digit) "
        "a pracovní uspořádání [[i]]WA[[/i]][[sub]]i[[/sub]] (remote). "
        "Binární logit (4.1) lze pak ekvivalentně zapsat ve tvaru:"
    )
)

parts.append(
    eq(
        "[[i]]P[[/i]]\u2009([[c]]has_ai[[/c]][[sub]]i[[/sub]] = 1 | "
        "[[i]]JF[[/i]][[sub]]i[[/sub]], [[i]]HC[[/i]][[sub]]i[[/sub]], "
        "[[i]]FC[[/i]][[sub]]i[[/sub]], [[i]]SE[[/i]][[sub]]i[[/sub]], "
        "[[i]]WA[[/i]][[sub]]i[[/sub]]) = "
        "[[i]]F[[/i]]\u2009([[i]]\u03b1[[/i]] + "
        "[[i]]\u03b2[[/i]]\u00b7[[i]]JF[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b3[[/i]]\u00b7[[i]]HC[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b4[[/i]]\u00b7[[i]]FC[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b6[[/i]]\u00b7[[i]]SE[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b7[[/i]]\u00b7[[i]]WA[[/i]][[sub]]i[[/sub]])"
    )
)
parts.append(eq_label("(4.1\u2032)"))

parts.append(
    p(
        "kde [[i]]F[[/i]]\u2009(\u00b7) = \u039b(\u00b7) je kumulativní "
        "standardní logistická distribuční funkce a [[i]]\u03b1[[/i]], "
        "[[i]]\u03b2[[/i]], [[i]]\u03b3[[/i]], [[i]]\u03b4[[/i]], "
        "[[i]]\u03b6[[/i]], [[i]]\u03b7[[/i]] jsou konstanta a vektory "
        "odhadovaných parametrů. Parametry jsou odhadovány metodou "
        "maximální věrohodnosti (MLE). Pro přímou interpretaci jsou "
        "reportovány průměrné marginální efekty:"
    )
)

parts.append(
    eq(
        "AME[[sub]]k[[/sub]] = (1\u2009/\u2009[[i]]N[[/i]])\u2009"
        "\u03a3[[sub]]i=1[[/sub]][[sup]]N[[/sup]]\u2009"
        "\u2202[[i]]P[[/i]]\u2009([[c]]has_ai[[/c]][[sub]]i[[/sub]] = 1 | "
        "[[i]]x[[/i]][[sub]]i[[/sub]]) / \u2202[[i]]x[[/i]][[sub]]k[[i]]i[[/i]][[/sub]]"
    )
)
parts.append(eq_label("(4.3)"))

parts.append(
    p(
        "Koeficient AME[[sub]]k[[/sub]] udává průměrnou změnu pravděpodobnosti "
        "AI požadavku (v procentních bodech) odpovídající změně proměnné "
        "[[i]]x[[/i]][[sub]]k[[/sub]] o jednotku, resp. přechodu z referenční "
        "kategorie u kategoriálních proměnných."
    )
)


# -------------------- Multinomial logit --------------------
parts.append(h2("4.4.2 Multinomická logistická regrese"))

parts.append(
    p(
        "Multinomický logit rozšiřuje binární model na tříhodnotovou závislou "
        "proměnnou [[c]]ai_level[[/c]][[sub]]i[[/sub]] \u2208 \u007b0, 1, 2\u007d, "
        "kde 0 = None, 1 = AI Integration a 2 = Applied/Core AI. Referenční "
        "kategorií je [[i]]j[[/i]] = 0 (None), pro kterou platí identifikační "
        "omezení [[i]]\u03b2[[/i]][[sub]]0[[/sub]] = 0. Pravděpodobnosti "
        "jednotlivých úrovní jsou:"
    )
)

parts.append(
    eq(
        "[[i]]P[[/i]]\u2009([[c]]ai_level[[/c]][[sub]]i[[/sub]] = [[i]]j[[/i]] | "
        "[[i]]x[[/i]][[sub]]i[[/sub]]) = exp([[i]]x[[/i]][[sub]]i[[/sub]]\u2032 "
        "[[i]]\u03b2[[/i]][[sub]]j[[/sub]])\u2009/\u2009"
        "[1 + \u03a3[[sub]]m=1[[/sub]][[sup]]2[[/sup]] "
        "exp([[i]]x[[/i]][[sub]]i[[/sub]]\u2032 [[i]]\u03b2[[/i]][[sub]]m[[/sub]])],"
        "\u2003 [[i]]j[[/i]] \u2208 \u007b1, 2\u007d"
    )
)
parts.append(eq_label("(4.4)"))

parts.append(
    eq(
        "[[i]]P[[/i]]\u2009([[c]]ai_level[[/c]][[sub]]i[[/sub]] = 0 | "
        "[[i]]x[[/i]][[sub]]i[[/sub]]) = 1\u2009/\u2009"
        "[1 + \u03a3[[sub]]m=1[[/sub]][[sup]]2[[/sup]] "
        "exp([[i]]x[[/i]][[sub]]i[[/sub]]\u2032 [[i]]\u03b2[[/i]][[sub]]m[[/sub]])]"
    )
)
parts.append(eq_label("(4.5)"))

parts.append(
    p(
        "Analogicky k zápisu (4.1\u2032) lze multinomický logit pro "
        "[[i]]j[[/i]] \u2208 \u007b1, 2\u007d přepsat v explicitně vektorové "
        "formě, kde [[i]]SC[[/i]][[sub]]i[[/sub]] označuje vektor 21 skill "
        "clusterů a ostatní vektory ([[i]]HC[[/i]][[sub]]i[[/sub]], "
        "[[i]]FC[[/i]][[sub]]i[[/sub]], [[i]]SE[[/i]][[sub]]i[[/sub]], "
        "[[i]]WA[[/i]][[sub]]i[[/sub]]) jsou definovány jako v (4.1\u2032):"
    )
)

parts.append(
    eq(
        "[[i]]P[[/i]]\u2009([[c]]ai_level[[/c]][[sub]]i[[/sub]] = [[i]]j[[/i]] | "
        "[[i]]SC[[/i]][[sub]]i[[/sub]], [[i]]HC[[/i]][[sub]]i[[/sub]], "
        "[[i]]FC[[/i]][[sub]]i[[/sub]], [[i]]SE[[/i]][[sub]]i[[/sub]], "
        "[[i]]WA[[/i]][[sub]]i[[/sub]]) = "
        "\u039b[[sub]]j[[/sub]]\u2009([[i]]\u03b1[[/i]][[sub]]j[[/sub]] + "
        "[[i]]\u03b2[[/i]][[sub]]j[[/sub]]\u00b7[[i]]SC[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b3[[/i]][[sub]]j[[/sub]]\u00b7[[i]]HC[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b4[[/i]][[sub]]j[[/sub]]\u00b7[[i]]FC[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b6[[/i]][[sub]]j[[/sub]]\u00b7[[i]]SE[[/i]][[sub]]i[[/sub]] + "
        "[[i]]\u03b7[[/i]][[sub]]j[[/sub]]\u00b7[[i]]WA[[/i]][[sub]]i[[/sub]])"
    )
)
parts.append(eq_label("(4.4\u2032)"))

parts.append(
    p(
        "kde \u039b[[sub]]j[[/sub]]\u2009(\u00b7) je multinomická logistická "
        "funkce pro kategorii [[i]]j[[/i]] (srov. 4.4) a "
        "[[i]]\u03b1[[/i]][[sub]]j[[/sub]], [[i]]\u03b2[[/i]][[sub]]j[[/sub]], "
        "[[i]]\u03b3[[/i]][[sub]]j[[/sub]], [[i]]\u03b4[[/i]][[sub]]j[[/sub]], "
        "[[i]]\u03b6[[/i]][[sub]]j[[/sub]], [[i]]\u03b7[[/i]][[sub]]j[[/sub]] "
        "jsou kategorie-specifické vektory koeficientů."
    )
)

parts.append(
    p(
        "Každá z kategorií [[i]]j[[/i]] \u2208 \u007b1, 2\u007d má vlastní "
        "vektor koeficientů [[i]]\u03b2[[/i]][[sub]]j[[/sub]], model tedy "
        "umožňuje oddělit determinanty povrchové integrace AI "
        "([[i]]\u03b2[[/i]][[sub]]1[[/sub]]) od determinantů hluboké AI expertízy "
        "([[i]]\u03b2[[/i]][[sub]]2[[/sub]]). Vektor [[i]]x[[/i]][[sub]]i[[/sub]] "
        "obsahuje všech 21 skill clusterů (včetně Generative AI a Data Science"
        "\u2009/\u2009ML, jelikož cílem je právě popsat, jakým profilům "
        "dovedností LLM jednotlivé AI úrovně přisuzuje), zkušenosti "
        "([[c]]exp_category[[/c]]), NACE sektor, typ a velikost firmy "
        "a indikátor remote práce. Vzdělání ([[c]]edu_logit[[/c]]) je "
        "z multinomického logitu záměrně vyřazeno, protože kombinace "
        "kategorie High School\u2009/\u2009Associate s úrovní Applied/Core AI "
        "má v německém a indickém vzorku méně než 25 pozorování a model by byl "
        "ohrožen quasi-complete separation. Stejně jako u binárního logitu "
        "jsou výsledky interpretovány prostřednictvím AME zvlášť pro každou "
        "úroveň [[i]]j[[/i]] (pomocí [[i]]margins[[/i]] [[i]]dydx[[/i]](*) "
        "[[i]]predict[[/i]](outcome([[i]]j[[/i]]))). Předpoklad nezávislosti "
        "irelevantních alternativ (IIA) je ověřen Hausmanovým testem "
        "(viz Příloha)."
    )
)


# -------------------- OLS Mincer wage regression --------------------
parts.append(h2("4.4.3 OLS mzdová regrese (Mincerova specifikace)"))

parts.append(
    p(
        "OLS regrese kvantifikuje čistou mzdovou prémii za AI dovednosti "
        "po kontrole pozorovatelných faktorů. Vychází z Mincerovy mzdové rovnice "
        "(Mincer, 1974), rozšířené o úrovně AI a bohatou sadu kontrol na úrovni "
        "inzerátu a firmy:"
    )
)

parts.append(
    eq(
        "ln([[i]]w[[/i]][[sub]]i[[/sub]]) = [[i]]\u03b1[[/i]] + "
        "[[i]]\u03b3[[/i]][[sub]]1[[/sub]]\u00b7AI_Integration[[sub]]i[[/sub]] + "
        "[[i]]\u03b3[[/i]][[sub]]2[[/sub]]\u00b7Applied_Core_AI[[sub]]i[[/sub]] + "
        "[[i]]x[[/i]][[sub]]i[[/sub]]\u2032 [[i]]\u03b2[[/i]] + "
        "[[i]]\u03b5[[/i]][[sub]]i[[/sub]]"
    )
)
parts.append(eq_label("(4.6)"))

parts.append(p("kde jednotlivé symboly znamenají:"))

parts.append(
    bullet(
        "[[i]]w[[/i]][[sub]]i[[/sub]] \u2014 inzerovaná roční mzda "
        "v USD u [[i]]i[[/i]]-tého inzerátu (pro DE a IN přepočtena fixními "
        "kurzy, viz §4.3);"
    )
)
parts.append(
    bullet(
        "AI_Integration[[sub]]i[[/sub]] a Applied_Core_AI[[sub]]i[[/sub]] "
        "\u2014 binární indikátory úrovně AI s referenční kategorií None; "
        "koeficienty [[i]]\u03b3[[/i]][[sub]]1[[/sub]] a "
        "[[i]]\u03b3[[/i]][[sub]]2[[/sub]] zachycují mzdovou prémii za dané "
        "úrovně oproti pozicím bez AI požadavku;"
    )
)
parts.append(
    bullet(
        "[[i]]x[[/i]][[sub]]i[[/sub]] \u2014 vektor kontrolních proměnných: "
        "vzdělání ([[c]]edu_ols[[/c]], 5 úrovní s referenční kategorií "
        "Bachelor+), zkušenosti ([[c]]exp_category[[/c]], 4 úrovně s referenční "
        "kategorií Mid 3\u20135 let), 19 skill clusterů (všech 21 rodin "
        "technických dovedností kromě Generative AI a Data Science\u2009/\u2009ML, "
        "viz odůvodnění níže), typ a velikost firmy, NACE sektor a indikátor "
        "remote práce. Profesní skupina ([[c]]job_family[[/c]]) není v Tabulce 5 "
        "zahrnuta, protože její efekt je z velké části zachycen 19 skill "
        "clustery a jednotná specifikace napříč USA, Německem a Indií zajišťuje "
        "symetrickou interpretaci. Varianta pro USA doplněná o fixní efekty "
        "čtyř regionů US Census je uvedena v Příloze D jako robustnostní "
        "kontrola regionální heterogenity;"
    )
)
parts.append(
    bullet(
        "[[i]]\u03b5[[/i]][[sub]]i[[/sub]] \u2014 náhodná chyba; standardní "
        "chyby odhadů jsou klastrované na úrovni firmy "
        "([[i]]vce[[/i]](cluster [[i]]firm[[/i]]))."
    )
)

parts.append(
    p(
        "Ekvivalentně lze (4.6) zapsat v explicitně vektorovém tvaru, ve kterém "
        "je kontrolní vektor [[i]]x[[/i]][[sub]]i[[/sub]] rozložen do tematických "
        "skupin \u2014 vektor úrovní AI [[i]]AI[[/i]][[sub]]i[[/sub]] "
        "(indikátory [[c]]AI_Integration[[/c]] a [[c]]Applied_Core_AI[[/c]]), "
        "vektor lidského kapitálu [[i]]HC[[/i]][[sub]]i[[/sub]] (vzdělání "
        "a zkušenost), vektor 19 dovednostních klastrů "
        "[[i]]SC[[/i]][[sub]]i[[/sub]], vektor firemních charakteristik "
        "[[i]]FC[[/i]][[sub]]i[[/sub]] (typ a velikost firmy), sektorový "
        "vektor [[i]]SE[[/i]][[sub]]i[[/sub]] (NACE) a indikátor pracovního "
        "uspořádání [[i]]WA[[/i]][[sub]]i[[/sub]] (remote):"
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
parts.append(eq_label("(4.6\u2032)"))

parts.append(
    p(
        "kde [[i]]\u03b3[[/i]] = ([[i]]\u03b3[[/i]][[sub]]1[[/sub]], "
        "[[i]]\u03b3[[/i]][[sub]]2[[/sub]])\u2032 je vektor hlavních "
        "zájmových koeficientů mzdové prémie za AI úrovně a "
        "[[i]]\u03b2[[/i]][[sub]]1[[/sub]] až [[i]]\u03b2[[/i]][[sub]]5[[/sub]] "
        "jsou vektory koeficientů odpovídající jednotlivým skupinám kontrolních "
        "proměnných."
    )
)

parts.append(
    p(
        "Koeficienty [[i]]\u03b3[[/i]][[sub]]1[[/sub]] a "
        "[[i]]\u03b3[[/i]][[sub]]2[[/sub]] se při závislé proměnné "
        "ln([[i]]w[[/i]][[sub]]i[[/sub]]) interpretují jako semielasticita. "
        "Pro malé hodnoty lze aproximovat:"
    )
)

parts.append(
    eq(
        "mzdová prémie \u2248 [[i]]\u03b3[[/i]]\u00b7100\u2009%,\u2003"
        "přesně: 100\u00b7(exp([[i]]\u03b3[[/i]]) \u2212 1)\u2009%"
    )
)
parts.append(eq_label("(4.7)"))

parts.append(
    p(
        "Důležitým rozhodnutím bylo vyřadit z hlavní OLS specifikace dovednostní "
        "klastry Generative AI a Data Science\u2009/\u2009ML, protože jsou "
        "mechanicky propojeny s klasifikací [[c]]ai_level[[/c]] (LLM je při "
        "klasifikaci implicitně používá) a jejich ponechání by vedlo "
        "k endogenitě \u2014 koeficienty [[i]]\u03b3[[/i]][[sub]]1[[/sub]] "
        "a [[i]]\u03b3[[/i]][[sub]]2[[/sub]] by byly zkresleny směrem k nule, "
        "protože část AI prémie by absorbovaly právě tyto klastry. Robustnostní "
        "specifikace s plnou sadou 21 klastrů je součástí přílohy."
    )
)


# -------------------- Closing: clustered SE + descriptive tests --------------------
parts.append(h2("4.4.4 Standardní chyby a doplňkové testy"))

parts.append(
    p(
        "Ve všech třech regresních modelech jsou standardní chyby klastrované "
        "na úrovni firmy, protože inzeráty téže firmy nejsou nezávislá "
        "pozorování (stejné HR oddělení, stejná mzdová politika). Klastrovaný "
        "odhadce kovarianční matice má tvar (Wooldridge, 2010):"
    )
)

parts.append(
    eq(
        "Var\u0302([[i]]\u03b2\u0302[[/i]]) = "
        "([[i]]X[[/i]]\u2032[[i]]X[[/i]])[[sup]]\u22121[[/sup]]\u2009"
        "[\u03a3[[sub]]c=1[[/sub]][[sup]]C[[/sup]]\u2009"
        "[[i]]X[[/i]][[sub]]c[[/sub]]\u2032 [[i]]u\u0302[[/i]][[sub]]c[[/sub]] "
        "[[i]]u\u0302[[/i]][[sub]]c[[/sub]]\u2032 [[i]]X[[/i]][[sub]]c[[/sub]]] "
        "([[i]]X[[/i]]\u2032[[i]]X[[/i]])[[sup]]\u22121[[/sup]]"
    )
)
parts.append(eq_label("(4.8)"))

parts.append(
    p(
        "kde [[i]]c[[/i]] = 1, \u2026, [[i]]C[[/i]] indexuje firmy, "
        "[[i]]X[[/i]][[sub]]c[[/sub]] je matice regresorů odpovídající firmě "
        "[[i]]c[[/i]] a [[i]]u\u0302[[/i]][[sub]]c[[/sub]] je vektor reziduí "
        "této firmy. Stejná logika je uplatněna i u MLE modelů (binární "
        "a multinomický logit) prostřednictvím robustního klastrovaného "
        "sandwich estimátoru."
    )
)

parts.append(
    p(
        "V deskriptivní části (kapitola 5.1) jsou pro testování rozdílů "
        "mezi skupinami použity \u03c7\u00b2 testy nezávislosti "
        "(Pearsonův chí-kvadrát) u kategoriálních proměnných, dvouvýběrový "
        "[[i]]t[[/i]]-test a neparametrický Mann\u2013Whitney U test "
        "pro spojité proměnné a ANOVA s Bonferroniho korekcí pro vícenásobná "
        "srovnání."
    )
)


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------
parts.append(FOOTER)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("".join(parts), encoding="cp1250", errors="xmlcharrefreplace")
print(f"Wrote: {OUT}  ({OUT.stat().st_size} bytes)")
