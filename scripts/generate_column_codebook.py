#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a detailed DOCX codebook describing how each final column was created.

This document is intended for senior academics who will develop
the master thesis into a peer-reviewed journal article.

Usage:
    python3 scripts/generate_column_codebook.py
"""

from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from pathlib import Path
import datetime

doc = Document()

# -- Styles ---------------------------------------------------------------
style = doc.styles["Normal"]
font = style.font
font.name = "Calibri"
font.size = Pt(11)

for level in range(1, 4):
    hstyle = doc.styles[f"Heading {level}"]
    hstyle.font.name = "Calibri"
    hstyle.font.color.rgb = RGBColor(0x1A, 0x23, 0x7E)

# -- Helper ----------------------------------------------------------------

def add_column_entry(name, dtype, source, description,
                     construction, decisions="", notes=""):
    """Add a formatted entry for one column."""
    doc.add_heading(name, level=3)

    tbl = doc.add_table(rows=0, cols=2)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT

    def _row(label, value):
        row = tbl.add_row()
        row.cells[0].text = label
        row.cells[1].text = value
        for cell in row.cells:
            for p in cell.paragraphs:
                p.style.font.size = Pt(10)
        row.cells[0].paragraphs[0].runs[0].bold = True

    _row("Datový typ", dtype)
    _row("Zdroj", source)
    _row("Popis", description)
    _row("Konstrukce", construction)
    if decisions:
        _row("Klíčová rozhodnutí", decisions)
    if notes:
        _row("Poznámky", notes)
    doc.add_paragraph()


# =========================================================================
# TITLE PAGE
# =========================================================================
title = doc.add_heading("Codebook: Konstrukce finálních proměnných", level=1)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.add_run(
    "Diplomová práce \u2013 Analýza požadavků na AI dovednosti v IT pracovních inzerátech\n"
    + f"Vygenerováno: {datetime.date.today().isoformat()}\n"
    + "Autor datové pipeline: Jakub Murcek\n"
).font.size = Pt(10)

doc.add_paragraph(
    "Tento dokument detailně popisuje, jak byl každý sloupec ve finálním "
    "datasetu zkonstruován \u2013 jaká data byla vstupem, jaký algoritmus (deterministický "
    "vs. LLM) byl použit, jaká rozhodnutí byla učiněna a proč. Slouží jako podklad "
    "pro zkušenější akademiky, kteří z práce budou tvořit odborný článek."
)

# =========================================================================
# 1. OVERVIEW
# =========================================================================
doc.add_heading("1. Přehled pipeline a zdrojových dat", level=1)

doc.add_paragraph(
    'Zdrojová data pocházejí z platformy Glassdoor a byla získána web-scrapingem. '
    'Každý záznam odpovídá jednomu pracovnímu inzerátu v IT sektoru. '
    'Data byla sbírána plošným vyhledáváním předem definovaných IT pracovních pozic '
    '(Software Engineer, Data Scientist apod.), čímž byly zachyceny IT role i '
    'v ne-IT sektorech (banky, výroba).'
)

doc.add_paragraph(
    'Surový CSV soubor obsahuje cca 33 sloupců (metadata inzerátu, text popisu, '
    'strukturované pole "skills" a "educations" z Glassdooru). Pipeline je obohatí '
    'na celkových 75 sloupců ve finálním Stata-ready datasetu.'
)

doc.add_heading("Schéma pipeline", level=2)
doc.add_paragraph(
    'CSV vstup \u2192 Deterministická anotace AI ze sloupce "skills" \u2192 '
    'LLM analýza textu inzerátu (3 paralelní úlohy: AI tier, Skills, Education) \u2192 '
    'Deterministická extrakce hardskills/softskills ze sloupce "job_desc_text" \u2192 '
    'Normalizace a sjednocení (Union deterministických + LLM skills) \u2192 '
    'Skill clustering (dummy proměnné) \u2192 Hybridní proměnná vzdělání \u2192 '
    'Clean-Stata krok (job_family, region, sector_nace, drop bulky columns) \u2192 '
    'Finální CSV pro Statu'
)

doc.add_heading("Použité modely a nástroje", level=2)
doc.add_paragraph(
    '\u2022 LLM: OpenAI gpt-5.4-nano (temperature = 0.1, batch size = 20, structured output s Pydantic)\n'
    '\u2022 Embedding model pro semantickou normalizaci: all-MiniLM-L6-v2 (sentence-transformers)\n'
    '\u2022 Deterministická extrakce: Slovník ~900 pojmů s word-boundary regex matchingem\n'
    '\u2022 Finální analýza: Stata 15.1'
)

# =========================================================================
# 2. INPUT COLUMNS
# =========================================================================
doc.add_heading("2. Vstupní sloupce (ze scrape Glassdoor)", level=1)

doc.add_paragraph(
    'Tyto sloupce jsou převzaty přímo z datasetu staženého z Glassdooru. '
    'Pipeline je nemění, pouze je využívá jako vstupy pro odvozené proměnné.'
)

input_cols = [
    ("id", "string", "Unikátní identifikátor inzerátu z Glassdooru."),
    ("job_title", "string", "Originální název pozice z Glassdooru (nepřeložený)."),
    ("location", "string", "Lokace pozice (město, stát, země)."),
    ("company_id", "integer", "ID zaměstnavatele v databázi Glassdoor."),
    ("company", "string", "Název zaměstnavatele."),
    ("age_in_days", "integer", "Stáří inzerátu ve dnech v okamžiku stažení."),
    ("pay_currency", "string", "Měna platu (USD, EUR, INR)."),
    ("pay_period", "string", "Perioda výplaty (yearly, monthly, hourly)."),
    ("salary_min/mid/max", "float", "Spodní, střední a horní hranice uváděného platového rozmezí."),
    ("rating", "float", "Hodnocení firmy na Glassdooru (1\u20135)."),
    ("discover_date", "date", "Datum, kdy byl inzerát nalezen scraperem."),
    ("job_types", "string", "Typ úvazku (Full-time, Part-time, Contract...)."),
    ("remote_work_types", "string", "Remote/Hybrid/On-site."),
    ("educations", "string", 'Strukturovaná pole vzdělání z Glassdooru (např. "Bachelor\'s degree, Master\'s degree").'),
    ("skills", "string", 'Čárkou oddělený seznam dovedností přiřazených Glassdoorem k inzerátu.'),
    ("job_desc_html/text", "string", "HTML resp. plaintext verze plného popisu pozice."),
    ("city / state / country", "string", "Rozložená lokace."),
    ("latitude / longitude", "float", "GPS souřadnice lokace."),
    ("ceo", "string", "Jméno CEO firmy."),
    ("headquarters", "string", "Sídlo firmy."),
    ("industry", "string", "Odvětví firmy dle Glassdooru."),
    ("sector", "string", "Sektor firmy dle Glassdooru (např. Information Technology)."),
    ("revenue", "string", "Příjmové pásmo firmy."),
    ("size", "string", "Velikostní pásmo firmy (počet zaměstnanců)."),
    ("type", "string", "Typ firmy (Company - Public, Company - Private...)."),
    ("website", "string", "Webová stránka firmy."),
    ("year_founded", "integer", "Rok založení firmy."),
]

tbl = doc.add_table(rows=1, cols=3)
tbl.style = "Table Grid"
hdr = tbl.rows[0]
for i, label in enumerate(["Sloupec", "Typ", "Popis"]):
    hdr.cells[i].text = label
    hdr.cells[i].paragraphs[0].runs[0].bold = True

for name, dtype, desc in input_cols:
    row = tbl.add_row()
    row.cells[0].text = name
    row.cells[1].text = dtype
    row.cells[2].text = desc

doc.add_paragraph()

# =========================================================================
# 3. DERIVED COLUMNS
# =========================================================================
doc.add_heading("3. Odvozené sloupce \u2013 detailní konstrukce", level=1)

# -- 3.1 Deterministic from skills column --
doc.add_heading('3.1 Deterministická analýza sloupce "skills" (z Glassdooru)', level=2)

doc.add_paragraph(
    'Glassdoor ke každému inzerátu přiřazuje strukturovaný seznam dovedností '
    '(sloupec "skills"), který je čárkou oddělený. Tato fáze z něj vytahuje '
    'AI-relevantní dovednosti pomocí předem definovaného slovníku.'
)

add_column_entry(
    name="skills_ai_det",
    dtype="string (čárkou oddělený seznam)",
    source='Sloupec "skills" (vstupní CSV)',
    description=(
        'AI/ML dovednosti nalezené v Glassdoor-provided sloupci "skills". '
        'Čistě deterministický výstup bez LLM.'
    ),
    construction=(
        '1. Sloupec "skills" se tokenizuje (split čárkami, lowercase, strip).\n'
        '2. Každý token se porovná proti seznamu AI_SKILLS (~900 termínů definovaných '
        'v config.py, pokrývajících: core AI/ML, neural architectures, GenAI/LLMs, '
        'RAG & embeddings, AI agents, fine-tuning, NLP, computer vision, speech, '
        'recommendation, time series, AutoML, ML frameworks, MLOps, cloud AI, GPU, '
        'edge AI, responsible AI, evaluation metrics).\n'
        '3. Shodné tokeny se deduplikují a seřadí abecedně.\n'
        'Implementace: skill_processing.py -> find_ai_matches().'
    ),
    decisions=(
        'Slovník AI_SKILLS byl ručně kurátorován. Obsahuje ~900 termínů a pokrývá '
        'i zkratky (llm, nlp, rag), frameworky (tensorflow, pytorch), '
        'platformy (sagemaker, vertex ai), koncepty (reinforcement learning), '
        'i buzzwords (ai-powered, ai-driven). Tyto buzzwords mohou generovat false '
        'positives, proto se v dalších krocích používá přísnější sada REAL_AI_SKILLS.'
    )
)

add_column_entry(
    name="skills_hasai_det",
    dtype="integer (0/1 dummy)",
    source="Odvozeno z skills_ai_det",
    description='Binární indikátor: 1 pokud skills_ai_det není prázdný (tj. Glassdoor "skills" obsahují alespoň jeden AI termín).',
    construction="skills_hasai_det = 1 pokud len(skills_ai_det) > 0, jinak 0."
)

# -- 3.2 LLM Analysis --
doc.add_heading("3.2 LLM analýza textu inzerátu (OpenAI gpt-5.4-nano)", level=2)

doc.add_paragraph(
    'Text popisu pozice (job_desc_text) a název pozice (job_title) se odesílají '
    'do OpenAI API jako "structured output" (Pydantic modely). Pipeline používá '
    '"task decomposition" \u2013 místo jednoho obřího promptu se pro každý inzerát '
    'spouštějí 3 nezávislé úlohy paralelně (batched po 20 inzerátech):'
)

doc.add_paragraph(
    '\u2022 Task 1: AI Tier Classification (ai_tier, ai_skills_mentioned, confidence)\n'
    '\u2022 Task 2: Skills Extraction (hardskills_raw, softskills_raw, ai_skills_mentioned)\n'
    '\u2022 Task 3: Education & Experience (min_education_level, min_years_experience)'
)

doc.add_paragraph(
    'Parametry LLM: model = gpt-5.4-nano, temperature = 0.1 (nízká pro konzistenci), '
    'service_tier = "flex" (batch API pricing), timeout = 900s, max_retries = 3.'
)

add_column_entry(
    name="desc_tier_llm",
    dtype="string (enum: core_ai | applied_ai | ai_integration | none)",
    source="LLM analýza job_desc_text + job_title",
    description=(
        'Klasifikace úrovně zapojení AI v pracovní pozici. Klíčová proměnná pro '
        'celou analýzu.'
    ),
    construction=(
        'LLM (Task 1) dostane systémový prompt s definicemi 4 tierů:\n'
        '\u2022 core_ai: Vývoj AI modelů od nuly (trénování foundation modelů, '
        'nové architektury, ML research) \u2013 EXTREMELY RARE.\n'
        '\u2022 applied_ai: Hands-on práce s ML modely (fine-tuning, training pipelines, '
        'feature engineering, MLOps, model deployment). MUSÍ modifikovat modely.\n'
        '\u2022 ai_integration: Používá AI jako black-box (volání OpenAI/Claude API, '
        'integrace AI služeb, prompt engineering bez fine-tuningu).\n'
        '\u2022 none: Žádná zmínka o AI v celém inzerátu.\n\n'
        'LLM hodnotí skutečné PRACOVNÍ POVINNOSTI, nikoliv jen zmínku AI v popisu firmy. '
        'Při nejistotě se nakloní k VYŠŠÍMU tieru (false positives se lépe kontrolují).'
    ),
    decisions=(
        'ZÁSADNÍ ROZHODNUTÍ: Ve finálních modelech byla hodnota tieru ponechána čistě '
        'dle LLM klasifikace bez jakéhokoli manuálního přepisování. Starší verze pipeline '
        'uvažovaly průnik tierů a deterministicky nalezených skills jako filtr buzzwords, '
        'ale od toho bylo upuštěno po feedbacku vedoucího práce.\n\n'
        'Ve Stata kódu se tier převádí na ordinální proměnnou: '
        'none = 0, ai_integration = 1, applied_ai/core_ai = 2 (sloučeny kvůli řídkosti core_ai).'
    )
)

add_column_entry(
    name="desc_conf_llm / ai_confidence",
    dtype="float (0.0 \u2013 1.0)",
    source="LLM analýza job_desc_text",
    description='Sebejistota LLM v klasifikaci tier. Oba sloupce obsahují stejnou hodnotu (historický důvod \u2013 ai_confidence je alias).',
    construction='LLM vrací float v rozmezí 0\u20131 jako součást structured output. Hodnota se ořezává na [0, 1] Pydantic validátorem.',
    decisions=(
        'Ve Stata kroku se vyřazují inzeráty s confidence < 0.7 (Data trimming \u2013 '
        'odfiltrování nekvalitních LLM odpovědí).'
    )
)

add_column_entry(
    name="desc_ai_llm",
    dtype="string (čárkou oddělený seznam)",
    source="LLM analýza job_desc_text",
    description='AI/ML specifické dovednosti identifikované LLM z textu inzerátu (tensorflow, pytorch, llm, genai, rag atd.).',
    construction=(
        'LLM (Task 1, ai_skills_mentioned) skenuje CELÝ text inzerátu (včetně popisu firmy) '
        'a vrací seznam AI/ML skills. Prompt explicitně uvádí kategorie: frameworky, koncepty, '
        'nástroje. Vrací prázdný seznam pouze pokud nikde v textu není zmínka o AI/ML.'
    )
)

add_column_entry(
    name="desc_rationale_llm",
    dtype="string",
    source="LLM analýza (jen v monolitickém režimu)",
    description='Textové zdůvodnění LLM pro zvolenou klasifikaci tieru. V decomposed režimu se nepoužívá.',
    construction='V decomposed režimu (výchozí) zůstává prázdný. Zachován pro zpětnou kompatibilitu.',
    notes="Ve finálním Stata datasetu je tento sloupec odstraněn (drop bulky columns)."
)

# -- 3.3 Deterministic extraction from job description --
doc.add_heading("3.3 Deterministická extrakce dovedností z textu inzerátu", level=2)

doc.add_paragraph(
    'Paralelně s LLM analýzou se na fulltext popisu (job_desc_text) i na '
    'Glassdoor "skills" sloupec aplikuje deterministický slovníkový matcher.'
)

add_column_entry(
    name="desc_hard_det",
    dtype="string (čárkou oddělený seznam)",
    source="Deterministická extrakce z job_desc_text + skills",
    description='Technické (hard) dovednosti nalezené slovníkovým matchingem v textu inzerátu + skills sloupci.',
    construction=(
        '1. Text inzerátu (job_desc_text) se prohledá proti slovníku HARDSKILL_VARIANTS '
        '(~970 variant -> ~700 kanonických názvů) s word-boundary regex.\n'
        '2. Stejný postup se aplikuje na sloupec "skills".\n'
        '3. Výsledky z obou zdrojů se sloučí (Union, case-insensitive deduplikace).\n'
        '4. Řeší se překryvy algoritmem "Longest Match Wins" \u2013 např. nalezení '
        '"SQL Server" zabrání samostatnému matchi "SQL", "React Native" potlačí "React".\n\n'
        'Implementace: deterministic_extractor.py -> extract_hardskills_deterministic() '
        's _extract_skills_with_spans() (span-based overlap resolution).'
    ),
    decisions=(
        'Ambiguózní krátké termíny (c, r, js, go) jsou záměrně vyloučeny ze slovníku \u2013 '
        'ty řeší LLM s kontextem. Varianty mapují na kanonické názvy (např. "reactjs", '
        '"react.js", "react js" -> "react").'
    )
)

add_column_entry(
    name="desc_soft_det",
    dtype="string (čárkou oddělený seznam)",
    source="Deterministická extrakce z job_desc_text + skills",
    description='Měkké (soft) dovednosti nalezené slovníkovým matchingem.',
    construction='Identický postup jako desc_hard_det, ale proti slovníku SOFTSKILL_VARIANTS (~100 variant -> ~65 kanonických názvů).'
)

add_column_entry(
    name="desc_hard_llm",
    dtype="string (čárkou oddělený seznam)",
    source="LLM analýza job_desc_text (Task 2)",
    description='Technické dovednosti extrahované LLM z textu inzerátu.',
    construction=(
        '1. LLM (Task 2, hardskills_raw) extrahuje všechny technické dovednosti z textu.\n'
        '2. Výstup se normalizuje: a) regex kanonizace (např. ".NET Core" -> "dotnet"), '
        'b) slovníková kanonizace (HARDSKILL_CANONICALIZATION), '
        'c) sémantická normalizace přes embeddings (threshold >= 0.90 pro zamezení '
        'falešných shod jako Java <-> JavaScript).\n'
        '3. Deduplikace a řazení.\n\n'
        'Implementace: skill_normalizer.py -> normalize_hardskills(use_semantic=True)'
    ),
    decisions=(
        'Sémantický matching s prahem 0.90 zabraňuje agresivnímu sloučení skutečně '
        'odlišných technologií. Nižší prahy (0.85) vedly k falešným pozitivům.'
    )
)

add_column_entry(
    name="desc_soft_llm",
    dtype="string (čárkou oddělený seznam)",
    source="LLM analýza job_desc_text (Task 2)",
    description='Měkké dovednosti extrahované LLM z textu inzerátu.',
    construction='Identický postup jako desc_hard_llm, ale pro softskills. Sémantický threshold = 0.85 (nižší, protože soft skills jsou inherentně méně přesné).'
)

# -- 3.4 Merged/Hybrid columns --
doc.add_heading("3.4 Sloučené (hybridní) sloupce dovedností", level=2)

doc.add_paragraph(
    'Klíčové odvozené proměnné, které kombinují deterministické a LLM výsledky '
    'strategií sjednocení (Union) pro maximální pokrytí.'
)

add_column_entry(
    name="hardskills",
    dtype="string (čárkou oddělený seznam)",
    source="Sloučení desc_hard_det + desc_hard_llm",
    description=(
        'FINÁLNÍ seznam technických dovedností pro daný inzerát. Toto je hlavní '
        'proměnná používaná pro všechny skill-based analýzy.'
    ),
    construction=(
        '1. Union (sjednocení) deterministicky nalezených skills (desc_hard_det) '
        'a LLM-extrahovaných skills (desc_hard_llm).\n'
        '2. Case-insensitive deduplikace s prioritou kanonického tvaru ze slovníku.\n'
        '3. Abecední řazení.\n\n'
        'Implementace: deterministic_extractor.py -> merge_skills(dict_skills, llm_skills)\n'
        'Princip: Deterministické skills mají prioritu v pojmenování (canonical casing), '
        'LLM skills přidávají pokrytí pro termíny, které nejsou ve slovníku.'
    ),
    decisions=(
        'ROZHODNUTÍ: Union strategie (ne průnik). Důvod: Průnik by vedl k příliš '
        'konzervativním výsledkům a ztratil by se přínos obou metod. Union poskytuje '
        'maximum recall; precision je zajištěna normalizací a validací v obou větvích.'
    )
)

add_column_entry(
    name="softskills",
    dtype="string (čárkou oddělený seznam)",
    source="Sloučení desc_soft_det + desc_soft_llm",
    description='FINÁLNÍ seznam měkkých dovedností. Identický Union strategie jako hardskills.',
    construction="Union(desc_soft_det, desc_soft_llm) -> deduplikace -> sort."
)

# -- 3.5 AI classification derived --
doc.add_heading("3.5 Odvozené klasifikační proměnné AI", level=2)

add_column_entry(
    name="ai_det_llm_match",
    dtype="integer (0/1 dummy)",
    source="Porovnání skills_hasai_det vs. desc_tier_llm",
    description=(
        'Indikátor shody mezi deterministickým detektorem a LLM klasifikací. '
        '1 = oba se shodují, 0 = neshodují.'
    ),
    construction=(
        'ai_det_llm_match = 1 pokud skills_hasai_det == (desc_tier_llm != "none").\n'
        'Jinými slovy: Shodují se, pokud oba říkají "má AI" nebo oba říkají "nemá AI".'
    ),
    notes='Slouží jako kvalitativní metrika pipeline. Vysoká shoda (>80 %) indikuje konzistenci obou přístupů.'
)

add_column_entry(
    name="is_real_ai",
    dtype="integer (0/1 dummy)",
    source="Hybridní: desc_tier_llm + hardskills + REAL_AI_SKILLS sada",
    description=(
        'Binární indikátor "skutečného AI" \u2013 pozice, kde se AI BUDUJE/TRÉNUJE/NASAZUJE '
        '(ne jen používá jako nástroj).'
    ),
    construction=(
        'is_real_ai = 1 pokud:\n'
        '  a) desc_tier_llm in {core_ai, applied_ai}, NEBO\n'
        '  b) průnik hardskills a REAL_AI_SKILLS je neprázdný.\n\n'
        'REAL_AI_SKILLS je přísnější podmnožina (~45 termínů) zahrnující pouze:\n'
        '\u2022 ML/DL frameworky: tensorflow, pytorch, keras, scikit-learn, jax...\n'
        '\u2022 Specializované knihovny: huggingface, transformers, diffusers, detectron2...\n'
        '\u2022 Core aktivity: model training, fine-tuning, model serving, model deployment...\n'
        '\u2022 DL koncepty: deep learning, neural networks, reinforcement learning\n'
        '\u2022 GenAI/LLM: llm, generative ai, foundation model, rag\n'
        '\u2022 MLOps: mlops, mlflow, kubeflow, sagemaker, vertex ai, wandb...\n'
        '\u2022 Optimalizace: onnx, tensorrt, triton inference server'
    ),
    decisions=(
        'ROZHODNUTÍ: Dvojitá podmínka (tier NEBO skills). Důvod: LLM může inzerát '
        'klasifikovat jako "none" i přesto, že se v hardskills nachází framework jako '
        'PyTorch \u2013 k tomu dochází, když LLM pracuje s kratším nebo nejednoznačným textem. '
        'Skill-based záchranná síť toto řeší.\n\n'
        'REAL_AI_SKILLS je úmyslně přísnější než AI_SKILLS \u2013 neobsahuje buzzwords '
        '(ai-powered, ai-driven, copilot, chatgpt), protože ty neindikují "budování AI".'
    )
)

# -- 3.6 Education --
doc.add_heading("3.6 Proměnné vzdělání", level=2)

add_column_entry(
    name="edu_level_det",
    dtype="string (enum: highschool | associate | bachelor | master | phd | prázdný)",
    source='Deterministická extrakce ze sloupce "educations" (Glassdoor metadata)',
    description=(
        'Nejnižší explicitně zmíněná úroveň vzdělání ve strukturovaném poli Glassdooru. '
        'Extrakce dle metodiky vedoucího: EDUCATION2 = lowest explicitly mentioned level.'
    ),
    construction=(
        '1. Text sloupce "educations" se normalizuje (collapse whitespace).\n'
        '2. Regex patterny detekují jednotlivé úrovně:\n'
        '   \u2022 highschool: high school, GED, HSD\n'
        '   \u2022 associate: associate\'s degree, A.A., A.S.\n'
        '   \u2022 bachelor: bachelor\'s, undergraduate degree, B.A., B.S.\n'
        '   \u2022 master: master\'s, graduate degree, MBA, M.A., M.S.\n'
        '   \u2022 phd: Ph.D., doctorate, doctoral degree\n'
        '3. Ze všech nalezených úrovní se vrátí NEJNIŽŠÍ dle hierarchie.\n\n'
        'Implementace: education_extractor.py -> extract_education_from_row()'
    ),
    decisions='Rozhodnutí dle profesora: extrahuje se NEJNIŽŠÍ úroveň (ne nejvyšší). Logika: minimum qualification = entrance barrier.',
    notes='Pokud "educations" sloupec chybí nebo je prázdný, vrací se prázdný string.'
)

add_column_entry(
    name="edulevel_llm",
    dtype='string (High School | Associate | Bachelor\'s | Master\'s | PhD | "-")',
    source="LLM analýza job_desc_text (Task 3)",
    description=(
        'Minimální požadavek na vzdělání extrahovaný LLM z volného textu popisu pozice. '
        'Nezávislý na Glassdoor metadata "educations".'
    ),
    construction=(
        '1. LLM (Task 3) dostane prompt s přesnými pravidly:\n'
        '   \u2022 Extrahovat POUZE pokud je EXPLICITNĚ uvedeno v textu.\n'
        '   \u2022 Pokud více úrovní (např. "Bachelor\'s or Master\'s"), vrátit NEJNIŽŠÍ.\n'
        '   \u2022 Pokud nic neuvádí, vrátit null (uloženo jako "-").\n'
        '   \u2022 Nepředpokládat Bachelor\'s jen protože jde o tech pozici.\n'
        '   \u2022 Německé termíny: Abgeschlossenes Studium -> Bachelor\'s, Ausbildung -> Associate, Promotion -> PhD.\n'
        '2. LLM záměrně NEDOSTÁVÁ sloupec "educations" jako vstup \u2013 to zajišťuje nezávislost od Glassdoor metadata.\n'
        '3. Hodnota "-" indikuje, že LLM v textu nenašel žádný explicitní požadavek na vzdělání.'
    ),
    decisions=(
        'ROZHODNUTÍ: LLM NESMÍ vidět sloupec "educations". Důvod: Zajištění nezávislosti '
        'pro hybridní proměnnou education_hybrid, kde se edu_level_det a edulevel_llm '
        'vzájemně doplňují.'
    )
)

add_column_entry(
    name="education_hybrid",
    dtype="string",
    source="Kombinace edu_level_det + edulevel_llm",
    description=(
        'Hybridní proměnná vzdělání: primárně bere deterministický výsledek '
        '(ze strukturovaných Glassdoor dat), a pokud chybí, doplní LLM výsledkem z textu.'
    ),
    construction=(
        '1. Primární zdroj: edu_level_det (strukturovaná metadata).\n'
        '2. Pokud edu_level_det je prázdný -> fallback na edulevel_llm.\n'
        '3. Normalizace: odebrání "\'s", "degree", "diploma" (např. "Bachelor\'s degree" -> "bachelor").\n'
        '4. Výstup lowercase.\n\n'
        'Implementace: pipeline.py -> _apply_stata_transformations() -> get_education_hybrid()'
    ),
    decisions='Priorita deterministického zdroje, protože Glassdoor metadata jsou strukturovanější a konzistentnější než LLM extrakce z volného textu.',
    notes='Ve finálním Stata datasetu je tento sloupec odstraněn \u2013 Stata kód si tvoří vlastní vzdělávací proměnné (edu_ols, edu_logit) přímo z edu_level_det a edulevel_llm.'
)

# -- 3.7 Experience --
doc.add_heading("3.7 Proměnná praxe", level=2)

add_column_entry(
    name="experience_min_llm",
    dtype='float | "-"',
    source="LLM analýza job_desc_text (Task 3)",
    description='Minimální požadovaný počet let praxe extrahovaný LLM z textu inzerátu.',
    construction=(
        '1. LLM extrahuje float hodnotu dle pravidel v promptu:\n'
        '   \u2022 Explicitní zmínky: "3+ years" -> 3.0, "3-5 years" -> 3.0 (minimum rozmezí).\n'
        '   \u2022 "Preferred" i "required" se extrahují stejně.\n'
        '   \u2022 Entry level / Recent grad / Junior -> 0.0.\n'
        '2. Implicitní inference z titulku pozice (pokud text neuvádí roky):\n'
        '   \u2022 "Senior" -> 5.0, "Staff/Principal/Lead/Architect" -> 7.0, "Junior/Associate" -> 0.0.\n'
        '3. Sekundární signály: "extensive experience" -> 5.0, "some experience" -> 1.0.\n'
        '4. Pokud nic neindikuje úroveň praxe -> null (uloženo jako "-").\n\n'
        'Německé termíny: "Jahre" = years, "Mehrjährige Berufserfahrung" -> 3.0.'
    ),
    decisions=(
        'ROZHODNUTÍ: Inference z titulku je kontroverzní ale byla zachována, '
        'protože velká část inzerátů (zejm. DE) neuvádí explicitní roky. '
        'Bez inference by experience_min_llm bylo prázdné pro >40 % dat.'
    )
)

# -- 3.8 Skill Cluster Dummies --
doc.add_heading("3.8 Skill Cluster dummy proměnné (cluster_*)", level=2)

doc.add_paragraph(
    'Pro ekonometrickou analýzu se hardskills převádějí na 24 binárních dummy '
    'proměnných (skill clusters/families). Každý cluster odpovídá jedné technologické '
    'rodině.'
)

clusters = [
    ("cluster_systems_programming", "Systems Programming", "c++, rust, golang, assembly..."),
    ("cluster_enterprise__managed", "Enterprise & Managed", "java, java ee, kotlin, scala, c#, dotnet..."),
    ("cluster_dynamic__web", "Dynamic & Web", "javascript, typescript, python, ruby, php..."),
    ("cluster_scripting__shell", "Scripting & Shell", "bash, powershell, shell scripting..."),
    ("cluster_data_analysis__stats", "Data Analysis & Stats", "r, matlab, stata, spss, sas..."),
    ("cluster_legacy__mainframe", "Legacy & Mainframe", "cobol, fortran, vba, visual basic..."),
    ("cluster_frontend_development", "Frontend Development", "react, angular, vue, next.js, css, html..."),
    ("cluster_backend_development", "Backend Development", "node.js, django, flask, spring boot, express..."),
    ("cluster_mobile__desktop", "Mobile & Desktop", "react native, flutter, swift, swiftui, android..."),
    ("cluster_databases__storage", "Databases & Storage", "postgresql, mongodb, redis, elasticsearch, sql server..."),
    ("cluster_data_engineering", "Data Engineering", "spark, airflow, kafka, snowflake, databricks, etl, dbt..."),
    ("cluster_data_science__ml", "Data Science & ML", "tensorflow, pytorch, scikit-learn, pandas, numpy, keras..."),
    ("cluster_generative_ai", "Generative AI", "llm, gpt, chatgpt, openai, langchain, huggingface, rag..."),
    ("cluster_bi__analytics", "BI & Analytics", "tableau, power bi, looker, excel, qlik..."),
    ("cluster_cloud_computing", "Cloud Computing", "aws, azure, gcp, cloud native, multi-cloud..."),
    ("cluster_devops__containers", "DevOps & Containers", "docker, kubernetes, terraform, ci/cd, jenkins, helm..."),
    ("cluster_os__embedded", "OS & Embedded", "linux, windows, embedded, firmware, iot..."),
    ("cluster_networking", "Networking", "tcp/ip, dns, vpn, routing, firewall, cdn..."),
    ("cluster_security__identity", "Security & Identity", "cybersecurity, penetration testing, siem, oauth, sso..."),
    ("cluster_testing_qa__debugging", "Testing, QA & Debugging", "selenium, jest, pytest, junit, cypress..."),
    ("cluster_architecture__methods", "Architecture & Methods", "microservices, agile, scrum, design patterns, devops..."),
    ("cluster_enterprise_platforms", "Enterprise Platforms", "salesforce, sap, sharepoint, dynamics..."),
    ("cluster_certifications", "Certifications", "aws certified, pmp, cissp, ccna..."),
    ("cluster_tools__editors", "Tools & Editors", "vscode, intellij, git, jira, figma..."),
]

add_column_entry(
    name="cluster_* (24 sloupcu)",
    dtype="integer (0/1 dummy)",
    source='Odvozeno ze sloupce "hardskills" + slovník SKILL_TO_FAMILY',
    description=(
        'Binární dummy proměnné indikující přítomnost alespoň jedné dovednosti '
        'z dané technologické rodiny v inzerátu.'
    ),
    construction=(
        '1. Sloupec "hardskills" (merged/hybrid) se tokenizuje (split čárkami, lowercase).\n'
        '2. Každý skill se vyhledá ve slovníku SKILL_TO_FAMILY (~1000 kanonických skills -> 24 rodin).\n'
        '3. Pokud se alespoň 1 skill z rodiny nachází v inzerátu -> cluster_X = 1, jinak 0.\n'
        '4. Názvy sloupců: regex sanitizace (lowercase, podtržítka místo mezer/speciálních znaků).\n\n'
        'Implementace: pipeline.py -> _apply_stata_transformations() a cli.py -> _handle_clean_stata()\n'
        '(Duplikováno v obou místech pro konzistenci analyze i clean-stata příkazů).'
    ),
    decisions=(
        'ROZHODNUTÍ PRO STATA: Ze 24 clusterů se ve finálním Stata kódu 3 explicitně '
        'mažou kvůli řídkosti (legacy__mainframe, data_analysis__stats, tools__editors). '
        'Do regresí tedy vstupuje 21 clusterů.\n\n'
        'ROZHODNUTÍ PRO OLS MZDY: Clustery generative_ai a data_science__ml jsou záměrně '
        'vyřazeny z mzdové regrese, aby nedocházelo k cirkularitě s proměnnou ai_level '
        '(LLM definuje AI tier z velké části právě na základě těchto dovedností).'
    )
)

doc.add_paragraph("Přehled 24 clusterů:")
tbl2 = doc.add_table(rows=1, cols=3)
tbl2.style = "Table Grid"
hdr2 = tbl2.rows[0]
for i, label in enumerate(["Název sloupce", "Rodina", "Příklady skills"]):
    hdr2.cells[i].text = label
    hdr2.cells[i].paragraphs[0].runs[0].bold = True
for col_name, family, examples in clusters:
    row = tbl2.add_row()
    row.cells[0].text = col_name
    row.cells[1].text = family
    row.cells[2].text = examples

doc.add_paragraph()

# -- 3.9 Clean-Stata columns --
doc.add_heading("3.9 Sloupce vytvořené v Clean-Stata kroku", level=2)

doc.add_paragraph(
    'Příkaz "clean-stata" provádí finální transformace pro import do Stata: '
    'vytvoří region, job_family, sector_nace, a odstraní bulky sloupce.'
)

add_column_entry(
    name="region",
    dtype="string (Northeast | Midwest | South | West | Unknown)",
    source='Odvozeno ze sloupce "state" (US Census Regions)',
    description='US Census Bureau region pro daný stát. Používá se jako fixní efekt v robustness testu.',
    construction=(
        'Deterministické mapování 50 US států + DC do 4 regionů dle US Census Bureau:\n'
        '\u2022 Northeast: CT, ME, MA, NH, RI, VT, NJ, NY, PA\n'
        '\u2022 Midwest: IL, IN, MI, OH, WI, IA, KS, MN, MO, NE, ND, SD\n'
        '\u2022 South: DE, FL, GA, MD, NC, SC, VA, DC, WV, AL, KY, MS, TN, AR, LA, OK, TX\n'
        '\u2022 West: AZ, CO, ID, MT, NV, NM, UT, WY, AK, CA, HI, OR, WA\n'
        'Pokud stát není nalezen -> "Unknown".'
    )
)

add_column_entry(
    name="job_family",
    dtype="string (10 kategorii + Other)",
    source='Regex klasifikace sloupce "job_title"',
    description=(
        'Normalizovaná rodina pozice. Slouží jako klíčová kontrolní proměnná '
        'v regresních modelech.'
    ),
    construction=(
        'Regex patterny aplikované v pořadí priority (první match vyhrává):\n'
        '1. Management (manager, director, architect, tech lead, VP, head of...)\n'
        '2. Security (secur, cyber, SOC, SIEM, penetration, firewall...)\n'
        '3. QA & Testing (qa, tester, quality assurance, sdet...)\n'
        '4. DevOps & Cloud (devops, devsecops, SRE, cloud, platform, infrastructure...)\n'
        '5. Data & AI (data eng, data scien, machine learn, AI, ML, BI...)\n'
        '6. Systems & Embedded (system eng, embedded, firmware, mainframe, DBA...)\n'
        '7. Frontend & Design (frontend, UI/UX, web design, game design...)\n'
        '8. Sr+ Software Engineer (senior/staff/principal/lead + engineer/developer)\n'
        '9. Software Developer (full-stack, .net dev, java dev, web dev, programmer...)\n'
        '10. Software Engineer (catch-all: software eng, backend eng...)\n'
        'Pokud žádný match -> "Other".\n\n'
        'Patterny pokrývají i německé varianty (Entwickler, Programmierer, Informatik...).\n'
        'Implementace: cli.py -> _handle_clean_stata() -> _JOB_FAMILY_PATTERNS'
    ),
    decisions=(
        'ROZHODNUTÍ: Software Engineer je BASELINE (referenční kategorie) v regresích. '
        'Důvod: neutrální "střed" IT pozic \u2013 ani specializovaný, ani manažerský.'
    ),
    notes=(
        'Existuje i detailnější normalizátor (job_title_normalizer.py s ~240 regex patterny '
        '-> ~40 kategorií), ale pro regresní modely se používá hrubší 10-kategoriální verze '
        'z clean-stata kroku, protože jemnější dělení by vedlo k příliš řídkým kategoriím.'
    )
)

add_column_entry(
    name="sector_nace",
    dtype="string (jednopismenovy kod: A\u2013S, Unknown)",
    source='Mapování sloupce "sector" na NACE rev. 2 sekce',
    description='Sektor firmy překódovaný do standardu NACE (Eurostat klasifikace ekonomických činností).',
    construction=(
        'Deterministické mapování Glassdoor sektoru na NACE sekce:\n'
        '\u2022 Information Technology -> J (Informační a komunikační činnosti)\n'
        '\u2022 Manufacturing -> C\n'
        '\u2022 Financial Services / Insurance -> K\n'
        '\u2022 Healthcare -> Q\n'
        '\u2022 Management & Consulting -> M\n'
        '\u2022 Retail & Wholesale -> G\n'
        '\u2022 Education -> P\n'
        '... a další. Pokud sektor nenalezen -> "Unknown".\n'
        'Pokrývá i německé názvy sektoru (Informationstechnologie -> J).'
    ),
    decisions='ROZHODNUTÍ: Referenční sektor v regresích = J (IT). Logika: náš dataset obsahuje primárně IT pozice, J je nejčastější.'
)

# -- 3.10 skill_cluster (text) --
doc.add_heading("3.10 Další odvozené sloupce", level=2)

add_column_entry(
    name="skill_cluster",
    dtype="string (formatovany text)",
    source="hardskills + SKILL_TO_FAMILY + sémantická kategorizace",
    description=(
        'Textová reprezentace skill clusteru pro daný inzerát. '
        'Formát: "Family1: skill1, skill2; Family2: skill3".'
    ),
    construction=(
        '1. Merged hardskills se kategorizují přes sémantický normalizer '
        '(SemanticSkillNormalizer.categorize_skills()).\n'
        '2. Kategorizace: primárně slovník SKILL_TO_FAMILY, fallback = embedding centroidy '
        'rodin (cosine similarity, threshold > 0.4).\n'
        '3. Skills se seskupí dle rodin a zformátují jako čitelný string.'
    ),
    notes='Tento sloupec je primárně pro debugging/vizualizaci. Pro regresi se používají binární cluster_* dummy proměnné.'
)

# =========================================================================
# 4. DATA TRIMMING & STATA
# =========================================================================
doc.add_heading("4. Očištění dat a transformace pro Statu", level=1)

doc.add_paragraph(
    'Následující kroky se provádějí ve Stata kódu (ai_skills_thesis_final.do), '
    'nikoliv v Pythonu. Jsou zde dokumentovány pro úplnost.'
)

trimming_items = [
    ("Filtr confidence", "Smazány inzeráty s desc_conf_llm < 0.7 (nespolehlivé LLM klasifikace)."),
    ("Filtr data", "Vyřazeny inzeráty starší než rok 2024."),
    ("Platy \u2013 měnový převod",
     "EUR a INR převedeny na roční USD dle fixních kurzů z období sběru. "
     "Hodinové/měsíční přepočteny dle lokálních norem (US: 2080h, DE: 1607h, IN: 1920h/rok)."),
    ("Platy \u2013 outliery",
     "IN: smazáno pod $2,000/rok. US/DE: pod $3,000/rok. Maximum zastropováno na $500,000."),
    ("ai_level (ordinální)",
     'Nová proměnná: none=0, ai_integration=1, applied_ai + core_ai=2 (sloučeny kvůli řídkosti core_ai).'),
    ("has_ai (binární)", "has_ai = 1 pokud ai_level > 0."),
    ("edu_ols (5 úrovní)",
     "Kódování vzdělání pro OLS: highschool, associate, bachelor (ref.), master, phd. "
     "Missing = samostatna kategorie."),
    ("edu_logit (3 úrovně)",
     "Zjednodušené vzdělání pro logit: sub_bachelor (HS+Associate), bachelor_plus (ref.), "
     "postgraduate (Master+PhD). Missing = samostatna kategorie."),
    ("Drop řídké clustery",
     "Odstraněny cluster_legacy__mainframe, cluster_data_analysis__stats, cluster_tools__editors "
     "(prilis malo pozorovani pro stabilni odhady)."),
    ("Drop bulky sloupce",
     "job_desc_text, job_desc_html, desc_rationale_llm, educations, education_hybrid, ceo_photo, id_posting."),
]

tbl3 = doc.add_table(rows=1, cols=2)
tbl3.style = "Table Grid"
hdr3 = tbl3.rows[0]
hdr3.cells[0].text = "Krok"
hdr3.cells[1].text = "Popis"
for cell in hdr3.cells:
    cell.paragraphs[0].runs[0].bold = True

for step, desc in trimming_items:
    row = tbl3.add_row()
    row.cells[0].text = step
    row.cells[1].text = desc

doc.add_paragraph()

# =========================================================================
# 5. SUMMARY OF KEY METHODOLOGICAL DECISIONS
# =========================================================================
doc.add_heading("5. Shrnutí klíčových metodologických rozhodnutí", level=1)

decisions_list = [
    ("Hybridní extrakce (Union)",
     "Deterministické + LLM skills se SJEDNOCUJÍ (ne průnik). Maximální recall; "
     "precision zajištěna normalizací a validací v obou větvích."),
    ("AI tier = čistě LLM",
     "Bez manuálního override. Starší verze zvažovaly průnik s deterministickými "
     "skills jako filtr buzzwords, ale od toho bylo upuštěno po feedbacku vedoucího."),
    ('Vzdělání: LLM nevidí "educations"',
     "Záměrné oddělení zdrojů. LLM extrahuje z textu, deterministika z metadat Glassdooru. "
     "Hybridní proměnná pak kombinuje oba zdroje s prioritou metadata."),
    ("REAL_AI_SKILLS = přísnější sada",
     "~45 termínů vs. ~900 v AI_SKILLS. Vylučuje buzzwords (ai-powered, copilot) "
     'a zaměřuje se na hands-on AI práci (frameworky, training, MLOps).'),
    ("Longest Match Wins",
     "Span-based overlap resolution v deterministické extrakci. Zabraňuje fragmentaci "
     'víceslovných termínů (SQL Server != SQL + Server).'),
    ("Sémantický threshold 0.90 (hard), 0.85 (soft)",
     "Pro LLM skill normalizaci. Vysoký threshold zabraňuje falešným shodám "
     "(Java <-> JavaScript, React <-> React Native)."),
    ("Task decomposition pro LLM",
     "Místo jednoho obřího promptu 3 fokusované úlohy. Lepší přesnost za cenu "
     "mírně vyšších nákladů (3x API call)."),
    ("Cirkularity guard v OLS",
     "GenAI a Data Science/ML clustery vyřazeny z mzdové regrese, protože de facto "
     "konstruují definici AI tieru. Test robustnosti (Příloha C) ověřuje stabilitu."),
    ("Drop 3 řídkých clusterů ve Stata",
     "Legacy/Mainframe, Data Analysis/Stats, Tools/Editors \u2013 příliš málo pozorování "
     "pro stabilní koeficienty. Zůstává 21 z 24 clusterů."),
    ("Experience inference z titulku",
     "Kontroverzní ale nutné \u2013 bez inference by >40 % dat nemělo experience hodnotu. "
     "Senior -> 5.0, Staff -> 7.0, Junior -> 0.0."),
]

for ttl, detail in decisions_list:
    p = doc.add_paragraph()
    run_title = p.add_run("\u2022 " + ttl + ": ")
    run_title.bold = True
    p.add_run(detail)

# =========================================================================
# 6. REPRODUCIBILITY
# =========================================================================
doc.add_heading("6. Reprodukovatelnost", level=1)

doc.add_paragraph(
    'Pipeline je plně reprodukovatelná pomocí CLI:\n\n'
    '# 1. Příprava vzorku\n'
    'uv run python -m ai_skills.cli prepare-inputs --rows 30000 --source data/inputs/us_relevant.csv\n\n'
    '# 2. Analýza (s resumable checkpoints)\n'
    'uv run python -m ai_skills.cli analyze --input-csv data/inputs/us_relevant_30000.csv --resume\n\n'
    '# 3. Clean-Stata\n'
    'uv run python -m ai_skills.cli clean-stata --input-csv data/outputs/us_relevant_30000_ai.csv\n\n'
    '# 4. Stata modely\n'
    '# do ai_skills_thesis_final.do\n\n'
    'Konfigurace: config/settings.toml (tracked), config/settings.local.toml (gitignored overrides). '
    'API klíč výhradně z environmentu (OPENAI_API_KEY).'
)

# =========================================================================
# SAVE
# =========================================================================
output_path = Path(__file__).parent.parent / "docs" / "Codebook_Konstrukce_Promennych.docx"
output_path.parent.mkdir(exist_ok=True)
doc.save(str(output_path))
print(f"Codebook saved to: {output_path}")
print(f"   Size: {output_path.stat().st_size / 1024:.0f} KB")
