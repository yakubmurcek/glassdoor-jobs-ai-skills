********************************************************************************
* AI SKILLS IN IT JOB POSTINGS - STATA ANALYSIS
* ==============================================================================
* Dataset: us_relevant_ai.csv
* Autor: [Yakub Murcek]
* Datum: Leden 2026
* 
* Tento do-file obsahuje kompletní analýzu datasetu IT pracovních inzerátů
* s fokusem na požadavky na AI dovednosti.
*
* TESTOVÁNO PRO: STATA 15.1 (IC/SE verze)
********************************************************************************

* ==============================================================================
* 1. NASTAVENÍ PROSTŘEDÍ
* ==============================================================================
* Vyčistíme paměť a nastavíme working directory
* Toto je důležité pro reprodukovatelnost - každý běh začíná čistě

version 15.1                              
clear all
set more off                              

* Nastav cestu k datům - UPRAV PODLE SVÉ STRUKTURY
global datadir "/Users/yakub/Python/ai-skills/data/outputs"
global outdir "/Users/yakub/Python/ai-skills/analysis/stata/output"

* Vytvoř output složku pokud neexistuje
capture mkdir "$outdir"

* Log file - zaznamenává veškerý výstup pro pozdější kontrolu
capture log close
log using "$outdir/ai_skills_analysis.log", replace text


* ==============================================================================
* 2. IMPORT DAT
* ==============================================================================
* CSV soubor používá středník jako oddělovač (typické pro evropské formáty)

import delimited "$datadir/us_relevant_ai.csv", delimiter(";") clear varnames(1) encoding(utf8)

* Základní kontrola úspěšného importu
describe, short
display "Počet pozorování (job postů): " _N


* ==============================================================================
* 3. ČIŠTĚNÍ A PŘÍPRAVA DAT
* ==============================================================================
* Tato sekce převádí textové proměnné na numerické kde je to potřeba
* a vytváří nové analytické proměnné

* --- 3.1 AI Tier klasifikace ---
* desc_tier_llm obsahuje 3 kategorie: "none", "ai_integration", "ai_focused"
* Vytvoříme numerickou proměnnou pro statistické analýzy

* Ošetříme prázdné hodnoty před encode
replace desc_tier_llm = "missing" if desc_tier_llm == ""
encode desc_tier_llm, generate(ai_tier_num)

* Binární dummy: má pozice AI požadavky? (ai_integration nebo ai_focused)
gen has_ai = (desc_tier_llm != "none" & desc_tier_llm != "missing")
label variable has_ai "Pozice vyžaduje AI dovednosti (1=ano)"

* --- 3.2 Vzdělání ---
* edulevel_llm obsahuje: "-", "High School", "Associate's", "Bachelor's", "Master's", "Ph.D."
* Převedeme na ordinální škálu

gen edu_level = .
replace edu_level = 0 if edulevel_llm == "-" | edulevel_llm == ""
replace edu_level = 1 if strpos(lower(edulevel_llm), "high") > 0
replace edu_level = 2 if strpos(lower(edulevel_llm), "associate") > 0
replace edu_level = 3 if strpos(lower(edulevel_llm), "bachelor") > 0
replace edu_level = 4 if strpos(lower(edulevel_llm), "master") > 0
replace edu_level = 5 if strpos(lower(edulevel_llm), "ph") > 0 | strpos(lower(edulevel_llm), "doctor") > 0

label define edu_lbl 0 "Neuvedeno" 1 "Stredoskolske" 2 "Vyssi odborne" 3 "Bakalarske" 4 "Magisterske" 5 "Doktorske"
label values edu_level edu_lbl
label variable edu_level "Pozadovane vzdelani (ordinalni)"

* --- 3.3 Zkušenosti ---
* experience_min_llm je float - minimální požadované roky zkušeností

destring experience_min_llm, replace force
label variable experience_min_llm "Min. pozadovane roky zkusenosti"

* Kategorie zkušeností pro kontingenční tabulky
gen exp_category = .
replace exp_category = 1 if experience_min_llm == 0 | experience_min_llm == .
replace exp_category = 2 if experience_min_llm > 0 & experience_min_llm <= 2
replace exp_category = 3 if experience_min_llm > 2 & experience_min_llm <= 5
replace exp_category = 4 if experience_min_llm > 5 & experience_min_llm <= 10
replace exp_category = 5 if experience_min_llm > 10 & experience_min_llm < .

label define exp_lbl 1 "Entry (0)" 2 "Junior (1-2)" 3 "Mid (3-5)" 4 "Senior (6-10)" 5 "Expert (10+)"
label values exp_category exp_lbl
label variable exp_category "Kategorie seniority"

* --- 3.4 Plat ---
* Destring salary proměnných

destring salary_min salary_mid salary_max, replace force

* Filtrujeme nereálné hodnoty (příliš nízké nebo vysoké)
replace salary_mid = . if salary_mid < 20000 | salary_mid > 500000
label variable salary_mid "Rocni plat - stredni hodnota (USD)"

* --- 3.5 Sektor a Industrie ---
* Zjednodušené kategorie pro analýzu
* Ošetříme prázdné hodnoty

replace sector = "Unknown" if sector == ""
replace industry = "Unknown" if industry == ""
encode sector, generate(sector_num)
encode industry, generate(industry_num)

* --- 3.6 Lokace ---
* Extrahujeme stát z location nebo použijeme sloupec state

replace state = "Unknown" if state == ""
encode state, generate(state_num)

* --- 3.7 Remote práce ---
* remote_work_types obsahuje informaci o možnosti remote

gen is_remote = 0
replace is_remote = 1 if strpos(lower(remote_work_types), "home") > 0
replace is_remote = 1 if strpos(lower(remote_work_types), "remote") > 0
label variable is_remote "Moznost remote prace (1=ano)"


* ==============================================================================
* 4. DESKRIPTIVNÍ STATISTIKA
* ==============================================================================
* Základní přehled datasetu - toto jde typicky do první tabulky v diplomce

display _n "=============================================================="
display "4. DESKRIPTIVNI STATISTIKA"
display "=============================================================="

* --- 4.1 Frekvence AI tier klasifikace ---
* KLÍČOVÁ TABULKA: Kolik % pozic vyžaduje AI dovednosti?
display _n "--- 4.1 Distribuce AI pozadavku v IT pozicich ---"
tab desc_tier_llm, missing
tab desc_tier_llm if desc_tier_llm != "missing", sort

* Procentuální rozdělení (pro text diplomky)
count if has_ai == 1
local n_ai = r(N)
count
local n_total = r(N)
display _n "Podil pozic s AI pozadavky: " %5.2f (`n_ai'/`n_total')*100 "%"

* --- 4.2 Požadované vzdělání ---
display _n "--- 4.2 Distribuce vzdelavacich pozadavku ---"
tab edu_level, missing
tab edu_level if edu_level > 0

* Porovnání vzdělání podle AI tier
display _n "Vzdelani x AI tier (kontingencni tabulka):"
tab edu_level ai_tier_num if edu_level > 0, chi2 column

* --- 4.3 Požadované zkušenosti ---
display _n "--- 4.3 Distribuce pozadavku na zkusenosti ---"
summarize experience_min_llm, detail

tab exp_category, missing
tab exp_category ai_tier_num, chi2 column

* --- 4.4 Platy ---
display _n "--- 4.4 Distribuce platu ---"
summarize salary_mid, detail

* Platy podle AI tier - DŮLEŽITÉ pro argument o "AI premium"
display _n "Plat podle AI tier:"
tabstat salary_mid, by(desc_tier_llm) statistics(count mean sd min p25 p50 p75 max)

* --- 4.5 Sektory a industrie ---
display _n "--- 4.5 Top 10 sektoru ---"
tab sector if sector != "Unknown", sort

display _n "--- 4.6 Remote prace ---"
tab is_remote
tab is_remote has_ai, chi2 row


* ==============================================================================
* 5. ANALYTICKÉ TESTY - HYPOTÉZY
* ==============================================================================
* Statistické testy pro ověření hypotéz diplomové práce

display _n "=============================================================="
display "5. STATISTICKE TESTY"
display "=============================================================="

* --- 5.1 T-test: Liší se platy AI vs non-AI pozic? ---
* H0: Průměrný plat AI pozic = průměrný plat non-AI pozic
* H1: Průměrné platy se liší

display _n "--- 5.1 T-test: Plat AI vs non-AI pozic ---"
ttest salary_mid, by(has_ai)

* Efekt size (Cohenovo d) - důležité pro interpretaci praktické významnosti
quietly summarize salary_mid if has_ai == 0
local mean_no_ai = r(mean)
local sd_no_ai = r(sd)
local n_no_ai = r(N)

quietly summarize salary_mid if has_ai == 1  
local mean_ai = r(mean)
local sd_ai = r(sd)
local n_ai = r(N)

* Pooled standard deviation
local sd_pooled = sqrt(((`n_no_ai'-1)*`sd_no_ai'^2 + (`n_ai'-1)*`sd_ai'^2) / (`n_no_ai'+`n_ai'-2))
local cohens_d = (`mean_ai' - `mean_no_ai') / `sd_pooled'

display _n "Cohenovo d (effect size): " %5.3f `cohens_d'
display "Interpretace: |d| < 0.2 = maly, 0.2-0.8 = stredni, > 0.8 = velky efekt"

* --- 5.2 ANOVA: Liší se platy mezi AI tiers? ---
* Testuje rozdíly mezi none, ai_integration, ai_focused

display _n "--- 5.2 ANOVA: Plat podle AI tier ---"
oneway salary_mid ai_tier_num, tabulate bonferroni

* --- 5.3 Chi-square: Vzdělání a AI požadavky ---
* Jsou AI pozice náročnější na vzdělání?

display _n "--- 5.3 Chi-square: Vzdelani x AI tier ---"
tab edu_level has_ai if edu_level > 0, chi2 expected

* --- 5.4 Chi-square: Zkušenosti a AI požadavky ---
display _n "--- 5.4 Chi-square: Zkusenosti x AI tier ---"
tab exp_category has_ai, chi2 expected

* --- 5.5 Mann-Whitney U test (neparametrický) ---
* Pro případ, že platy nemají normální rozdělení

display _n "--- 5.5 Mann-Whitney U test: Plat AI vs non-AI ---"
ranksum salary_mid, by(has_ai)


* ==============================================================================
* 6. REGRESNÍ ANALÝZA
* ==============================================================================
* Modelování faktorů ovlivňujících plat a AI požadavky

display _n "=============================================================="
display "6. REGRESNI ANALYZA"
display "=============================================================="

* --- 6.1 OLS regrese: Co ovlivňuje plat? ---
* Závislá proměnná: salary_mid
* Nezávislé: has_ai, edu_level, experience_min_llm, is_remote

display _n "--- 6.1 OLS regrese: Determinanty platu ---"

regress salary_mid has_ai i.edu_level experience_min_llm is_remote

* Robustní standardní chyby (heteroskedasticita)
display _n "--- 6.1b OLS s robustnimi standardnimi chybami ---"
regress salary_mid has_ai i.edu_level experience_min_llm is_remote, vce(robust)

* --- 6.2 Rozšířený model s interakcemi ---
* AI × vzdělání interakce - má AI větší vliv u vyššího vzdělání?

display _n "--- 6.2 Model s interakcemi ---"
regress salary_mid c.has_ai##i.edu_level experience_min_llm is_remote, vce(robust)

* --- 6.3 Logistická regrese: Co predikuje AI požadavky? ---
* Které charakteristiky pozice souvisí s AI požadavky?

display _n "--- 6.3 Logisticka regrese: Prediktory AI pozadavku ---"
logit has_ai i.edu_level experience_min_llm is_remote, or

* Marginální efekty (interpretovatelné jako procentní změny)
display _n "--- 6.3b Marginalni efekty ---"
margins, dydx(*) atmeans


* ==============================================================================
* 7. ANALÝZA HARD SKILLS
* ==============================================================================
* Rozbor nejčastějších technických dovedností

display _n "=============================================================="
display "7. ANALYZA HARD SKILLS"
display "=============================================================="

* Poznámka: hardskills je textový sloupec s čárkami oddělenými skills
* Pro detailní analýzu je lepší použít Python k vytvoření dummy proměnných
* Zde ukážeme základní exploraci

* Počet unique skills na pozici (aproximace)
gen skill_count = 1 + length(hardskills) - length(subinstr(hardskills, ",", "", .))
replace skill_count = 0 if hardskills == ""
label variable skill_count "Pocet pozadovanych hard skills"

display _n "--- 7.1 Pocet skills na pozici ---"
summarize skill_count, detail
tabstat skill_count, by(desc_tier_llm) statistics(count mean sd min max)

* T-test: Vyžadují AI pozice více skills?
display _n "--- 7.2 T-test: Pocet skills AI vs non-AI ---"
ttest skill_count, by(has_ai)


* ==============================================================================
* 8. EXPORTY PRO TABULKY A GRAFY
* ==============================================================================
* Příprava dat pro publikovatelné tabulky

display _n "=============================================================="
display "8. EXPORTY"
display "=============================================================="

* --- 8.1 Summary statistics ---
display _n "--- 8.1 Summary statistics pro tabulky ---"
summarize salary_mid experience_min_llm edu_level skill_count has_ai is_remote

* Export do CSV pro další zpracování (grafy v Excelu/R/Python)
export delimited desc_tier_llm salary_mid experience_min_llm edulevel_llm state sector using "$outdir/summary_for_charts.csv" if salary_mid != ., delimiter(",") replace


* ==============================================================================
* 9. VIZUALIZACE
* ==============================================================================
* Základní grafy pro diplomovou práci

display _n "=============================================================="
display "9. VIZUALIZACE"
display "=============================================================="

* --- 9.1 Sloupcový graf: AI tier distribuce ---
graph bar (count), over(desc_tier_llm) ///
    title("Distribuce AI pozadavku v IT pozicich") ///
    ytitle("Pocet pozic") ///
    bar(1, color(navy))
graph export "$outdir/ai_tier_distribution.png", replace width(1200)

* --- 9.2 Box plot: Platy podle AI tier ---
graph box salary_mid, over(desc_tier_llm) ///
    title("Rozlozeni platu podle AI pozadavku") ///
    ytitle("Rocni plat (USD)")
graph export "$outdir/salary_by_ai_tier.png", replace width(1200)

* --- 9.3 Histogram: Požadované zkušenosti ---
histogram experience_min_llm if experience_min_llm < 15, ///
    by(has_ai, title("Pozadovane zkusenosti") note("")) ///
    xtitle("Roky zkusenosti") percent bin(15)
graph export "$outdir/experience_histogram.png", replace width(1200)

* --- 9.4 Vzdělání podle AI tier ---
graph bar (count), over(edu_level) over(has_ai) ///
    title("Vzdelavaci pozadavky: AI vs non-AI pozice") ///
    ytitle("Pocet pozic") ///
    legend(label(1 "non-AI") label(2 "AI pozice"))
graph export "$outdir/education_by_ai.png", replace width(1200)


* ==============================================================================
* 10. ZÁVĚR A ULOŽENÍ
* ==============================================================================

display _n "=============================================================="
display "ANALYZA DOKONCENA"
display "=============================================================="
display "Vystupy ulozeny do: $outdir"
display "Log soubor: $outdir/ai_skills_analysis.log"

* Ulož zpracovaný dataset pro další práci
save "$outdir/ai_skills_processed.dta", replace

* Zavři log
log close

* ==============================================================================
* KONEC DO-FILU
* ==============================================================================
