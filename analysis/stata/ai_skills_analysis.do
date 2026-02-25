********************************************************************************
* AI SKILLS IN IT JOB POSTINGS - STATA ANALYSIS
* ==============================================================================
* Dataset: us_relevant_ai_stata.csv
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
* AUTO-SETUP: Pokus o automatické nastavení složky
capture {
    local userprofile : environment USERPROFILE
    * Převod zpětných lomítek na dopředná pro jistotu
    local userprofile = subinstr("`userprofile'", "\\", "/", .)
    cd "`userprofile'/Projects/glassdoor-jobs-ai-skills/analysis/stata"
}
if _rc {
    display as text "Pozor: Nepodařilo se automaticky nastavit složku. Ujistěte se, že jste v 'analysis/stata'"
}
display "Aktuální pracovní složka: " c(pwd)

version 15.1                              
clear all
set more off                              
set max_memory 8g, permanently            

* Nastav cestu k datům - UPRAV PODLE SVÉ STRUKTURY
global datadir "../../data/outputs"
global outdir "./output"

* Vytvoř output složku pokud neexistuje
capture mkdir "$outdir"

* Log file - zaznamenává veškerý výstup pro pozdější kontrolu
capture log close
* Generovani casoveho razitka pro unikatni nazev log souboru
local c_date = c(current_date)
local c_time = c(current_time)
local time_string = subinstr("`c_date'_`c_time'", " ", "_", .)
local time_string = subinstr("`time_string'", ":", "-", .)
log using "$outdir/ai_skills_analysis_`time_string'.log", replace text


* ==============================================================================
* 2. IMPORT DAT
* ==============================================================================
* Používáme soubor 'us_relevant_ai_stata.csv'
import delimited "$datadir/us_relevant_ai_stata.csv", delimiter(";") clear varnames(1) encoding(utf8)

* Základní kontrola úspěšného importu
describe
display "Počet pozorování (job postů): " _N


* ==============================================================================
* 3. ČIŠTĚNÍ A PŘÍPRAVA DAT
* ==============================================================================

* --- 3.0 Filtrování (podle požadavku uživatele) ---
* Ponecháme pouze pozice s vysokou confidencí modelu
keep if desc_conf_llm >= 0.7
display "Počet pozorování po filtrování confidence >= 0.7: " _N

* Vyloučení starých okrajových inzerátů (roky 2022 a 2023)
* Získání reálného data publikace odečtením stáří
gen date_format_discover = date(substr(discover_date, 1, 10), "YMD")
format date_format_discover %td
gen real_post_date = date_format_discover - age_in_days
format real_post_date %td
gen post_year = year(real_post_date)
drop if post_year <= 2023
display "Počet pozorování po vyřazení inzerátů starých 3+ roky (2022, 2023): " _N

* --- 3.1 AI Tier klasifikace ---
* desc_tier_llm obsahuje kategorie: "none", "ai_integration", "applied_ai", "core_ai"
replace desc_tier_llm = "missing" if desc_tier_llm == ""

* Merge core_ai into applied_ai (only 6 obs — below 50-obs threshold)
replace desc_tier_llm = "applied_ai" if desc_tier_llm == "core_ai"

encode desc_tier_llm, generate(ai_tier_num)

* --- 3.2 AI Flag (Strict Intersection + Buzzword Filter) ---
* 1. Sloučime zdroje skills do jednoho retezce pro kontrolu
gen skills_combined = lower(skills_ai_det + " " + desc_ai_llm)

* 2. Odstranime obecne buzzwords (AI, ML, Artificial Intelligence atd.)
* Pouzivame regex s word boundaries \b aby se smazalo jen "AI" a ne "OpenAI"
gen skills_no_buzz = ustrregexra(skills_combined, "(?i)\b(ai|ml|artificial intelligence|machine learning|genai)\b", "")

* 3. Odstranime interpunkci (carky, mezery), abychom zarucili ze zbyva neco realneho
gen skills_cleaned = subinstr(skills_no_buzz, ",", "", .)
replace skills_cleaned = subinstr(skills_cleaned, " ", "", .)
replace skills_cleaned = subinstr(skills_cleaned, ";", "", .)

* 4. Definice has_ai_flag
* Tier != None A ZAROVEN zbyly nejake specificke skills (delka > 1 znaku)
gen has_ai_flag = ((desc_tier_llm != "none" & desc_tier_llm != "missing") & length(skills_cleaned) > 1)

label variable has_ai_flag "AI Job (Tier + Valid Skills, no buzzwords)"

* Kompatibilita pro starší skripty (volitelné)
gen has_ai = has_ai_flag 

* --- 3.3 Vzdělání (Hybridní) ---
* Vytvoříme 'education_hybrid' z edulevel_llm (primární) a edu_level_det (fallback)
* Výsledné hodnoty: "missing", "highschool", "associate", "bachelor", "master", "phd"
* POZOR: encode řadí abecedně, proto kódujeme manuálně (ordinální pořadí)

* Krok 1: Normalizujeme edulevel_llm na lowercase a sjednotíme hodnoty
gen education_hybrid = lower(edulevel_llm)
replace education_hybrid = subinstr(education_hybrid, "'s", "", .)
replace education_hybrid = "highschool" if education_hybrid == "high school"
replace education_hybrid = "missing" if education_hybrid == "-" | education_hybrid == ""

* Krok 2: Fallback na edu_level_det pokud LLM chybí
replace education_hybrid = edu_level_det if education_hybrid == "missing" & edu_level_det != ""

* Krok 3: Finální missing pro prázdné
replace education_hybrid = "missing" if education_hybrid == ""

gen edu_cat = .
replace edu_cat = 0 if inlist(education_hybrid, "missing", "", "highschool", "associate")
replace edu_cat = 1 if inlist(education_hybrid, "bachelor", "master", "phd")

label define edu_lbl 0 "No Degree / Missing" 1 "Bachelor or Higher"
label values edu_cat edu_lbl
label variable edu_cat "Pozadovane vzdelani (Binary)"

* --- 3.4 Zkušenosti ---
* experience_min_llm by měl být již numerický, ale pro jistotu
destring experience_min_llm, replace force
label variable experience_min_llm "Min. pozadovane roky zkusenosti"

* Kategorie zkušeností
* Missing = neuvedeno v inzerátu (samostatná kategorie, jako u vzdělání)
gen exp_category = .
replace exp_category = 0 if experience_min_llm == .
* Slouceni Entry s Junior a Expert se Senior
replace exp_category = 2 if experience_min_llm >= 0 & experience_min_llm <= 2
replace exp_category = 3 if experience_min_llm > 2 & experience_min_llm <= 5
replace exp_category = 4 if experience_min_llm > 5 & experience_min_llm < .

label define exp_lbl 0 "Missing" 2 "Junior (0-2)" 3 "Mid (3-5)" 4 "Senior+ (6+)"
label values exp_category exp_lbl
label variable exp_category "Kategorie seniority"

* --- 3.5 Plat ---
* Převod na roční plat podle pay_period
* US standard: 2080 hodin/rok (40h/týden × 52 týdnů), 12 měsíců/rok
destring salary_min salary_mid salary_max, replace force

* Přepočet hodinových mezd na roční (HOURLY × 2080)
replace salary_min = salary_min * 2080 if pay_period == "HOURLY"
replace salary_mid = salary_mid * 2080 if pay_period == "HOURLY"
replace salary_max = salary_max * 2080 if pay_period == "HOURLY"

* Přepočet měsíčních mezd na roční (MONTHLY × 12)
replace salary_min = salary_min * 12 if pay_period == "MONTHLY"
replace salary_mid = salary_mid * 12 if pay_period == "MONTHLY"
replace salary_max = salary_max * 12 if pay_period == "MONTHLY"

* Filtr outlierů (po přepočtu na roční bázi)
replace salary_mid = . if salary_mid < 20000 | salary_mid > 500000
label variable salary_mid "Rocni plat - stredni hodnota (USD)"

* --- 3.6 Sektor a Lokace ---
replace sector = "Unknown" if sector == ""
replace industry = "Unknown" if industry == ""
encode sector, generate(sector_num)

replace state = "Unknown" if state == ""
encode state, generate(state_num)

* --- 3.8 Velikost firmy (ordinální) ---
replace size = "Unknown" if size == ""
gen size_cat = .
replace size_cat = 0 if size == "Unknown"
replace size_cat = 1 if size == "1 to 50 Employees"
replace size_cat = 2 if size == "51 to 200 Employees"
replace size_cat = 3 if size == "201 to 500 Employees"
replace size_cat = 4 if size == "501 to 1000 Employees"
replace size_cat = 5 if size == "1001 to 5000 Employees"
replace size_cat = 6 if size == "5001 to 10000 Employees"
replace size_cat = 7 if size == "10000+ Employees"

label define size_lbl 0 "Unknown" 1 "1-50" 2 "51-200" 3 "201-500" ///
    4 "501-1000" 5 "1001-5000" 6 "5001-10000" 7 "10000+"
label values size_cat size_lbl
label variable size_cat "Velikost firmy (ordinalni)"

* --- 3.9 Typ firmy (nominální, sloučený) ---
gen type_cat = .
replace type_cat = 0 if inlist(type, "", "Unknown", "Contract", "Self-employed", "Private Practice / Firm", "Franchise") // Merge Other to Unknown
replace type_cat = 1 if inlist(type, "Company - Private", "Subsidiary or Business Segment") // Merge Subsidiary to Private
replace type_cat = 2 if type == "Company - Public"
replace type_cat = 4 if inlist(type, "Nonprofit Organization", "Government", ///
    "College / University", "School / School District", "Hospital")

label define type_lbl 0 "Unknown/Other" 1 "Private/Subsidiary" 2 "Public" ///
    4 "Nonprofit/Gov/Edu"
label values type_cat type_lbl
label variable type_cat "Typ firmy"

* --- 3.10 NACE sektor (přichází z clean-stata CLI, jen encode) ---
replace sector_nace = "Unknown" if sector_nace == ""

* Zachovat top 5 sektorů + Unknown, vše ostatní sloučit do "Other" (kvůli < 50 obs v Applied AI)
replace sector_nace = "Other" if !inlist(sector_nace, "J", "C", "K", "M", "Q", "Unknown")

encode sector_nace, generate(sector_nace_num)
label variable sector_nace_num "NACE sektor"

* --- 3.11 Rok založení firmy ---
destring year_founded, replace force
label variable year_founded "Rok zalozeni firmy"

* --- 3.7 Remote práce ---
gen is_remote = 0
replace is_remote = 1 if strpos(lower(remote_work_types), "home") > 0
replace is_remote = 1 if strpos(lower(remote_work_types), "remote") > 0

label variable is_remote "Moznost remote prace (1=ano)"

* --- 3.12 AI Level (multinomiální závislá proměnná) ---
* Tříkategoriální proměnná pro multinomiální logit:
*   0 = žádné AI požadavky (none)
*   1 = používá AI nástroje (ai_integration)
*   2 = vyvíjí/buduje AI (applied_ai — core_ai already merged above)
gen ai_level = 0
replace ai_level = 1 if desc_tier_llm == "ai_integration"
replace ai_level = 2 if desc_tier_llm == "applied_ai"

label define ailevel_lbl 0 "None" 1 "AI Integration" 2 "Applied/Core AI"
label values ai_level ailevel_lbl
label variable ai_level "Uroven AI pozadavku (0/1/2)"

* --- 3.13 Census Region (přichází z clean-stata CLI, jen encode) ---
replace region = "Unknown" if region == ""
encode region, generate(region_num)
label variable region_num "Census region"

* --- 3.14 Job Family (přichází z clean-stata CLI, jen encode) ---
replace job_family = "Unknown" if job_family == ""

* Sloučení řídkých technických rolí do "Other" (kvůli < 50 obs v Applied AI)
replace job_family = "Other" if inlist(job_family, "Frontend & Design", "QA & Testing", "Security", "Systems & Embedded")

encode job_family, generate(job_family_num)
label variable job_family_num "Rodina pozice"

* --- 3.15 Vyřazení neplatných skill clusterů ---
* Tyto clustery mají v kategorii Applied/Core AI méně než 50 pozorování, což by narušilo model.
* Proto je mažeme, aby je nezahrnoval wildcard cluster_* v regresních modelech.
drop cluster_legacy__mainframe
drop cluster_data_analysis__stats
drop cluster_tools__editors

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
tab edu_cat, missing
tab edu_cat if edu_cat > 0

* Porovnání vzdělání podle AI tier
display _n "Vzdelani x AI tier (kontingencni tabulka):"
tab edu_cat ai_tier_num if edu_cat > 0, chi2 column

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

* --- 4.7 Velikost firmy ---
display _n "--- 4.7 Velikost firmy ---"
tab size_cat, missing

* --- 4.8 Typ firmy ---
display _n "--- 4.8 Typ firmy ---"
tab type_cat, missing

* --- 4.9 NACE sektor ---
display _n "--- 4.9 NACE sektor ---"
tab sector_nace if sector_nace != "Unknown", sort

* --- 4.10 Rok zalozeni ---
display _n "--- 4.10 Rok zalozeni firmy ---"
summarize year_founded, detail

* --- 4.11 Census Region ---
display _n "--- 4.11 Census Region ---"
tab region, missing
tab region has_ai, chi2 column

* --- 4.12 Job Family ---
display _n "--- 4.12 Job Family ---"
tab job_family, missing
tab job_family has_ai, chi2 column

* --- 4.13 Skill Cluster frequencies ---
display _n "--- 4.13 Pocet pozic s danym skill clusterem ---"
foreach var of varlist cluster_* {
    quietly count if `var' == 1
    display "`var': " r(N) " jobs (" %4.1f r(N)/_N*100 "%)"
}

* --- 4.14 Cross-tabs of model IVs with ai_level (check for empty cells) ---
display _n "--- 4.14 Krizove tabulky pro kontrolu prazdnych bunek ---"
display _n "NACE sektor x AI level:"
tab sector_nace ai_level, column
display _n "Typ firmy x AI level:"
tab type_cat ai_level, column
display _n "Velikost firmy x AI level:"
tab size_cat ai_level, column
display _n "Region x AI level:"
tab region ai_level, column
display _n "Job Family x AI level:"
tab job_family ai_level, column


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
tab edu_cat has_ai if edu_cat > 0, chi2 expected

* --- 5.4 Chi-square: Zkušenosti a AI požadavky ---
display _n "--- 5.4 Chi-square: Zkusenosti x AI tier ---"
tab exp_category has_ai, chi2 expected

* --- 5.5 Mann-Whitney U test (neparametrický) ---
* Pro případ, že platy nemají normální rozdělení

display _n "--- 5.5 Mann-Whitney U test: Plat AI vs non-AI ---"
ranksum salary_mid, by(has_ai)


* ==============================================================================
* 6. REGRESNÍ ANALÝZA — MODELY VEDOUCÍHO
* ==============================================================================
* log(mzda) = skill clustery + AI tier + sektor + lokace + remote + typ + velikost
*             + (vzdělání, zkušenosti, pozice)
* Závorka = Model B (plný), bez závorky = Model A (base)

display _n "=============================================================="
display "6. REGRESNI ANALYZA — MODELY VEDOUCIHO"
display "=============================================================="

* --- 6.0 Příprava: log transformace platu ---
gen ln_salary = ln(salary_mid)
label variable ln_salary "Prirozeny logaritmus platu"

summarize ln_salary, detail
display _n "Pozn: koeficient b v log modelu = (exp(b)-1)*100 % zmena platu"

* --- 6.1 Model A: Base model (bez individuálních charakteristik) ---
* DV: ln(salary_mid)
* IV: 24 skill clusterů + AI level + sektor NACE + region + remote + typ + velikost

display _n "--- 6.1 Model A: Base model ---"
display "ln(plat) = skill clustery + AI level + sektor + region + remote + typ + velikost"

regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    i.sector_nace_num ///
    i.region_num ///
    is_remote ///
    i.type_cat ///
    i.size_cat ///
    if ln_salary != ., vce(robust)

estimates store model_a
display _n "Model A: R2 = " e(r2) ", Adj R2 = " e(r2_a) ", N = " e(N)

* --- 6.1b VIF diagnostika (kolinearita) ---
display _n "--- 6.1b VIF diagnostika Model A ---"
quietly regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    i.sector_nace_num ///
    i.region_num ///
    is_remote ///
    i.type_cat ///
    i.size_cat ///
    if ln_salary != .
vif

* --- 6.2 Model B: Plný model (+vzdělání, zkušenosti, pozice) ---
display _n "--- 6.2 Model B: Plny model ---"
display "ln(plat) = Model A + vzdelani + zkusenosti + pozice (job family)"

regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    i.sector_nace_num ///
    i.region_num ///
    is_remote ///
    i.type_cat ///
    i.size_cat ///
    i.edu_cat ///
    i.exp_category ///
    i.job_family_num ///
    if ln_salary != ., vce(robust)

estimates store model_b
display _n "Model B: R2 = " e(r2) ", Adj R2 = " e(r2_a) ", N = " e(N)

* --- 6.2b VIF diagnostika Model B ---
display _n "--- 6.2b VIF diagnostika Model B ---"
quietly regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    i.sector_nace_num ///
    i.region_num ///
    is_remote ///
    i.type_cat ///
    i.size_cat ///
    i.edu_cat ///
    i.exp_category ///
    i.job_family_num ///
    if ln_salary != .
vif

* --- 6.3 Srovnání modelů A vs B (F-test nested models) ---
display _n "--- 6.3 Srovnani modelu A vs B ---"
display "F-test: testuje zda edu + exp + job_family vyznamne zlepsuje model"

quietly regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    i.sector_nace_num ///
    i.region_num ///
    is_remote ///
    i.type_cat ///
    i.size_cat ///
    if ln_salary != .
estimates store model_a_lr

quietly regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    i.sector_nace_num ///
    i.region_num ///
    is_remote ///
    i.type_cat ///
    i.size_cat ///
    i.edu_cat ///
    i.exp_category ///
    i.job_family_num ///
    if ln_salary != .
estimates store model_b_lr

lrtest model_a_lr model_b_lr
estimates drop model_a_lr model_b_lr

* --- 6.4 Porovnávací tabulka modelů ---
display _n "--- 6.4 Porovnani koeficientu Model A vs B ---"
estimates table model_a model_b, star stats(N r2 r2_a)


* ==============================================================================
* 6B. JEDNODUCHÉ MODELY (původní — pro srovnání)
* ==============================================================================

display _n "=============================================================="
display "6B. JEDNODUCHE MODELY (puvodni)"
display "=============================================================="

* --- 6B.1 OLS: plat ~ has_ai + edu + exp + remote ---
display _n "--- 6B.1 OLS: Determinanty platu (jednoduchy model) ---"
regress salary_mid has_ai ib1.edu_cat ib3.exp_category is_remote, vce(robust)

* --- 6B.2 Logistická regrese: Prediktory AI požadavků ---
display _n "--- 6B.2 Logisticka regrese: Prediktory AI pozadavku ---"
logit has_ai ib1.edu_cat ib3.exp_category is_remote, or
display _n "--- 6B.2b Marginalni efekty ---"
margins, dydx(*) atmeans

* --- 6B.3 Multinomiální logit ---
display _n "--- 6B.3 Multinomialni logit: Uroven AI pozadavku ---"
mlogit ai_level ib1.edu_cat ib3.exp_category is_remote, baseoutcome(0) rrr
display _n "--- 6B.3b Marginalni efekty: P(AI Integration) ---"
margins, dydx(*) predict(outcome(1)) atmeans
display _n "--- 6B.3c Marginalni efekty: P(Applied/Core AI) ---"
margins, dydx(*) predict(outcome(2)) atmeans

* Hausman test IIA
display _n "--- 6B.3d Hausman test IIA ---"
quietly mlogit ai_level ib1.edu_cat ib3.exp_category is_remote, baseoutcome(0)
estimates store full_model
quietly mlogit ai_level ib1.edu_cat ib3.exp_category is_remote if ai_level != 1, baseoutcome(0)
estimates store reduced_model
capture hausman reduced_model full_model, alleqs constant
if _rc {
    display "Hausman test nelze provest (typicke pro male vzorky v nekterych kategoriich)"
}
estimates drop full_model reduced_model


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
summarize salary_mid experience_min_llm edu_cat exp_category skill_count has_ai is_remote

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
graph bar (count), over(edu_cat) over(has_ai) ///
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
display "Log soubor: $outdir/ai_skills_analysis_`time_string'.log"

* Ulož zpracovaný dataset pro další práci
save "$outdir/ai_skills_processed.dta", replace

* Zavři log
log close

* ==============================================================================
* KONEC DO-FILU
* ==============================================================================
