********************************************************************************
* AI SKILLS IN IT JOB POSTINGS - STATA ANALYSIS
* ==============================================================================
* Dataset: us_relevant_ai_stata.csv
* Autor: Yakub Murcek
* Datum: Leden 2026
* Stata 15.1 (IC/SE)
********************************************************************************

* ==============================================================================
* 1. NASTAVENÍ PROSTŘEDÍ
* ==============================================================================
* Pokus o automatické nastavení working directory
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

* Generovani casoveho razitka pro unikatni nazev slozky
local c_date = c(current_date)
local c_time = c(current_time)
local time_string = subinstr("`c_date'_`c_time'", " ", "_", .)
local time_string = subinstr("`time_string'", ":", "-", .)

* Nastav cestu k datům - UPRAV PODLE SVÉ STRUKTURY
global datadir "../../data/outputs"
global outdir "./output/run_`time_string'"

* Vytvoř output složku pokud neexistuje
capture mkdir "./output"
capture mkdir "$outdir"

* Log file - zaznamenává veškerý výstup pro pozdější kontrolu
capture log close
log using "$outdir/ai_skills_analysis.log", replace text


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

* --- 3.2 AI Flag (consistent with tier classification) ---
* has_ai is a simple alias of "tier is a real AI tier".
* Counts therefore equal ai_integration + applied_ai exactly,
* and match the totals used by ai_tier_num / ai_level downstream.
gen has_ai = (desc_tier_llm != "none" & desc_tier_llm != "missing")
label variable has_ai "AI Job (desc_tier_llm in {ai_integration, applied_ai})"

* --- 3.3 Vzdělání (Hybridní) ---
* Vytvoříme 'education_hybrid' z edulevel_llm (primární) a edu_level_det (fallback)
* POZOR: OLS (mzdový) model používá GRANULÁRNÍ vzdělání (edu_ols, 5 úrovní),
*        Mlogit model používá 2-úrovňové (edu_logit: Below_BA_or_Missing / BA+),
*        viz důvod v sekci 3.3b — HS/Assoc × Applied AI mělo jen 23 pozorování.
*        Missing je v obou samostatná kategorie — sloučení s "bez titulu" by
*        zkreslovalo závěry, protože AI inzeráty mají významně vyšší missingness.

* Krok 1: Normalizujeme edulevel_llm na lowercase a sjednotíme hodnoty
gen education_hybrid = lower(edulevel_llm)
replace education_hybrid = subinstr(education_hybrid, "'s", "", .)
replace education_hybrid = "highschool" if education_hybrid == "high school"
replace education_hybrid = "missing" if education_hybrid == "-" | education_hybrid == ""

* Krok 2: Fallback na edu_level_det pokud LLM chybí
replace education_hybrid = edu_level_det if education_hybrid == "missing" & edu_level_det != ""

* Krok 3: Finální missing pro prázdné
replace education_hybrid = "missing" if education_hybrid == ""

* Krok 4: Globální sloučení PhD → Master
replace education_hybrid = "master" if education_hybrid == "phd"

* --- 3.3a edu_ols: Granulární proměnná pro OLS (mzdovou) regresi ---
* Úrovně: 0=Missing, 1=High School, 2=Associate, 3=Bachelor, 4=Master(+PhD)
* Granulární 4 úrovně — zachování detailů zlepšuje model
gen edu_ols = .
replace edu_ols = 0 if education_hybrid == "missing"
replace edu_ols = 1 if education_hybrid == "highschool"
replace edu_ols = 2 if education_hybrid == "associate"
replace edu_ols = 3 if education_hybrid == "bachelor"
replace edu_ols = 4 if education_hybrid == "master"

label define edu_ols_lbl 0 "Missing" 1 "High School" 2 "Associate" 3 "Bachelor" 4 "Master+"
label values edu_ols edu_ols_lbl
label variable edu_ols "Vzdelani (granularni pro OLS)"

* --- 3.3b edu_logit: 2-úrovňová proměnná pro Mlogit ---
* Úrovně: 0 = Below BA or Missing (neuvedeno NEBO HS/Associate), 1 = Bachelor+
* Reference v mlogit: ib1.edu_logit (Bachelor+).
*
* Proč 2 úrovně místo 3 (puvodne Missing / HS+Assoc / BA+):
*   Krizova tabulka edu_logit_3level x ai_tier_num (log run 7-Apr-2026 14-04-33,
*   r. 642–667) ukazuje, ze HS/Associate × Applied AI = 23 a × AI Integration = 45.
*   Obe bunky jsou pod hranici 50 pozorovani potrebnou pro stabilni odhady v
*   multinomialnich modelech (viz § 5.1.1 prakticka_cast). Zhrubnuti na 2 urovne
*   posune nejmensi bunku Below_BA_or_Missing × Applied AI na 534+23 = 557 (>>50).
*
* Semanticka interpretace: dummy "Below_BA_or_Missing" je dominovana missingness
* signalem (713 HS/Assoc vs 6 364 Missing → ~90 % nove kategorie je Missing).
* Mlogit AME pro tuto dummy se proto interpretuje jako "BA+ vs unknown/below BA",
* nikoliv jako ciste "BA+ vs sub-BA". Pro popisnou analyzu vzdelani v § 5.1.3
* pouzivame dale granularni edu_ols (5 urovni), kde tato kompromisnost neni nutna.
gen edu_logit = .
replace edu_logit = 0 if inlist(education_hybrid, "missing", "", "highschool", "associate")
replace edu_logit = 1 if inlist(education_hybrid, "bachelor", "master")

label define edu_logit_lbl 0 "Below BA or Missing" 1 "Bachelor or Higher"
label values edu_logit edu_logit_lbl
label variable edu_logit "Vzdelani (2 urovne pro Mlogit)"

* Deskriptivni 3-urovnova promenna pro § 5.1.3 (NEZAVISLA na edu_logit).
* Drzime ji 3-urovnovou (Missing / HS+Assoc / BA+), aby v § 5.1.3 zustal
* zachovan chi-kvadrat test χ²(2) = 161,3 a aby missingness asymetrie zustala
* viditelna v deskriptivni tabulce.
gen edu_cat = .
replace edu_cat = 0 if inlist(education_hybrid, "missing", "")
replace edu_cat = 1 if inlist(education_hybrid, "highschool", "associate")
replace edu_cat = 2 if inlist(education_hybrid, "bachelor", "master")
label define edu_cat_lbl 0 "Missing" 1 "HS / Associate" 2 "Bachelor or Higher"
label values edu_cat edu_cat_lbl
label variable edu_cat "Pozadovane vzdelani (3 urovne, deskriptivni)"

* --- 3.4 Zkušenosti ---
* experience_min_llm by měl být již numerický, ale pro jistotu
destring experience_min_llm, replace force
label variable experience_min_llm "Min. pozadovane roky zkusenosti"

* Kategorie zkušeností
* Missing = neuvedeno v inzerátu (samostatná kategorie, jako u vzdělání)
gen exp_category = .
replace exp_category = 0 if experience_min_llm == .
* Slouceni Entry+Junior a Senior+Expert do spolecnych kategorii
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

* --- 3.16 Počet hard skills na pozici (kontinuální proměnná) ---
* Přesunuto sem ze sekce 7 — potřebujeme pro deskriptivní i regresní analýzy
gen skill_count = 1 + length(hardskills) - length(subinstr(hardskills, ",", "", .))
replace skill_count = 0 if hardskills == ""
label variable skill_count "Pocet pozadovanych hard skills"

* --- 3.17 Kvadratický člen zkušeností (Mincerova specifikace) ---
gen experience_sq = experience_min_llm^2
label variable experience_sq "Zkusenosti na druhou (Mincer)"

* --- 3.18 Logaritmus platu (pro OLS) ---
* Mincerova mzdová rovnice standardně používá ln(plat)
gen ln_salary = ln(salary_mid)
label variable ln_salary "Prirozeny logaritmus platu"

* --- 3.19 ID firmy (pro clusterované SE) ---
* company_id jiz existuje v importovanem CSV — neni treba encode
destring company_id, replace force
label variable company_id "ID firmy (pro cluster SE)"

* ==============================================================================
* 4. DESKRIPTIVNÍ STATISTIKA
* ==============================================================================
* Základní přehled datasetu

display _n "=============================================================="
display "4. DESKRIPTIVNI STATISTIKA"
display "=============================================================="

* --- 4.1 Frekvence AI tier klasifikace ---
* Kolik % pozic vyzaduje AI dovednosti?
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
display _n "Granularni distribuce vzdelani (vcetne Missing):"
tab edu_ols, missing

* Porovnání vzdělání podle AI tier (3 urovne + Missing jako vlastni kategorie)
display _n "Vzdelani x AI tier (kontingencni tabulka, 3 urovne):"
tab edu_cat ai_tier_num, chi2 column
display _n "Granularni vzdelani x AI tier (pro detail):"
tab edu_ols ai_tier_num, chi2 column

* --- 4.3 Požadované zkušenosti ---
display _n "--- 4.3 Distribuce pozadavku na zkusenosti ---"
summarize experience_min_llm, detail

tab exp_category, missing
tab exp_category ai_tier_num, chi2 column

* --- 4.4 Platy ---
display _n "--- 4.4 Distribuce platu ---"
summarize salary_mid, detail

* Platy podle AI tier — vychozi cisla pro argument o "AI premium"
display _n "Plat podle AI tier:"
tabstat salary_mid, by(desc_tier_llm) statistics(count mean sd min p25 p50 p75 max)

* --- 4.5 Sektory a industrie ---
display _n "--- 4.5 Top 10 sektoru ---"
tab sector if sector != "Unknown", sort

display _n "--- 4.6 Remote prace ---"
tab is_remote
display _n "Remote x AI (sloupcova procenta — % AI/non-AI pozic s remote):"
tab is_remote has_ai, chi2 column
display _n "Remote x AI tier (sloupcova procenta):"
tab is_remote ai_level, chi2 column

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

* --- 4.14b Seniority x AI (sloupcova procenta — slozeni AI/non-AI workforce) ---
display _n "--- 4.14b Seniority x AI (sloupcova procenta) ---"
display _n "Seniority x has_ai:"
tab exp_category has_ai, column chi2
display _n "Seniority x AI tier:"
tab exp_category ai_level, column chi2

* --- 4.14c Seniority x AI (radkova procenta — % AI penetrace v kazde urovni) ---
* Pro Word tabulky: kazdy radek (seniority level) se scita do 100 %
display _n "--- 4.14c Seniority x AI (radkova procenta) ---"
display _n "Seniority x has_ai (radkova %):"
tab exp_category has_ai, row chi2
display _n "Seniority x AI tier (radkova %):"
tab exp_category ai_level, row chi2

* --- 4.15 Diagnostika chybejicich platu ---
* POZN: nesignifikance testu nize NEZNAMENA absenci selection bias (ne MCAR).
* Testujeme pouze, zda missingness platu souvisi s AI tierem — ne s dalsimi
* observables. Viz sekce 4.15b nize, kde modelujeme missingness proti vice
* promennym, abychom ukazali, ze data NEJSOU MCAR.
display _n "--- 4.15 Diagnostika chybejicich platu (podle AI) ---"
gen has_salary = (salary_mid != .)
label variable has_salary "Ma uvedeny plat (1=ano)"
display _n "Chybejici platy x AI flag:"
tab has_salary has_ai, chi2 row
display _n "Chybejici platy x AI tier:"
tab has_salary desc_tier_llm, chi2 row
display "Pozn: nesignifikance zde znamena pouze, ze missingness se nelisi"
display "napric AI tiery — nikoli, ze neexistuje selection bias obecne."

* --- 4.15b Missingness platu x dalsi observables ---
* Ukazka, ze missingness platu NENI MCAR: lisi se napric regiony,
* sektory, velikosti firmy i remote statusem. Formalne testujeme logitem.
display _n "--- 4.15b Missingness platu x observables (ne-MCAR diagnostika) ---"
display _n "has_salary x region:"
tab has_salary region_num, chi2 row
display _n "has_salary x company size:"
tab has_salary size_cat, chi2 row
display _n "has_salary x sector (top-level):"
tab has_salary sector_nace_num, chi2 row
display _n "has_salary x remote:"
tab has_salary is_remote, chi2 row

display _n "--- 4.15c Logit modelu missingness: has_salary ~ observables ---"
display "Pokud jsou prediktory spolecne signifikantni, data NEJSOU MCAR."
logit has_salary ///
    i.ai_level ///
    i.sector_nace_num ///
    i.region_num ///
    i.size_cat ///
    i.type_cat ///
    is_remote ///
    i.job_family_num ///
    ib1.edu_logit ///
    ib3.exp_category, vce(robust)
display _n "Wald test spolecne signifikance vsech observables (krome AI):"
testparm i.sector_nace_num i.region_num i.size_cat i.type_cat is_remote ///
    i.job_family_num i.edu_logit i.exp_category


* ==============================================================================
* 5. ANALYTICKÉ TESTY — HYPOTÉZY
* ==============================================================================

display _n "=============================================================="
display "5. STATISTICKE TESTY"
display "=============================================================="

* --- 5.1 T-test: Plat AI vs non-AI ---
* H0: Průměrný plat AI pozic = průměrný plat non-AI pozic
* H1: Průměrné platy se liší

display _n "--- 5.1a T-test: Plat AI vs non-AI pozic (rovne variance) ---"
ttest salary_mid, by(has_ai)

display _n "--- 5.1b Welch t-test (nerovne variance) ---"
ttest salary_mid, by(has_ai) unequal

* Cohenovo d (effect size)
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

display _n "Cohenovo d: " %5.3f `cohens_d'
* Interpretace: |d| < 0.2 = maly, 0.2-0.8 = stredni, > 0.8 = velky efekt

* --- 5.2 ANOVA: Plat podle AI tier ---
* Rozdíly mezi none, ai_integration, applied_ai

display _n "--- 5.2a ANOVA: Plat podle AI tier ---"
oneway salary_mid ai_tier_num, tabulate bonferroni

* Neparametrická kontrola (Bartlett test je signifikantní, variance nejsou rovné)
display _n "--- 5.2b Robustni ANOVA (Kruskal-Wallis) ---"
kwallis salary_mid, by(ai_tier_num)

* --- 5.3 Chi-square: Vzdelani x AI tier ---
* Jsou AI pozice narocnejsi na vzdelani?
display _n "--- 5.3 Chi-square: Vzdelani x AI tier ---"
tab edu_cat has_ai, chi2 expected
display _n "Granularni vzdelani x AI (pro informaci):"
tab edu_ols has_ai, chi2 column

* --- 5.4 Chi-square: Zkušenosti a AI požadavky ---
display _n "--- 5.4 Chi-square: Zkusenosti x AI tier ---"
tab exp_category has_ai, chi2 expected

* --- 5.5 Mann-Whitney U test (neparametrický) ---
* Pro případ, že platy nemají normální rozdělení

display _n "--- 5.5 Mann-Whitney U test: Plat AI vs non-AI ---"
ranksum salary_mid, by(has_ai)


* ==============================================================================
* 6. REGRESNÍ ANALÝZA — OLS MODELY
* ==============================================================================
* Závislá proměnná: ln_salary
* Model A = firemní profil; Model B = Model A + lidský kapitál (edu, exp, job_family)

display _n "=============================================================="
display "6. REGRESNI ANALYZA — OLS MODELY"
display "=============================================================="

* --- 6.0 Kontrola log transformace (ln_salary vytvořeno v sekci 3.18) ---
* Koeficient b v log modelu: (exp(b)-1)*100 = % zmena platu
summarize ln_salary, detail

* --- 6.1 Model A: Základní OLS (Firemní a technologický profil) ---
* DV: ln_salary
* IV: cluster_* (soft/hard skills bločky), i.ai_level, i.sector_nace_num, i.region_num, is_remote, i.type_cat, i.size_cat

display _n "--- 6.1 Model A: Zakladni OLS ---"
display "ln(plat) ~ cluster_* + AI_level + sektor + region + remote + typ_firmy + velikost_firmy"

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

* --- 6.1b VIF diagnostika Model A ---
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

* --- 6.2 Model B: Rozšířený OLS (Lidský kapitál a přesná pozice) ---
display _n "--- 6.2 Model B: Rozsireny OLS (Profil uchazece) ---"
display "ln(plat) ~ Model A + job_family_num + edu_ols + exp_category"

regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    i.sector_nace_num ///
    i.region_num ///
    is_remote ///
    i.type_cat ///
    i.size_cat ///
    i.job_family_num ///
    i.edu_ols ///
    ib3.exp_category ///
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
    i.job_family_num ///
    i.edu_ols ///
    ib3.exp_category ///
    if ln_salary != .
vif

* -----------------------------------------------------------------------
* VARIANTY MODELU B
* 6.2e-f: Model B bez job_family (test mediace)
* 6.2g-h: Model B-Mincer (kontinualni zkusenosti)
* 6.2i:   Srovnavaci tabulka vsech OLS modelu
* Puvodni robustnostni kontroly nasleduji v 6.2c-d
* -----------------------------------------------------------------------

* --- 6.2e Model B bez job_family (test mediace) ---
* job_family muze byt mediator AI pozadavku (napr. "ML Engineer" v sobe primo kóduje AI skill).
display _n "--- 6.2e Model B bez job_family ---"
display "ln(plat) ~ Model B - job_family (test zda job_family mediuje AI premii)"
regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    i.sector_nace_num ///
    i.region_num ///
    is_remote ///
    i.type_cat ///
    i.size_cat ///
    i.edu_ols ///
    ib3.exp_category ///
    if ln_salary != ., vce(robust)
estimates store model_b_nojf
display _n "Model B-nojf: R2 = " e(r2) ", Adj R2 = " e(r2_a) ", N = " e(N)

* VIF pro Model B-nojf
display _n "--- 6.2f VIF diagnostika Model B bez job_family ---"
quietly regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    i.sector_nace_num ///
    i.region_num ///
    is_remote ///
    i.type_cat ///
    i.size_cat ///
    i.edu_ols ///
    ib3.exp_category ///
    if ln_salary != .
vif

* --- 6.2g Model B s kontinualni zkusenosti (Mincerova specifikace) ---
* Pridavame experience_min_llm + experience_sq misto kategoricke exp_category
* — kontinualni promenna pro spravnou specifikaci Mincerovy rovnice
* POZOR: Tento model MA JINY VZOREK nez Model B! Model B pouziva exp_category,
* kde missing zkusenosti = kategorie 0 (zachovano v regresi). Model B-Mincer
* pouziva kontinualni experience_min_llm, kde missing = . (Stata automaticky
* dropne). Proto N v B-Mincer bude nizsi. Porovnani R2 je orientacni.
display _n "--- 6.2g Model B s Mincerovou specifikaci (experience + experience^2) ---"
display "ln(plat) ~ Model B s kontinualni experience misto kategoricke"
display "POZN: Jiny vzorek — missing experience = dropped (vs. kategorie v Model B)"
regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    i.sector_nace_num ///
    i.region_num ///
    is_remote ///
    i.type_cat ///
    i.size_cat ///
    i.job_family_num ///
    i.edu_ols ///
    experience_min_llm experience_sq ///
    if ln_salary != ., vce(robust)
estimates store model_b_mincer
display _n "Model B-Mincer: R2 = " e(r2) ", Adj R2 = " e(r2_a) ", N = " e(N)

* VIF pro Model B-Mincer
display _n "--- 6.2h VIF diagnostika Model B Mincer ---"
quietly regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    i.sector_nace_num ///
    i.region_num ///
    is_remote ///
    i.type_cat ///
    i.size_cat ///
    i.job_family_num ///
    i.edu_ols ///
    experience_min_llm experience_sq ///
    if ln_salary != .
vif

* --- 6.2i Srovnavaci tabulka vsech OLS modelu ---
display _n "--- 6.2i Porovnani vsech OLS modelu ---"
estimates table model_a model_b_nojf model_b model_b_mincer, star stats(N r2 r2_a)

* --- 6.2c Robustnostni kontrola: Clusterovane SE ---
* Vice inzeratu od stejne firmy → korelovane chyby. Testujeme zda se SE meni.
display _n "--- 6.2c Robustnostni kontrola: Clusterovane SE ---"
regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    i.sector_nace_num ///
    i.region_num ///
    is_remote ///
    i.type_cat ///
    i.size_cat ///
    i.job_family_num ///
    i.edu_ols ///
    ib3.exp_category ///
    if ln_salary != ., vce(cluster company_id)
estimates store model_b_cluster
display _n "Porovnani: robust SE (Model B) vs cluster SE pro AI koeficienty"
estimates table model_b model_b_cluster, star stats(N r2 r2_a)

* --- 6.2d Interakcni efekt: AI x zkusenosti ---
* Testujeme zda AI premie se lisi podle seniority
display _n "--- 6.2d Interakcni efekt: AI x zkusenosti ---"
regress ln_salary ///
    cluster_* ///
    i.ai_level##ib3.exp_category ///
    i.sector_nace_num i.region_num is_remote ///
    i.type_cat i.size_cat ///
    i.job_family_num i.edu_ols ///
    if ln_salary != ., vce(robust)
display "Testujeme zda AI premie se lisi podle seniority"
testparm i.ai_level#ib3.exp_category

* --- 6.3 Srovnání modelů A vs B (Wald F-test nested models) ---
display _n "--- 6.3 Srovnani OLS modelu A vs B ---"
display "Wald F-test na robustnich SE: testuje zda edu + exp + job_family vyznamne"
display "zlepsuje model oproti zakladni specifikaci (cluster_* + firm profile)."
display "Pouzivame testparm po jedine regresi s plnou specifikaci a vce(robust),"
display "coz je korektni pristup pri heteroskedasticite (na rozdil od lrtest)."

quietly regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    i.sector_nace_num ///
    i.region_num ///
    is_remote ///
    i.type_cat ///
    i.size_cat ///
    i.job_family_num ///
    i.edu_ols ///
    ib3.exp_category ///
    if ln_salary != ., vce(robust)

display _n "Wald F-test: spolecna signifikance pridanych bloku (job_family + edu + exp)"
testparm i.job_family_num i.edu_ols i.exp_category

* --- 6.4 Porovnávací tabulka OLS modelů ---
display _n "--- 6.4 Porovnani koeficientu OLS Model A vs B ---"
estimates table model_a model_b, star stats(N r2 r2_a)


* ==============================================================================
* 6B. PRAVDĚPODOBNOSTNÍ MODELY (Multinomiální Logit)
* ==============================================================================
* DV: ai_level (multinomiální: 0=None, 1=AI Integration, 2=Applied/Core AI)
* is_remote zde NENÍ zahrnuta (je spíše výsledek než příčina AI požadavku).
*
* Model je exploratorní, nikoliv kauzální: cluster_* jsou extrahovány
* ze stejného textu inzerátu, ze kterého LLM přidělil ai_level.

display _n "=============================================================="
display "6B. PRAVDEPODOBNOSTNI MODELY — MULTINOMIALNI LOGIT"
display "=============================================================="

* -----------------------------------------------------------------------
* MODEL 1: Základní profil firmy (Sektor, Typ, Velikost, Lokace)
* -----------------------------------------------------------------------
* Jaký typ firmy a v jaké lokalitě vyžaduje AI?
display _n "--- 6B.1a Mlogit Model 1: Profil firmy ---"
mlogit ai_level ///
    i.sector_nace_num ///
    i.type_cat ///
    i.size_cat ///
    i.region_num, baseoutcome(0) rrr
estimates store mlogit_m1
display _n "--- 6B.1b Marginalni efekty Mlogit M1: P(AI Integration) ---"
margins, dydx(*) predict(outcome(1))
display _n "--- 6B.1c Marginalni efekty Mlogit M1: P(Applied/Core AI) ---"
margins, dydx(*) predict(outcome(2))

* -----------------------------------------------------------------------
* MODEL 2: Profil role a člověka (Skills, Pozice, Vzdělání, Praxe)
* -----------------------------------------------------------------------
* Souvisí požadavek na AI s typem dovedností a profilem uchazeče?
display _n "--- 6B.2a Mlogit Model 2: Profil role a cloveka ---"
mlogit ai_level ///
    cluster_* ///
    i.job_family_num ///
    ib1.edu_logit ///
    ib3.exp_category, baseoutcome(0) rrr
estimates store mlogit_m2
display _n "--- 6B.2b Marginalni efekty Mlogit M2: P(AI Integration) ---"
margins, dydx(*) predict(outcome(1))
display _n "--- 6B.2c Marginalni efekty Mlogit M2: P(Applied/Core AI) ---"
margins, dydx(*) predict(outcome(2))

* -----------------------------------------------------------------------
* MODEL 3: Kompletní (Model 1 + Model 2)
* -----------------------------------------------------------------------
* Firemní profil i profil role a uchazeče dohromady.
display _n "--- 6B.3a Mlogit Model 3: Kompletni ---"
mlogit ai_level ///
    i.sector_nace_num ///
    i.type_cat ///
    i.size_cat ///
    i.region_num ///
    cluster_* ///
    i.job_family_num ///
    ib1.edu_logit ///
    ib3.exp_category, baseoutcome(0) rrr
estimates store mlogit_m3
display _n "--- 6B.3b Marginalni efekty Mlogit M3: P(AI Integration) ---"
margins, dydx(*) predict(outcome(1))
display _n "--- 6B.3c Marginalni efekty Mlogit M3: P(Applied/Core AI) ---"
margins, dydx(*) predict(outcome(2))

* -----------------------------------------------------------------------
* DIAGNOSTIKA: Mlogit M3 BEZ edu_logit (rozhodovaci test)
* -----------------------------------------------------------------------
* Ucel: Zjistit, zda je legitimni zcela vyradit edu_logit z mlogit kvuli
* problemu s malymi bunkami (HS/Associate × Applied AI mel v puvodnim
* 3-urovnovem edu_logit jen 23 pozorovani — pod prahem 50).
*
* Rozhodovaci pravidlo:
*   Pokud Pseudo R² poklesne o <= 1 pb A marginalni efekty klicovych skill
*   clusteru (GenAI, DS/ML, Cloud, Dynamic Web) se posunou o < 0.5 pb oproti
*   M3 (s 2-urovnovym edu_logit), pak edu_logit z mlogit trvale odstranime
*   a v thesis explicitne uvedeme duvod.
display _n "--- 6B.3d DIAGNOSTIKA: Mlogit M3 BEZ edu_logit ---"
mlogit ai_level ///
    i.sector_nace_num ///
    i.type_cat ///
    i.size_cat ///
    i.region_num ///
    cluster_* ///
    i.job_family_num ///
    ib3.exp_category, baseoutcome(0) rrr
estimates store mlogit_m3_noedu
display _n "--- 6B.3e Marginalni efekty Mlogit M3 noedu: P(AI Integration) ---"
margins, dydx(*) predict(outcome(1))
display _n "--- 6B.3f Marginalni efekty Mlogit M3 noedu: P(Applied/Core AI) ---"
margins, dydx(*) predict(outcome(2))

display _n "--- 6B.3g Porovnani M3 vs M3_noedu (Pseudo R², LL, koeficienty) ---"
estimates table mlogit_m3 mlogit_m3_noedu, star stats(N ll chi2 r2_p)

* -----------------------------------------------------------------------
* Srovnávací tabulky Mlogit modelů
* -----------------------------------------------------------------------
display _n "--- 6B.4 Porovnani Mlogit modelu 1, 2, 3 ---"
estimates table mlogit_m1 mlogit_m2 mlogit_m3, star stats(N ll chi2)

* Hausman test IIA (na kompletním modelu 3)
display _n "--- 6B.6 Hausman test IIA (Model 3) ---"
quietly mlogit ai_level ///
    i.sector_nace_num i.type_cat i.size_cat i.region_num ///
    cluster_* i.job_family_num ib1.edu_logit ib3.exp_category, baseoutcome(0)
estimates store hausman_full
quietly mlogit ai_level ///
    i.sector_nace_num i.type_cat i.size_cat i.region_num ///
    cluster_* i.job_family_num ib1.edu_logit ib3.exp_category ///
    if ai_level != 1, baseoutcome(0)
estimates store hausman_reduced
capture noisily hausman hausman_reduced hausman_full, alleqs constant
if _rc {
    display _n "Hausman test selhal (return code: " _rc ")"
    display "Pozn: Hausman test casto selhava u mlogit s mnoha kategoriemi"
    display "kvuli non-positive-definite variancni matici (V_b-V_B)."
    display "Toto je znama limitace — neznamena to problem s modelem."
}
estimates drop hausman_full hausman_reduced

* -----------------------------------------------------------------------
* MODEL 3a: Kompletní BEZ job_family (test mediace)
* -----------------------------------------------------------------------
* job_family muze byt mediator — primo v sobe zahrnuje typ dovednosti
display _n "--- 6B.7a Mlogit Model 3a: Kompletni bez job_family ---"
mlogit ai_level ///
    i.sector_nace_num ///
    i.type_cat ///
    i.size_cat ///
    i.region_num ///
    cluster_* ///
    ib1.edu_logit ///
    ib3.exp_category, baseoutcome(0) rrr
estimates store mlogit_m3a
display _n "--- 6B.7b Marginalni efekty Mlogit M3a: P(AI Integration) ---"
margins, dydx(*) predict(outcome(1))
display _n "--- 6B.7c Marginalni efekty Mlogit M3a: P(Applied/Core AI) ---"
margins, dydx(*) predict(outcome(2))

* -----------------------------------------------------------------------
* MODEL 3b: Kompletní BEZ job_family A BEZ seniority (test mediace)
* -----------------------------------------------------------------------
* job_family a seniorita mohou primo v sobe zahrnovat pozadavky na skills — potencialni mediatory
display _n "--- 6B.8a Mlogit Model 3b: Bez job_family a seniority ---"
mlogit ai_level ///
    i.sector_nace_num ///
    i.type_cat ///
    i.size_cat ///
    i.region_num ///
    cluster_* ///
    ib1.edu_logit, baseoutcome(0) rrr
estimates store mlogit_m3b
display _n "--- 6B.8b Marginalni efekty Mlogit M3b: P(AI Integration) ---"
margins, dydx(*) predict(outcome(1))
display _n "--- 6B.8c Marginalni efekty Mlogit M3b: P(Applied/Core AI) ---"
margins, dydx(*) predict(outcome(2))

* -----------------------------------------------------------------------
* Srovnávací tabulky: Model 3 vs 3a vs 3b
* -----------------------------------------------------------------------
display _n "--- 6B.5 Porovnani Mlogit M3 vs M3a vs M3b ---"
estimates table mlogit_m3 mlogit_m3a mlogit_m3b, star stats(N ll chi2)

* -----------------------------------------------------------------------
* CITLIVOSTNÍ ANALÝZA: Mlogit BEZ cluster_generative_ai a cluster_data_science__ml
* -----------------------------------------------------------------------
* Test cirkularity: cluster_generative_ai a cluster_data_science__ml primo implikuji
* AI pozadavek (GPT, LLM, TensorFlow, PyTorch...). Vyradime je pres docasne prejmenovani,
* aby je cluster_* wildcard nezachytil.
* POZOR: pokud kod selze mezi rename a unrename, promenne zustanou prejmenovane
* a zbytek do-filu nepujde spustit. V pripade chyby rucne spustte:
*   rename _excl_genai cluster_generative_ai
*   rename _excl_dsml cluster_data_science__ml

display _n "=============================================================="
display "6C. CITLIVOSTNI ANALYZA — MLOGIT BEZ GenAI A DS/ML CLUSTERU"
display "=============================================================="

rename cluster_generative_ai _excl_genai
rename cluster_data_science__ml _excl_dsml

display _n "--- 6C.1a Mlogit M3 bez GenAI a DS/ML ---"
mlogit ai_level ///
    i.sector_nace_num ///
    i.type_cat ///
    i.size_cat ///
    i.region_num ///
    cluster_* ///
    i.job_family_num ///
    ib1.edu_logit ///
    ib3.exp_category, baseoutcome(0) rrr
estimates store mlogit_m3_nocirc
display _n "--- 6C.1b Marginalni efekty Mlogit M3 nocirc: P(AI Integration) ---"
margins, dydx(*) predict(outcome(1))
display _n "--- 6C.1c Marginalni efekty Mlogit M3 nocirc: P(Applied/Core AI) ---"
margins, dydx(*) predict(outcome(2))

* Vratit prejmenovane clustery
rename _excl_genai cluster_generative_ai
rename _excl_dsml cluster_data_science__ml

* Srovnani koeficientu s a bez cirkularnich prediktoru
display _n "--- 6C.2 Porovnani Mlogit M3 vs M3-nocirc ---"
estimates table mlogit_m3 mlogit_m3_nocirc, star stats(N ll chi2)


* ==============================================================================
* 7. ANALÝZA HARD SKILLS
* ==============================================================================
* Rozbor nejčastějších technických dovedností

display _n "=============================================================="
display "7. ANALYZA HARD SKILLS"
display "=============================================================="

* hardskills je textovy sloupec s carkami oddelenymi skills; pro detailni
* analyzu (dummy promenne per skill) je lepsi pouzit Python.
* skill_count je jiz vytvoren v sekci 3.16

display _n "--- 7.1 Pocet skills na pozici ---"
summarize skill_count, detail
tabstat skill_count, by(desc_tier_llm) statistics(count mean sd min max)

* Vyžadují AI pozice více skills než non-AI pozice?
display _n "--- 7.2a T-test: Pocet skills AI vs non-AI (rovne variance) ---"
ttest skill_count, by(has_ai)

display _n "--- 7.2b Welch t-test: Pocet skills (nerovne variance) ---"
ttest skill_count, by(has_ai) unequal


* ==============================================================================
* 8. EXPORTY PRO TABULKY A GRAFY
* ==============================================================================

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
display "Log soubor: $outdir/ai_skills_analysis.log"

* Ulož zpracovaný dataset pro další práci
save "$outdir/ai_skills_processed.dta", replace

log close
