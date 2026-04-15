********************************************************************************
* AI SKILLS — FINALNI ANALYZA PRO DIPLOMOVOU PRACI (US vs DE vs IN)
* ==============================================================================
* Cilem tohoto do-file je generovat 5 hlavnich tabulek a grafy pro praktickou
* cast diplomky. Vsechny modely jsou PLNE (ne inkrementalni), reportuji se AME
* pro logit i mlogit a separatne per zeme tam, kde to zadani pozaduje.
*
* Vystupy:
*   Tabulka 1: Vyskyt AI pozadavku po zemich (sloupcova %)
*   Tabulka 2: Binarni logit P(AI) ~ job family + controls (AME, per zeme)
*   Tabulka 3: Binarni logit P(AI) ~ skill clustery + controls (AME, per zeme)
*   Tabulka 4: Mlogit P(AI tier) ~ skill clustery + controls (AME, 9 sloupcu)
*   Tabulka 5: OLS ln(plat) ~ skill clustery + AI tiery + controls (per zeme)
*   Grafy:     kernel density ln(plat) x ai_level per zeme
*
* Kontroly (neukazovane v tabulkach): NACE sektor, typ firmy, velikost firmy,
*   remote, region FE (jen US v OLS). Vzdelani: edu_bin (Bc.+). Praxe: exp_bin
*   (>=1 rok). Referencni zeme v pooled modelech je US.
*
* Klicova uprava oproti hlavnimu do-file:
*   - cluster_generative_ai -> AI Integration tier override (§3)
*   - Pouze jedna specifikace per tabulka (plny model), zadne inkrementalni varianty
*   - cluster_generative_ai vyrazen z RHS logit/mlogit/OLS kvuli cirkularite
*
* Autor: Yakub Murcek
* Datum: Duben 2026
* Stata 15.1
********************************************************************************


* ==============================================================================
* 1. NASTAVENI PROSTREDI
* ==============================================================================
capture {
    local userprofile : environment USERPROFILE
    local userprofile = subinstr("`userprofile'", "\\", "/", .)
    cd "`userprofile'/Projects/glassdoor-jobs-ai-skills/analysis/stata"
}
if _rc {
    display as text "Pozor: Nepodarilo se automaticky nastavit slozku. Ujistete se, ze jste v 'analysis/stata'"
}
display "Aktualni pracovni slozka: " c(pwd)

version 15.1
clear all
set more off
set max_memory 8g, permanently

local c_date = c(current_date)
local c_time = c(current_time)
local time_string = subinstr("`c_date'_`c_time'", " ", "_", .)
local time_string = subinstr("`time_string'", ":", "-", .)

global datadir "../../data/outputs"
global outdir "./output/thesis_final_run_`time_string'"

capture mkdir "./output"
capture mkdir "$outdir"

capture log close
log using "$outdir/ai_skills_thesis_final.log", replace text


* ==============================================================================
* 2. IMPORT A APPEND TREMI DATASETU (US + DE + IN)
* ==============================================================================
display _n "=============================================================="
display "2. IMPORT A APPEND DATASETU"
display "=============================================================="

import delimited "$datadir/us_relevant_ai_stata.csv", delimiter(";") clear varnames(1) encoding(utf8)
capture drop country
gen country = "US"
display "US import: " _N " pozorovani"
tempfile us_data
save `us_data'

import delimited "$datadir/de/de_relevant_ai_stata.csv", delimiter(";") clear varnames(1) encoding(utf8)
capture drop country
gen country = "DE"
display "DE import: " _N " pozorovani"
tempfile de_data
save `de_data'

import delimited "$datadir/in_relevant_ai_stata.csv", delimiter(";") clear varnames(1) encoding(utf8)
capture drop country
gen country = "IN"
display "IN import: " _N " pozorovani"
tempfile in_data
save `in_data'

use `us_data', clear
append using `de_data', force
append using `in_data', force

display _n "Pocet pozorovani celkem po spojeni (US+DE+IN): " _N
tab country, missing

encode country, generate(country_id)
label variable country_id "Zeme povodu inzeratu (DE=1, IN=2, US=3)"
* POZN: V regresich pouzivame ib3.country_id (baseline = US).


* ==============================================================================
* 3. CISTENI A PRIPRAVA DAT (vcetne GenAI override)
* ==============================================================================
display _n "=============================================================="
display "3. CISTENI A PRIPRAVA DAT"
display "=============================================================="

* --- 3.0 Confidence filter a vyrazeni starych inzeratu ---
keep if desc_conf_llm >= 0.7
display "Po filtru confidence >= 0.7: " _N

gen date_format_discover = date(substr(discover_date, 1, 10), "YMD")
format date_format_discover %td
destring age_in_days, replace force
gen real_post_date = date_format_discover - age_in_days
format real_post_date %td
gen post_year = year(real_post_date)
drop if post_year <= 2023
display "Po vyrazeni <= 2023: " _N

* --- 3.1 AI tier klasifikace ---
replace desc_tier_llm = "missing" if desc_tier_llm == ""
replace desc_tier_llm = "applied_ai" if desc_tier_llm == "core_ai"

* --- 3.2 has_ai + ai_level ---
gen has_ai = (desc_tier_llm != "none" & desc_tier_llm != "missing")
label variable has_ai "AI Job (1=AI Integration nebo Applied/Core AI)"

gen ai_level = 0
replace ai_level = 1 if desc_tier_llm == "ai_integration"
replace ai_level = 2 if desc_tier_llm == "applied_ai"
label define ailevel_lbl 0 "None" 1 "AI Integration" 2 "Applied/Core AI"
label values ai_level ailevel_lbl
label variable ai_level "Uroven AI pozadavku (0/1/2)"

* --- 3.2b GenAI skill cluster -> AI Integration tier override ---
* Pokud inzerat ma cluster_generative_ai == 1 a je v None tieru, prirad do AI Integration.
* Pravidlo NEPONIZUJE inzeraty, ktere uz jsou v Applied/Core AI.
* Dusledek: cluster_generative_ai je mechanicky vazan na ai_level>=1 a musi byt
* vyrazen z RHS regresnich modelu, kde ai_level/has_ai je LHS (cirkularita).
display _n "--- 3.2b GenAI -> AI Integration override ---"
display "Pred override: inzeraty s cluster_generative_ai==1 podle ai_level:"
tab ai_level if cluster_generative_ai == 1, missing

count if cluster_generative_ai == 1 & ai_level == 0
local n_genai_override = r(N)
replace ai_level = 1 if cluster_generative_ai == 1 & ai_level == 0
replace has_ai = 1 if ai_level >= 1
replace desc_tier_llm = "ai_integration" if ai_level == 1 & desc_tier_llm == "none"

display "Pocet inzeratu presunutych z None -> AI Integration: `n_genai_override'"
display _n "Po override: inzeraty s cluster_generative_ai==1 podle ai_level:"
tab ai_level if cluster_generative_ai == 1, missing

* --- 3.2c Diagnosticke crosstaby (dokumentace rozhodnuti, ML override neprovadime) ---
display _n "--- 3.2c Crosstab: cluster_data_science__ml x ai_level (ML NENI overridnut) ---"
tab cluster_data_science__ml ai_level, row col
display _n "--- 3.2c Crosstab: cluster_generative_ai x ai_level (post-override, kontrola) ---"
tab cluster_generative_ai ai_level, row col

preserve
    contract cluster_generative_ai ai_level
    export delimited "$outdir/Crosstab_GenAI_Tier.csv", replace delimiter(";")
restore
preserve
    contract cluster_data_science__ml ai_level
    export delimited "$outdir/Crosstab_ML_Tier.csv", replace delimiter(";")
restore

* --- 3.3 Vzdelani: edu_bin (Bc.+ ano/ne) ---
gen education_hybrid = lower(edulevel_llm)
replace education_hybrid = subinstr(education_hybrid, "'s", "", .)
replace education_hybrid = "highschool" if education_hybrid == "high school"
replace education_hybrid = "missing" if education_hybrid == "-" | education_hybrid == ""
replace education_hybrid = edu_level_det if education_hybrid == "missing" & edu_level_det != ""
replace education_hybrid = "missing" if education_hybrid == ""
replace education_hybrid = "master" if education_hybrid == "phd"
replace education_hybrid = "associate" if education_hybrid == "diploma"
replace education_hybrid = "missing" if !inlist(education_hybrid, "highschool", "associate", "bachelor", "master", "missing")

gen edu_bin = inlist(education_hybrid, "bachelor", "master")
label define edu_bin_lbl 0 "Nizsi/chybejici" 1 "Bc.+"
label values edu_bin edu_bin_lbl
label variable edu_bin "Bakalar+ pozadovan (binarni)"

display _n "--- 3.3 edu_bin rozdeleni po zemich ---"
tab edu_bin country, col

* --- 3.4 Zkusenosti: exp_bin (>=1 rok ano/ne) ---
destring experience_min_llm, replace force
gen exp_bin = (experience_min_llm >= 1) if experience_min_llm != .
replace exp_bin = 0 if missing(exp_bin)
label define exp_bin_lbl 0 "< 1 rok / chybejici" 1 ">= 1 rok praxe"
label values exp_bin exp_bin_lbl
label variable exp_bin "Praxe >=1 rok pozadovana (binarni)"

display _n "--- 3.4 exp_bin rozdeleni po zemich ---"
tab exp_bin country, col

* --- 3.5 Plat — prevod na USD a rocni bazi ---
destring salary_min salary_mid salary_max, replace force

* Pevne kurzy (prumer Sep-Oct 2025)
local eur_usd = 1.165
local inr_usd = 88

drop if country == "DE" & !inlist(pay_currency, "EUR", "")
drop if country == "IN" & pay_currency == "" & salary_mid != .

foreach var of varlist salary_min salary_mid salary_max {
    replace `var' = `var' * `eur_usd' if pay_currency == "EUR"
    replace `var' = `var' / `inr_usd' if pay_currency == "INR"
}

* US: 2080 h/rok, DE: 1607 h (OECD 2024), IN: 1920 h
foreach var of varlist salary_min salary_mid salary_max {
    replace `var' = `var' * 2080 if pay_period == "HOURLY" & country == "US"
    replace `var' = `var' * 1607 if pay_period == "HOURLY" & country == "DE"
    replace `var' = `var' * 1920 if pay_period == "HOURLY" & country == "IN"
    replace `var' = `var' * 12   if pay_period == "MONTHLY"
}

replace salary_mid = . if salary_mid < 3000 | salary_mid > 500000
label variable salary_mid "Rocni plat - stredni hodnota (USD)"

gen ln_salary = ln(salary_mid)
label variable ln_salary "Prirozeny logaritmus platu (USD)"

gen has_salary = (salary_mid != .)
label variable has_salary "Ma uvedeny plat (1=ano)"

* --- 3.6 Velikost firmy ---
replace size = "Unknown" if size == ""
gen size_cat = .
replace size_cat = 0 if inlist(size, "Unknown", "Unbekannt")
replace size_cat = 1 if inlist(size, "1 to 50 Employees", "1 bis 50 Mitarbeiter")
replace size_cat = 2 if inlist(size, "51 to 200 Employees", "51 bis 200 Mitarbeiter")
replace size_cat = 3 if inlist(size, "201 to 500 Employees", "201 bis 500 Mitarbeiter")
replace size_cat = 4 if inlist(size, "501 to 1000 Employees", "501 bis 1.000 Mitarbeiter")
replace size_cat = 5 if inlist(size, "1001 to 5000 Employees", "1.001 bis 5.000 Mitarbeiter")
replace size_cat = 6 if inlist(size, "5001 to 10000 Employees", "5.001 bis 10.000 Mitarbeiter")
replace size_cat = 7 if inlist(size, "10000+ Employees", "Mehr als 10.000 Mitarbeiter")
label define size_lbl 0 "Unknown" 1 "1-50" 2 "51-200" 3 "201-500" ///
    4 "501-1000" 5 "1001-5000" 6 "5001-10000" 7 "10000+"
label values size_cat size_lbl
label variable size_cat "Velikost firmy (ordinalni)"

* --- 3.7 Typ firmy ---
gen type_cat = .
replace type_cat = 0 if inlist(type, "", "Unknown", "Contract", "Self-employed", "Private Practice / Firm", "Franchise")
replace type_cat = 0 if inlist(type, "Unbekannt", "Auftragsunternehmen", "Selbstständig", "Privatpraxis/Kanzlei")
replace type_cat = 1 if inlist(type, "Company - Private", "Subsidiary or Business Segment", "Privatunternehmen", "Tochtergesellschaft oder Geschäftsbereich")
replace type_cat = 2 if inlist(type, "Company - Public", "Aktiengesellschaft")
replace type_cat = 4 if inlist(type, "Nonprofit Organization", "Non-profit Organisation", "Government", "College / University", "School / School District", "Hospital")
replace type_cat = 4 if inlist(type, "Gemeinnützige Organisation", "Öffentlicher Dienst", "Hochschule/Universität", "Schule/Schulbezirk", "Krankenhaus")
label define type_lbl 0 "Unknown/Other" 1 "Private/Subsidiary" 2 "Public" 4 "Nonprofit/Gov/Edu"
label values type_cat type_lbl
label variable type_cat "Typ firmy"

* --- 3.8 NACE sektor ---
replace sector_nace = "Unknown" if sector_nace == ""
replace sector_nace = "Other" if !inlist(sector_nace, "J", "C", "K", "M", "Q", "Unknown")
encode sector_nace, generate(sector_nace_num)
label variable sector_nace_num "NACE sektor"

* --- 3.9 Remote prace ---
gen is_remote = 0
replace is_remote = 1 if strpos(lower(remote_work_types), "home") > 0
replace is_remote = 1 if strpos(lower(remote_work_types), "remote") > 0
label variable is_remote "Moznost remote prace (1=ano)"

* --- 3.10 Job family ---
replace job_family = "Unknown" if job_family == ""
replace job_family = "Other" if inlist(job_family, "Frontend & Design", "QA & Testing", "Security", "Systems & Embedded")
encode job_family, generate(job_family_num)
label variable job_family_num "Rodina pozice"

* --- 3.11 Region (jen pro US) ---
replace region = "Unknown" if region == ""
encode region, generate(region_num)
label variable region_num "Census region (US-specific)"

* --- 3.12 Vyrazeni ridkych skill clusteru (konzistentne s hlavni analyzou) ---
capture drop cluster_legacy__mainframe
capture drop cluster_data_analysis__stats
capture drop cluster_tools__editors

* --- 3.13 Company ID (pro potencialni cluster SE) ---
destring company_id, replace force
label variable company_id "ID firmy"


* ==============================================================================
* 4. TABULKA 1 — VYSKYT AI POZADAVKU PO ZEMICH (DESKRIPCE)
* ==============================================================================
display _n "=============================================================="
display "4. TABULKA 1 — VYSKYT AI PO ZEMICH"
display "=============================================================="

display _n "--- Sloupcova % (v ramci zeme soucet = 100%) ---"
tab ai_level country, column
display _n "--- Absolutni pocty ---"
tab ai_level country

preserve
    contract ai_level country
    bysort country: egen country_total = total(_freq)
    gen pct_col = 100 * _freq / country_total
    keep country ai_level _freq pct_col
    reshape wide _freq pct_col, i(ai_level) j(country) string
    order ai_level _freqUS pct_colUS _freqDE pct_colDE _freqIN pct_colIN
    export delimited "$outdir/Tabulka_1_Vyskyt_AI.csv", replace delimiter(";")
restore


* ==============================================================================
* 5. PRIPRAVA PRO LOGIT/MLOGIT — vyrazeni cluster_generative_ai z RHS
* ==============================================================================
* cluster_generative_ai je po §3.2b override mechanicky vazan na ai_level>=1
* (vsechny inzeraty s tim clusterem maji ai_level v {1,2}). Proto ho v logit/
* mlogit modelech vyrazujeme z RHS. cluster_data_science__ml zustava jako bezny
* skill cluster (override se neprovadi).
rename cluster_generative_ai _excl_genai


* ==============================================================================
* 6. TABULKA 2 — BINARNI LOGIT P(AI) ~ JOB FAMILY + CONTROLS (AME, per zeme)
* ==============================================================================
display _n "=============================================================="
display "6. TABULKA 2 — BINARNI LOGIT, JOB FAMILY"
display "=============================================================="

foreach c in US DE IN {
    display _n "=========== Zeme = `c' (Tabulka 2) ==========="
    capture noisily logit has_ai ///
        i.job_family_num ///
        edu_bin ///
        exp_bin ///
        i.sector_nace_num ///
        ib1.type_cat ///
        ib5.size_cat ///
        is_remote ///
        if country == "`c'", vce(robust)
    if _rc == 0 {
        quietly margins, dydx(*) post
        estimates store ame_t2_`c'
    }
    else {
        display as error "Logit Tabulka 2 pro `c' selhal (rc=" _rc ")"
    }
}

esttab ame_t2_US ame_t2_DE ame_t2_IN using "$outdir/Tabulka_2_Logit_JobFamily.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    keep(*.job_family_num edu_bin exp_bin) ///
    order(*.job_family_num edu_bin exp_bin) ///
    stats(N, fmt(0) labels("N")) ///
    mtitles("USA" "Nemecko" "Indie") ///
    title("Tabulka 2: Binarni logit P(AI pozadavek = ano), AME podle zeme — job family") ///
    addnotes("Zavisla promenna: has_ai (1=AI Integration nebo Applied/Core AI, 0=None)." ///
             "Prumerne marginalni efekty (AME) z logitu s robustnimi SE." ///
             "Kontroly (neukazovane v tabulce): NACE sektor, typ firmy, velikost firmy, remote." ///
             "Referencni job family: Data & AI.")


* ==============================================================================
* 7. TABULKA 3 — BINARNI LOGIT P(AI) ~ SKILL CLUSTERY + CONTROLS (AME, per zeme)
* ==============================================================================
display _n "=============================================================="
display "7. TABULKA 3 — BINARNI LOGIT, SKILL CLUSTERY"
display "=============================================================="

foreach c in US DE IN {
    display _n "=========== Zeme = `c' (Tabulka 3) ==========="
    capture noisily logit has_ai ///
        cluster_* ///
        edu_bin ///
        exp_bin ///
        i.sector_nace_num ///
        ib1.type_cat ///
        ib5.size_cat ///
        is_remote ///
        if country == "`c'", vce(robust)
    if _rc == 0 {
        quietly margins, dydx(*) post
        estimates store ame_t3_`c'
    }
    else {
        display as error "Logit Tabulka 3 pro `c' selhal (rc=" _rc ")"
    }
}

esttab ame_t3_US ame_t3_DE ame_t3_IN using "$outdir/Tabulka_3_Logit_SkillClusters.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    keep(cluster_* edu_bin exp_bin) ///
    order(cluster_* edu_bin exp_bin) ///
    stats(N, fmt(0) labels("N")) ///
    mtitles("USA" "Nemecko" "Indie") ///
    title("Tabulka 3: Binarni logit P(AI pozadavek = ano), AME podle zeme — skill clustery") ///
    addnotes("Zavisla promenna: has_ai (1=AI Integration nebo Applied/Core AI, 0=None)." ///
             "Prumerne marginalni efekty (AME) z logitu s robustnimi SE." ///
             "Kontroly (neukazovane): NACE sektor, typ firmy, velikost firmy, remote." ///
             "cluster_generative_ai vyrazen z RHS kvuli cirkularite (GenAI -> AI Integration override).")


* ==============================================================================
* 8. TABULKA 4 — MLOGIT P(AI tier) ~ SKILL CLUSTERY (AME, 9 sloupcu)
* ==============================================================================
display _n "=============================================================="
display "8. TABULKA 4 — MLOGIT, SKILL CLUSTERY"
display "=============================================================="

foreach c in US DE IN {
    display _n "=========== Zeme = `c' (Tabulka 4) ==========="
    * Tri beh modelu (pro kazdy outcome zvlast), protoze margins post prepisuje
    * e(b). Pouzivame AME predict(outcome(k)) pro k=0,1,2.
    foreach o in 0 1 2 {
        capture noisily mlogit ai_level ///
            cluster_* ///
            edu_bin ///
            exp_bin ///
            i.sector_nace_num ///
            ib1.type_cat ///
            ib5.size_cat ///
            is_remote ///
            if country == "`c'", baseoutcome(0) vce(robust)
        if _rc == 0 {
            quietly margins, dydx(*) predict(outcome(`o')) post
            estimates store ame_t4_`c'_`o'
        }
        else {
            display as error "Mlogit Tabulka 4 pro `c' outcome `o' selhal (rc=" _rc ")"
        }
    }
}

esttab ame_t4_US_0 ame_t4_US_1 ame_t4_US_2 ///
       ame_t4_DE_0 ame_t4_DE_1 ame_t4_DE_2 ///
       ame_t4_IN_0 ame_t4_IN_1 ame_t4_IN_2 ///
       using "$outdir/Tabulka_4_Mlogit_SkillClusters.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    keep(cluster_* edu_bin exp_bin) ///
    order(cluster_* edu_bin exp_bin) ///
    stats(N, fmt(0) labels("N")) ///
    mtitles("US:None" "US:AI Integ" "US:Applied" ///
            "DE:None" "DE:AI Integ" "DE:Applied" ///
            "IN:None" "IN:AI Integ" "IN:Applied") ///
    title("Tabulka 4: Multinomialni logit P(AI tier), AME podle zeme a tieru — skill clustery") ///
    addnotes("Zavisla promenna: ai_level (0=None, 1=AI Integration, 2=Applied/Core AI)." ///
             "Prumerne marginalni efekty (AME) z mlogitu s robustnimi SE, base outcome = None." ///
             "Kontroly (neukazovane): NACE sektor, typ firmy, velikost firmy, remote." ///
             "Soucet AME napric tremi outcomes pro kazdou promennou je 0.")


* ==============================================================================
* 9. TABULKA 5 — OLS ln(plat) ~ SKILL CLUSTERY + AI TIERY (per zeme)
* ==============================================================================
display _n "=============================================================="
display "9. TABULKA 5 — OLS ln(plat) per zeme"
display "=============================================================="

* US (vc. region FE)
display _n "=========== Zeme = US (Tabulka 5) ==========="
capture noisily regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    edu_bin ///
    exp_bin ///
    i.sector_nace_num ///
    i.region_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    if country == "US" & ln_salary != ., vce(robust)
if _rc == 0 {
    estimates store ols_t5_US
}

foreach c in DE IN {
    display _n "=========== Zeme = `c' (Tabulka 5) ==========="
    capture noisily regress ln_salary ///
        cluster_* ///
        i.ai_level ///
        edu_bin ///
        exp_bin ///
        i.sector_nace_num ///
        is_remote ///
        ib1.type_cat ///
        ib5.size_cat ///
        if country == "`c'" & ln_salary != ., vce(robust)
    if _rc == 0 {
        estimates store ols_t5_`c'
    }
}

esttab ols_t5_US ols_t5_DE ols_t5_IN using "$outdir/Tabulka_5_OLS_lnMzda.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    keep(cluster_* 1.ai_level 2.ai_level edu_bin exp_bin) ///
    order(1.ai_level 2.ai_level cluster_* edu_bin exp_bin) ///
    stats(N r2, fmt(0 3) labels("N" "R2")) ///
    mtitles("USA" "Nemecko" "Indie") ///
    title("Tabulka 5: OLS ln(plat) — skill clustery + AI tiery, separatne per zeme") ///
    addnotes("Zavisla promenna: ln(rocni plat v USD)." ///
             "Robustni standardni chyby v zavorkach." ///
             "Kontroly (neukazovane): NACE sektor, typ firmy, velikost firmy, remote (+ region FE pro USA)." ///
             "Referencni AI uroven: None." ///
             "cluster_generative_ai vyrazen z RHS kvuli cirkularite s ai_level (GenAI override).")


* ==============================================================================
* 10. GRAFY — ROZLOZENI ln(plat) PODLE AI UROVNE (kernel density per zeme)
* ==============================================================================
display _n "=============================================================="
display "10. GRAFY — ROZLOZENI ln(plat) PODLE AI UROVNE"
display "=============================================================="

foreach c in US DE IN {
    capture noisily twoway ///
        (kdensity ln_salary if country == "`c'" & ai_level == 0, lcolor(gs10) lpattern(solid)) ///
        (kdensity ln_salary if country == "`c'" & ai_level == 1, lcolor(navy) lpattern(dash)) ///
        (kdensity ln_salary if country == "`c'" & ai_level == 2, lcolor(maroon) lpattern(shortdash)), ///
        title("`c': Rozlozeni ln(plat) podle urovne AI pozadavku", size(medium)) ///
        xtitle("ln(rocni plat, USD)") ytitle("Hustota") ///
        legend(order(1 "None" 2 "AI Integration" 3 "Applied/Core AI") rows(1) size(small)) ///
        graphregion(color(white)) bgcolor(white)
    if _rc == 0 {
        graph export "$outdir/Graf_Mzda_AI_`c'.png", replace width(1200)
    }
}

* Doplnkova deskripce platu po ai_level a zemich (pro text kapitoly Mzdy)
display _n "--- Deskripce platu (USD) po ai_level a zemich ---"
foreach c in US DE IN {
    display _n "Zeme = `c'"
    tabstat salary_mid ln_salary if country == "`c'", by(ai_level) ///
        stat(count mean sd p25 p50 p75) columns(statistics)
}


* ==============================================================================
* 11. ZAVER
* ==============================================================================
* Vratit cluster_generative_ai pro pripad dalsi prace s datasetem
rename _excl_genai cluster_generative_ai

display _n "=============================================================="
display "FINALNI TEZISTOVA ANALYZA DOKONCENA"
display "=============================================================="
display "Vystupy ulozeny do: $outdir"
display "Log: $outdir/ai_skills_thesis_final.log"

save "$outdir/ai_skills_thesis_final_processed.dta", replace

log close
