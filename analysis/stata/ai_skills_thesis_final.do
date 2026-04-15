********************************************************************************
* AI SKILLS — FINALNI ANALYZA PRO DIPLOMOVOU PRACI (US vs DE vs IN)
* ==============================================================================
* Cilem tohoto do-file je generovat 5 hlavnich tabulek a grafy pro praktickou
* cast diplomky. Vsechny modely jsou PLNE (ne inkrementalni), reportuji se AME
* pro logit i mlogit a separatne per zeme tam, kde to zadani pozaduje.
*
* Vystupy (RTF do Wordu):
*   Tabulka 1:  Vyskyt AI pozadavku po zemich (sloupcova %, N + %)
*   Tabulka 2:  Binarni logit P(AI) — job family ^ skill clustery (AME, 6 sloupcu)
*   Tabulka 3:  Mlogit P(AI tier) ~ skill clustery (AME, 9 sloupcu)
*   Tabulka 4:  OLS ln(plat) ~ skill clustery + AI tiery (3 sloupce, per zeme)
*   Graf_*:     Frekvencni kernel density ln(plat) x ai_level per zeme
*   Priloha A:  Heckman selection model pro ln(plat) — robustness
*   Priloha B:  Cross-country Wald testy (v logu, viz sekce 12)
*
* Kontroly (neukazovane v tabulkach): NACE sektor, typ firmy, velikost firmy,
*   remote, region FE (jen US v OLS). Vzdelani: edu_bin (Bc.+) + edu_missing.
*   Praxe: exp_bin (>=1 rok) + exp_missing. Referencni zeme v pooled modelech
*   je US; referencni job family je Software Engineer.
*
* Klicove upravy:
*   - cluster_generative_ai -> AI Integration tier override (§3.2b)
*   - cluster_generative_ai vyrazen z RHS logit/mlogit/OLS kvuli cirkularite
*   - SE clusterovane na firmu (firm_cluster) ve vsech regresich
*   - edu_missing a exp_missing dummy odliseni "nepozaduje se" vs. "neuvedeno"
*   - Baseline job family = Software Engineer (nejcastejsi, neutralni)
*   - Priloha A: Heckman pro selection bias v ln(plat)
*   - Priloha B: pooled modely s country interakcemi + Wald testy
*   - Pouze jedna specifikace per tabulka (plny model), zadne inkrementalni varianty
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
label variable country_id "Země původu inzerátu (DE=1, IN=2, US=3)"
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
label variable ai_level "Úroveň AI požadavku (0/1/2)"

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
label define edu_bin_lbl 0 "Nižší/neuvedeno" 1 "Bc.+"
label values edu_bin edu_bin_lbl
label variable edu_bin "Bakalář+ explicitně požadován"

* edu_missing: dummy odliseni "nepozaduje se" vs. "neuvedeno"
gen edu_missing = (education_hybrid == "missing")
label define edu_miss_lbl 0 "Vzdělání uvedeno" 1 "Neuvedeno"
label values edu_missing edu_miss_lbl
label variable edu_missing "Vzdělání v inzerátu neuvedeno"

display _n "--- 3.3 edu_bin + edu_missing rozdeleni po zemich ---"
tab edu_bin country, col
tab edu_missing country, col

* --- 3.4 Zkusenosti: exp_bin (>=1 rok ano/ne) ---
destring experience_min_llm, replace force
gen exp_missing = missing(experience_min_llm)
gen exp_bin = (experience_min_llm >= 1) if !missing(experience_min_llm)
replace exp_bin = 0 if missing(exp_bin)
label define exp_bin_lbl 0 "< 1 rok / neuvedeno" 1 ">= 1 rok praxe"
label values exp_bin exp_bin_lbl
label variable exp_bin "Praxe ≥ 1 rok explicitně požadována"

label define exp_miss_lbl 0 "Praxe uvedena" 1 "Neuvedeno"
label values exp_missing exp_miss_lbl
label variable exp_missing "Praxe v inzerátu neuvedena"

display _n "--- 3.4 exp_bin + exp_missing rozdeleni po zemich ---"
tab exp_bin country, col
tab exp_missing country, col

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
label variable salary_mid "Roční plat – střední hodnota (USD)"

gen ln_salary = ln(salary_mid)
label variable ln_salary "Přirozený logaritmus platu (USD)"

gen has_salary = (salary_mid != .)
label variable has_salary "Má uvedený plat (1=ano)"

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
label variable size_cat "Velikost firmy (ordinální)"

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
label variable is_remote "Možnost remote práce (1=ano)"

* --- 3.10 Job family ---
* Ponechavame plnou taxonomii z Python pipeline (10 kategorii). Drive byly
* Frontend & Design / QA & Testing / Security / Systems & Embedded slouceny do
* "Other" kvuli malym N; pri soucasnem vzorku (US 17k, DE 6.4k, IN 14k) je
* drzet oddelene informativnejsi. "Other" si ponechavame jen pro inzeraty,
* ktere regex neklasifikoval.
replace job_family = "Unknown" if job_family == ""
encode job_family, generate(job_family_num)
label variable job_family_num "Rodina pozice (job family)"

display _n "--- 3.10 Diagnosticke rozlozeni job_family x country (pro kontrolu N) ---"
tab job_family country, missing

* --- 3.11 Region (jen pro US) ---
replace region = "Unknown" if region == ""
encode region, generate(region_num)
label variable region_num "Census region (US-specific)"

* --- 3.12 Vyrazeni ridkych skill clusteru (konzistentne s hlavni analyzou) ---
capture drop cluster_legacy__mainframe
capture drop cluster_data_analysis__stats
capture drop cluster_tools__editors

* --- 3.12b Ciste CJ labely skill clusteru (pro esttab vystupy) ---
capture label var cluster_architecture__methods  "Architektura & metody"
capture label var cluster_bi__analytics          "BI & analytika"
capture label var cluster_backend_development    "Backend"
capture label var cluster_certifications         "Certifikace"
capture label var cluster_cloud_computing        "Cloud"
capture label var cluster_data_engineering       "Datové inženýrství"
capture label var cluster_data_science__ml       "Data science & ML"
capture label var cluster_databases__storage     "Databáze & storage"
capture label var cluster_devops__containers     "DevOps & kontejnery"
capture label var cluster_dynamic__web           "Dynamický web"
capture label var cluster_enterprise__managed    "Enterprise (managed)"
capture label var cluster_enterprise_platforms   "Enterprise platformy"
capture label var cluster_frontend_development   "Frontend"
capture label var cluster_generative_ai          "Generativní AI"
capture label var cluster_mobile__desktop        "Mobile & desktop"
capture label var cluster_networking             "Síťování"
capture label var cluster_os__embedded           "OS & embedded"
capture label var cluster_scripting__shell       "Skriptování & shell"
capture label var cluster_security__identity     "Bezpečnost & identita"
capture label var cluster_systems_programming    "Systémové programování"
capture label var cluster_testing_qa__debugging  "Testování & QA"

* --- 3.13 Company ID + firm_cluster pro clustered SE ---
destring company_id, replace force
label variable company_id "ID firmy"

* firm_cluster: unikatni per (zeme, firma). Inzeraty bez company_id
* dostanou kazdy svuj vlastni cluster id (cili tam se clustering neaplikuje).
egen firm_cluster = group(country company_id) if !missing(company_id)
quietly sum firm_cluster
local max_fc = r(max)
if missing(`max_fc') local max_fc = 0
replace firm_cluster = _n + `max_fc' if missing(firm_cluster)
label variable firm_cluster "Firm cluster ID (per-country company_id)"

* --- 3.14 Baseline job family = Software Engineer (nejcastejsi, neutralni) ---
quietly levelsof job_family_num if job_family == "Software Engineer", local(_sw_list)
local sw_base : word 1 of `_sw_list'
display "Baseline job_family_num (Software Engineer) = `sw_base'"


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

* --- Export Tabulky 1 do RTF pres matrix ---
matrix T1 = J(3, 6, .)
local row = 0
foreach lv_num in 0 1 2 {
    local ++row
    local col = 0
    foreach c in US DE IN {
        count if country == "`c'"
        local total = r(N)
        count if country == "`c'" & ai_level == `lv_num'
        local n = r(N)
        local ++col
        matrix T1[`row', `col'] = `n'
        local ++col
        matrix T1[`row', `col'] = 100 * `n' / `total'
    }
}
matrix rownames T1 = "None" "AI_Integration" "Applied_Core_AI"
matrix colnames T1 = "USA_N" "USA_pct" "DE_N" "DE_pct" "IN_N" "IN_pct"

esttab matrix(T1, fmt(%9.0fc %5.2f %9.0fc %5.2f %9.0fc %5.2f)) ///
    using "$outdir/Tabulka_1_Vyskyt_AI.rtf", replace ///
    mgroups("USA" "Nemecko" "Indie", pattern(1 0 1 0 1 0)) ///
    collabels("N" "%" "N" "%" "N" "%") ///
    title("Tabulka 1: Rozložení úrovní AI v IT inzerátech podle země") ///
    addnotes("N = počet inzerátů s danou úrovní AI v rámci země." ///
             "% = sloupcové procento (součet v rámci země = 100 %)." ///
             "Po GenAI override (§3.2b): inzeráty s cluster_generative_ai == 1 původně v None tieru byly přesunuty do AI Integration.")


* ==============================================================================
* 5. PRIPRAVA PRO LOGIT/MLOGIT — vyrazeni cluster_generative_ai z RHS
* ==============================================================================
* cluster_generative_ai je po §3.2b override mechanicky vazan na ai_level>=1
* (vsechny inzeraty s tim clusterem maji ai_level v {1,2}). Proto ho v logit/
* mlogit modelech vyrazujeme z RHS. cluster_data_science__ml zustava jako bezny
* skill cluster (override se neprovadi).
rename cluster_generative_ai _excl_genai


* ==============================================================================
* 6. TABULKA 2 — BINARNI LOGIT P(AI) ~ JOB FAMILY ^ SKILL CLUSTERY (AME, per zeme)
* ==============================================================================
* Dva komplementarni modely per zeme (celkem 6 sloupcu):
*   - Levy panel (3 sloupce): P(AI) ~ job_family + controls
*   - Pravy panel (3 sloupce): P(AI) ~ skill clustery + controls
* Reporting: prumerne marginalni efekty (AME), robustni SE clusterovane na firmu.
display _n "=============================================================="
display "6. TABULKA 2 — BINARNI LOGIT, JOB FAMILY + SKILL CLUSTERY (6 sloupcu)"
display "=============================================================="

* --- 6a. Job family panel ---
foreach c in US DE IN {
    display _n "=========== Zeme = `c' (T2 job family) ==========="
    capture noisily logit has_ai ///
        ib`sw_base'.job_family_num ///
        edu_bin edu_missing ///
        exp_bin exp_missing ///
        i.sector_nace_num ///
        ib1.type_cat ///
        ib5.size_cat ///
        is_remote ///
        if country == "`c'", vce(cluster firm_cluster)
    if _rc == 0 {
        quietly margins, dydx(*) post
        estadd local ctrl_nace   "Ano"
        estadd local ctrl_type   "Ano"
        estadd local ctrl_size   "Ano"
        estadd local ctrl_remote "Ano"
        estimates store ame_t2jf_`c'
    }
    else {
        display as error "Logit T2 (job family) pro `c' selhal (rc=" _rc ")"
    }
}

* --- 6b. Skill clustery panel ---
foreach c in US DE IN {
    display _n "=========== Zeme = `c' (T2 skill clustery) ==========="
    capture noisily logit has_ai ///
        cluster_* ///
        edu_bin edu_missing ///
        exp_bin exp_missing ///
        i.sector_nace_num ///
        ib1.type_cat ///
        ib5.size_cat ///
        is_remote ///
        if country == "`c'", vce(cluster firm_cluster)
    if _rc == 0 {
        quietly margins, dydx(*) post
        estadd local ctrl_nace   "Ano"
        estadd local ctrl_type   "Ano"
        estadd local ctrl_size   "Ano"
        estadd local ctrl_remote "Ano"
        estimates store ame_t2sk_`c'
    }
    else {
        display as error "Logit T2 (skill clustery) pro `c' selhal (rc=" _rc ")"
    }
}

* --- 6c. Export Tabulky 2 (merged 6 sloupcu) ---
esttab ame_t2jf_US ame_t2jf_DE ame_t2jf_IN ame_t2sk_US ame_t2sk_DE ame_t2sk_IN ///
    using "$outdir/Tabulka_2_Logit_AI.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    keep(*.job_family_num cluster_* edu_bin edu_missing exp_bin exp_missing) ///
    order(*.job_family_num cluster_* edu_bin edu_missing exp_bin exp_missing) ///
    stats(ctrl_nace ctrl_type ctrl_size ctrl_remote N, ///
          fmt(%s %s %s %s %9.0fc) ///
          labels("NACE sektor" "Typ firmy" "Velikost firmy" "Remote" "N")) ///
    mgroups("Job family" "Skill clustery", pattern(1 0 0 1 0 0)) ///
    mtitles("USA" "Německo" "Indie" "USA" "Německo" "Indie") ///
    title("Tabulka 2: Binární logit P(AI požadavek = ano) — AME podle země") ///
    addnotes("Závislá proměnná: has_ai (1=AI Integration nebo Applied/Core AI, 0=None)." ///
             "Průměrné marginální efekty (AME) z logitu, SE klastrované na firmu v závorkách." ///
             "Referenční job family: Software Engineer." ///
             "cluster_generative_ai vyřazen z RHS kvůli cirkularitě (GenAI → AI Integration override)." ///
             "edu_missing / exp_missing: dummy pro inzeráty bez uvedeného vzdělání / praxe.")


* ==============================================================================
* 7. TABULKA 3 — MLOGIT P(AI tier) ~ SKILL CLUSTERY (AME, 9 sloupcu)
* ==============================================================================
display _n "=============================================================="
display "7. TABULKA 3 — MLOGIT, SKILL CLUSTERY"
display "=============================================================="

foreach c in US DE IN {
    display _n "=========== Zeme = `c' (Tabulka 3) ==========="
    * Tri beh modelu (pro kazdy outcome zvlast), protoze margins post prepisuje
    * e(b). Pouzivame AME predict(outcome(k)) pro k=0,1,2.
    foreach o in 0 1 2 {
        capture noisily mlogit ai_level ///
            cluster_* ///
            edu_bin edu_missing ///
            exp_bin exp_missing ///
            i.sector_nace_num ///
            ib1.type_cat ///
            ib5.size_cat ///
            is_remote ///
            if country == "`c'", baseoutcome(0) vce(cluster firm_cluster)
        if _rc == 0 {
            quietly margins, dydx(*) predict(outcome(`o')) post
            estadd local ctrl_nace   "Ano"
            estadd local ctrl_type   "Ano"
            estadd local ctrl_size   "Ano"
            estadd local ctrl_remote "Ano"
            estimates store ame_t3_`c'_`o'
        }
        else {
            display as error "Mlogit Tabulka 3 pro `c' outcome `o' selhal (rc=" _rc ")"
        }
    }
}

esttab ame_t3_US_0 ame_t3_US_1 ame_t3_US_2 ///
       ame_t3_DE_0 ame_t3_DE_1 ame_t3_DE_2 ///
       ame_t3_IN_0 ame_t3_IN_1 ame_t3_IN_2 ///
       using "$outdir/Tabulka_3_Mlogit_AI_Tier.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    keep(cluster_* edu_bin edu_missing exp_bin exp_missing) ///
    order(cluster_* edu_bin edu_missing exp_bin exp_missing) ///
    stats(ctrl_nace ctrl_type ctrl_size ctrl_remote N, ///
          fmt(%s %s %s %s %9.0fc) ///
          labels("NACE sektor" "Typ firmy" "Velikost firmy" "Remote" "N")) ///
    mgroups("USA" "Německo" "Indie", pattern(1 0 0 1 0 0 1 0 0)) ///
    mtitles("None" "Integ." "Applied" "None" "Integ." "Applied" "None" "Integ." "Applied") ///
    title("Tabulka 3: Multinomiální logit P(AI tier) — AME podle země a úrovně") ///
    addnotes("Závislá proměnná: ai_level (0=None, 1=AI Integration, 2=Applied/Core AI)." ///
             "Průměrné marginální efekty (AME) z mlogitu, SE klastrované na firmu v závorkách, base outcome = None." ///
             "cluster_generative_ai vyřazen z RHS kvůli cirkularitě (GenAI override)." ///
             "Součet AME napříč třemi outcomes pro každou proměnnou je 0.")


* ==============================================================================
* 8. TABULKA 4 — OLS ln(plat) ~ SKILL CLUSTERY + AI TIERY (per zeme)
* ==============================================================================
display _n "=============================================================="
display "8. TABULKA 4 — OLS ln(plat) per zeme"
display "=============================================================="

* US (vc. region FE)
display _n "=========== Zeme = US (Tabulka 4) ==========="
capture noisily regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    edu_bin edu_missing ///
    exp_bin exp_missing ///
    i.sector_nace_num ///
    i.region_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    if country == "US" & ln_salary != ., vce(cluster firm_cluster)
if _rc == 0 {
    estadd local ctrl_nace   "Ano"
    estadd local ctrl_type   "Ano"
    estadd local ctrl_size   "Ano"
    estadd local ctrl_remote "Ano"
    estadd local ctrl_region "Ano"
    estimates store ols_t4_US
    display _n "--- VIF Kontrola (US) ---"
    quietly regress ln_salary ///
        cluster_* ///
        i.ai_level ///
        edu_bin edu_missing ///
        exp_bin exp_missing ///
        i.sector_nace_num ///
        i.region_num ///
        is_remote ///
        ib1.type_cat ///
        ib5.size_cat ///
        if country == "US" & ln_salary != .
    capture noisily estat vif
}

foreach c in DE IN {
    display _n "=========== Zeme = `c' (Tabulka 4) ==========="
    capture noisily regress ln_salary ///
        cluster_* ///
        i.ai_level ///
        edu_bin edu_missing ///
        exp_bin exp_missing ///
        i.sector_nace_num ///
        is_remote ///
        ib1.type_cat ///
        ib5.size_cat ///
        if country == "`c'" & ln_salary != ., vce(cluster firm_cluster)
    if _rc == 0 {
        estadd local ctrl_nace   "Ano"
        estadd local ctrl_type   "Ano"
        estadd local ctrl_size   "Ano"
        estadd local ctrl_remote "Ano"
        estadd local ctrl_region "-"
        estimates store ols_t4_`c'
        display _n "--- VIF Kontrola (`c') ---"
        quietly regress ln_salary ///
            cluster_* ///
            i.ai_level ///
            edu_bin edu_missing ///
            exp_bin exp_missing ///
            i.sector_nace_num ///
            is_remote ///
            ib1.type_cat ///
            ib5.size_cat ///
            if country == "`c'" & ln_salary != .
        capture noisily estat vif
    }
}

esttab ols_t4_US ols_t4_DE ols_t4_IN using "$outdir/Tabulka_4_OLS_lnMzda.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    keep(cluster_* 1.ai_level 2.ai_level edu_bin edu_missing exp_bin exp_missing) ///
    order(1.ai_level 2.ai_level cluster_* edu_bin edu_missing exp_bin exp_missing) ///
    stats(ctrl_nace ctrl_type ctrl_size ctrl_remote ctrl_region N r2, ///
          fmt(%s %s %s %s %s %9.0fc %5.3f) ///
          labels("NACE sektor" "Typ firmy" "Velikost firmy" "Remote" "Region FE (US)" "N" "R²")) ///
    mtitles("USA" "Německo" "Indie") ///
    title("Tabulka 4: OLS ln(plat) — skill clustery + AI tiery, separátně podle země") ///
    addnotes("Závislá proměnná: ln(roční plat v USD)." ///
             "Standardní chyby klastrované na firmu v závorkách." ///
             "Referenční AI úroveň: None." ///
             "cluster_generative_ai vyřazen z RHS kvůli cirkularitě s ai_level (GenAI override)." ///
             "Pozor: v DE je pokrytí platu velmi nízké — viz Heckman robustness v Příloze A.")


* ==============================================================================
* 10. GRAFY — ROZLOZENI ln(plat) PODLE AI UROVNE (frekvencni kernel density per zeme)
* ==============================================================================
* Smooth kernel density krivky preskalovane na frekvenci (pocet pozorovani na
* jednotku sirky binu), nikoli na hustotu (integral = 1). Preskalujeme vynasobenim
* density hodnot poctem pozorovani ve skupine: f_freq(x) = f_density(x) * N_group.
display _n "=============================================================="
display "10. GRAFY — ROZLOZENI ln(plat) PODLE AI UROVNE (FREKVENCE)"
display "=============================================================="

foreach c in US DE IN {
    * Spocitat N v kazdem ai_level pro danou zemi a pripravit scaled kdensity
    capture drop _kx_* _kf_*
    local plot_cmd = ""
    local plot_ok = 1

    foreach lv in 0 1 2 {
        count if country == "`c'" & ai_level == `lv' & ln_salary != .
        local n_`lv' = r(N)
        if `n_`lv'' < 10 {
            display as text "Skupina ai_level=`lv' v `c' ma jen `n_`lv'' obs — preskakujeme kdensity."
            local plot_ok = 0
            continue
        }
        capture noisily kdensity ln_salary if country == "`c'" & ai_level == `lv', ///
            generate(_kx_`lv' _kf_`lv') nograph
        if _rc != 0 {
            display as error "kdensity selhal pro `c' ai_level=`lv' (rc=" _rc ")"
            local plot_ok = 0
            continue
        }
        * Preskalovat density -> frekvence
        replace _kf_`lv' = _kf_`lv' * `n_`lv''
    }

    if `plot_ok' == 1 {
        capture noisily twoway ///
            (line _kf_0 _kx_0, lcolor(gs10) lpattern(solid)) ///
            (line _kf_1 _kx_1, lcolor(navy) lpattern(dash)) ///
            (line _kf_2 _kx_2, lcolor(maroon) lpattern(shortdash)), ///
            title("`c': Rozložení ln(plat) podle úrovně AI požadavku", size(medium)) ///
            xtitle("ln(roční plat, USD)") ytitle("Frekvence") ///
            legend(order(1 "None" 2 "AI Integration" 3 "Applied/Core AI") rows(1) size(small)) ///
            graphregion(color(white)) bgcolor(white)
        if _rc == 0 {
            graph export "$outdir/Graf_Mzda_AI_`c'.png", replace width(1200)
        }
    }
    capture drop _kx_* _kf_*
}

* Doplnkova deskripce platu po ai_level a zemich (pro text kapitoly Mzdy)
display _n "--- Deskripce platu (USD) po ai_level a zemich ---"
foreach c in US DE IN {
    display _n "Zeme = `c'"
    tabstat salary_mid ln_salary if country == "`c'", by(ai_level) ///
        stat(count mean sd p25 p50 p75) columns(statistics)
}


* ==============================================================================
* 11. PRILOHA A — HECKMAN SELECTION MODEL PRO ln(plat) (robustness)
* ==============================================================================
* Motivace: v DE je pokryti platu velmi nizke (~8%). OLS z Tabulky 4 muze byt
* vychyleny selekci (firmy, ktere plat inzeruji, nejsou nahodny vzorek).
* Heckman selection model odhaduje pravdepodobnost inzerovani platu v 1. fazi
* a koeficienty ln(plat) ve 2. fazi s korekci inverse Mills ratio.
* Identifikace: bez formalni exclusion restrikce jsme odkazani na funkcni formu
* (non-linearita IMR). Vysledky proto slouzi ciste jako robustness check.
display _n "=============================================================="
display "11. HECKMAN SELECTION — robustness check pro Tabulku 4"
display "=============================================================="

foreach c in US DE IN {
    display _n "=========== Zeme = `c' (Heckman) ==========="
    capture noisily heckman ln_salary ///
        cluster_* ///
        i.ai_level ///
        edu_bin edu_missing ///
        exp_bin exp_missing ///
        i.sector_nace_num ///
        ib1.type_cat ///
        ib5.size_cat ///
        is_remote ///
        if country == "`c'", ///
        select(has_salary = cluster_* ///
                            i.ai_level ///
                            edu_bin edu_missing ///
                            exp_bin exp_missing ///
                            i.sector_nace_num ///
                            ib1.type_cat ///
                            ib5.size_cat ///
                            is_remote) ///
        vce(cluster firm_cluster)
    if _rc == 0 {
        estimates store heck_t4_`c'
    }
    else {
        display as error "Heckman pro `c' selhal (rc=" _rc ")"
    }
}

capture esttab heck_t4_US heck_t4_DE heck_t4_IN using "$outdir/Priloha_A_Heckman_lnMzda.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    keep(cluster_* 1.ai_level 2.ai_level edu_bin edu_missing exp_bin exp_missing) ///
    order(1.ai_level 2.ai_level cluster_* edu_bin edu_missing exp_bin exp_missing) ///
    stats(N, fmt(0) labels("N")) ///
    mtitles("USA" "Německo" "Indie") ///
    title("Příloha A: Heckman selection model — ln(plat) s korekcí selekce") ///
    addnotes("Rovnice výběru: P(plat inzerován) — Probit se stejnými kontrolami jako ve 2. fázi." ///
             "Identifikace jen funkční formou IMR (bez exclusion restrikce) — slouží jako robustness." ///
             "Porovnat směr a velikost koeficientů s OLS v Tabulce 4, zvláště pro DE.")


* ==============================================================================
* 12. PRILOHA B — CROSS-COUNTRY TEST (pooled model s interakcemi)
* ==============================================================================
* Motivace: Tabulky 2-4 poskytuji koeficienty separatne per zeme, ale neobsahuji
* formalni test, ze se lisi. Zde odhadneme pooled logit/OLS s interakcemi country
* x klicove regresory a Wald testujeme vyznamnost interakci. Vyznamny test =
* koeficienty se napric zememi statisticky lisi; nevyznamny = nelze zamitnout
* homogenitu.
display _n "=============================================================="
display "12. CROSS-COUNTRY TEST (pooled models s interakcemi)"
display "=============================================================="

* --- 12.1 Pooled logit: job family x country ---
display _n "--- 12.1 Pooled logit has_ai ~ job_family x country (Tabulka 2 JF test) ---"
capture noisily logit has_ai ///
    ib`sw_base'.job_family_num##ib3.country_id ///
    edu_bin edu_missing exp_bin exp_missing ///
    i.sector_nace_num ib1.type_cat ib5.size_cat is_remote, ///
    vce(cluster firm_cluster)
if _rc == 0 {
    display _n "Wald test: vsechny interakce job_family x country = 0"
    testparm i.job_family_num#i.country_id
}

* --- 12.2 Pooled logit: skill clustery x country ---
* Hlavni efekty vsech skill clusteru pres cluster_*; navic pridavame jen
* interakce (# misto ##) pro tri klicove clustery, aby se main effects
* nedublikovaly.
display _n "--- 12.2 Pooled logit has_ai ~ skill_clusters x country (Tabulka 2 skill test) ---"
capture noisily logit has_ai ///
    cluster_* ///
    ib3.country_id ///
    c.cluster_cloud_computing#ib3.country_id ///
    c.cluster_data_science__ml#ib3.country_id ///
    c.cluster_backend_development#ib3.country_id ///
    edu_bin edu_missing exp_bin exp_missing ///
    i.sector_nace_num ib1.type_cat ib5.size_cat is_remote, ///
    vce(cluster firm_cluster)
if _rc == 0 {
    display _n "Wald test: klicove skill cluster interakce = 0"
    testparm c.cluster_cloud_computing#i.country_id ///
             c.cluster_data_science__ml#i.country_id ///
             c.cluster_backend_development#i.country_id
}

* --- 12.3 Pooled OLS ln(plat): AI tier x country ---
display _n "--- 12.3 Pooled OLS ln_salary ~ ai_level x country (Tabulka 4 test) ---"
capture noisily regress ln_salary ///
    i.ai_level##ib3.country_id ///
    cluster_* ///
    edu_bin edu_missing exp_bin exp_missing ///
    i.sector_nace_num ib1.type_cat ib5.size_cat is_remote ///
    if ln_salary != ., vce(cluster firm_cluster)
if _rc == 0 {
    display _n "Wald test: interakce ai_level x country = 0 (AI-premium se mezi zememi lisi?)"
    testparm i.ai_level#i.country_id
}


* ==============================================================================
* 13. ZAVER
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
