********************************************************************************
* AI SKILLS — FINALNI ANALYZA PRO DIPLOMOVOU PRACI (US vs DE vs IN)
* ==============================================================================
* Cilem tohoto do-file je generovat 5 hlavnich tabulek a grafy pro praktickou
* cast diplomky. Vsechny modely jsou PLNE (ne inkrementalni), reportuji se AME
* pro logit i mlogit a separatne per zeme tam, kde to zadani pozaduje.
*
* Vystupy (RTF do Wordu):
*   Tabulka 1:  Vyskyt AI pozadavku po zemich (sloupcova %, N + %)
*   Tabulka 2:  Binarni logit P(AI) — job family (AME, 3 sloupce per zeme)
*   Tabulka 3:  Binarni logit P(AI) — skill clustery (AME, 3 sloupce per zeme)
*   Tabulka 4:  Mlogit P(AI tier) ~ skill clustery (AME, 9 sloupcu)
*   Tabulka 5:  OLS ln(plat) ~ skill clustery + AI tiery (3 sloupce, per zeme)
*   Graf_*:     Kernel density rocnich platu (USD) x ai_level per zeme
*   Priloha A:  Heckman selection model pro ln(plat) — robustness
*   Priloha B:  Cross-country Wald testy (v logu, viz sekce 12)
*   Priloha C:  OLS ln(plat) s plnou sadou skill clusteru (robustness k Tabulce 5)
*   Priloha D:  US OLS s region FE (robustness vuci regionalni heterogenite)
*
* Kontroly (neukazovane v tabulkach): NACE sektor, typ firmy, velikost firmy,
*   remote. Region FE (US Census) pouze v Priloze D jako robustness — hlavni
*   Tabulka 5 je symetricka napric zememi bez region FE (DE/IN ekvivalentni
*   promennou v datech nemaji). Vzdelani: edu_ols (5 urovni, ref. Bachelor)
*   pouzito v OLS / Heckman; edu_logit (3 urovne, ref. Bachelor+) pouzito v
*   binarnim logitu; v mlogitu edu vyrazeno kvuli sparse cells (viz sekce 7).
*   Praxe: exp_category (4 urovne, ref. Mid 3-5 let) ve vsech modelech.
*   Referencni zeme v pooled modelech je US; ref. job family Software Engineer.
*
* Klicove upravy (dle feedbacku vedouciho):
*   - AI tiers a skill clustery ponechany v puvodni LLM definici (zadny override)
*   - Logit + mlogit: VSECHNY skill clustery v RHS vcetne cluster_generative_ai
*     a cluster_data_science__ml. Mlogit = reverse-engineering AI tierov (popisne,
*     nekauzalni — co typicky charakterizuje inzeraty v kazdem tieru).
*   - OLS ln(plat): hlavni specifikace BEZ cluster_generative_ai a
*     cluster_data_science__ml (jsou konstrukcne cast ai_level -> cirkularita).
*     Interpretace: mzdova premie role vcetne znalosti AI technologii. Plna
*     specifikace s temito clustery jako robustness v Priloze C.
*   - SE clusterovane na firmu (firm_cluster) ve vsech regresich
*   - Vzdelani/praxe jako kategoricke (edu_ols 5 ur., edu_logit 3 ur., exp_category 4 ur.);
*     kategorie "Missing" odlisena od "nepozaduje se", shodne s verzi 2024
*   - Baseline job family = Software Engineer (nejcastejsi, neutralni)
*   - Priloha A: Heckman pro selection bias v ln(plat) (stejna spec. jako Tabulka 5)
*   - Priloha B: pooled modely s country interakcemi + Wald testy
*   - Priloha C: OLS ln(plat) s plnou sadou clusteru (robustness check)
*   - Priloha D: US OLS s region FE (robustness vuci regionalni heterogenite)
*   - Pouze jedna hlavni specifikace per tabulka; robustness v priloze
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
set max_memory 8g

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
* 3. CISTENI A PRIPRAVA DAT
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
* Inzeraty bez LLM klasifikace nelze spolehlive zaradit do AI tier (nejsou
* to autenticke "none" ale chybejici hodnoty) — vyradime je z analyzy.
count if desc_tier_llm == ""
display "Inzeraty bez LLM klasifikace (desc_tier_llm prazdne): " r(N)
drop if desc_tier_llm == ""
display "Po vyrazeni chybejici LLM klasifikace: " _N
replace desc_tier_llm = "applied_ai" if desc_tier_llm == "core_ai"

* --- 3.2 has_ai + ai_level ---
gen has_ai = (desc_tier_llm != "none")
label variable has_ai "AI Job (1=AI Integration nebo Applied/Core AI)"

gen ai_level = 0
replace ai_level = 1 if desc_tier_llm == "ai_integration"
replace ai_level = 2 if desc_tier_llm == "applied_ai"
label define ailevel_lbl 0 "None" 1 "AI Integration" 2 "Applied/Core AI"
label values ai_level ailevel_lbl
label variable ai_level "Úroveň AI požadavku (0/1/2)"

* --- 3.2b Diagnostika prekryvu GenAI / ML clusteru x AI tier ---
* AI tiers (ai_level, has_ai) zustavaji odvozene pouze z LLM klasifikace
* desc_tier_llm; zadny override neprovadime (pozadavek vedouciho — zachovat
* puvodni definice). Clustery cluster_generative_ai a cluster_data_science__ml
* vsak maji silnou konstrukcni vazbu na AI tiery (LLM je pri klasifikaci
* implicitne zohlednuje), coz je duvod, proc jsou v OLS mezd vyrazeny z RHS
* (viz sekce 8). V logit/mlogit zustavaji — tam je to zadouci, protoze mlogit
* slouzi jako reverse-engineering AI tieru.
display _n "--- 3.2b Crosstab: cluster_generative_ai x ai_level ---"
tab cluster_generative_ai ai_level, row col
display _n "--- 3.2b Crosstab: cluster_data_science__ml x ai_level ---"
tab cluster_data_science__ml ai_level, row col

preserve
    contract cluster_generative_ai ai_level
    export delimited "$outdir/Crosstab_GenAI_Tier.csv", replace delimiter(";")
restore
preserve
    contract cluster_data_science__ml ai_level
    export delimited "$outdir/Crosstab_ML_Tier.csv", replace delimiter(";")
restore

* --- 3.3 Vzdelani: edu_ols (5 urovni) + edu_logit (3 urovne) ---
* Shodne s `ai_skills_analysis_comparative.do` (2024): granularni edu_ols
* pro OLS (zachovava nelinearity HS -> Master), komprimovana edu_logit
* pro binarni logit (sloucene HS/Assoc kvuli sparse cells). Missing se
* v obou skalach drzi jako samostatna kategorie ("nepozaduje se"
* vs. konkretni uroven).
gen education_hybrid = lower(edulevel_llm)
replace education_hybrid = subinstr(education_hybrid, "'s", "", .)
replace education_hybrid = "highschool" if education_hybrid == "high school"
replace education_hybrid = "missing" if education_hybrid == "-" | education_hybrid == ""
replace education_hybrid = edu_level_det if education_hybrid == "missing" & edu_level_det != ""
replace education_hybrid = "missing" if education_hybrid == ""
replace education_hybrid = "master" if education_hybrid == "phd"
replace education_hybrid = "associate" if education_hybrid == "diploma"
replace education_hybrid = "missing" if !inlist(education_hybrid, "highschool", "associate", "bachelor", "master", "missing")

* edu_ols: 5-urovnova granularni (pro OLS — Tabulka 5, Priloha A Heckman, Priloha C)
gen edu_ols = .
replace edu_ols = 0 if education_hybrid == "missing"
replace edu_ols = 1 if education_hybrid == "highschool"
replace edu_ols = 2 if education_hybrid == "associate"
replace edu_ols = 3 if education_hybrid == "bachelor"
replace edu_ols = 4 if education_hybrid == "master"
label define edu_ols_lbl 0 "Missing" 1 "High School" 2 "Associate" 3 "Bachelor" 4 "Master+"
label values edu_ols edu_ols_lbl
label variable edu_ols "Vzdělání (5 úrovní pro OLS)"

* edu_logit: 3-urovnova pro binarni logit (Tabulka 2) — HS+Associate sloucene.
* Reference: ib2.edu_logit (Bachelor or Higher).
gen edu_logit = .
replace edu_logit = 0 if inlist(education_hybrid, "missing", "")
replace edu_logit = 1 if inlist(education_hybrid, "highschool", "associate")
replace edu_logit = 2 if inlist(education_hybrid, "bachelor", "master")
label define edu_logit_lbl 0 "Missing" 1 "HS / Associate" 2 "Bachelor or Higher"
label values edu_logit edu_logit_lbl
label variable edu_logit "Vzdělání (3 úrovně pro binární logit)"

display _n "--- 3.3 edu_ols rozdělení po zemích ---"
tab edu_ols country, col
display _n "--- 3.3 edu_logit rozdělení po zemích ---"
tab edu_logit country, col

* --- 3.4 Zkusenosti: exp_category (4 urovne) ---
* Shodne s `ai_skills_analysis_comparative.do` (2024). Reference: ib3.exp_category
* (Mid 3-5 let) — statisticky nejcastejsi i vecne nejprimerenejsi kategorie.
destring experience_min_llm, replace force
label variable experience_min_llm "Min. požadované roky zkušeností"

* Kvadraticky clen pro Mincerovu specifikaci (sekce 13)
gen experience_sq = experience_min_llm^2
label variable experience_sq "Zkušenosti na druhou (Mincer)"

gen exp_category = .
replace exp_category = 0 if experience_min_llm == .
replace exp_category = 2 if experience_min_llm >= 0 & experience_min_llm <= 2
replace exp_category = 3 if experience_min_llm >  2 & experience_min_llm <= 5
replace exp_category = 4 if experience_min_llm >  5 & experience_min_llm <  .
label define exp_lbl 0 "Missing" 2 "Junior (0-2)" 3 "Mid (3-5)" 4 "Senior+ (6+)"
label values exp_category exp_lbl
label variable exp_category "Seniorita (4 úrovně, min. roky praxe)"

display _n "--- 3.4 exp_category rozdělení po zemích ---"
tab exp_category country, col

* --- 3.5 Plat — prevod na USD a rocni bazi ---
destring salary_min salary_mid salary_max, replace force

* Pevne kurzy (prumer Sep-Oct 2025)
local eur_usd = 1.165
local inr_usd = 88

* Neplatny/nesedici currency neznamena, ze cely inzerat mame vyhodit
* (porad je pouzitelny pro logit/mlogit na has_ai). Pouze invalidujeme plat.
count if country == "DE" & !inlist(pay_currency, "EUR", "") & salary_mid != .
display "DE inzeraty s neocekavanou menou (plat nastaven na missing): " r(N)
foreach var of varlist salary_min salary_mid salary_max {
    replace `var' = . if country == "DE" & !inlist(pay_currency, "EUR", "")
}

count if country == "IN" & pay_currency == "" & salary_mid != .
display "IN inzeraty bez uvedene meny (plat nastaven na missing): " r(N)
foreach var of varlist salary_min salary_mid salary_max {
    replace `var' = . if country == "IN" & pay_currency == ""
}

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

* Country-specific salary floor. Cilem je vyradit zjevne chybne konverze
* a neserozni inzeraty, nikoli legitimni entry-level pozice.
* IN: ~170 000 INR/rok (~1 933 USD) = reportovana minimalni mzda IT juniora;
* DE/US: 3 000 USD/rok = filtr absurdne nizkych castek (ucednicke pozice,
* chyby v konverzi). Horni hranice 500 000 USD spolecna (outlier cap).
count if salary_mid != . & country == "IN" & salary_mid < 2000
display "IN platy pod 2000 USD/rok (vyrazeno jako outliery): " r(N)
count if salary_mid != . & inlist(country, "US", "DE") & salary_mid < 3000
display "US/DE platy pod 3000 USD/rok (vyrazeno jako outliery): " r(N)
count if salary_mid != . & salary_mid > 500000
display "Platy nad 500 000 USD/rok (vyrazeno jako outliery): " r(N)
replace salary_mid = . if country == "IN" & salary_mid < 2000
replace salary_mid = . if inlist(country, "US", "DE") & salary_mid < 3000
replace salary_mid = . if salary_mid > 500000 & salary_mid != .
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
replace size_cat = 3 if inlist(size, "201 to 500 Employees", "201 bis 500 Mitarbeiter", "501 to 1000 Employees", "501 bis 1.000 Mitarbeiter")
replace size_cat = 5 if inlist(size, "1001 to 5000 Employees", "1.001 bis 5.000 Mitarbeiter")
replace size_cat = 6 if inlist(size, "5001 to 10000 Employees", "5.001 bis 10.000 Mitarbeiter")
replace size_cat = 7 if inlist(size, "10000+ Employees", "Mehr als 10.000 Mitarbeiter")
label define size_lbl 0 "Unknown" 1 "1-50" 2 "51-200" 3 "201-1000" ///
    5 "1001-5000" 6 "5001-10000" 7 "10000+"
label values size_cat size_lbl
label variable size_cat "Velikost firmy (ordinální)"

* --- 3.7 Typ firmy ---
gen type_cat = .
replace type_cat = 0 if inlist(type, "", "Unknown", "Contract", "Self-employed", "Private Practice / Firm", "Franchise")
replace type_cat = 0 if inlist(type, "Unbekannt", "Auftragsunternehmen", "Selbstständig", "Privatpraxis/Kanzlei")
replace type_cat = 1 if inlist(type, "Company - Private", "Subsidiary or Business Segment", "Privatunternehmen", "Tochtergesellschaft oder Geschäftsbereich")
replace type_cat = 2 if inlist(type, "Company - Public", "Aktiengesellschaft")
replace type_cat = 0 if inlist(type, "Nonprofit Organization", "Non-profit Organisation", "Government", "College / University", "School / School District", "Hospital")
replace type_cat = 0 if inlist(type, "Gemeinnützige Organisation", "Öffentlicher Dienst", "Hochschule/Universität", "Schule/Schulbezirk", "Krankenhaus")
label define type_lbl 0 "Unknown/Other/Gov" 1 "Private/Subsidiary" 2 "Public"
label values type_cat type_lbl
label variable type_cat "Typ firmy"

* --- 3.8 NACE sektor ---
* Sektory s menším počtem N (< 20) sloučeny přímo do Other.
replace sector_nace = "Unknown" if sector_nace == ""
replace sector_nace = "Other" if !inlist(sector_nace, "J", "C", "Unknown")
encode sector_nace, generate(sector_nace_num)
label variable sector_nace_num "NACE sektor"

* --- 3.9 Remote prace ---
gen is_remote = 0
replace is_remote = 1 if strpos(lower(remote_work_types), "home") > 0
replace is_remote = 1 if strpos(lower(remote_work_types), "remote") > 0
label variable is_remote "Možnost remote práce (1=ano)"

* --- 3.10 Job family ---
* Sloučení malých kategorií do "Other" kvůli sparse cells v mlogit
* (ai_level=2 × job_family pro DE/IN: Security=1, Frontend=3, QA=7).
* Shodně s verzí 2024 (ai_skills_analysis_comparative.do ř. 296-300)
* a s doporučením vedoucího "raději slučovat než vyhazovat".
* Výsledná taxonomie: 7 kategorií - Data & AI, DevOps & Cloud,
* Management, Software Developer, Software Engineer, Sr+ SW Engineer,
* Other. Všechny klíčové AI-signálové kategorie zachovány.
replace job_family = "Unknown" if job_family == ""
replace job_family = "Other" if inlist(job_family, ///
    "Frontend & Design", "QA & Testing", "Security", "Systems & Embedded")
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

* --- 3.12b Anglicke labely skill clusteru (pro esttab vystupy) ---
* Shodne s anglickymi nazvy pouzitymi v prozaickem textu kapitoly 5, aby
* nedochazelo k nekonzistenci mezi labely v tabulkach a nazvy v textu.
capture label var cluster_architecture__methods  "Architecture & Methods"
capture label var cluster_bi__analytics          "BI & Analytics"
capture label var cluster_backend_development    "Backend Development"
capture label var cluster_certifications         "Certifications"
capture label var cluster_cloud_computing        "Cloud Computing"
capture label var cluster_data_engineering       "Data Engineering"
capture label var cluster_data_science__ml       "Data Science / ML"
capture label var cluster_databases__storage     "Databases & Storage"
capture label var cluster_devops__containers     "DevOps & Containers"
capture label var cluster_dynamic__web           "Dynamic Web"
capture label var cluster_enterprise__managed    "Enterprise / Managed"
capture label var cluster_enterprise_platforms   "Enterprise Platforms"
capture label var cluster_frontend_development   "Frontend Development"
capture label var cluster_generative_ai          "Generative AI"
capture label var cluster_mobile__desktop        "Mobile & Desktop"
capture label var cluster_networking             "Networking"
capture label var cluster_os__embedded           "OS & Embedded"
capture label var cluster_scripting__shell       "Scripting / Shell"
capture label var cluster_security__identity     "Security & Identity"
capture label var cluster_systems_programming    "Systems Programming"
capture label var cluster_testing_qa__debugging  "Testing / QA & Debugging"

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

* --- 3.15 Baseline NACE sektor = J (Information & Communication) ---
* Diplomka se zameruje na IT inzeraty, drtiva vetsina vzorku je J; baseline
* = J dava smysluplnou interpretaci "odchylka od IT sektoru".
quietly levelsof sector_nace_num if sector_nace == "J", local(_nace_list)
local nace_base : word 1 of `_nace_list'
if "`nace_base'" == "" {
    display as error "Sector_nace_num = J (Information & Communication) nenalezeno — fallback na default baseline"
    local nace_base 1
}
display "Baseline sector_nace_num (J = Information & Communication) = `nace_base'"

* --- 3.16 Baseline region_num = South (nejvetsi US region) ---
* Jen pro US specifikaci; pri absenci region == "South" padame zpet na
* region s nejvyssi cetnosti v US vzorku.
quietly levelsof region_num if region == "South", local(_reg_list)
local region_base : word 1 of `_reg_list'
if "`region_base'" == "" {
    display as error "Region == South nenalezeno — fallback na nejcetnejsi region"
    quietly levelsof region_num if country == "US", local(_us_reg_levels)
    local best_count = 0
    local best_level = 1
    foreach lv of local _us_reg_levels {
        quietly count if region_num == `lv' & country == "US"
        if r(N) > `best_count' {
            local best_count = r(N)
            local best_level = `lv'
        }
    }
    local region_base `best_level'
}
display "Baseline region_num (South) = `region_base'"


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
             "AI tiery odvozeny pouze z LLM klasifikace desc_tier_llm (bez post-hoc override ze skill clusterů).")


* ==============================================================================
* 4b. KOMPARATIVNI DESKRIPCE — χ² ASOCIACE PRO §5.1
* ==============================================================================
* Doplnkove testy asociace klicovych deskriptivnich promennych s has_ai
* per zeme. Cisla pouzije text kapitoly 5.1 (Deskriptivni statistika).
*   - remote × has_ai per zeme
*   - job_family × has_ai per zeme
*   - exp_category × has_ai per zeme (pro US doplneni)
* Vysledky jdou pouze do logu, nikoli do RTF tabulky.

display _n "=============================================================="
display "4b. χ² ASOCIACE PRO §5.1 (per zeme)"
display "=============================================================="

foreach c in US DE IN {
    display _n "=== Zeme: `c' ==="
    display _n "--- remote × has_ai (`c') ---"
    tab is_remote has_ai if country == "`c'", col chi2
    display _n "--- job_family × has_ai (`c') ---"
    tab job_family has_ai if country == "`c'", col chi2
    display _n "--- exp_category × has_ai (`c') ---"
    tab exp_category has_ai if country == "`c'", col chi2
}

* Mzdovy t-test None vs AI (AI Integration + Applied/Core AI)
* kvantifikuje hrubou AI premii per zeme pred regresni analyzou.
display _n "--- Mzdovy t-test: None vs AI (salary_mid) per zeme ---"
foreach c in US DE IN {
    display _n "=== Zeme: `c' ==="
    ttest salary_mid if country == "`c'", by(has_ai) unequal
}


* ==============================================================================
* 6. TABULKY 2 + 3 — BINARNI LOGIT P(AI) ~ JOB FAMILY ^ SKILL CLUSTERY (AME, per zeme)
* ==============================================================================
* Dva komplementarni modely per zeme (celkem 6 sloupcu):
*   - Levy panel (3 sloupce): P(AI) ~ job_family + controls
*   - Pravy panel (3 sloupce): P(AI) ~ skill clustery + controls
* Reporting: prumerne marginalni efekty (AME), robustni SE clusterovane na firmu.

* --- 6.0 Crosstab diagnostika: kontrola minimálních počtů pozorování pro logit modely ---
display _n "--- 6.0 Crosstab diagnostika (minimum obs v logit/mlogit predictors per zeme) ---"
foreach c in US DE IN {
    display _n "=== Diagnostika pro zemi: `c' ==="
    foreach var in job_family_num size_cat type_cat is_remote edu_ols edu_logit exp_category sector_nace_num {
        display "Crosstab `var' vs has_ai (`c')"
        tab `var' has_ai if country == "`c'", missing
        display "Crosstab `var' vs ai_level (`c')"
        tab `var' ai_level if country == "`c'", missing
    }
}

* --- 6.0b Skill clustery vs ai_level (DE, IN) — overeni sparse cells ---
* US ma vzdy dost dat; u DE/IN kontrolujeme, ze zadny cluster x Applied AI
* nema pathologicky malou bunku (< 10 obs).
display _n "--- 6.0b Skill clustery vs ai_level (DE, IN) ---"
foreach c in DE IN {
    display _n "=== Skill clustery x ai_level pro `c' ==="
    foreach clvar of varlist cluster_* {
        display "Crosstab `clvar' vs ai_level (`c')"
        tab `clvar' ai_level if country == "`c'", row
    }
}

* --- 6.0c Pokryti platu (has_salary) per zeme — informativni pro Heckman / OLS ---
display _n "--- 6.0c Pokryti inzerovaneho platu per zeme ---"
tab country has_salary, row missing

display _n "=============================================================="
display "6. TABULKY 2 + 3 — BINARNI LOGIT, JOB FAMILY (T2) a SKILL CLUSTERY (T3)"
display "=============================================================="

* --- 6a. Job family panel ---
foreach c in US DE IN {
    display _n "=========== Zeme = `c' (T2 job family) ==========="
    capture noisily logit has_ai ///
        ib`sw_base'.job_family_num ///
        ib2.edu_logit ///
        ib3.exp_category ///
        ib`nace_base'.sector_nace_num ///
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
    display _n "=========== Zeme = `c' (T3 skill clustery) ==========="
    capture noisily logit has_ai ///
        cluster_* ///
        ib2.edu_logit ///
        ib3.exp_category ///
        ib`nace_base'.sector_nace_num ///
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

        * Hosmer-Lemeshow Goodness of Fit (vyzaduje model bez klastrovani)
        capture quietly logit has_ai ///
            cluster_* ///
            ib2.edu_logit ///
            ib3.exp_category ///
            ib`nace_base'.sector_nace_num ///
            ib1.type_cat ///
            ib5.size_cat ///
            is_remote ///
            if country == "`c'"
        if _rc == 0 {
            display _n ">> Hosmer-Lemeshow GOF test pro `c' (bez klastrovani vzorku)"
            capture noisily estat gof, group(10)
        }
    }
    else {
        display as error "Logit T3 (skill clustery) pro `c' selhal (rc=" _rc ")"
    }
}

* --- 6c. Export Tabulky 2 (job family, 3 sloupce) ---
esttab ame_t2jf_US ame_t2jf_DE ame_t2jf_IN ///
    using "$outdir/Tabulka_2_Logit_JobFamily.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    keep(*.job_family_num *.edu_logit *.exp_category) ///
    order(*.job_family_num *.edu_logit *.exp_category) ///
    stats(ctrl_nace ctrl_type ctrl_size ctrl_remote N, ///
          fmt(%s %s %s %s %9.0fc) ///
          labels("NACE sektor" "Typ firmy" "Velikost firmy" "Remote" "N")) ///
    mtitles("USA" "Německo" "Indie") ///
    title("Tabulka 2: Binární logit P(AI požadavek = ano) — job family (AME podle země)") ///
    addnotes("Závislá proměnná: has_ai (1=AI Integration nebo Applied/Core AI, 0=None)." ///
             "Průměrné marginální efekty (AME) z logitu, SE klastrované na firmu v závorkách." ///
             "Referenční job family: Software Engineer." ///
             "Tato specifikace neobsahuje skill clustery — ty jsou samostatně v Tabulce 3 se stejnými kontrolami. Oddělené modely umožňují čistou interpretaci profesních a dovednostních efektů bez vzájemné kanibalizace koeficientů (job family a skill clustery jsou silně korelované)." ///
             "edu_logit: 3 úrovně (referenční Bachelor+); exp_category: 4 úrovně (referenční Mid 3–5 let).")

* --- 6d. Export Tabulky 3 (skill clustery, 3 sloupce) ---
esttab ame_t2sk_US ame_t2sk_DE ame_t2sk_IN ///
    using "$outdir/Tabulka_3_Logit_SkillClusters.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    keep(cluster_* *.edu_logit *.exp_category) ///
    order(cluster_* *.edu_logit *.exp_category) ///
    stats(ctrl_nace ctrl_type ctrl_size ctrl_remote N, ///
          fmt(%s %s %s %s %9.0fc) ///
          labels("NACE sektor" "Typ firmy" "Velikost firmy" "Remote" "N")) ///
    mtitles("USA" "Německo" "Indie") ///
    title("Tabulka 3: Binární logit P(AI požadavek = ano) — skill clustery (AME podle země)") ///
    addnotes("Závislá proměnná: has_ai (1=AI Integration nebo Applied/Core AI, 0=None)." ///
             "Průměrné marginální efekty (AME) z logitu, SE klastrované na firmu v závorkách." ///
             "Všechny skill clustery (včetně cluster_generative_ai a cluster_data_science__ml) jsou součástí RHS jako binární (0/1) prediktory; AME udává změnu pravděpodobnosti AI požadavku při přítomnosti daného clusteru." ///
             "Tato specifikace neobsahuje job_family (zachycení profesních efektů viz Tabulka 2 se stejnými kontrolami)." ///
             "edu_logit: 3 úrovně (referenční Bachelor+); exp_category: 4 úrovně (referenční Mid 3–5 let).")


* ==============================================================================
* 7. TABULKA 4 — MLOGIT P(AI tier) ~ SKILL CLUSTERY (AME, 9 sloupcu)
* ==============================================================================
display _n "=============================================================="
display "7. TABULKA 4 — MLOGIT, SKILL CLUSTERY"
display "=============================================================="

foreach c in US DE IN {
    display _n "=========== Zeme = `c' (Tabulka 4) ==========="

    * POZN: vzdelani (edu_logit) z mlogitu ZAMERNE vyrazeno — kategorie
    * HS/Associate x Applied/Core AI ma v DE/IN < 25 pozorovani (quasi-
    * complete separation risk). Stejny pristup pouzila verze 2024
    * (ai_skills_analysis_comparative.do). Seniorita (exp_category) je
    * zachovana, ma dostatecne buňky napric outcomy.

    * IIA Hausman test (vyzaduje model bez vce(cluster), otestujeme vynechani Core AI)
    capture quietly mlogit ai_level cluster_* ib3.exp_category ib`nace_base'.sector_nace_num ib1.type_cat ib5.size_cat is_remote if country == "`c'", baseoutcome(0)
    if _rc == 0 {
        estimates store mfull_`c'
        capture quietly mlogit ai_level cluster_* ib3.exp_category ib`nace_base'.sector_nace_num ib1.type_cat ib5.size_cat is_remote if country == "`c'" & ai_level != 2, baseoutcome(0)
        if _rc == 0 {
            estimates store msub_`c'
            display _n ">> IIA Hausman Test pro `c' (Full vs subset bez Applied/Core AI)"
            capture noisily hausman msub_`c' mfull_`c', alleqs constant
        }
    }

    * Tri beh modelu (pro kazdy outcome zvlast), protoze margins post prepisuje
    * e(b). Pouzivame AME predict(outcome(k)) pro k=0,1,2.
    foreach o in 0 1 2 {
        capture noisily mlogit ai_level ///
            cluster_* ///
            ib3.exp_category ///
            ib`nace_base'.sector_nace_num ///
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
            display as error "Mlogit Tabulka 4 pro `c' outcome `o' selhal (rc=" _rc ")"
        }
    }
}

esttab ame_t3_US_0 ame_t3_US_1 ame_t3_US_2 ///
       ame_t3_DE_0 ame_t3_DE_1 ame_t3_DE_2 ///
       ame_t3_IN_0 ame_t3_IN_1 ame_t3_IN_2 ///
       using "$outdir/Tabulka_4_Mlogit_AI_Tier.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    keep(cluster_* *.exp_category) ///
    order(cluster_* *.exp_category) ///
    stats(ctrl_nace ctrl_type ctrl_size ctrl_remote N, ///
          fmt(%s %s %s %s %9.0fc) ///
          labels("NACE sektor" "Typ firmy" "Velikost firmy" "Remote" "N")) ///
    mgroups("USA" "Německo" "Indie", pattern(1 0 0 1 0 0 1 0 0)) ///
    mtitles("None" "Integ." "Applied" "None" "Integ." "Applied" "None" "Integ." "Applied") ///
    title("Tabulka 4: Multinomiální logit P(AI tier) — AME podle země a úrovně") ///
    addnotes("Závislá proměnná: ai_level (0=None, 1=AI Integration, 2=Applied/Core AI)." ///
             "Průměrné marginální efekty (AME) z mlogitu, SE klastrované na firmu v závorkách, base outcome = None." ///
             "Všechny skill clustery (včetně cluster_generative_ai a cluster_data_science__ml) jsou v RHS." ///
             "Vzdělání (edu_logit) z mlogitu vyřazeno kvůli sparse cells HS/Assoc × Applied AI v DE/IN (shodně s verzí 2024)." ///
             "Interpretace: reverzní inženýrství AI tiers — jaké inzeráty/skills LLM typicky klasifikuje do jednotlivých úrovní (popisné, nikoli kauzální)." ///
             "Součet AME napříč třemi outcomes pro každou proměnnou je 0.")


* ==============================================================================
* 8. TABULKA 5 — OLS ln(plat) ~ SKILL CLUSTERY + AI TIERY (per zeme)
* ==============================================================================
* Hlavni specifikace: cluster_generative_ai a cluster_data_science__ml JSOU
* VYRAZENY z RHS, protoze jsou konstrukcne cast ai_level (LLM je pri klasifikaci
* do AI tiers implicitne zohlednuje -> cirkularita). Koeficienty i.ai_level
* tim pádem zachytavaji mzdovou premii role *vcetne* znalosti AI technologii,
* ktere ta role implicitne predpoklada. Plna specifikace s temito clustery
* zpet v RHS = Priloha C (sekce 8b, robustness check).
display _n "=============================================================="
display "8. TABULKA 5 — OLS ln(plat) per zeme"
display "=============================================================="

* --- 8.0 Minimum-N diagnostika per zeme ---
* OLS mzdovy model je citlivy na maly vzorek a rozlozeni ai_level.
* Prah: alespon 200 inzeratu s platem per zeme + alespon 10 v kazdem
* ne-baseline ai_level (jinak koeficient 1.ai_level / 2.ai_level nelze
* spolehlive interpretovat). Pri nedostatku se varuje, ale OLS se presto
* spusti (Heckman v Priloze A slouzi jako robustness pro DE).
display _n "--- 8.0 Minimum-N diagnostika (OLS mzdovy model) ---"
foreach c in US DE IN {
    count if country == "`c'" & ln_salary != .
    local n_sal = r(N)
    count if country == "`c'" & ln_salary != . & ai_level == 0
    local n_sal_0 = r(N)
    count if country == "`c'" & ln_salary != . & ai_level == 1
    local n_sal_1 = r(N)
    count if country == "`c'" & ln_salary != . & ai_level == 2
    local n_sal_2 = r(N)
    display "`c': N(plat)=`n_sal' | ai_level 0=`n_sal_0', 1=`n_sal_1', 2=`n_sal_2'"
    if `n_sal' < 200 {
        display as error "  VAROVANI: `c' ma mene nez 200 inzeratu s platem — OLS podhodnoceny."
    }
    if `n_sal_1' < 10 | `n_sal_2' < 10 {
        display as error "  VAROVANI: `c' ma < 10 obs v nejakem ai_level tieru — koeficient na okraji interpretovatelnosti."
    }
}

* Lokalni makro: vsechny skill clustery krome cluster_generative_ai a
* cluster_data_science__ml (tj. RHS pro hlavni OLS specifikaci).
unab all_clusters : cluster_*
local ols_excl cluster_generative_ai cluster_data_science__ml
local ols_clusters : list all_clusters - ols_excl
display _n "OLS skill clustery (bez GenAI a DS/ML):"
display "`ols_clusters'"

* US (symetricka specifikace s DE/IN — bez region FE; region FE varianta viz Priloha D)
display _n "=========== Zeme = US (Tabulka 5) ==========="
capture noisily regress ln_salary ///
    `ols_clusters' ///
    i.ai_level ///
    ib3.edu_ols ///
    ib3.exp_category ///
    ib`nace_base'.sector_nace_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    if country == "US" & ln_salary != ., vce(cluster firm_cluster)
if _rc == 0 {
    estadd local ctrl_nace   "Ano"
    estadd local ctrl_type   "Ano"
    estadd local ctrl_size   "Ano"
    estadd local ctrl_remote "Ano"
    estimates store ols_t4_US
    display _n "--- VIF Kontrola (US) ---"
    quietly regress ln_salary ///
        `ols_clusters' ///
        i.ai_level ///
        ib3.edu_ols ///
        ib3.exp_category ///
        ib`nace_base'.sector_nace_num ///
        is_remote ///
        ib1.type_cat ///
        ib5.size_cat ///
        if country == "US" & ln_salary != .
    capture noisily estat vif
}

foreach c in DE IN {
    display _n "=========== Zeme = `c' (Tabulka 5) ==========="
    capture noisily regress ln_salary ///
        `ols_clusters' ///
        i.ai_level ///
        ib3.edu_ols ///
        ib3.exp_category ///
        ib`nace_base'.sector_nace_num ///
        is_remote ///
        ib1.type_cat ///
        ib5.size_cat ///
        if country == "`c'" & ln_salary != ., vce(cluster firm_cluster)
    if _rc == 0 {
        estadd local ctrl_nace   "Ano"
        estadd local ctrl_type   "Ano"
        estadd local ctrl_size   "Ano"
        estadd local ctrl_remote "Ano"
        estimates store ols_t4_`c'
        display _n "--- VIF Kontrola (`c') ---"
        quietly regress ln_salary ///
            `ols_clusters' ///
            i.ai_level ///
            ib3.edu_ols ///
            ib3.exp_category ///
            ib`nace_base'.sector_nace_num ///
            is_remote ///
            ib1.type_cat ///
            ib5.size_cat ///
            if country == "`c'" & ln_salary != .
        capture noisily estat vif
    }
}

esttab ols_t4_US ols_t4_DE ols_t4_IN using "$outdir/Tabulka_5_OLS_lnMzda.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    keep(cluster_* 1.ai_level 2.ai_level *.edu_ols *.exp_category) ///
    order(1.ai_level 2.ai_level cluster_* *.edu_ols *.exp_category) ///
    stats(ctrl_nace ctrl_type ctrl_size ctrl_remote N r2, ///
          fmt(%s %s %s %s %9.0fc %5.3f) ///
          labels("NACE sektor" "Typ firmy" "Velikost firmy" "Remote" "N" "R²")) ///
    mtitles("USA" "Německo" "Indie") ///
    title("Tabulka 5: OLS ln(plat) — skill clustery + AI tiery, separátně podle země") ///
    addnotes("Závislá proměnná: ln(roční plat v USD)." ///
             "Standardní chyby klastrované na firmu v závorkách." ///
             "Referenční AI úroveň: None." ///
             "Symetrická specifikace napříč zeměmi bez regionálních fixních efektů. Varianta s US Census region FE viz Příloha D (robustness)." ///
             "cluster_generative_ai a cluster_data_science__ml záměrně vynechány z RHS — jsou konstrukčně součástí klasifikace ai_level (cirkularita). Koeficienty i.ai_level tak zachycují mzdovou prémii role včetně implicitních AI znalostí." ///
             "Plná specifikace s oběma clustery zpět v RHS jako robustness viz Příloha C." ///
             "Pozor: v DE je pokrytí platu velmi nízké — viz Heckman robustness v Příloze A.")


* ==============================================================================
* 8b. PRILOHA C — OLS ln(plat) s plnou sadou skill clusteru (robustness)
* ==============================================================================
* Robustnostni kontrola k Tabulce 5: stejna specifikace, ale s navratem
* cluster_generative_ai a cluster_data_science__ml zpet do RHS. Pokud budou
* koeficienty i.ai_level stabilni oproti Tabulce 5, je to signal, ze efekt
* AI tieru zachycuje roli (nikoli jen prekryv s GenAI/ML skills).
display _n "=============================================================="
display "8b. PRILOHA C — OLS ln(plat) s plnou sadou skill clusteru (robustness)"
display "=============================================================="

* US (symetricka specifikace s DE/IN — bez region FE)
display _n "=========== Zeme = US (Priloha C) ==========="
capture noisily regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    ib3.edu_ols ///
    ib3.exp_category ///
    ib`nace_base'.sector_nace_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    if country == "US" & ln_salary != ., vce(cluster firm_cluster)
if _rc == 0 {
    estadd local ctrl_nace   "Ano"
    estadd local ctrl_type   "Ano"
    estadd local ctrl_size   "Ano"
    estadd local ctrl_remote "Ano"
    estimates store ols_robC_US
}

foreach c in DE IN {
    display _n "=========== Zeme = `c' (Priloha C) ==========="
    capture noisily regress ln_salary ///
        cluster_* ///
        i.ai_level ///
        ib3.edu_ols ///
        ib3.exp_category ///
        ib`nace_base'.sector_nace_num ///
        is_remote ///
        ib1.type_cat ///
        ib5.size_cat ///
        if country == "`c'" & ln_salary != ., vce(cluster firm_cluster)
    if _rc == 0 {
        estadd local ctrl_nace   "Ano"
        estadd local ctrl_type   "Ano"
        estadd local ctrl_size   "Ano"
        estadd local ctrl_remote "Ano"
        estimates store ols_robC_`c'
    }
}

esttab ols_robC_US ols_robC_DE ols_robC_IN using "$outdir/Priloha_C_OLS_FullClusters.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    keep(cluster_* 1.ai_level 2.ai_level *.edu_ols *.exp_category) ///
    order(1.ai_level 2.ai_level cluster_* *.edu_ols *.exp_category) ///
    stats(ctrl_nace ctrl_type ctrl_size ctrl_remote N r2, ///
          fmt(%s %s %s %s %9.0fc %5.3f) ///
          labels("NACE sektor" "Typ firmy" "Velikost firmy" "Remote" "N" "R²")) ///
    mtitles("USA" "Německo" "Indie") ///
    title("Příloha C: OLS ln(plat) — plná sada skill clusterů (robustness k Tabulce 5)") ///
    addnotes("Robustnostní specifikace k Tabulce 5 s návratem cluster_generative_ai a cluster_data_science__ml do RHS." ///
             "Stabilita koeficientů 1.ai_level a 2.ai_level mezi Tabulkou 4 a Přílohou C indikuje, že efekt AI tierů zachycuje roli, nikoli jen překryv s GenAI/ML skills." ///
             "Standardní chyby klastrované na firmu v závorkách.")


* ==============================================================================
* 8c. PRILOHA D — US OLS ln(plat) s region FE (robustness k Tabulce 5)
* ==============================================================================
* Robustnostni kontrola k Tabulce 5 pro USA: stejna specifikace jako hlavni US
* OLS, ale s pridanymi 4 US Census regions jako fixnimi efekty (South jako
* referencni). Cilem je doložit, ze regionalni heterogenita mezd v USA (West /
* Northeast maji vyssi mzdy a vyssi koncentraci AI) nezavadi podstatne zkresleni
* do AI koeficientu; hlavni Tabulka 5 je bez region FE kvuli symetrii s DE/IN
* (ty ekvivalentni regionalni promennou v datech nemaji).
display _n "=============================================================="
display "8c. PRILOHA D — US OLS ln(plat) s region FE (robustness)"
display "=============================================================="

display _n "=========== Zeme = US (Priloha D — s region FE) ==========="
capture noisily regress ln_salary ///
    `ols_clusters' ///
    i.ai_level ///
    ib3.edu_ols ///
    ib3.exp_category ///
    ib`nace_base'.sector_nace_num ///
    ib`region_base'.region_num ///
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
    estimates store ols_robD_US_regFE
}

* Export side-by-side: hlavni US OLS (bez region FE, z Tabulky 4) vs. Priloha D (s region FE).
capture esttab ols_t4_US ols_robD_US_regFE using "$outdir/Priloha_D_OLS_US_RegionFE.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    keep(cluster_* 1.ai_level 2.ai_level *.edu_ols *.exp_category) ///
    order(1.ai_level 2.ai_level cluster_* *.edu_ols *.exp_category) ///
    stats(ctrl_nace ctrl_type ctrl_size ctrl_remote ctrl_region N r2, ///
          fmt(%s %s %s %s %s %9.0fc %5.3f) ///
          labels("NACE sektor" "Typ firmy" "Velikost firmy" "Remote" "Region FE (US Census)" "N" "R²")) ///
    mtitles("US bez region FE" "US s region FE") ///
    title("Příloha D: USA OLS ln(plat) — robustness vůči regionální heterogenitě") ///
    addnotes("Levý sloupec: hlavní specifikace z Tabulky 4 (bez region FE, symetrická s DE/IN)." ///
             "Pravý sloupec: identická specifikace doplněná o fixní efekty pro 4 US Census regions (South jako referenční, dále Midwest, Northeast, West)." ///
             "Srovnání sloupců umožňuje kvantifikovat, o kolik procentních bodů se koeficienty AI tierů a skill clusterů posunou po zavedení regionálních FE." ///
             "Standardní chyby klastrované na firmu v závorkách.")


* ==============================================================================
* 10. GRAFY — ROZLOZENI ROCNICH PLATU (USD) PODLE AI UROVNE, per zeme
* ==============================================================================
* Kernel density krivky na raw USD skale (nikoliv ln). Kazda krivka je
* normalizovana samostatne (integral = 1), takze vsechny tri AI urovne maji
* srovnatelnou vizualni vahu a ctenar vidi primo posun peaku doprava u AI
* inzeratu — hlavni poselstvi grafu je tvar rozdeleni a poloha peaku, nikoli
* velikosti vzorku per AI uroven (ty jsou v deskriptivni tabulce §5.1).
* Vyplnene prekryvajici se plochy (area) se svetle seda -> svetle modra ->
* tmava modra, alpha transparentnost pro prekryvy. Osa X v USD s oddelovaci
* tisicu pro okamzitou citelnost (log skala v regresnim modelu OLS, ale ne
* v popisnem grafu).
display _n "=============================================================="
display "10. GRAFY — ROZLOZENI ROCNICH PLATU PODLE AI UROVNE (HUSTOTA)"
display "=============================================================="

foreach c in US DE IN {
    * Country label pro titulek grafu
    local c_label = ""
    if "`c'" == "US" local c_label "USA"
    if "`c'" == "DE" local c_label "Německo"
    if "`c'" == "IN" local c_label "Indie"

    capture drop _kx_* _kf_*
    local plot_ok = 1

    foreach lv in 0 1 2 {
        count if country == "`c'" & ai_level == `lv' & salary_mid != .
        local n_`lv' = r(N)
        if `n_`lv'' < 10 {
            display as text "Skupina ai_level=`lv' v `c' ma jen `n_`lv'' obs — preskakujeme kdensity."
            local plot_ok = 0
            continue
        }
        capture noisily kdensity salary_mid if country == "`c'" & ai_level == `lv', ///
            generate(_kx_`lv' _kf_`lv') nograph
        if _rc != 0 {
            display as error "kdensity selhal pro `c' ai_level=`lv' (rc=" _rc ")"
            local plot_ok = 0
            continue
        }
        * Per-group hustota (integral kazde krivky = 1). Nepreskalovavame
        * podle podilu skupiny — vsechny tri krivky jsou tak vizualne
        * srovnatelne a zdurazni se posun peaku u AI inzeratu.
    }

    if `plot_ok' == 1 {
        * Vykresleni: velka skupina (None) na pozadi, AI skupiny nahore
        * s alpha transparentnosti. Stata 15+ podporuje alpha via %XX suffix.
        capture noisily twoway ///
            (area _kf_0 _kx_0, color(gs14) lcolor(gs10) lwidth(medthin)) ///
            (area _kf_1 _kx_1, color(eltblue%70) lcolor(edkblue) lwidth(medthin)) ///
            (area _kf_2 _kx_2, color(edkblue%60) lcolor(navy) lwidth(medthin)), ///
            title("`c_label': Rozdělení ročních platů podle úrovně AI", size(medium)) ///
            xtitle("Roční plat (USD)") ytitle("Hustota") ///
            xlabel(, format(%9.0fc)) ///
            ylabel(, format(%9.2e)) ///
            legend(order(1 "Bez AI" 2 "AI Integration" 3 "Applied/Core AI") rows(1) size(small)) ///
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
        `ols_clusters' ///
        i.ai_level ///
        ib3.edu_ols ///
        ib3.exp_category ///
        ib`nace_base'.sector_nace_num ///
        ib1.type_cat ///
        ib5.size_cat ///
        is_remote ///
        if country == "`c'", ///
        select(has_salary = `ols_clusters' ///
                            i.ai_level ///
                            ib3.edu_ols ///
                            ib3.exp_category ///
                            ib`nace_base'.sector_nace_num ///
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
    keep(cluster_* 1.ai_level 2.ai_level *.edu_ols *.exp_category) ///
    order(1.ai_level 2.ai_level cluster_* *.edu_ols *.exp_category) ///
    stats(N, fmt(0) labels("N")) ///
    mtitles("USA" "Německo" "Indie") ///
    title("Příloha A: Heckman selection model — ln(plat) s korekcí selekce") ///
    addnotes("Stejná specifikace jako Tabulka 5 (bez cluster_generative_ai a cluster_data_science__ml)." ///
             "Rovnice výběru: P(plat inzerován) — Probit se stejnými kontrolami jako ve 2. fázi." ///
             "Identifikace jen funkční formou IMR (bez exclusion restrikce) — slouží jako robustness." ///
             "Porovnat směr a velikost koeficientů s OLS v Tabulce 5, zvláště pro DE.")


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
    ib2.edu_logit ib3.exp_category ///
    ib`nace_base'.sector_nace_num ib1.type_cat ib5.size_cat is_remote, ///
    vce(cluster firm_cluster)
if _rc == 0 {
    display _n "Wald test: vsechny interakce job_family x country = 0"
    testparm i.job_family_num#i.country_id
}

* --- 12.2 Pooled logit: skill clustery x country ---
* Hlavni efekty vsech skill clusteru pres cluster_*; navic pridavame jen
* interakce (# misto ##) pro tri klicove clustery, aby se main effects
* nedublikovaly.
display _n "--- 12.2 Pooled logit has_ai ~ skill_clusters x country (Tabulka 3 test) ---"
capture noisily logit has_ai ///
    cluster_* ///
    ib3.country_id ///
    c.cluster_cloud_computing#ib3.country_id ///
    c.cluster_data_science__ml#ib3.country_id ///
    c.cluster_backend_development#ib3.country_id ///
    ib2.edu_logit ib3.exp_category ///
    ib`nace_base'.sector_nace_num ib1.type_cat ib5.size_cat is_remote, ///
    vce(cluster firm_cluster)
if _rc == 0 {
    display _n "Wald test: klicove skill cluster interakce = 0"
    testparm c.cluster_cloud_computing#i.country_id ///
             c.cluster_data_science__ml#i.country_id ///
             c.cluster_backend_development#i.country_id
}

* --- 12.3 Pooled OLS ln(plat): AI tier x country ---
* POZN: Stejna specifikace jako Tabulka 5 (bez cluster_generative_ai a
* cluster_data_science__ml) kvuli konzistenci s hlavnim OLS modelem.
display _n "--- 12.3 Pooled OLS ln_salary ~ ai_level x country (Tabulka 5 test) ---"
capture noisily regress ln_salary ///
    i.ai_level##ib3.country_id ///
    `ols_clusters' ///
    ib3.edu_ols ib3.exp_category ///
    ib`nace_base'.sector_nace_num ib1.type_cat ib5.size_cat is_remote ///
    if ln_salary != ., vce(cluster firm_cluster)
if _rc == 0 {
    display _n "Wald test: interakce ai_level x country = 0 (AI-premium se mezi zememi lisi?)"
    testparm i.ai_level#i.country_id
}


* ==============================================================================
* 13. INKREMENTALNI MODELY NA US PODVZORKU (pro praktickou cast §5.3 - §5.5)
* ==============================================================================
* Kontext: text v docs/prakticka_cast.md je strukturovan jako inkrementalni
* vypravovani Model A -> B -> C (OLS) a M1 -> M2 -> M3 (logit/mlogit) na
* US datech. Tato sekce replikuje spec ze starsiho `ai_skills_analysis.do`
* (verzija 2025-04-14), ale:
*   - na US podvzorku (`if country == "US"`) pro zachovani srovnatelnosti
*     s textem, kde US je hlavni pripadova studie,
*   - s vce(cluster firm_cluster) — stejny cluster-robust SE jako v hlavnich
*     tabulkach 2-5 a Prilohach A-C. Firma je prirozeny cluster (inzeraty
*     od stejne firmy sdileji firemni unobservables: kulturu, region, tech
*     stack, HR slovnik), US vzorek ma ~2 inzeraty/firma. Klastrovani
*     pres firmu tedy odpovida datove strukture a zajistuje konzistenci
*     SE napric vsemi tabulkami v kapitole 5.
*   - se stejnym kategorickym edu_ols / edu_logit / exp_category schematem.
*
* Per-country finalni tabulky 2–4 (sekce 6–8) a srovnani US vs DE vs IN
* (sekce 12 / Priloha B) zustavaji nedotcene — tato sekce je aditivni.
*
* Rename trick: cluster_generative_ai a cluster_data_science__ml docasne
* prejmenujeme, aby `cluster_*` v hlavnich spec A/B/C a M1/M2/M3 reflektovaly
* pouze NE-cirkularni clustery. Sensitivitní modely (Model C-nocirc, M3-circ)
* pak tyto clustery vrati zpet pro kvantifikaci cirkularity.

display _n "=============================================================="
display "13. INKREMENTALNI MODELY NA US PODVZORKU"
display "=============================================================="

* Docasne prejmenovani cirkularnich clusteru (aby cluster_* je neobsahovalo)
rename cluster_generative_ai    _excl_genai_s13
rename cluster_data_science__ml _excl_dsml_s13

* ------------------------------------------------------------------
* 13.1 OLS Model A: Firemni profil (US)
* ------------------------------------------------------------------
display _n "--- 13.1 OLS Model A: Firemni profil (US, ln_salary) ---"
display "ln(plat) ~ i.ai_level + sektor + region + remote + typ_firmy + velikost_firmy"
regress ln_salary ///
    i.ai_level ///
    ib`nace_base'.sector_nace_num ///
    ib`region_base'.region_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    if country == "US" & ln_salary != ., vce(cluster firm_cluster)
estimates store s13_ols_A
display _n "Model A: R2 = " e(r2) ", Adj R2 = " e(r2_a) ", N = " e(N)

* ------------------------------------------------------------------
* 13.2 OLS Model B: + Lidsky kapital (US)
* ------------------------------------------------------------------
display _n "--- 13.2 OLS Model B: + Lidsky kapital (US) ---"
display "ln(plat) ~ Model A + edu_ols + exp_category"
regress ln_salary ///
    i.ai_level ///
    ib`nace_base'.sector_nace_num ///
    ib`region_base'.region_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib3.edu_ols ///
    ib3.exp_category ///
    if country == "US" & ln_salary != ., vce(cluster firm_cluster)
estimates store s13_ols_B
display _n "Model B: R2 = " e(r2) ", Adj R2 = " e(r2_a) ", N = " e(N)

* ------------------------------------------------------------------
* 13.3 OLS Model C: Plny model (US, bez GenAI/DS-ML)
* ------------------------------------------------------------------
display _n "--- 13.3 OLS Model C: Plny model (US, tech skills + job_family) ---"
display "ln(plat) ~ Model B + cluster_* (bez GenAI/DS-ML) + job_family"
regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    ib`nace_base'.sector_nace_num ///
    ib`region_base'.region_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib`sw_base'.job_family_num ///
    ib3.edu_ols ///
    ib3.exp_category ///
    if country == "US" & ln_salary != ., vce(cluster firm_cluster)
estimates store s13_ols_C
display _n "Model C: R2 = " e(r2) ", Adj R2 = " e(r2_a) ", N = " e(N)

* VIF pro Model C
display _n "--- 13.3b VIF diagnostika Model C (US) ---"
quietly regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    ib`nace_base'.sector_nace_num ///
    ib`region_base'.region_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib`sw_base'.job_family_num ///
    ib3.edu_ols ///
    ib3.exp_category ///
    if country == "US" & ln_salary != .
vif

* ------------------------------------------------------------------
* 13.4 OLS Model C-nojf: Mediace pres job_family
* ------------------------------------------------------------------
display _n "--- 13.4 OLS Model C bez job_family (mediace) ---"
regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    ib`nace_base'.sector_nace_num ///
    ib`region_base'.region_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib3.edu_ols ///
    ib3.exp_category ///
    if country == "US" & ln_salary != ., vce(cluster firm_cluster)
estimates store s13_ols_Cnojf
display _n "Model C-nojf: R2 = " e(r2) ", Adj R2 = " e(r2_a) ", N = " e(N)

* ------------------------------------------------------------------
* 13.5 OLS Model C-Mincer: kontinualni zkusenost + exp^2
* ------------------------------------------------------------------
* POZN: Jiny vzorek nez Model C — missing experience_min_llm = drop (vs.
* kategorie 0 u exp_category). Srovnani R2 je orientacni.
display _n "--- 13.5 OLS Model C-Mincer (experience + experience^2) ---"
display "POZN: Jiny vzorek (missing exp dropped misto kategorie)"
regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    ib`nace_base'.sector_nace_num ///
    ib`region_base'.region_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib`sw_base'.job_family_num ///
    ib3.edu_ols ///
    experience_min_llm experience_sq ///
    if country == "US" & ln_salary != ., vce(cluster firm_cluster)
estimates store s13_ols_Cmincer
display _n "Model C-Mincer: R2 = " e(r2) ", Adj R2 = " e(r2_a) ", N = " e(N)

* ------------------------------------------------------------------
* 13.6 Srovnani OLS modelu
* ------------------------------------------------------------------
display _n "--- 13.6 Srovnani OLS Model A/B/C/C-nojf/C-Mincer (US) ---"
estimates table s13_ols_A s13_ols_B s13_ols_Cnojf s13_ols_C s13_ols_Cmincer, ///
    star stats(N r2 r2_a) b(%7.4f)

* ------------------------------------------------------------------
* 13.7 Wald F-testy: nested bloky (A->B, B->C)
* ------------------------------------------------------------------
* Testujeme, zda pridane bloky (lidsky kapital / tech skills + pozice)
* vyznamne zlepsuji model. Postup: plna specifikace s testparm pro
* jednotlive bloky, vce(cluster firm_cluster).
display _n "--- 13.7 Wald F-testy nested modelu (A -> B -> C) ---"
quietly regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    ib`nace_base'.sector_nace_num ///
    ib`region_base'.region_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib`sw_base'.job_family_num ///
    ib3.edu_ols ///
    ib3.exp_category ///
    if country == "US" & ln_salary != ., vce(cluster firm_cluster)

display _n "Wald F-test A -> B: spolecna signifikance lidskeho kapitalu (edu + exp)"
testparm ib3.edu_ols i.exp_category

display _n "Wald F-test B -> C: spolecna signifikance tech skills + pozice (cluster_* + job_family)"
testparm cluster_* i.job_family_num

* ------------------------------------------------------------------
* 13.8 OLS Model C-nocirc: sensitivity s GenAI + DS/ML
* ------------------------------------------------------------------
* Vratime cirkularni clustery zpet a odhadneme plny model pro srovnani
* dopadu cirkularity. Shodne s principem Prilohy C (ale zde jen US).
display _n "--- 13.8 OLS Model C-circ (vse vcetne GenAI + DS/ML, US) ---"
rename _excl_genai_s13  cluster_generative_ai
rename _excl_dsml_s13   cluster_data_science__ml

regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    ib`nace_base'.sector_nace_num ///
    ib`region_base'.region_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib`sw_base'.job_family_num ///
    ib3.edu_ols ///
    ib3.exp_category ///
    if country == "US" & ln_salary != ., vce(cluster firm_cluster)
estimates store s13_ols_Ccirc
display _n "Model C-circ (s GenAI + DS/ML): R2 = " e(r2) ", N = " e(N)

display _n "--- 13.8b Srovnani Model C (bez) vs C-circ (s cirkularnimi clustery) ---"
estimates table s13_ols_C s13_ols_Ccirc, star stats(N r2 r2_a) b(%7.4f)

* Ponechame cluster_* KOMPLETNI (s GenAI/DS-ML) pro M3-circ sensitivity.
* Pred logit M1/M2/M3 je zase prejmenujeme pryc.

* ------------------------------------------------------------------
* 13.9 LOGIT M1 / M2 / M3 — inkrementalni (US, bez GenAI/DS-ML)
* ------------------------------------------------------------------
rename cluster_generative_ai    _excl_genai_s13
rename cluster_data_science__ml _excl_dsml_s13

display _n "--- 13.9a Logit M1: Profil firmy (US) ---"
logit has_ai ///
    ib`nace_base'.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib`region_base'.region_num ///
    if country == "US", or vce(cluster firm_cluster)
estimates store s13_lg_M1
display _n "AME Logit M1 (US):"
margins, dydx(*) post
estimates store s13_lg_M1_ame

display _n "--- 13.9b Logit M2: Profil role (US) ---"
logit has_ai ///
    cluster_* ///
    ib`sw_base'.job_family_num ///
    ib2.edu_logit ///
    ib3.exp_category ///
    if country == "US", or vce(cluster firm_cluster)
estimates store s13_lg_M2
display _n "AME Logit M2 (US):"
margins, dydx(*) post
estimates store s13_lg_M2_ame

display _n "--- 13.9c Logit M3: Kompletni (US) ---"
logit has_ai ///
    ib`nace_base'.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib`region_base'.region_num ///
    cluster_* ///
    ib`sw_base'.job_family_num ///
    ib2.edu_logit ///
    ib3.exp_category ///
    if country == "US", or vce(cluster firm_cluster)
estimates store s13_lg_M3
display _n "AME Logit M3 (US):"
margins, dydx(*) post
estimates store s13_lg_M3_ame

* ------------------------------------------------------------------
* 13.10 Logit M3: Linktest (spec test)
* ------------------------------------------------------------------
display _n "--- 13.10 Logit M3 linktest (US) ---"
quietly logit has_ai ///
    ib`nace_base'.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib`region_base'.region_num ///
    cluster_* ///
    ib`sw_base'.job_family_num ///
    ib2.edu_logit ///
    ib3.exp_category ///
    if country == "US", vce(cluster firm_cluster)
linktest

* ------------------------------------------------------------------
* 13.11 Logit M3: ROC / AUC + classification
* ------------------------------------------------------------------
display _n "--- 13.11a Logit M3 lroc (AUC) / US ---"
quietly logit has_ai ///
    ib`nace_base'.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib`region_base'.region_num ///
    cluster_* ///
    ib`sw_base'.job_family_num ///
    ib2.edu_logit ///
    ib3.exp_category ///
    if country == "US"
lroc, nograph
display _n "--- 13.11b Logit M3 classification table (US) ---"
estat classification

display _n "--- 13.11c Logit M3 Hosmer-Lemeshow GOF (US) ---"
estat gof, group(10)

* ------------------------------------------------------------------
* 13.12 Logit M3 bez job_family (test mediace)
* ------------------------------------------------------------------
display _n "--- 13.12a Logit M3 bez job_family (US) ---"
logit has_ai ///
    ib`nace_base'.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib`region_base'.region_num ///
    cluster_* ///
    ib2.edu_logit ///
    ib3.exp_category ///
    if country == "US", or vce(cluster firm_cluster)
estimates store s13_lg_M3nojf
display _n "AME Logit M3-nojf (US):"
margins, dydx(*) post
estimates store s13_lg_M3nojf_ame

display _n "--- 13.12b Logit M3 bez job_family a seniority (US) ---"
logit has_ai ///
    ib`nace_base'.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib`region_base'.region_num ///
    cluster_* ///
    ib2.edu_logit ///
    if country == "US", or vce(cluster firm_cluster)
estimates store s13_lg_M3nojfnoexp
display _n "AME Logit M3-nojf-noexp (US):"
margins, dydx(*) post
estimates store s13_lg_M3nojfnoexp_ame

* ------------------------------------------------------------------
* 13.13 MLOGIT M1 / M2 / M3 — inkrementalni (US, bez GenAI/DS-ML)
* ------------------------------------------------------------------
* Mlogit (stejne jako Tabulka 4): edu vyrazeno kvuli sparse cells u
* HS/Associate × Applied/Core AI. Kontrola vzdelani je v binarnim logitu.
display _n "--- 13.13a Mlogit M1: Profil firmy (US) ---"
mlogit ai_level ///
    ib`nace_base'.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib`region_base'.region_num ///
    if country == "US", baseoutcome(0) rrr vce(cluster firm_cluster)
estimates store s13_ml_M1

display _n "--- 13.13b Mlogit M2: Profil role (US) ---"
mlogit ai_level ///
    cluster_* ///
    ib`sw_base'.job_family_num ///
    ib3.exp_category ///
    if country == "US", baseoutcome(0) rrr vce(cluster firm_cluster)
estimates store s13_ml_M2

display _n "--- 13.13c Mlogit M3: Kompletni (US) ---"
mlogit ai_level ///
    ib`nace_base'.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib`region_base'.region_num ///
    cluster_* ///
    ib`sw_base'.job_family_num ///
    ib3.exp_category ///
    if country == "US", baseoutcome(0) rrr vce(cluster firm_cluster)
estimates store s13_ml_M3

display _n "--- 13.13d Mlogit M3 AME: P(AI Integration) ---"
margins, dydx(*) predict(outcome(1))
display _n "--- 13.13e Mlogit M3 AME: P(Applied/Core AI) ---"
margins, dydx(*) predict(outcome(2))

* ------------------------------------------------------------------
* 13.14 Mlogit M3a / M3b — mediace (job_family, seniority)
* ------------------------------------------------------------------
display _n "--- 13.14a Mlogit M3a (bez job_family) ---"
mlogit ai_level ///
    ib`nace_base'.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib`region_base'.region_num ///
    cluster_* ///
    ib3.exp_category ///
    if country == "US", baseoutcome(0) rrr vce(cluster firm_cluster)
estimates store s13_ml_M3a
display _n "--- 13.14a1 Mlogit M3a AME: P(AI Integration) ---"
margins, dydx(*) predict(outcome(1))
display _n "--- 13.14a2 Mlogit M3a AME: P(Applied/Core AI) ---"
margins, dydx(*) predict(outcome(2))

display _n "--- 13.14b Mlogit M3b (bez job_family a seniority) ---"
mlogit ai_level ///
    ib`nace_base'.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib`region_base'.region_num ///
    cluster_* ///
    if country == "US", baseoutcome(0) rrr vce(cluster firm_cluster)
estimates store s13_ml_M3b
display _n "--- 13.14b1 Mlogit M3b AME: P(AI Integration) ---"
margins, dydx(*) predict(outcome(1))
display _n "--- 13.14b2 Mlogit M3b AME: P(Applied/Core AI) ---"
margins, dydx(*) predict(outcome(2))

* ------------------------------------------------------------------
* 13.15 Sensitivity: Logit M3-circ (s GenAI + DS/ML)
* ------------------------------------------------------------------
rename _excl_genai_s13  cluster_generative_ai
rename _excl_dsml_s13   cluster_data_science__ml

display _n "--- 13.15 Logit M3-circ (US, cluster_* vcetne GenAI + DS/ML) ---"
logit has_ai ///
    ib`nace_base'.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib`region_base'.region_num ///
    cluster_* ///
    ib`sw_base'.job_family_num ///
    ib2.edu_logit ///
    ib3.exp_category ///
    if country == "US", or vce(cluster firm_cluster)
estimates store s13_lg_M3circ
display _n "AME Logit M3-circ (US, ilustrace cirkularity):"
margins, dydx(*) post
estimates store s13_lg_M3circ_ame

* ------------------------------------------------------------------
* 13.16 Analyza skill_count: prumery per AI tier + t-test (US)
* ------------------------------------------------------------------
* Generovani skill_count z `hardskills` (carkami oddelene skills).
* Shodne se starym ai_skills_analysis.do (§3.16). capture confirm osetri
* pripad, kdy uz existuje z drivejsiho behu nebo kdy hardskills nejsou
* stringove (napr. po nejakem destring).
capture confirm variable skill_count
if _rc != 0 {
    capture confirm string variable hardskills
    if _rc == 0 {
        gen skill_count = 1 + length(hardskills) - length(subinstr(hardskills, ",", "", .))
        replace skill_count = 0 if hardskills == ""
        label variable skill_count "Pocet pozadovanych hard skills"
        display "skill_count vygenerovan z hardskills (string)."
    }
    else {
        display as error "hardskills neni stringova nebo neexistuje — skill_count preskocen."
        display as error "Overte import CSV v sekci 2 — hardskills musi byt textovy sloupec."
    }
}

* Pokud se skill_count nepodarilo vytvorit, cely blok 13.16 preskocime.
capture confirm variable skill_count
if _rc == 0 {
    display _n "--- 13.16a Prumery skill_count per ai_level (US, pro text §5.5.1) ---"
    tabstat skill_count if country == "US", by(ai_level) statistics(n mean sd min max)

    display _n "--- 13.16b T-test skill_count by has_ai (rovne variance, US) ---"
    ttest skill_count if country == "US", by(has_ai)

    display _n "--- 13.16c Welch t-test skill_count by has_ai (nerovne variance, US) ---"
    ttest skill_count if country == "US", by(has_ai) unequal
}
else {
    display as error "13.16 preskocen — skill_count neni dostupny."
}

* ------------------------------------------------------------------
* 13.17 Exporty: RTF tabulky pro inkrementalni modely
* ------------------------------------------------------------------
display _n "--- 13.17 Export inkrementalnich modelu do RTF ---"

* Tabulka 13a: OLS Model A / B / C-nojf / C / C-Mincer
esttab s13_ols_A s13_ols_B s13_ols_Cnojf s13_ols_C s13_ols_Cmincer ///
    using "$outdir/Tabulka_13a_OLS_incremental_US.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    stats(N r2 r2_a, fmt(0 3 3) labels("N" "R2" "Adj R2")) ///
    mtitles("Model A" "Model B" "C - bez JF" "Model C" "C - Mincer") ///
    title("Tabulka 13a: Inkrementalni OLS modely ln(plat) - US podvzorek") ///
    addnotes("SE klastrovane na firm_cluster. Reference: ai_level=0 (None), edu_ols=Bachelor, exp_category=Mid (3-5 let).")

* Tabulka 13b: Logit M1 / M2 / M3 / M3-nojf / M3-nojf-noexp / M3-circ (AME)
esttab s13_lg_M1_ame s13_lg_M2_ame s13_lg_M3_ame ///
       s13_lg_M3nojf_ame s13_lg_M3nojfnoexp_ame s13_lg_M3circ_ame ///
    using "$outdir/Tabulka_13b_Logit_incremental_US_AME.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    stats(N, fmt(0) labels("N")) ///
    mtitles("M1" "M2" "M3" "M3-nojf" "M3-nojf-noexp" "M3-circ") ///
    title("Tabulka 13b: Inkrementalni logit P(AI) AME - US podvzorek") ///
    addnotes("AME (dy/dx). SE klastrovane na firm_cluster. M3-circ = sensitivity s GenAI+DS/ML (cirkularni).")

* Tabulka 13c: Mlogit M1 / M2 / M3 / M3a / M3b (koeficienty, ne AME — AME
* per outcome by explodovalo tabulku na 10 sloupcu; pro text sekce 5.5.1–
* 5.5.3 staci koeficienty + LL)
esttab s13_ml_M1 s13_ml_M2 s13_ml_M3 s13_ml_M3a s13_ml_M3b ///
    using "$outdir/Tabulka_13c_Mlogit_incremental_US.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    stats(N ll chi2, fmt(0 3 3) labels("N" "Log-likelihood" "chi2")) ///
    mtitles("M1" "M2" "M3" "M3a (bez JF)" "M3b (bez JF+exp)") ///
    title("Tabulka 13c: Inkrementalni mlogit P(AI tier) - US podvzorek") ///
    addnotes("RRR koeficienty (baseoutcome=None). SE klastrovane na firm_cluster. Dve rovnice per model: AI Integration (vs None), Applied/Core AI (vs None).")

display _n "--- Sekce 13 dokoncena ---"
display "RTF tabulky: Tabulka_13a_OLS_incremental_US.rtf,"
display "             Tabulka_13b_Logit_incremental_US_AME.rtf,"
display "             Tabulka_13c_Mlogit_incremental_US.rtf"


* ==============================================================================
* 14. EXPORT PRO PYTHON GRAFY (CSV data pro analysis/charts/build_charts.py)
* ==============================================================================
* Tento blok NEMENI zadne odhady — pouze cte `e(b)`, `e(V)` z uz ulozenych
* estimates a zapisuje hodnoty do CSV, ktere pak nacte Python skript.
* Vystupy: 7 CSV souboru v $outdir/charts_data/
display _n "=============================================================="
display "14. EXPORT PRO PYTHON GRAFY"
display "=============================================================="

capture mkdir "$outdir/charts_data"

* --- Pomocny program: zapsat koeficienty + SE + p-value + 95% CI do CSV ---
capture program drop _export_est
program define _export_est
    args ename outfile keeppat
    capture estimates restore `ename'
    if _rc != 0 {
        display as error "Estimate `ename' nenalezen — preskakuji `outfile'."
        exit 0
    }
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colnames `b'
    local K = colsof(`b')
    tempname fh
    file open `fh' using "`outfile'", write replace
    file write `fh' "coef,b,se,z,p,ci_low,ci_high" _n
    forvalues i = 1/`K' {
        local nm : word `i' of `names'
        if "`keeppat'" != "" {
            if !strmatch("`nm'", "`keeppat'") continue
        }
        local bi = `b'[1, `i']
        local vi = `V'[`i', `i']
        if `vi' <= 0 | missing(`vi') continue
        local sei = sqrt(`vi')
        local zi  = `bi' / `sei'
        local pi  = 2 * (1 - normal(abs(`zi')))
        local lo  = `bi' - 1.96 * `sei'
        local hi  = `bi' + 1.96 * `sei'
        file write `fh' "`nm',`bi',`sei',`zi',`pi',`lo',`hi'" _n
    }
    file close `fh'
    display "Exported: `outfile'"
end

* ------------------------------------------------------------------
* GRAF 1 — AI tier × country (crosstab pocty a %)
* ------------------------------------------------------------------
preserve
    keep country ai_level
    contract country ai_level
    bysort country: egen _total = total(_freq)
    gen pct = 100 * _freq / _total
    export delimited country ai_level _freq pct ///
        using "$outdir/charts_data/g1_ai_tier_by_country.csv", replace
restore

* ------------------------------------------------------------------
* GRAF 2 — AI share podle job_family x country
* ------------------------------------------------------------------
preserve
    collapse (mean) ai_share=has_ai (count) n=has_ai, by(country job_family_num)
    replace ai_share = 100 * ai_share
    capture decode job_family_num, gen(job_family)
    if _rc != 0 gen job_family = string(job_family_num)
    export delimited country job_family_num job_family ai_share n ///
        using "$outdir/charts_data/g2_ai_share_by_jobfamily.csv", replace
restore

* ------------------------------------------------------------------
* GRAF 3 — binarni logit US AME skill clusteru
* ------------------------------------------------------------------
_export_est ame_t2sk_US "$outdir/charts_data/g3_logit_ame_us.csv" "cluster_*"

* ------------------------------------------------------------------
* GRAF 4 — mlogit US AME (Integration vs Applied/Core AI)
* ------------------------------------------------------------------
_export_est ame_t3_US_1 "$outdir/charts_data/g4_mlogit_us_integration.csv" "cluster_*"
_export_est ame_t3_US_2 "$outdir/charts_data/g4_mlogit_us_applied.csv"     "cluster_*"

* ------------------------------------------------------------------
* GRAF 5 — binarni logit AME skill clusteru napric zememi (pro heatmap)
* ------------------------------------------------------------------
foreach c in US DE IN {
    _export_est ame_t2sk_`c' "$outdir/charts_data/g5_logit_ame_`c'.csv" "cluster_*"
}

* ------------------------------------------------------------------
* GRAF 6 — OLS inkrementalni A / B / C (US) — ai_level koeficienty
* ------------------------------------------------------------------
_export_est s13_ols_A "$outdir/charts_data/g6_ols_A.csv" "*ai_level*"
_export_est s13_ols_B "$outdir/charts_data/g6_ols_B.csv" "*ai_level*"
_export_est s13_ols_C "$outdir/charts_data/g6_ols_C.csv" "*ai_level*"

* ------------------------------------------------------------------
* GRAF 7 — OLS cross-country (US / DE / IN) — ai_level koeficienty
* ------------------------------------------------------------------
foreach c in US DE IN {
    _export_est ols_t4_`c' "$outdir/charts_data/g7_ols_`c'.csv" "*ai_level*"
}

display _n "Export CSV dokoncen: $outdir/charts_data/"


* ==============================================================================
* 15. ZAVER
* ==============================================================================
display _n "=============================================================="
display "FINALNI TEZISTOVA ANALYZA DOKONCENA"
display "=============================================================="
display "Vystupy ulozeny do: $outdir"
display "Log: $outdir/ai_skills_thesis_final.log"

save "$outdir/ai_skills_thesis_final_processed.dta", replace

log close
