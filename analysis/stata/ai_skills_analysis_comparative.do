********************************************************************************
* AI SKILLS IN IT JOB POSTINGS - COMPARATIVE STATA ANALYSIS (US VS DE VS IN)
* ==============================================================================
* Datasets: us_relevant_ai_stata.csv, de_relevant_ai_stata.csv, in_relevant_ai_stata.csv
* Autor: [Yakub Murcek]
* Datum: Leden 2026/Duben 2026
* 
* Tento do-file obsahuje POOLED (sdilenou) analyzu datasetu IT pracovnich inzeratu
* pro USA, Nemecko a Indii, s cilem zkoumat vliv zeme na AI mzdove premie.
*
* TESTOVANO PRO: STATA 15.1 (IC/SE verze)
********************************************************************************

* ==============================================================================
* 1. NASTAVENÍ PROSTŘEDÍ
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
global outdir "./output/comparative_run_`time_string'"

capture mkdir "./output"
capture mkdir "$outdir"

capture log close
log using "$outdir/ai_skills_comparative_analysis.log", replace text


* ==============================================================================
* 2. IMPORT DAT A APPENDING
* ==============================================================================

* 2.1 US Dataset
import delimited "$datadir/us_relevant_ai_stata.csv", delimiter(";") clear varnames(1) encoding(utf8)
capture drop country
gen country = "US"
tempfile us_data
save `us_data'

* 2.2 DE Dataset
import delimited "$datadir/de/de_relevant_ai_stata.csv", delimiter(";") clear varnames(1) encoding(utf8)
capture drop country
gen country = "DE"
tempfile de_data
save `de_data'

* 2.3 IN Dataset
import delimited "$datadir/in_relevant_ai_stata.csv", delimiter(";") clear varnames(1) encoding(utf8)
capture drop country
gen country = "IN"
tempfile in_data
save `in_data'

* 2.4 Merge do jednoho analyzovaneho poolu
use `us_data', clear
append using `de_data', force
append using `in_data', force

display "Pocet pozorovani celkem po spojeni (US+DE+IN): " _N
tab country

* Vytvoreni country_id faktorove promenne pro regrese
encode country, generate(country_id)
label variable country_id "Zeme povodu inzeratu (US/DE/IN)"


* ==============================================================================
* 3. ČIŠTĚNÍ A PŘÍPRAVA DAT (Identicke jako v base modelu)
* ==============================================================================
* Ponecháme pouze pozice s vysokou confidencí modelu
keep if desc_conf_llm >= 0.7

* Zbaveni se starych if existuji
gen date_format_discover = date(substr(discover_date, 1, 10), "YMD")
format date_format_discover %td
destring age_in_days, replace force
gen real_post_date = date_format_discover - age_in_days
format real_post_date %td
gen post_year = year(real_post_date)
drop if post_year <= 2023

* Zpracovani AI Tier a AI Flag
replace desc_tier_llm = "missing" if desc_tier_llm == ""
replace desc_tier_llm = "applied_ai" if desc_tier_llm == "core_ai"
encode desc_tier_llm, generate(ai_tier_num)

gen skills_combined = lower(skills_ai_det + " " + desc_ai_llm)
gen skills_no_buzz = ustrregexra(skills_combined, "(?i)\b(ai|ml|artificial intelligence|machine learning|genai)\b", "")
gen skills_cleaned = subinstr(skills_no_buzz, ",", "", .)
replace skills_cleaned = subinstr(skills_cleaned, " ", "", .)
replace skills_cleaned = subinstr(skills_cleaned, ";", "", .)

gen has_ai = ((desc_tier_llm != "none" & desc_tier_llm != "missing") & length(skills_cleaned) > 1)
label variable has_ai "AI Job (Tier + Valid Skills, no buzzwords)"

* Vzdelani
gen education_hybrid = lower(edulevel_llm)
replace education_hybrid = subinstr(education_hybrid, "'s", "", .)
replace education_hybrid = "highschool" if education_hybrid == "high school"
replace education_hybrid = "missing" if education_hybrid == "-" | education_hybrid == ""
replace education_hybrid = edu_level_det if education_hybrid == "missing" & edu_level_det != ""
replace education_hybrid = "missing" if education_hybrid == ""
replace education_hybrid = "master" if education_hybrid == "phd"

gen edu_ols = .
replace edu_ols = 0 if education_hybrid == "missing"
replace edu_ols = 1 if education_hybrid == "highschool"
replace edu_ols = 2 if education_hybrid == "associate"
replace edu_ols = 3 if education_hybrid == "bachelor"
replace edu_ols = 4 if education_hybrid == "master"
label define edu_ols_lbl 0 "Missing" 1 "High School" 2 "Associate" 3 "Bachelor" 4 "Master+"
label values edu_ols edu_ols_lbl

gen edu_logit = .
replace edu_logit = 0 if inlist(education_hybrid, "missing", "", "highschool", "associate")
replace edu_logit = 1 if inlist(education_hybrid, "bachelor", "master")

* Zkusenosti
destring experience_min_llm, replace force
gen exp_category = .
replace exp_category = 0 if experience_min_llm == .
replace exp_category = 2 if experience_min_llm >= 0 & experience_min_llm <= 2
replace exp_category = 3 if experience_min_llm > 2 & experience_min_llm <= 5
replace exp_category = 4 if experience_min_llm > 5 & experience_min_llm < .

* Plat
destring salary_min salary_mid salary_max, replace force

* --- Konverze men na USD (scraping period: Sep-Oct 2025) ---
* Pevne kurzy: prumerny ECB/RBI kurz za obdobi scrapingu
display _n "--- Konverze men na USD ---"
local eur_usd = 1.165
local inr_usd = 88

* Drop rows with non-standard or missing currencies
* DE: 1 COP + 1 USD row (no salary data anyway)
* IN: 2 rows with empty currency but non-empty salary (unknown currency)
drop if country == "DE" & !inlist(pay_currency, "EUR", "")
drop if country == "IN" & pay_currency == "" & salary_mid != .

* Konverze EUR na USD
foreach var of varlist salary_min salary_mid salary_max {
    replace `var' = `var' * `eur_usd' if pay_currency == "EUR"
}

* Konverze INR na USD
foreach var of varlist salary_min salary_mid salary_max {
    replace `var' = `var' / `inr_usd' if pay_currency == "INR"
}

display "Vsechny platy nyni v USD"

* Prepocet hodinovych mezd na rocni (country-specific hours)
* US: 2080 h/rok (40h x 52w), DE: 1607 h/rok (OECD 2024), IN: 1920 h/rok (48h x 40w)
replace salary_min = salary_min * 2080 if pay_period == "HOURLY" & country == "US"
replace salary_mid = salary_mid * 2080 if pay_period == "HOURLY" & country == "US"
replace salary_max = salary_max * 2080 if pay_period == "HOURLY" & country == "US"

replace salary_min = salary_min * 1607 if pay_period == "HOURLY" & country == "DE"
replace salary_mid = salary_mid * 1607 if pay_period == "HOURLY" & country == "DE"
replace salary_max = salary_max * 1607 if pay_period == "HOURLY" & country == "DE"

replace salary_min = salary_min * 1920 if pay_period == "HOURLY" & country == "IN"
replace salary_mid = salary_mid * 1920 if pay_period == "HOURLY" & country == "IN"
replace salary_max = salary_max * 1920 if pay_period == "HOURLY" & country == "IN"

* Prepocet mesicnich mezd na rocni (12 mesicu — univerzalni)
replace salary_min = salary_min * 12 if pay_period == "MONTHLY"
replace salary_mid = salary_mid * 12 if pay_period == "MONTHLY"
replace salary_max = salary_max * 12 if pay_period == "MONTHLY"

* Outliery (vsechny platy nyni v USD po konverzi)
* Dolni mez $3K: zachyti near-zero / chyby (indicke IT median ~$6.8K)
* Horni mez $500K: stejna jako US-only soubor
replace salary_mid = . if salary_mid < 3000 | salary_mid > 500000
gen ln_salary = ln(salary_mid)

* Ostatní categorical covariates
gen type_cat = .
replace type_cat = 0 if inlist(type, "", "Unknown", "Contract", "Self-employed", "Private Practice / Firm", "Franchise") 
replace type_cat = 1 if inlist(type, "Company - Private", "Subsidiary or Business Segment")
replace type_cat = 2 if type == "Company - Public"
replace type_cat = 3 if inlist(type, "Nonprofit Organization", "Government", "College / University", "School / School District", "Hospital")

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

* Audit: check missing coverage for categorical vars
display _n "--- Audit: missing obs in type_cat and size_cat ---"
tab type_cat, missing
tab size_cat, missing

replace sector_nace = "Unknown" if sector_nace == ""
replace sector_nace = "Other" if !inlist(sector_nace, "J", "C", "K", "M", "Q", "Unknown")
encode sector_nace, generate(sector_nace_num)

gen is_remote = 0
replace is_remote = 1 if strpos(lower(remote_work_types), "home") > 0 | strpos(lower(remote_work_types), "remote") > 0

gen ai_level = 0
replace ai_level = 1 if desc_tier_llm == "ai_integration"
replace ai_level = 2 if desc_tier_llm == "applied_ai"

replace job_family = "Unknown" if job_family == ""
replace job_family = "Other" if inlist(job_family, "Frontend & Design", "QA & Testing", "Security", "Systems & Embedded")
encode job_family, generate(job_family_num)

* Pro sjednoceni lokaci mezi zahranicnimi (nemaji US regiony)
gen location_market = ""
replace location_market = "US_" + region if country == "US"
replace location_market = country if country != "US"
encode location_market, generate(location_market_num)

drop cluster_legacy__mainframe
drop cluster_data_analysis__stats
drop cluster_tools__editors


* ==============================================================================
* 4. CROSS-COUNTRY DESKRIPTIVNI STATISTIKA
* ==============================================================================
display _n "=============================================================="
display "4. CROSS-COUNTRY DESKRIPTIVNI STATISTIKA"
display "=============================================================="

display _n "--- 4.1 Distribuce podilu pozic s AI mezi roznymi zememi ---"
tab country has_ai, row chi2

display _n "--- 4.2 Prumerny a medialni hruby rocni plat (v USD) podle zeme ---"
tabstat salary_mid, by(country) stat(count mean median p25 p75)

display _n "--- 4.3 Chybejici mzdova data podle zeme (% platu reportovanych) ---"
gen has_salary = (salary_mid != .)
tab country has_salary, row chi2

display _n "--- 4.4 Prumerny vzdelavaci narok podle zemes ---"
tab country edu_logit, row chi2


* ==============================================================================
* 5. CROSS-COUNTRY REGRESNI ANALYZA
* ==============================================================================
display _n "=============================================================="
display "5. CROSS-COUNTRY OLS WAGE REGRESSION"
display "=============================================================="
* Tato analyza kombinuje data ze tri statu. Pro to abychom vyrovnali ekonomicke
* rozdily a menove kurzy mezi zememi, zahrnujeme dummy promennou pro zemi
* i.country_id, ktera tento fixni rozdil pohlti do zkraceni (intercept).

* --- 5.1 Baseline "Slouceny" Model B (s fixed effects na zemi) ---
display _n "--- 5.1 Baseline Slouceny Model B (s country FEs) ---"
regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    i.country_id ///
    i.location_market_num ///
    i.sector_nace_num ///
    is_remote ///
    i.type_cat i.size_cat ///
    i.job_family_num ///
    i.edu_ols ib3.exp_category ///
    if ln_salary != ., vce(robust)
estimates store pooled_model_b


* --- 5.2 INTERAKCNI MODEL: AI mzdova premie podla zeme ---
* KLICOVA ANALYZA PRO DIPLOMKU: Pta se, zdali "wage premium za znalost AI"
* se statisticky signifikantne lisi podle toho, jestli pracujete v US, DE, nebo IN.
* Pouzivame GRADOVANY ai_level (0=zadny AI, 1=AI integrace, 2=Applied/Core AI),
* ktery je informativnejsi nez binarni has_ai a prokazal signifikanci v Modelu B.

display _n "--- 5.2 Interakcni Model: Wage AI Premium (ai_level) lomeno zemi ---"
display "Base level pro country je definovan Stata (abecedne = DE)."
display "Base level pro ai_level = 0 (zadna AI). Interakce testuje, zda se"
display "gradovana AI premie statisticky lisi mezi DE, IN a US."

regress ln_salary ///
    cluster_* ///
    i.country_id##i.ai_level ///
    i.location_market_num ///
    i.sector_nace_num ///
    is_remote ///
    i.type_cat i.size_cat ///
    i.job_family_num ///
    i.edu_ols ib3.exp_category ///
    if ln_salary != ., vce(robust)
estimates store interaction_model
* Test: jsou interakcni cleny country x ai_level spolecne signifikantni?
testparm i.country_id#i.ai_level


* ==============================================================================
* 6. CROSS-COUNTRY LOGISTICKE MODELY (PRAVDEPODOBNOST AI)
* ==============================================================================
display _n "=============================================================="
display "6. CROSS-COUNTRY LOGISTICKE MODELY"
display "=============================================================="

display _n "--- 6.1 Mlogit: Pravdepodobnost AI urovne x Zeme ---"
mlogit ai_level ///
    i.country_id ///
    i.sector_nace_num ///
    i.type_cat i.size_cat ///
    cluster_* ///
    i.job_family_num ///
    ib1.edu_logit ib3.exp_category, baseoutcome(0) rrr
estimates store pooled_mlogit
display _n "--- 6.2 Marginalni efekty Mlogit: P(AI Integration) ---"
margins, dydx(*) predict(outcome(1))
display _n "--- 6.3 Marginalni efekty Mlogit: P(Applied/Core AI) ---"
margins, dydx(*) predict(outcome(2))


* ==============================================================================
* ZÁVĚR
* ==============================================================================
display _n "=============================================================="
display "CROSS-COUNTRY ANALYZA DOKONCENA"
display "=============================================================="
estimates table pooled_model_b interaction_model, star stats(N r2 r2_a)
display _n "--- Mlogit vysledky ---"
estimates table pooled_mlogit, star stats(N ll chi2)
log close
