********************************************************************************
* AI SKILLS IN IT JOB POSTINGS — COMPARATIVE STATA ANALYSIS (US vs DE vs IN)
* ==============================================================================
* Datasety: us_relevant_ai_stata.csv, de/de_relevant_ai_stata.csv, in_relevant_ai_stata.csv
* Autor: Yakub Murcek
* Datum: Duben 2026
* Stata 15.1 (IC/SE)
*
* Tento skript je paralelní verze hlavního ai_skills_analysis.do pro sdružené
* (pooled) vzorky tří zemí. Úprava oproti hlavní analýze:
*   - sdruceny vzorek US + DE + IN (country_id fixed effects, baseline = US via ib3)
*   - platy konvertovane na USD pevnymi kurzy za obdobi scrapingu (Sep–Oct 2025)
*   - geograficka kontrola: v pooled OLS pouzivame country FE (i.country_id).
*     Pomocna promenna location_market (US_<region> pro US, country code pro DE/IN)
*     je vytvorena pouze pro deskriptivu — v regresich by byla perfektne kolinearni
*     s country_id, proto ji nepouzivame jako regresor. Region FE zustava jen
*     v US-only specifikaci (§6.6).
*   - hlavni zajem je test homogenity AI premie napric zememi (country × ai_level)
*     a kvantifikace rozdilu v pravdepodobnosti AI pozice napric zememi (mlogit)
*
* Struktura logicky zrcadli hlavni do-file:
*   1. Nastaveni prostredi
*   2. Import a append trech datasetu
*   3. Cisteni a priprava dat (vc. meny, lokalni trh, clustery)
*   4. Cross-country deskriptivni statistika
*   5. Cross-country statisticke testy (AI premie, missingness, vzdelani)
*   6. OLS wage modely: baseline FE model + plny Model C + interakce country×AI
*   6A. Binarni logit: P(has_ai) s country FE
*   6B. Multinomialni logit: P(ai_level) s country FE + bez JF / bez JF+exp
*   7. Robustnostni kontroly (clustered SE, bez cirkularnich clusteru)
*   8. Exporty tabulek a grafu
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
global outdir "./output/comparative_run_`time_string'"

capture mkdir "./output"
capture mkdir "$outdir"

capture log close
log using "$outdir/ai_skills_comparative_analysis.log", replace text


* ==============================================================================
* 2. IMPORT DAT A APPEND
* ==============================================================================
display _n "=============================================================="
display "2. IMPORT A APPEND DATASETU (US + DE + IN)"
display "=============================================================="

* 2.1 US
import delimited "$datadir/us_relevant_ai_stata.csv", delimiter(";") clear varnames(1) encoding(utf8)
capture drop country
gen country = "US"
display "US import: " _N " pozorovani"
tempfile us_data
save `us_data'

* 2.2 DE
import delimited "$datadir/de/de_relevant_ai_stata.csv", delimiter(";") clear varnames(1) encoding(utf8)
capture drop country
gen country = "DE"
display "DE import: " _N " pozorovani"
tempfile de_data
save `de_data'

* 2.3 IN
import delimited "$datadir/in_relevant_ai_stata.csv", delimiter(";") clear varnames(1) encoding(utf8)
capture drop country
gen country = "IN"
display "IN import: " _N " pozorovani"
tempfile in_data
save `in_data'

* 2.4 Append do sdruceneho datasetu
use `us_data', clear
append using `de_data', force
append using `in_data', force

display _n "Pocet pozorovani celkem po spojeni (US+DE+IN): " _N
tab country, missing

* Country jako faktorova promenna (encode abecedne: DE=1, IN=2, US=3)
encode country, generate(country_id)
label variable country_id "Zeme povodu inzeratu (DE=1, IN=2, US=3)"
* POZN: V regresich pouzivame ib3.country_id (baseline = US), protoze
* USA jsou hlavni referencni trh pro komparativni kapitolu. Koeficienty
* 1.country_id (DE) a 2.country_id (IN) tedy vyjadruji odchylku od US.


* ==============================================================================
* 3. CISTENI A PRIPRAVA DAT
* ==============================================================================
* Nasledujici kroky odpovidaji cisteni v ai_skills_analysis.do; komentare jsou
* zkracene, detailni vysvetleni viz hlavni do-file.

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
encode desc_tier_llm, generate(ai_tier_num)

* --- 3.2 has_ai + ai_level ---
gen has_ai = (desc_tier_llm != "none" & desc_tier_llm != "missing")
label variable has_ai "AI Job (desc_tier_llm in {ai_integration, applied_ai})"

gen ai_level = 0
replace ai_level = 1 if desc_tier_llm == "ai_integration"
replace ai_level = 2 if desc_tier_llm == "applied_ai"
label define ailevel_lbl 0 "None" 1 "AI Integration" 2 "Applied/Core AI"
label values ai_level ailevel_lbl
label variable ai_level "Uroven AI pozadavku (0/1/2)"

* --- 3.3 Vzdelani (hybridni) ---
gen education_hybrid = lower(edulevel_llm)
replace education_hybrid = subinstr(education_hybrid, "'s", "", .)
replace education_hybrid = "highschool" if education_hybrid == "high school"
replace education_hybrid = "missing" if education_hybrid == "-" | education_hybrid == ""
replace education_hybrid = edu_level_det if education_hybrid == "missing" & edu_level_det != ""
replace education_hybrid = "missing" if education_hybrid == ""
replace education_hybrid = "master" if education_hybrid == "phd"

* edu_ols: granularni pro OLS
gen edu_ols = .
replace edu_ols = 0 if education_hybrid == "missing"
replace edu_ols = 1 if education_hybrid == "highschool"
replace edu_ols = 2 if education_hybrid == "associate"
replace edu_ols = 3 if education_hybrid == "bachelor"
replace edu_ols = 4 if education_hybrid == "master"
label define edu_ols_lbl 0 "Missing" 1 "High School" 2 "Associate" 3 "Bachelor" 4 "Master+"
label values edu_ols edu_ols_lbl
label variable edu_ols "Vzdelani (granularni pro OLS)"

* edu_logit: 3-urovnova pro binarni logit
gen edu_logit = .
replace edu_logit = 0 if inlist(education_hybrid, "missing", "")
replace edu_logit = 1 if inlist(education_hybrid, "highschool", "associate")
replace edu_logit = 2 if inlist(education_hybrid, "bachelor", "master")
label define edu_logit_lbl 0 "Missing" 1 "HS / Associate" 2 "Bachelor or Higher"
label values edu_logit edu_logit_lbl
label variable edu_logit "Vzdelani (3 urovne pro binarni logit)"

* Deskriptivni 3-urovnova kategorie (nezavisla na edu_logit)
gen edu_cat = edu_logit
label values edu_cat edu_logit_lbl
label variable edu_cat "Pozadovane vzdelani (3 urovne, deskriptivni)"

* --- 3.4 Zkusenosti ---
destring experience_min_llm, replace force
label variable experience_min_llm "Min. pozadovane roky zkusenosti"

gen exp_category = .
replace exp_category = 0 if experience_min_llm == .
replace exp_category = 2 if experience_min_llm >= 0 & experience_min_llm <= 2
replace exp_category = 3 if experience_min_llm > 2 & experience_min_llm <= 5
replace exp_category = 4 if experience_min_llm > 5 & experience_min_llm < .
label define exp_lbl 0 "Missing" 2 "Junior (0-2)" 3 "Mid (3-5)" 4 "Senior+ (6+)"
label values exp_category exp_lbl
label variable exp_category "Kategorie seniority"

gen experience_sq = experience_min_llm^2
label variable experience_sq "Zkusenosti na druhou (Mincer)"

* --- 3.5 Plat — prevod na USD a rocni bazi ---
destring salary_min salary_mid salary_max, replace force

display _n "--- 3.5 Konverze men na USD (Sep-Oct 2025 prumer) ---"
* Pevne kurzy: prumer za obdobi scrapingu (ECB / RBI).
local eur_usd = 1.165
local inr_usd = 88

* Vyrazeni radku s nestandardnimi / chybejicimi menami
*   DE: COP + USD radky s null mzdou
*   IN: 2 radky s prazdnou menou, ale nenulovou mzdou (neznama mena)
drop if country == "DE" & !inlist(pay_currency, "EUR", "")
drop if country == "IN" & pay_currency == "" & salary_mid != .

foreach var of varlist salary_min salary_mid salary_max {
    replace `var' = `var' * `eur_usd' if pay_currency == "EUR"
    replace `var' = `var' / `inr_usd' if pay_currency == "INR"
}
display "Vsechny platy nyni v USD."

* Prepocet hodinovych mezd na rocni (country-specific)
* US: 2080 h/rok (40h x 52w), DE: 1607 h (OECD 2024), IN: 1920 h (48h x 40w)
foreach var of varlist salary_min salary_mid salary_max {
    replace `var' = `var' * 2080 if pay_period == "HOURLY" & country == "US"
    replace `var' = `var' * 1607 if pay_period == "HOURLY" & country == "DE"
    replace `var' = `var' * 1920 if pay_period == "HOURLY" & country == "IN"
    replace `var' = `var' * 12   if pay_period == "MONTHLY"
}

* Outliers (vsechny platy uz v USD)
* Dolni mez 3000 USD chyti near-zero / chyby (median IT v Indii ~ 6.8k USD)
* Horni mez 500000 USD stejna jako v US-only analyze
replace salary_mid = . if salary_mid < 3000 | salary_mid > 500000
label variable salary_mid "Rocni plat - stredni hodnota (USD)"

gen ln_salary = ln(salary_mid)
label variable ln_salary "Prirozeny logaritmus platu (USD)"

gen has_salary = (salary_mid != .)
label variable has_salary "Ma uvedeny plat (1=ano)"

* --- 3.6 Velikost firmy (ordinalni) ---
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

* --- 3.7 Typ firmy ---
gen type_cat = .
replace type_cat = 0 if inlist(type, "", "Unknown", "Contract", "Self-employed", "Private Practice / Firm", "Franchise")
replace type_cat = 1 if inlist(type, "Company - Private", "Subsidiary or Business Segment")
replace type_cat = 2 if type == "Company - Public"
replace type_cat = 4 if inlist(type, "Nonprofit Organization", "Government", ///
    "College / University", "School / School District", "Hospital")
label define type_lbl 0 "Unknown/Other" 1 "Private/Subsidiary" 2 "Public" ///
    4 "Nonprofit/Gov/Edu"
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

* --- 3.11 Lokalni trh (pomocna promenna, NEPOUZIVANA v regresich) ---
* Region je US-specific; pro DE/IN pouzivame country code.
* POZN: Tato promenna je perfektne kolinearni s country_id (DE/IN maji 1 uroven,
* US ma 4 regiony jejichz soucet = country_id=US). Proto ji v regresich
* NEPOUZIVAME — pouzivame pouze i.country_id jako unifikovany FE. Ponechavame
* ji pro deskriptivni ucely (pripadne robustnostni spec pouze na US subvzorku).
gen location_market = ""
replace location_market = "US_" + region if country == "US"
replace location_market = country if country != "US"
replace location_market = "Unknown" if location_market == "" | location_market == "US_"
encode location_market, generate(location_market_num)
label variable location_market_num "Lokalni trh (US region + country, deskriptivne)"

* Zachovani puvodniho region kodu pro US-only robustnostni spec
replace region = "Unknown" if region == ""
encode region, generate(region_num)
label variable region_num "Census region (US-specific)"

* --- 3.12 Cistka cirkularnich / prilis ridkych clusteru ---
* Stejne jako v hlavni analyze: tyto clustery maji < 50 obs v Applied AI
* nebo jsou artefaktem datasetu — odstranujeme, aby je nevzal cluster_*.
capture drop cluster_legacy__mainframe
capture drop cluster_data_analysis__stats
capture drop cluster_tools__editors

* --- 3.13 Skill count ---
gen skill_count = 1 + length(hardskills) - length(subinstr(hardskills, ",", "", .))
replace skill_count = 0 if hardskills == ""
label variable skill_count "Pocet pozadovanych hard skills"

* --- 3.14 Company ID (pro cluster SE) ---
destring company_id, replace force
label variable company_id "ID firmy (pro cluster SE)"


* ==============================================================================
* 4. CROSS-COUNTRY DESKRIPTIVNI STATISTIKA
* ==============================================================================
display _n "=============================================================="
display "4. CROSS-COUNTRY DESKRIPTIVNI STATISTIKA"
display "=============================================================="

* --- 4.1 Zakladni sample sizes ---
display _n "--- 4.1 Sample sizes podle zeme ---"
tab country, missing

* --- 4.2 Podil AI pozic podle zeme ---
display _n "--- 4.2 Podil AI pozic podle zeme ---"
tab country has_ai, row chi2
display _n "--- 4.2b AI tier distribuce podle zeme (sloupcova %) ---"
tab ai_level country, column chi2

* --- 4.3 Mzdy podle zeme ---
display _n "--- 4.3 Hrubý rocni plat (USD) podle zeme ---"
tabstat salary_mid, by(country) stat(count mean sd min p25 p50 p75 max)

* --- 4.4 Pokryti mzdovych dat podle zeme ---
display _n "--- 4.4 Pokryti mzdovych dat (%) podle zeme ---"
tab country has_salary, row chi2

* --- 4.5 AI premia v hrubych cislech (deskriptivni, bez kontrol) ---
display _n "--- 4.5 Plat (USD) podle zeme x AI tier ---"
foreach c in US DE IN {
    display _n "Country = `c'"
    tabstat salary_mid if country == "`c'", by(ai_level) stat(count mean p50 sd)
}

* --- 4.6 Vzdelani podle zeme ---
display _n "--- 4.6 Pozadovane vzdelani (3 urovne) podle zeme ---"
tab country edu_cat, row chi2
display _n "Granularni vzdelani podle zeme:"
tab country edu_ols, row

* --- 4.7 Seniorita podle zeme ---
display _n "--- 4.7 Seniorita podle zeme ---"
tab country exp_category, row chi2

* --- 4.8 Remote podle zeme ---
display _n "--- 4.8 Remote podle zeme ---"
tab country is_remote, row chi2

* --- 4.9 Sektorova struktura podle zeme ---
display _n "--- 4.9 NACE sektor podle zeme ---"
tab country sector_nace, row

* --- 4.10 Job family podle zeme ---
display _n "--- 4.10 Job family podle zeme ---"
tab country job_family, row

* --- 4.11 Skill clustery: prumerna frekvence podle zeme ---
display _n "--- 4.11 Prumerna frekvence skill clusteru podle zeme ---"
tabstat cluster_*, by(country) stat(mean) columns(statistics)

* --- 4.12 Missingness platu: ne-MCAR diagnostika na sdruceneho vzorku ---
display _n "--- 4.12 Missingness platu x observables ---"
display _n "has_salary x country:"
tab has_salary country, chi2 column
display _n "has_salary x AI level:"
tab has_salary ai_level, chi2 column

display _n "--- 4.12b Logit missingness has_salary ~ observables + country FE ---"
logit has_salary ///
    i.ai_level ///
    ib3.country_id ///
    i.sector_nace_num ///
    ib5.size_cat ///
    ib1.type_cat ///
    is_remote ///
    i.job_family_num ///
    ib2.edu_logit ///
    ib3.exp_category, vce(robust)
display _n "Wald test spolecne signifikance vsech observables krome ai_level:"
testparm ib3.country_id i.sector_nace_num ib5.size_cat ///
    ib1.type_cat is_remote i.job_family_num i.edu_logit i.exp_category


* ==============================================================================
* 5. CROSS-COUNTRY STATISTICKE TESTY
* ==============================================================================
display _n "=============================================================="
display "5. STATISTICKE TESTY — HRUBE ROZDILY MEZI ZEMEMI"
display "=============================================================="

* --- 5.1 Chi-square: AI vs country ---
display _n "--- 5.1 Chi-square: country x has_ai ---"
tab country has_ai, chi2 expected

* --- 5.2 ANOVA: Log(salary) podle zeme ---
display _n "--- 5.2 ANOVA: ln(salary) podle zeme ---"
oneway ln_salary country_id, tabulate bonferroni
display _n "--- 5.2b Kruskal-Wallis (neparametricka kontrola) ---"
kwallis ln_salary, by(country_id)

* --- 5.3 ANOVA: ln(salary) podle AI tier v kazde zemi zvlast ---
display _n "--- 5.3 ANOVA ln(salary) podle AI tier — podle zeme ---"
foreach c in US DE IN {
    display _n "Country = `c'"
    oneway ln_salary ai_tier_num if country == "`c'", tabulate bonferroni
}

* --- 5.4 Chi-square: vzdelani x country ---
display _n "--- 5.4 Chi-square: country x edu_cat ---"
tab country edu_cat, chi2 expected


* ==============================================================================
* 6. CROSS-COUNTRY OLS WAGE REGRESSION
* ==============================================================================
* Jadro komparativni analyzy: testujeme, zda AI premie se lisi mezi USA, DE a IN.
* Baseline country = US (ib3.country_id). Koef. 1.country_id = DE vs US, 2.country_id = IN vs US.
display _n "=============================================================="
display "6. CROSS-COUNTRY OLS WAGE REGRESSION"
display "=============================================================="

* --- 6.1 Model FE-A: Zakladni pooled FE (Firemni profil + country) ---
display _n "--- 6.1 Model FE-A: pooled OLS s country FE (firemni profil) ---"
regress ln_salary ///
    i.ai_level ///
    ib3.country_id ///
    i.sector_nace_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    if ln_salary != ., vce(robust)
estimates store pooled_fe_a
display _n "Model FE-A: R2 = " e(r2) ", Adj R2 = " e(r2_a) ", N = " e(N)

* --- 6.2 Model FE-B: + lidsky kapital ---
display _n "--- 6.2 Model FE-B: FE-A + vzdelani + zkusenosti ---"
regress ln_salary ///
    i.ai_level ///
    ib3.country_id ///
    i.sector_nace_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    ib3.edu_ols ///
    ib3.exp_category ///
    if ln_salary != ., vce(robust)
estimates store pooled_fe_b
display _n "Model FE-B: R2 = " e(r2) ", Adj R2 = " e(r2_a) ", N = " e(N)

* --- 6.3 Model FE-C: plny model s tech skills + job_family ---
display _n "--- 6.3 Model FE-C: FE-B + cluster_* + job_family ---"
regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    ib3.country_id ///
    i.sector_nace_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    i.job_family_num ///
    ib3.edu_ols ///
    ib3.exp_category ///
    if ln_salary != ., vce(robust)
estimates store pooled_fe_c
display _n "Model FE-C: R2 = " e(r2) ", Adj R2 = " e(r2_a) ", N = " e(N)

* --- 6.3b VIF diagnostika Model FE-C ---
display _n "--- 6.3b VIF diagnostika Model FE-C ---"
quietly regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    ib3.country_id ///
    i.sector_nace_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    i.job_family_num ///
    ib3.edu_ols ///
    ib3.exp_category ///
    if ln_salary != .
vif

* --- 6.4 Porovnani pooled modelu A / B / C ---
display _n "--- 6.4 Porovnani pooled modelu FE-A / FE-B / FE-C ---"
estimates table pooled_fe_a pooled_fe_b pooled_fe_c, star stats(N r2 r2_a)

* --- 6.5 Interakcni model: country x ai_level ---
* Klicovy test homogenity AI premie napric zememi.
* Baseline: country = US (ib3), ai_level = 0 (None).
display _n "--- 6.5 Interakcni model: country x ai_level (FE-C specifikace) ---"
regress ln_salary ///
    cluster_* ///
    ib3.country_id##i.ai_level ///
    i.sector_nace_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    i.job_family_num ///
    ib3.edu_ols ///
    ib3.exp_category ///
    if ln_salary != ., vce(robust)
estimates store pooled_interaction
display _n "--- 6.5b Wald F-test homogenity AI premie (country x ai_level) ---"
testparm ib3.country_id#i.ai_level

* --- 6.5c Marginalni efekty ai_level podle zeme ---
display _n "--- 6.5c Marginalni efekty ai_level podle zeme ---"
margins country_id, dydx(ai_level)

* --- 6.6 Per-country OLS (Model C specifikace zvlast na kazdem vzorku) ---
* V US specifikaci pridavame i.region_num (4 regiony). V DE/IN nelze
* (jen 1 uroven), proto tam vynechavame geografickou kontrolu. Slouzi
* k vizualni kontrole, zda AI koeficienty odpovidaji interakcnimu modelu.
display _n "--- 6.6 Per-country OLS (Model C, separately) ---"
* US (vcetne region FE)
display _n "=========== Country = US ==========="
capture noisily regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    i.sector_nace_num ///
    i.region_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    i.job_family_num ///
    ib3.edu_ols ///
    ib3.exp_category ///
    if country == "US" & ln_salary != ., vce(robust)
if _rc == 0 {
    estimates store ols_US
}

* DE a IN (bez region FE)
foreach c in DE IN {
    display _n "=========== Country = `c' ==========="
    capture noisily regress ln_salary ///
        cluster_* ///
        i.ai_level ///
        i.sector_nace_num ///
        is_remote ///
        ib1.type_cat ///
        ib5.size_cat ///
        i.job_family_num ///
        ib3.edu_ols ///
        ib3.exp_category ///
        if country == "`c'" & ln_salary != ., vce(robust)
    if _rc == 0 {
        estimates store ols_`c'
    }
    else {
        display as error "Model pro `c' selhal (return code: " _rc ") — pravdepodobne maly N."
    }
}

display _n "--- 6.6b Per-country porovnani AI koeficientu ---"
capture estimates table ols_US ols_DE ols_IN, ///
    keep(1.ai_level 2.ai_level) star stats(N r2)

* --- 6.7 Robustnostni kontrola: Clusterovane SE na urovni firmy ---
display _n "--- 6.7 Model FE-C s clusterovanymi SE (company_id) ---"
regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    ib3.country_id ///
    i.sector_nace_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    i.job_family_num ///
    ib3.edu_ols ///
    ib3.exp_category ///
    if ln_salary != ., vce(cluster company_id)
estimates store pooled_fe_c_cluster

display _n "--- 6.7b Porovnani robust vs cluster SE ---"
estimates table pooled_fe_c pooled_fe_c_cluster, star stats(N r2 r2_a)

* --- 6.8 Robustnostni kontrola: Model FE-C bez cirkularnich clusteru ---
* cluster_generative_ai a cluster_data_science__ml primo implikuji AI pozadavek.
display _n "--- 6.8 Model FE-C bez GenAI a DS/ML clusteru ---"
rename cluster_generative_ai _excl_genai_ols
rename cluster_data_science__ml _excl_dsml_ols

regress ln_salary ///
    cluster_* ///
    i.ai_level ///
    ib3.country_id ///
    i.sector_nace_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    i.job_family_num ///
    ib3.edu_ols ///
    ib3.exp_category ///
    if ln_salary != ., vce(robust)
estimates store pooled_fe_c_nocirc

rename _excl_genai_ols cluster_generative_ai
rename _excl_dsml_ols cluster_data_science__ml

display _n "--- 6.8b Porovnani Model FE-C vs FE-C-nocirc ---"
estimates table pooled_fe_c pooled_fe_c_nocirc, ///
    keep(1.ai_level 2.ai_level 1.country_id 2.country_id) star stats(N r2 r2_a)

* --- 6.9 Heckmanova korekce na sdruceni vzorku (robustnost) ---
* Vzhledem k obrovsky odlisnemu pokryti mzdovych dat napric zememi (US ~82 %,
* IN ~65 %, DE ~8 %) je Heckmanova korekce zvlast relevantni. Identifikace je,
* stejne jako v hlavni analyze, jen z funkcionalni formy probit modelu.
display _n "--- 6.9 Heckman MLE (Model FE-C specifikace) ---"
heckman ln_salary ///
    cluster_* ///
    i.ai_level ///
    ib3.country_id ///
    i.sector_nace_num ///
    is_remote ///
    ib1.type_cat ///
    ib5.size_cat ///
    i.job_family_num ///
    ib3.edu_ols ///
    ib3.exp_category, ///
    select(has_salary = ///
        i.ai_level ///
        ib3.country_id ///
        i.sector_nace_num ///
        is_remote ///
        ib1.type_cat ///
        ib5.size_cat ///
        i.job_family_num ///
        ib2.edu_logit ///
        ib3.exp_category) ///
    vce(robust)
estimates store pooled_heckman

display _n "--- 6.9b Porovnani OLS FE-C vs Heckman (AI a country koef.) ---"
estimates table pooled_fe_c pooled_heckman, ///
    keep(1.ai_level 2.ai_level 1.country_id 2.country_id) star


* ==============================================================================
* 6A. BINARNI LOGIT: P(has_ai) S COUNTRY FE
* ==============================================================================
display _n "=============================================================="
display "6A. BINARNI LOGIT — P(has_ai) s country FE"
display "=============================================================="

* --- Vyrazeni cirkularnich clusteru pro logit/mlogit ---
rename cluster_generative_ai _excl_genai_logit
rename cluster_data_science__ml _excl_dsml_logit

* --- 6A.1 Logit M1: firemni profil + country ---
display _n "--- 6A.1a Logit M1: profil firmy + country FE ---"
logit has_ai ///
    ib3.country_id ///
    i.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat, or vce(robust)
estimates store clogit_m1
display _n "--- 6A.1b AME Logit M1 ---"
margins, dydx(*)

* --- 6A.2 Logit M2: role + lidsky kapital + country ---
display _n "--- 6A.2a Logit M2: role + HC + country FE ---"
logit has_ai ///
    ib3.country_id ///
    cluster_* ///
    i.job_family_num ///
    ib2.edu_logit ///
    ib3.exp_category, or vce(robust)
estimates store clogit_m2
display _n "--- 6A.2b AME Logit M2 ---"
margins, dydx(*)

* --- 6A.3 Logit M3: kompletni ---
display _n "--- 6A.3a Logit M3: kompletni ---"
logit has_ai ///
    ib3.country_id ///
    i.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat ///
    cluster_* ///
    i.job_family_num ///
    ib2.edu_logit ///
    ib3.exp_category, or vce(robust)
estimates store clogit_m3
display _n "--- 6A.3b AME Logit M3 ---"
margins, dydx(*)

* --- 6A.3c AME pro country dummy (klicove pro komparativni interpretaci) ---
display _n "--- 6A.3c Country AME (oproti US, plny model) ---"
margins, dydx(country_id)

* --- 6A.4 Porovnani logit modelu ---
display _n "--- 6A.4 Porovnani Logit M1 / M2 / M3 ---"
estimates table clogit_m1 clogit_m2 clogit_m3, star stats(N ll chi2 r2_p)


* ==============================================================================
* 6B. MULTINOMIALNI LOGIT: P(ai_level) S COUNTRY FE
* ==============================================================================
display _n "=============================================================="
display "6B. MULTINOMIALNI LOGIT — P(ai_level) s country FE"
display "=============================================================="
* POZN: edu_logit zde neni zahrnuta (stejne jako v hlavni analyze) — HS/Assoc
* × Applied AI ma < 50 obs na nekterych zemich. Vzdelani je kontrolovano
* v binarnim logitu (§6A) a v OLS (§6).

* --- 6B.1 Mlogit M1: firemni profil + country ---
display _n "--- 6B.1a Mlogit M1: profil firmy + country FE ---"
mlogit ai_level ///
    ib3.country_id ///
    i.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat, baseoutcome(0) rrr vce(robust)
estimates store cmlogit_m1
display _n "--- 6B.1b AME Mlogit M1: P(AI Integration) ---"
margins, dydx(*) predict(outcome(1))
display _n "--- 6B.1c AME Mlogit M1: P(Applied/Core AI) ---"
margins, dydx(*) predict(outcome(2))

* --- 6B.2 Mlogit M2: role + country ---
display _n "--- 6B.2a Mlogit M2: role + country FE ---"
mlogit ai_level ///
    ib3.country_id ///
    cluster_* ///
    i.job_family_num ///
    ib3.exp_category, baseoutcome(0) rrr vce(robust)
estimates store cmlogit_m2
display _n "--- 6B.2b AME Mlogit M2: P(AI Integration) ---"
margins, dydx(*) predict(outcome(1))
display _n "--- 6B.2c AME Mlogit M2: P(Applied/Core AI) ---"
margins, dydx(*) predict(outcome(2))

* --- 6B.3 Mlogit M3: kompletni ---
display _n "--- 6B.3a Mlogit M3: kompletni ---"
mlogit ai_level ///
    ib3.country_id ///
    i.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat ///
    cluster_* ///
    i.job_family_num ///
    ib3.exp_category, baseoutcome(0) rrr vce(robust)
estimates store cmlogit_m3
display _n "--- 6B.3b AME Mlogit M3: P(AI Integration) ---"
margins, dydx(*) predict(outcome(1))
display _n "--- 6B.3c AME Mlogit M3: P(Applied/Core AI) ---"
margins, dydx(*) predict(outcome(2))

* --- 6B.3d Country AME (klicove cislo pro diplomku) ---
display _n "--- 6B.3d Country AME: P(AI Integration) ---"
margins, dydx(country_id) predict(outcome(1))
display _n "--- 6B.3e Country AME: P(Applied/Core AI) ---"
margins, dydx(country_id) predict(outcome(2))

* --- 6B.4 Porovnani mlogit modelu ---
display _n "--- 6B.4 Porovnani Mlogit M1 / M2 / M3 ---"
estimates table cmlogit_m1 cmlogit_m2 cmlogit_m3, star stats(N ll chi2)

* --- 6B.5 Mediacni varianty M3 ---
display _n "--- 6B.5a Mlogit M3a: M3 bez job_family ---"
mlogit ai_level ///
    ib3.country_id ///
    i.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat ///
    cluster_* ///
    ib3.exp_category, baseoutcome(0) rrr vce(robust)
estimates store cmlogit_m3a

display _n "--- 6B.5b Mlogit M3b: M3 bez job_family a seniority ---"
mlogit ai_level ///
    ib3.country_id ///
    i.sector_nace_num ///
    ib1.type_cat ///
    ib5.size_cat ///
    cluster_*, baseoutcome(0) rrr vce(robust)
estimates store cmlogit_m3b

display _n "--- 6B.5c Porovnani Mlogit M3 / M3a / M3b ---"
estimates table cmlogit_m3 cmlogit_m3a cmlogit_m3b, star stats(N ll chi2)

* Vratit vyrazene clustery pro dalsi analyzu / export
rename _excl_genai_logit cluster_generative_ai
rename _excl_dsml_logit cluster_data_science__ml


* ==============================================================================
* 7. EXPORTY TABULEK A GRAFU
* ==============================================================================
display _n "=============================================================="
display "7. EXPORTY"
display "=============================================================="

capture ssc install estout
capture ssc install coefplot

* --- 7.1 Tabulka 5: Pooled OLS s country FE + interakce (hlavni komparativni tabulka) ---
esttab pooled_fe_a pooled_fe_b pooled_fe_c pooled_interaction using "$outdir/Tabulka_5_Cross_Country_OLS.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    drop(_cons) ///
    order(1.ai_level 2.ai_level 1.country_id 2.country_id ///
          1.country_id#1.ai_level 1.country_id#2.ai_level ///
          2.country_id#1.ai_level 2.country_id#2.ai_level ///
          is_remote ///
          0.edu_ols 1.edu_ols 2.edu_ols 4.edu_ols ///
          0.exp_category 2.exp_category 4.exp_category ///
          cluster_*) ///
    refcat(1.ai_level "{it:AI uroven (ref: None)}" ///
           1.country_id "{it:Zeme (ref: US)}" ///
           0.edu_ols "{it:Vzdelani (ref: Bachelor)}" ///
           0.exp_category "{it:Zkusenosti (ref: Mid 3-5 let)}" ///
           cluster_architecture__methods "{it:Skill clustery}", nolabel) ///
    indicate("NACE sektor = *.sector_nace_num" ///
             "Typ firmy = *.type_cat" ///
             "Velikost firmy = *.size_cat" ///
             "Job family = *.job_family_num") ///
    stats(N r2 r2_a, fmt(0 3 3) labels("N" "R2" "Adj. R2")) ///
    mtitles("FE-A" "FE-B" "FE-C" "Interakce") ///
    title("Tabulka 5: Cross-country OLS pro ln(plat) — US/DE/IN") ///
    addnotes("Robustni standardni chyby v zavorkach." ///
             "Zavisla promenna: ln(rocni plat v USD, po konverzi EUR 1.165, INR /88)." ///
             "Referencni zeme: US. Zaporne koeficienty 1.country_id (DE) a 2.country_id (IN) = nizsi prumery nez US.")

* --- 7.2 Tabulka 5b: Country x AI AME (z interakcniho modelu) ---
estimates restore pooled_interaction
quietly margins country_id, dydx(ai_level) post
estimates store ame_country_ai
esttab ame_country_ai using "$outdir/Tabulka_5b_Country_AI_AME.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    mtitles("AME ai_level podle zeme") ///
    title("Tabulka 5b: AI mzdova premie podle zeme (marginalni efekty ai_level)")

* --- 7.3 Tabulka 6: Binarni logit — AME ---
foreach m in clogit_m1 clogit_m2 clogit_m3 {
    estimates restore `m'
    local n = e(N)
    local ll = e(ll)
    local r2_p = e(r2_p)
    quietly margins, dydx(*) post
    estadd scalar ll = `ll'
    estadd scalar r2_p = `r2_p'
    estimates store ame_`m'
}
esttab ame_clogit_m1 ame_clogit_m2 ame_clogit_m3 using "$outdir/Tabulka_6_Binarni_logit_country.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    stats(N ll r2_p, fmt(0 3 3) labels("N" "Log-likelihood" "Pseudo R2")) ///
    mtitles("M1 (Firma)" "M2 (Role)" "M3 (Kompletni)") ///
    title("Tabulka 6: Binarni logit s country FE — AME determinantu P(AI)")

* --- 7.4 Tabulka 7: Mlogit — AME M3 pro obe kategorie ---
foreach m in cmlogit_m3 cmlogit_m3a cmlogit_m3b {
    estimates restore `m'
    local n = e(N)
    local ll = e(ll)
    local r2_p = e(r2_p)
    quietly margins, dydx(*) predict(outcome(1)) post
    estadd scalar ll = `ll'
    estadd scalar r2_p = `r2_p'
    estimates store ame_`m'_1

    estimates restore `m'
    local n = e(N)
    local ll = e(ll)
    local r2_p = e(r2_p)
    quietly margins, dydx(*) predict(outcome(2)) post
    estadd scalar ll = `ll'
    estadd scalar r2_p = `r2_p'
    estimates store ame_`m'_2
}
esttab ame_cmlogit_m3_1 ame_cmlogit_m3_2 ///
       ame_cmlogit_m3a_1 ame_cmlogit_m3a_2 ///
       ame_cmlogit_m3b_1 ame_cmlogit_m3b_2 ///
       using "$outdir/Tabulka_7_Mlogit_country.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    stats(N ll r2_p, fmt(0 3 3) labels("N" "Log-likelihood" "Pseudo R2")) ///
    mtitles("M3: AI Integ" "M3: App AI" "M3a: AI Integ" "M3a: App AI" "M3b: AI Integ" "M3b: App AI") ///
    title("Tabulka 7: Multinomialni logit s country FE — AME")

* --- 7.5 Graf 13: Penetrace AI pozadavku podle zeme (vstup do externiho grafu) ---
preserve
contract country ai_level
export delimited "$outdir/Graf_13_Penetrace_AI.csv", replace
restore

* --- 7.6 Graf 9: Distribuce ln(plat) podle zeme + AI ---
preserve
collapse (mean) ln_salary (count) n=ln_salary if ln_salary != ., by(country ai_level)
export delimited "$outdir/Graf_9_ln_salary_country_ai.csv", replace
restore

* --- 7.7 Graf 10: Forest plot interakcnich koeficientu country x AI ---
estimates restore pooled_interaction
coefplot, keep(*.country_id#*.ai_level) ///
    xline(0, lpattern(dash) lcolor(red)) ///
    title("Interakce Country x AI level (OLS ln(plat)) — baseline US", size(medium)) ///
    xtitle("Koeficient (oproti US x None)", size(small)) ///
    msymbol(O) msize(small) mcolor(navy) ciopts(lcolor(navy) lwidth(thin)) ///
    graphregion(color(white)) bgcolor(white) plotregion(fcolor(white) lcolor(white))
graph export "$outdir/Graf_10_country_ai_interaction.png", replace

* --- 7.8 Summary statistics podle zeme ---
display _n "--- 7.8 Summary statistics pro tabulku ---"
tabstat salary_mid ln_salary experience_min_llm edu_ols exp_category ///
    skill_count has_ai is_remote, by(country) stat(count mean sd)

* Ulozeni zpracovaneho datasetu
save "$outdir/ai_skills_comparative_processed.dta", replace


* ==============================================================================
* 8. ZAVER
* ==============================================================================
display _n "=============================================================="
display "CROSS-COUNTRY ANALYZA DOKONCENA"
display "=============================================================="
display "Vystupy ulozeny do: $outdir"
display "Log: $outdir/ai_skills_comparative_analysis.log"

display _n "--- Finalni shrnuti klicovych estimaci ---"
estimates table pooled_fe_a pooled_fe_b pooled_fe_c pooled_interaction, ///
    keep(1.ai_level 2.ai_level 1.country_id 2.country_id ///
         1.country_id#1.ai_level 1.country_id#2.ai_level ///
         2.country_id#1.ai_level 2.country_id#2.ai_level) ///
    star stats(N r2 r2_a)

display _n "--- Finalni shrnuti mlogit (M3) ---"
estimates table cmlogit_m3, star stats(N ll r2_p)

log close
