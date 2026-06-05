# TAHÁK NA SCHŮZKU (TL;DR Metodika)

## 1. Data a Vzorek
* **Zdroj:** Glassdoor.
* **Země:** USA, Německo (DE), Indie (IN).
* **N (po čištění):** 38 436 inzerátů (US: 17k, IN: 14k, DE: 6k).
* **Sběr:** Plošně přes klíčové IT pozice.

## 2. Extrakce Dovedností (Hybridní)
* **Deterministická:** Regex hledání (hranice slov) vs. slovník (900+ pojmů).
* **LLM (gpt-4o-mini):** Task-based architektura (oddělené prompty pro Tier, Skilly a Vzdělání).
* **Výsledek:** Průnik obou metod do kanonických názvů.

## 3. Validace Dat a LLM Klasifikace (Klíčové testy kvality)
* **Manuální validace:** Na stratifikovaném vzorku 50 inzerátů dosáhl LLM **98% přesnosti** (49/50 správně zařazeno do tieru).
* **Cohen's Kappa (0.757):** Statisticky prokázaná silná shoda (91,7 %) mezi tupým "hledáním slov" a LLM úsudkem.
* **McNemarův test:** Prokázal, že LLM je systematicky citlivější než klasické regexy (najde AI kontext i tam, kde chybí přesné buzzwords).
* **Confidence Kalibrace:** Model umí říct "nevím" (vysvětluje nízkou jistotu u sporných inzerátů).

## 4. Klasifikace a Proměnné
* **AI Tiers (`ai_level`):** Čistě podle LLM (bez přepisování).
  * `0` = None
  * `1` = AI Integration (uživatel nástrojů)
  * `2` = Applied/Core AI (vývojář AI/ML)
* **Skill Clustery:** Celkem 24 → 21 v modelech (3 smazány pro řídkost).
* **Vzdělání (`edu_ols`):** 5 úrovní, referenční = Bachelor.
* **Baselines (Reference):** *Software Engineer* (Role) a *Sektor J* (IT).

## 4. Očištění Dat
* **Filtry:** Jen LLM confidence ≥ 0.7; Rok vydání ≥ 2024.
* **Úpravy mezd:** Na roční USD. Smazány extrémy (US/DE < $3k, IN < $2k, maximum $500k).

## 5. Ekonometrické Modely (Stata)
*Standardní chyby klastrované na firmu.*
* **Binární Logit:** Jaká je šance, že inzerát požaduje AI.
* **Multinomiální Logit:** Proč inzerát dostal konkrétní Tier. *(Vzdělání vyřazeno z modelu kvůli sparse cells).*
* **OLS Mzdy:**
  * Závislá: `ln(salary)`.
  * *Zásadní věc:* GenAI a DS/ML clustery **vyřazeny z rovnice**, aby nekanibalizovaly hlavní AI prémii (prevence cirkularity).

## 7. Prezentace Textu (Narativ)
* Kapitoly 5.3 až 5.5 = Case study jen pro **USA**.
* **Inkrementální budování:** M1 -> M3 (Logit) a A -> C (OLS). Přidávají se postupně kontroly a dovednosti.
* Až po tomto příběhu následuje mezinárodní srovnání zemí.

## 8. Robustnostní Testy Ekonometrie (Obrana modelu)
* **Heckman Selection:**
  * Řeší tajení platů u firem (selection bias).
  * Bias se potvrdil v IN.
  * *Výsledek:* I po odfiltrování biasu AI prémie všude drží (US +12,6 %, DE +14,7 %, IN +11,5 %).
* **Test cirkularity:**
  * Vráceny GenAI/ML clustery zpět do OLS.
  * *Výsledek:* AI prémie se nezhroutila.
* **US Region FE:**
  * Odfiltrovány drahé regiony v USA.
  * *Výsledek:* AI prémie není jen "Silicon Valley" efekt.
* **Cross-country modely:**
  * Všechny země hozeny do jednoho modelu (Pooled).
  * *Výsledek:* Wald testy matematicky potvrdily, že trhy se od sebe strukturálně liší.
