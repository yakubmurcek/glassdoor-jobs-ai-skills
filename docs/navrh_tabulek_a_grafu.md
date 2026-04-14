# Návrh tabulek a grafů pro diplomovou práci

Tento dokument obsahuje návrh na vizualizaci výsledků analýzy pro kapitolu "5 Výsledky empirické analýzy". Grafy a tabulky jsou navrženy tak, aby čtenáři co nejsrozumitelněji předaly klíčová zjištění a zabránily zahlcení textu čísly.

## 1. Rozvržení tabulek a grafů dle kapitol

### Pomocné datové exporty
* **Data 1: Agregovaný dataset pro externí grafy**
  * **Export v do-filu:** `Data_1_summary_for_charts.csv`
  * **Cíl:** Poskytnout čistá data (AI tier, plat, zkušenosti, odvětví) pro jednoduché flexibilní ploty v MS Excel / Python.

### 5.1 Deskriptivní statistika
* **Graf 1: Distribuce ročních platů napříč úrovněmi AI (Křivka hustoty / KDE)** 
  * **Umístění:** Sekce 5.1.2.
  * **Cíl:** Vizuálně zachytit komplexní posun mzdové masy k vyšším částkám u AI pozic.
  * **Vzhled:** Elegantní překryvný graf hustoty se 3 vrstvami. Pomocí poloprůhledných moderních barev krásně ukazuje, jak se střed distribuce u AI dovedností masivně posouvá doprava. Je to esteticky i analyticky mnohem silnější než standardní boxplot.

* **Graf 2: Podíl AI inzerátů podle *Job Family***
  * **Umístění:** Sekce 5.1.5.
  * **Cíl:** Nahradit zdlouhavý text sestupným žebříčkem na základě čistých vizualizovatelných dat z vygenerovaného CSV.
  * **Export v do-filu:** `Graf_2_job_family_ai.csv`

* **Graf 3: Podíl inzerátů nabízejících remote práci**
  * **Umístění:** Sekce 5.1.4.
  * **Cíl:** Rychlé vizuální porovnání (např. vysoká flexibilita pro AI pozice vs. zbytek). 
  * **Vzhled:** Čisté sloupce (bez rušivého šedého/modrého gridu typického pro Statu) navázané na sémantické popisky ("Běžné pozice" / "AI pozice").

* **Graf 4: Kompozice seniority podle úrovně AI (Skládaný 100% sloupcový graf)**
  * **Umístění:** Sekce 5.1.5.
  * **Cíl:** Ukázat kompozici pracovních sil. Osa X: None, AI Integration, Applied AI; Osa Y: poměr Junior vs Mid vs Senior. Z grafu tak na první pohled vyplyne, že AI role preferují zkušenější kandidáty.
  * **Export v do-filu:** `Graf_4_seniority_by_ai.csv`

* **Tabulka 1: Základní charakteristiky vzorku podle AI požadavků (křížová tabulka)**
  * **Umístění:** Napříč sekcemi 5.1.3 až 5.1.6.
  * **Cíl:** Seskupit tunu deskriptivních procent z textu do jedné přehledné matice.
  * **Struktura:** Sloupce: *Všekny inzeráty | Bez AI | AI Integration | Applied/Core AI*. Řádky: *Vzdělání (%)*, *Zkušenosti (%)* a *Remote work (%)*.
  * **Export v do-filu:** `Tabulka_1_zakladni_charakteristiky.csv`

### 5.2 Statistické testy
* V této sekci se tabulky většinou nedoporučují, pokud k tomu není speciální důvod. Výsledky jednoduchých porovnání (t-testy, ANOVA) obvykle stačí popsat v textu i s příslušnými hodnotami (*t*, *p*, *d*).

### 5.3 AI prémie na platu (OLS regresní analýza)
* **Tabulka 2: OLS modely determinantů log(platu) — hlavní hierarchická tabulka**
  * **Umístění:** Shrnutí sekcí 5.3.1–5.3.3.
  * **Cíl:** Standardní ekonometrická tabulka reportující 4 modely vedle sebe. Vedoucí doporučuje reportovat B vs C pro ukázku efektu job_family na AI prémii.
  * **Struktura:**

    | | Model A | Model B | C (bez JF) | Model C |
    |---|---|---|---|---|
    | AI Integration | β (SE) | β (SE) | β (SE) | β (SE) |
    | Applied/Core AI | β (SE) | β (SE) | β (SE) | β (SE) |
    | Remote | β (SE) | β (SE) | β (SE) | β (SE) |
    | Vzdělání (4 kat.) | — | β (SE) | β (SE) | β (SE) |
    | Zkušenosti (3 kat.) | — | β (SE) | β (SE) | β (SE) |
    | NACE sektor | Yes | Yes | Yes | Yes |
    | Region | Yes | Yes | Yes | Yes |
    | Typ firmy | Yes | Yes | Yes | Yes |
    | Velikost firmy | Yes | Yes | Yes | Yes |
    | Skill clustery | — | — | Yes | Yes |
    | Job Family | — | — | — | Yes |
    | N | | | | |
    | R² | | | | |
    | Adj. R² | | | | |

  * **Export v do-filu:** §6.8 → `Tabulka_2_OLS_hlavni.rtf`
  * **Poznámka:** Reference: AI level = None, Vzdělání = Missing, Zkušenosti = Mid (3–5 let). Robustní SE.

* **Graf 5: Hierarchie mzdových efektů (Koeficientový / Forest plot)**
  * **Umístění:** Sekce 5.3.3.
  * **Cíl:** Vizuálně srovnat relativní sílu AI prémie vůči jiným mzdovým charakteristikám.
  * **Vzhled:** Bodový regresní graf vybraných klíčových předpovědí (koeficienty z Modelu B s 95% intervaly spolehlivosti). To vizuálně doloží pointu, že klasický lidský kapitál a lokace mají větší vliv než samotná AI prémie.
  * **Export v do-filu:** §6.8 → `Graf_5_coefplot_ols.png`

* **Graf 6: Predikované průměrné platy podle AI úrovně (Marginsplot)**
  * **Umístění:** Diskuze po Modelu B.
  * **Cíl:** Převést abstraktní logaritmické procentuální nárůsty na čtivé predikované dolarové rozdíly (Adjusted Predictions).
  * **Vzhled:** Bodové průměry pro tři AI úrovně na ose X a predikovaný plat na ose Y s chybovými úsečkami (95 % CI).
  * **Export v do-filu:** §6.8 → `Graf_6_margins_ai.png`

* **Graf 7: Vývoj křivky platu napříč senioritou (Marginsplot vrstvený podle AI)**
  * **Umístění:** Sekce 5.3.5.
  * **Cíl:** Odkrýt vzorec rozevírajících se mzdových nůžek, ve kterém plat za AI roste rapidněji s rostoucí praxí. 
  * **Vzhled:** Liniový interakční graf. 3 separátní průběhy (pro „No AI", „AI Integration" a „Applied/Core AI").
  * **Export v do-filu:** §6.8 → `Graf_7_margins_seniority.png`

* **Graf 8: Predikce mzdy dle Vzdělání (Marginsplot)**
  * **Umístění:** OLS Analýza (konec sekce 5.3).
  * **Cíl:** Vizualizovat trend "Skills-based hiring". Graf odpoví na otázku, zda kandidát s nižším vzděláním, ale AI dovednostmi vydělává více, než tradičně vzdělaný s magisterským titulem.
  * **Export v do-filu:** `Graf_8_margins_education.png`

* **Graf 9: Mzdová AI mezera napříč NACE sektory**
  * **Umístění:** OLS Analýza (konec sekce 5.3).
  * **Cíl:** Tzv. Marginsplot (obdoba Cleveland Dot Plot). Zjišťuje, ve kterém odvětví roste "AI prémie" nejsilněji (IT vs Finance vs Manufacturing).
  * **Export v do-filu:** `Graf_9_margins_sectors.png`

* **Graf 10: Mzdová prémie u Remote rolí podle AI**
  * **Umístění:** OLS Analýza (konec sekce 5.3).
  * **Cíl:** Interakční graf zkoumající, zda tvoří Remote práce a AI znalosti "dvojsmyslnou prémii" pro moderní digitalizovaný talent.
  * **Export v do-filu:** `Graf_10_margins_remote.png`

### 5.4 Determinanty AI požadavku (binární logistická regrese)
* **Tabulka 3: Binární logit — AME determinantů P(AI požadavek)**
  * **Umístění:** Sekce 5.4.
  * **Cíl:** Ukázat, které charakteristiky (firma, skills, pozice) predikují, zda inzerát vůbec požaduje AI.
  * **Struktura:** 4 sloupce: M1 (firma), M2 (role), M3 (kompletní), M3 bez JF (mediace). Řádky: AME jednotlivých prediktorů. Dole: N, Log-likelihood, Pseudo R².
  * **Klíčová zjištění:** M3 vs M3-nojf ukáže mediační efekt job_family.
  * **Zdroj v do-filu:** §6A.1–6A.4d (logit M1–M3 + M3-nojf, `margins, dydx(*)`) → `Tabulka_3_Binarni_logit.rtf`
  * **POZN:** Clustery `cluster_generative_ai` a `cluster_data_science__ml` jsou vyřazeny (tautologické s DV = AI tier).

* **Graf 11: ROC křivka a AUC (Binární logit Model M3)**
  * **Umístění:** Sekce 5.4.
  * **Cíl:** Graficky zhodnotit prediktivní sílu a přesnost kompletního logistického modelu pro P(AI požadavek).
  * **Export v do-filu:** `Graf_11_roc_logit_m3.png`

### 5.5 Dovednostní profil AI: Používání vs. Vývoj (multinomiální logit)
* **Tabulka 4: Multinomiální logit — AME pro P(AI Integration) a P(Applied/Core AI)**
  * **Umístění:** Sekce 5.5.3.
  * **Cíl:** Ukázat odlišnosti dovednostních profilů: které charakteristiky rozlišují „používání AI" od „vývoje AI".
  * **Struktura:** V řádcích skill clustery a organizační charakteristiky. Dva hlavní sloupce vyjadřují AME pro *P(AI Integration)* a *P(Applied/Core AI)*. Kompletní model M3 + mediační modely M3a (bez JF) a M3b (bez JF + seniority).
  * **Zdroj v do-filu:** §6B.1–6B.8 (mlogit M1–M3, M3a, M3b, `margins, dydx(*) predict(outcome(…))`) → `Tabulka_4_Multinomialni_logit.rtf`
  * **POZN:** Clustery GenAI a DS/ML vyřazeny (viz výše). Vzdělání (edu_logit) není zahrnuto v mlogit kvůli nízkému n v buňce HS/Assoc × Applied AI (23 < 50).

* **Graf 12: Vizuální profil AI rolí (Forest Plot marginálních efektů)**
  * **Umístění:** Sekce 5.5.3.
  * **Cíl:** Vizuálně srovnat marginální efekty z Tabulky 4. Modré body ukážou překlopení pravděpodobnosti do "AI Integration", červené body překlápění do "Applied/Core AI" napříč Skill clustery. Tím čtenář uvidí, jaká dovednost odděluje pouhé "užívání AI" od "vývoje AI".
  * **Export v do-filu:** `Graf_12_coefplot_mlogit.png`

### 5.6 Komparativní analýza zemí
* **Graf 13: Penetrace AI požadavků na světových trzích**
  * **Umístění:** Sekce 5.6.1.
  * **Cíl:** Názorné rozlišení podílů jednotlivých států v poptávce o AI role.
  * **Vzhled:** Skládaný 100% sloupcový graf (Stacked bar chart) pro USA, Německo a Indii.
  * **Export v do-filu:** `Graf_13_Penetrace_AI.csv`

* **Tabulka 5: Test homogenity mezinárodní AI prémie (OLS interakce)**
  * **Umístění:** Sekce 5.6.5 (může být odsunuta do přílohy).
  * **Cíl:** Transparentnost OLS regrese s fixními efekty a interakcemi country × ai_level.
  * **Zdroj v do-filu:** §5.2 v `ai_skills_analysis_comparative.do` → `Tabulka_5_Test_homogenity.rtf`

---

## 2. Doporučené technické workflow a Best Practices

Kopírovat tabulky či výsledky regresí manuálně je rizikové i zdlouhavé. Pro elegantní automatici je doporučeno využít balíčků v programu Stata:

### A) Generování tabulek (`estout` / `outreg2`)
Balíček `estout` automaticky formátuje tabulky (RTF nebo DOCX) z odhadnutých Stata modelů. Vše je přesně zarovnané včetně hvězdiček. Ukázka použití:
```stata
* Uložení modelů po regresi
eststo ModelA: regress ln_salary i.ai_level i.sector_nace_num is_remote ...
eststo ModelB: regress ln_salary i.ai_level i.sector_nace_num is_remote i.edu ...

* Export dokonalé rtf tabulky
esttab ModelA ModelB using "$outdir/mzdova_regrese.rtf", replace ///
    label b(3) se(3) star(* 0.05 ** 0.01 *** 0.001) ///
    keep(1.ai_level 2.ai_level is_remote *edu*) ///
    title("Tabulka X: OLS Mzdové modely determinantů log(platu)") ///
    addnotes("Robustní standardní chyby v závorkách.")
```

### B) Vytvoření Forest plotu (Graf koeficientů - `coefplot`)
Na základě vytvořené OLS regrese lze použít populární balíček `coefplot`, který ušetří spoustu nešikovného klikání. Příklad kódu:
```stata
coefplot, keep(1.ai_level 2.ai_level 4.exp_category 1.exp_category is_remote *region*) ///
    xline(0, lpattern(dash) lcolor(red)) ///
    sort ///
    title("Porovnání efektů v OLS Modelu platu") ///
    xtitle("Procentuální prémie (Koeficient)")
```

### C) Predikované mzdy (`margins` a `marginsplot`)
Místo suchých regresních koeficientů chceš občas ukázat přímo předpovídané číslo (Adjusted prediction). Skvělé využití pro nové grafy k Mincerově rovnici a AI prémii.
```stata
quietly regress ln_salary i.ai_level ib3.exp_category i.region_num ... , vce(robust)
* Spočítání předpovědi (margins drží všechny covariates na průměru vzorku)
margins i.ai_level
* Generování grafu s obálkou intervalů spolehlivosti
marginsplot, title("Predikovaná relace platu a AI") xtitle("Úroveň AI") ytitle("log(Plat)")
```

### D) Grafy doplňujících deskriptivních statistik pro Word
Pokud máš potíže dotvářet moderní layout grafů ve Statě (barvy a styly), jedním z efektivních triků je sestavení souhrnů příkazem `collapse`, export to tabulky `.csv` a následná blesková úprava na několik kliknutí moderní šablonou v MS Excel:
```stata
collapse (mean) percent_ai=has_ai, by(job_family)
export delimited "$outdir/job_family_ai.csv", replace
```

### E) Celkové estetické principy pro VŠKP
1. **Písmová unifikace**: Grafy a osy musí sdílet font použitý v hlavním textu práce (např. Times New Roman nebo Calibri).
2. **Konzistentní barvy**: Sledujte fixní sémantiku barev celou prací. Například: *Applied/Core AI* jako tmavě modrá, *AI Integration* jako světle modrá a *None* jako šedá. Tyto barvy nezaměnitelně ulehčují vizuální čtení.
3. Přehlednost: Všechny sloupcové tabulky regrese nebo AME by měly udržovat stejnou úroveň odsazení a unifikovaný popis sledovaných nezávislých proměnných.
