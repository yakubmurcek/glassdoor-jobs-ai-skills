# Návrh tabulek a grafů pro diplomovou práci

Tento dokument obsahuje návrh na vizualizaci výsledků analýzy pro kapitolu "5 Výsledky empirické analýzy". Grafy a tabulky jsou navrženy tak, aby čtenáři co nejsrozumitelněji předaly klíčová zjištění a zabránily zahlcení textu čísly.

## 1. Rozvržení tabulek a grafů dle kapitol

### 5.1 Deskriptivní statistika
* **Graf 1: Distribuce ročních platů napříč úrovněmi AI (Boxplot)** 
  * **Umístění:** Sekce 5.1.2.
  * **Cíl:** Vizualizovat vzestup mediánu a průměru mezd (Applied AI > AI Integration > None) i s rozptylem platů.
  * **Tip:** Doporučuje se graf nahoře oříznout (např. omezit osu Y na max. 300 000 USD), aby extrémní odlehlé hodnoty nezploštily vizualizované "krabičky".

* **Graf 2: Podíl AI inzerátů podle *Job Family***
  * **Umístění:** Sekce 5.1.5.
  * **Cíl:** Nahradit zdlouhavý text o podílu AI pozic v různých profesích.
  * **Vzhled:** Horizontální sloupcový graf (bar chart), seřazený sestupně (na prvním místě např. *Data & AI* s 55 %, na dalších *Software Engineers* s 26 %, atd.).

* **Graf 3: Podíl inzerátů nabízejících remote práci (Seskupený sloupcový graf)**
  * **Umístění:** Sekce 5.1.4.
  * **Cíl:** Rychlé vizuální porovnání (Remote 37,4 % pro AI pozice vs. 26,1 % pro non-AI pozice). 
  * **Vzhled:** Dva jasně odlišené sloupce (AI vs non-AI).

* **Graf 4: Kompozice seniority podle úrovně AI (Skládaný 100% sloupcový graf)**
  * **Umístění:** Sekce 5.1.5.
  * **Cíl:** Ukázat kompozici pracovních sil. Osa X: None, AI Integration, Applied AI; Osa Y: poměr Junior vs Mid vs Senior. Z grafu tak na první pohled vyplyne, že AI role preferují zkušenější kandidáty.

* **Tabulka 1: Základní charakteristiky vzorku podle AI požadavků (křížová tabulka)**
  * **Umístění:** Napříč sekcemi 5.1.3 až 5.1.6.
  * **Cíl:** Seskupit tunu deskriptivních procent z textu do jedné přehledné matice.
  * **Struktura:** Sloupce: *Všekny inzeráty | Bez AI | AI Integration | Applied/Core AI*. Řádky: *Vzdělání (%)*, *Zkušenosti (%)* a *Remote work (%)*.

### 5.2 Statistické testy
* V této sekci se tabulky většinou nedoporučují, pokud k tomu není speciální důvod. Výsledky jednoduchých porovnání (t-testy, ANOVA) obvykle stačí popsat v textu i s příslušnými hodnotami (*t*, *p*, *d*).

### 5.3 AI prémie na platu (OLS regresní analýza)
* **Tabulka 2: OLS modely vysvětlující mzdovou prémii za AI dovednosti**
  * **Umístění:** Shrnutí sekcí 5.3.1 a 5.3.2.
  * **Cíl:** Standardní ekonometrická tabulka reportující modely vedle sebe (Model A, Model B). 
  * **Struktura:** Viditelné hlavní proměnné (úrovně AI, remote, praxe), zatímco technické doplňky (skill clusters) lze logicky skrýt ("Controls: Yes"). Obsahuje koeficienty a standardní chyby v závorkách.

* **Graf 5: Hierarchie mzdových efektů (Koeficientový / Forest plot)**
  * **Umístění:** Sekce 5.3.3.
  * **Cíl:** Vizuálně srovnat relativní sílu AI prémie vůči jiným mzdovým charakteristikám.
  * **Vzhled:** Bodový regresní graf vybraných klíčových předpovědí (koeficienty z Modelu B s 95% intervaly spolehlivosti). To vizuálně doloží pointu, že klasický lidský kapitál a lokace mají větší vliv než samotná AI prémie.

* **Graf 6: Predikované průměrné platy podle AI úrovně (Marginsplot)**
  * **Umístění:** Diskuze po Modelu B.
  * **Cíl:** Převést abstraktní logaritmické procentuální nárůsty na čtivé predikované dolarové rozdíly ("Adjusted Predictions at means").
  * **Vzhled:** Bodové průměry pro tři AI úrovně na ose X a predikovaný plat na ose Y s chybovými úsečkami (95 % CI). Laický čtenář rovnou uvidí absolutní zisk z AI, protože efekt koeficientu je očištěn od ostatních zkreslení.

* **Graf 7: Klesající výnosy ze zkušeností (Kvadratická křivka z Mincerova modelu)**
  * **Umístění:** Sekce 5.3.5.
  * **Cíl:** Ukázat efekt let praxe na plat a dokázat vizuálně zpomalení nárůstu mzdy (diminishing returns).
  * **Vzhled:** Liniový graf závislosti predikovaného platu (Y) vůči rokům požadované praxe na ose X. Můžeme vykreslit dvě křivky (pozice s AI a bez AI), aby bylo vidět posunutí mzdové hladiny o danou prémii napříč veškerou praxí.

### 5.4 a 5.5 Determinanty AI požadavku a dovednostní profil
* **Tabulka 3: Porovnání dovednostních profilů (Multinomiální logit - marginální efekty)**
  * **Umístění:** Sekce 5.5.3 (používání AI vs. vývoj AI).
  * **Cíl:** Ukázat odlišnosti dovedností napříč tierovaným AI.
  * **Struktura:** V řádcích leží skill clustery. Dva hlavní sloupce vyjadřují marginální efekty (AME) pro zařazení inzerátu do *AI Integration* a do *Applied/Core AI*. Tabulka krásně ukáže, že vývojář potřebuje navíc fundamenty (*Systems Programming*, *Data Engineering*), zatímco konzumentovi stačí aplikační dovednosti (*Frontend*, *GenAI*).

### 5.6 Komparativní analýza zemí
* **Graf 4: Penetrace AI požadavků na světových trzích**
  * **Umístění:** Sekce 5.6.1.
  * **Cíl:** Názorné rozlišení podílů jednotlivých států v poptávce o AI role.
  * **Vzhled:** Skládaný 100% sloupcový graf (Stacked bar chart) ukáže podíly kategorií na trhu pro USA, Německo a Indii. Zvýrazní tak indický propad oproti západním trhům i strukturu poptávky.

* **Tabulka 4: Test homogenity mezinárodní AI prémie**
  * **Umístění:** Sekce 5.6.5 (může být odsunuta do případné přílohy).
  * **Cíl:** Slouží k transparentnosti OLS regrese, která obsahuje fixní efekty i interakční efekty států na to, že mzdová prémie za AI se mezi nimi proporčně neliší.

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
