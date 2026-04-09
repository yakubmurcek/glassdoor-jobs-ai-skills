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

* **Tabulka 1: Základní charakteristiky vzorku podle AI požadavků (křížová tabulka)**
  * **Umístění:** Napříč sekcemi 5.1.3 až 5.1.6.
  * **Cíl:** Seskupit velkou spoustu procent z textu do jedné přehledné matice.
  * **Struktura:** Sloupce: *Všekny inzeráty | Bez AI | AI Integration | Applied/Core AI*. Řádky: *Vzdělání (%)*, *Zkušenosti (%)* a *Remote work (%)*.

### 5.2 Statistické testy
* V této sekci se tabulky většinou nedoporučují, pokud k tomu není speciální důvod. Výsledky jednoduchých porovnání (t-testy, ANOVA) obvykle stačí popsat v textu i s příslušnými hodnotami (*t*, *p*, *d*).

### 5.3 AI prémie na platu (OLS regresní analýza)
* **Tabulka 2: OLS modely vysvětlující mzdovou prémii za AI dovednosti**
  * **Umístění:** Shrnutí sekcí 5.3.1 a 5.3.2.
  * **Cíl:** Standardní ekonometrická tabulka reportující robustní modely vedle sebe.
  * **Struktura:** Sloupce by měly zastupovat Model A (Základní) a Model B (Lidský kapitál). Měly by tu být zviditelněny zejména hlavní proměnné (úrovně AI, remote, praxe), zatímco technické proměnné (skill clusters) lze sdružit či logicky skrýt ("Controls: Yes"). Obsahuje koeficienty a standardní chyby s hvězdičkami statistické významnosti.

* **Graf 3: Hierarchie mzdových efektů (Koeficientový / Forest plot)**
  * **Umístění:** Sekce 5.3.3.
  * **Cíl:** Vizuálně srovnat relativní sílu AI prémie vůči jiným mzdovým charakteristikám.
  * **Vzhled:** Bodový regresní graf (koeficienty s intervalem spolehlivosti) vybraných klíčových předpovědí z Modelu B (např. *Region West*, *Seniorita 6+ let*, *Applied AI*, *AI Integration*, *Junior*). To skvěle komunikuje pointu, že "AI dovednosti jsou aditivní bonus, ale klasický lidský kapitál i lokalita mají na mzdu větší vliv".

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

### C) Grafy doplňujících deskriptivních statistik pro Word
Pokud máš potíže dotvářet moderní layout grafů ve Statě (barvy a styly), jedním z efektivních triků je sestavení souhrnů příkazem `collapse`, export to tabulky `.csv` a následná blesková úprava na několik kliknutí moderní šablonou v MS Excel:
```stata
collapse (mean) percent_ai=has_ai, by(job_family)
export delimited "$outdir/job_family_ai.csv", replace
```

### D) Celkové estetické principy pro VŠKP
1. **Písmová unifikace**: Grafy a osy musí sdílet font použitý v hlavním textu práce (např. Times New Roman nebo Calibri).
2. **Konzistentní barvy**: Sledujte fixní sémantiku barev celou prací. Například: *Applied/Core AI* jako tmavě modrá, *AI Integration* jako světle modrá a *None* jako šedá. Tyto barvy nezaměnitelně ulehčují vizuální čtení.
3. Přehlednost: Všechny sloupcové tabulky regrese nebo AME by měly udržovat stejnou úroveň odsazení a unifikovaný popis sledovaných nezávislých proměnných.
