# Metodika a Postupy: Analýza AI dovedností (Podklady pro schůzku)

Tento dokument stručně shrnuje celou datovou a analytickou pipeline projektu. Poslouží jako detailní podklad pro případné rozšíření práce a psaní akademického článku.

*(Poznámka na úvod: Přestože starší pracovní návrhy zmiňují ČR a USA, **finální dataset a naprogramované Stata modely (`ai_skills_thesis_final.do`) explicitně pracují se vzorkem pro USA, Německo (DE) a Indii (IN)**. Dokumentace níže reflektuje tento skutečně implementovaný a finální stav.)*

## 1. Zdroj dat a Architektura
* **Sběr dat:** Scrape pracovních inzerátů přímo z platformy Glassdoor. 
* **Velikost vzorku:** Počáteční hrubý dataset obsahoval necelých 45 tisíc inzerátů. Po vyčištění dat (viz sekce 4) vstupuje do ekonometrických modelů finální dataset o velikosti **N = 38 436** inzerátů (USA: 17 848, Německo: 6 402, Indie: 14 186).
* **Postup vyhledávání:** Jelikož Glassdoor neumožňuje stáhnout celý IT sektor, data byla získána plošným vyhledáváním předem definované sady **klíčových IT pracovních pozic** (Software Engineer, Data Scientist atd.). Výhoda: Zachycuje to IT role i v "ne-IT" sektorech (banky, výroba).
* **Technologický stack:** 
  * Zpracování a extrakce: Python (CLI nástroj `ai_skills`), závislosti řešeny přes balíčkovací systém `uv`.
  * Ekonometrická analýza: Stata 15.1.
* **Pipeline Flow:** Surové CSV -> Deterministická extrakce -> LLM Analýza -> Spojení výsledků -> Očištění pro Statu -> Ekonometrické modely.

## 2. Hybridní extrakce dovedností
Z textu inzerátů jsou dovednosti izolovány dvěma komplementárními způsoby, které se na konci spojí (Union) a deduplikují do kanonických názvů:

1. **Deterministická extrakce (Slovníková):**
   * Využívá regexy s ohledem na hranice slov (`word-boundary matching`) proti předem definovanému slovníku o cca **900 pojmech**.
   * Řeší překryvy "chytře" (např. shoda "SQL Server" má přednost před "SQL", "React Native" nevyvolá i "React" - nejdelší nalezený termín vítězí).
2. **LLM extrakce (OpenAI - model gpt-5-mini / gpt-4o-mini):**
   * **Decomposed task-based batching:** Kód využívá moderní přístup, kdy se pro jeden inzerát nedělá jeden obrovský prompt, ale proces je rozdělen do samostatných paralelních tasků (AI tier klasifikace, Skills extrakce, Education extrakce).

## 3. Taxonomie a Klasifikace AI Tierů
* **Skill Taxonomy (24 -> 21 clusterů):** Skript v Pythonu sice nejprve mapuje dovednosti do 24 sémantických rodin, nicméně **přímo ve Stata kódu se tři z nich pro řídkost dat explicitně mažou** (`legacy__mainframe`, `data_analysis__stats`, `tools__editors`). Do samotných regresních modelů tedy reálně vstupuje **21 clusterů** (např. *Frontend*, *GenAI*, *Data Science/ML*, *Cloud*, *DevOps* atd.) jako binární dummy proměnné.
* **Klasifikace AI úrovně pozice (`ai_level` a `has_ai`):**
  * LLM vyhodnocuje text inzerátu a přiděluje mu jednu ze 3 úrovní:
    * `0 = None` (Žádný AI požadavek)
    * `1 = AI Integration` (Aplikované použití nástrojů – např. Copilot).
    * `2 = Applied/Core AI` (Vývoj AI, ML modely, Data Science).
  * `has_ai`: Binární indikátor (1 = tier 1 nebo 2).
  * **ZÁSADNÍ ROZHODNUTÍ (dle feedbacku vedoucího):** Ve finálním Stata modelu byla hodnota AI úrovně ponechána **čistě podle definice LLM** (bez jakéhokoliv manuálního přepisování (override) podle toho, jestli se našel daný skill ve slovníku). *Starší dokumenty v repozitáři navrhovaly průnik tieru a skillů jako filtr buzzwords, avšak od toho bylo zjevně upuštěno právě po feedbacku.*

## 4. Očištění dat (Data prep & Trimming)
* **Kvalita dat:** Smazány všechny inzeráty s LLM skóre jistoty (`confidence`) < 0.7 a inzeráty s původem před rokem 2024. Vyřazeny byly také ty, kde LLM vrátil "missing" klasifikaci.
* **Vzdělání a praxe:** 
  * `edu_ols` (5 úrovní, referenční je Bachelor).
  * `edu_logit` (3 úrovně, sloučeny High School a Associate, referenční je Bachelor+).
  * Chybějící (Missing) vzdělání / praxe je chápáno jako samostatná smysluplná kategorie a nevyhazuje se.
* **Platy (Mzdová analýza):**
  * Převod měn (EUR, INR) na roční USD dle zafixovaných kurzů z období sběru dat. Hodinové a měsíční mzdy přepočítány dle lokálních norem odpracovaných hodin (US: 2080, DE: 1607, IN: 1920).
  * Odstraněny nesmyslné "outliery": v IN smazáno vše pod $2,000/rok, v US/DE pod $3,000/rok. Maximum zastropováno na $500,000.
* **Baseline (Referenční proměnné):** 
  * Základní obor: Sektor J (Informační a komunikační činnosti).
  * Základní pozice (`job_family`): Software Engineer (neutrální střed).

## 5. Ekonometrické Modely (Stata - `ai_skills_thesis_final.do`)
Všechny modely jsou analyzovány symetricky pro USA, Německo a Indii. Standardní chyby (SE) jsou klastrované na úroveň firmy. Jsou reportovány průměrné marginální efekty (AME).

1. **Deskriptiva (Tabulka 1):** Tabulka výskytu AI tierů po zemích (rozložení % a absolutní N).
2. **Binární logit P(AI) (Tabulky 2 a 3):**
   * Sleduje pravděpodobnost, že inzerát požaduje AI dovednosti.
   * Modelovány odděleně: jednou závisí primárně na rodině pozice (`job_family`), podruhé čistě na sadě `skill_clusters` (aby nedošlo ke kolinearitě/pohlcení efektů – tyto dvě skupiny prediktorů spolu přirozeně silně korelují).
3. **Multinomiální logit (Tabulka 4):**
   * Tzv. "Reverzní inženýrství" AI Tierů (proč dal LLM inzerátu konkrétní Tier?).
   * Vysvětluje šance zařazení do tieru 1 a 2 vs 0 na základě skill clusterů.
   * *Rozhodnutí:* Z tohoto modelu bylo záměrně vyřazeno "Vzdělání", protože v Německu a Indii je příliš málo inzerátů kombinujících "Applied/Core AI" a nižší vzdělání (riziko tzv. quasi-complete separation risk).
4. **OLS Mzdová regrese (Tabulka 5):**
   * Závislá proměnná: `ln(salary)`. 
   * Hlavní vysvětlující proměnné: Skill clustery + `ai_level` (AI prémie).
   * **ZÁSADNÍ METODOLOGICKÉ ROZHODNUTÍ:** Clustery `Generative AI` a `Data Science/ML` byly z finální specifikace pro OLS mzdy **záměrně vyřazeny**. Důvod: Tyto dovednosti de facto konstruují definici `Applied/Core AI` tieru. Kdyby byly ponechány v modelu současně s tierem, hrozila by cirkularita a proměnné by na sebe "kanibalizovaly" mzdovou prémii. Interpretace AI koeficientu v tabulce tedy znamená: *mzdová prémie za AI roli, která už implicitně obsahuje tyto úzce zaměřené AI technologie*.
5. **Robustness checky a kontroly:** (Níže podrobněji rozepsáno jako Sekce 6, jelikož jde o klíčovou obranu modelu).

---

## 6. Způsob prezentace výsledků (Inkrementální narativ)
Je extrémně důležité pro čtení Stata kódu a finálního textu vědět, že samotný text práce (kapitoly 5.3 až 5.5) prezentuje modely jako "příběh" budovaný od jednoduchého ke složitému. Z důvodu přehlednosti **je tento narativní postup postaven exkluzivně na americkém podvzorku (`country == "US"`)**. 

1. **Logit modely (M1 -> M2 -> M3):**
   * **M1:** Základní model (jen kontroly a `job_family`).
   * **M2:** Přidání hrubých IT dovedností.
   * **M3:** Finální, nejkomplexnější model se všemi specifickými AI skill clustery.
2. **OLS mzdové modely (Model A -> B -> C):**
   * **Model A:** Jen kontrolní proměnné (firmy, vzdělání) + `ai_level`.
   * **Model B:** Přidání `job_family`, aby se ukázalo, jak se prémie změní, když zohledníme typ práce.
   * **Model C:** Přidání *skill clusterů* (bez GenAI a ML, viz výše), což je finální specifikace odpovídající Tabulce 5.

Teprve po tomto inkrementálním vysvětlení na datech USA (sekce 13 v kódu) přistupuje práce k oněm symetrickým mezistátním srovnáním (Tabulky 2–5).

---

## 7. Testy Robustnosti a Kontroly (Přílohy)
Aby výsledky regresních modelů nebyly napadnutelné z hlediska zkreslení dat (bias) nebo metodologických voleb, obsahuje kód 4 detailní testy robustnosti (Přílohy A až D):

### A. Heckman Selection Model (Řešení Selection Biasu platů)
* **Problém:** Zejména v Německu je podíl inzerátů s uvedeným platem extrémně nízký (~8 %). Firmy, které uvádějí plat, nejsou náhodným vzorkem (může jít jen o velké korporace nebo určité sektory), což by OLS mzdový model statisticky zkreslilo (tzv. *selection bias*).
* **Řešení:** Heckmanův model to řeší ve dvou fázích. První fáze (Probit) modeluje už jen pouhou *pravděpodobnost, zda firma vůbec do inzerátu plat napíše*. Z toho se spočítá korekční faktor (Inverse Mills Ratio), který se vloží do druhé fáze (samotné mzdové OLS regrese). Slouží k ověření, zda se mzdová AI prémie dramaticky nezmění, když se tento bias ošetří.
* **Výsledek (TL;DR):** Signifikantní bias (IMR) se prokázal jen v Indii. I po jeho matematickém odfiltrování ale mzdová "AI prémie" neochvějně zůstala silná a signifikantní ve všech 3 zemích (US: +12,6 %, DE: +14,7 %, IN: +11,5 %).

### B. OLS s vrácenými GenAI/ML clustery (Test Cirkularity)
* **Problém:** Z hlavní mzdové OLS regrese byly vyřazeny dovednostní clustery `Generative AI` a `Data Science / ML`, aby nekanibalizovaly hlavní proměnnou `ai_level` (protože LLM určuje AI tier z velké části právě podle těchto dovedností, čímž vzniká matematická cirkularita).
* **Řešení:** V Příloze C je OLS model spuštěn znovu, ale s těmito dvěma clustery vrácenými zpět do rovnice. Cílem je dokázat stabilitu modelu – pokud se koeficient pro "AI prémii" nezhroutí, znamená to, že odměna za AI práci odráží celkovou komplexitu role, a není to jen prémie za pouhou znalost jednoho izolovaného GenAI nástroje.
* **Výsledek (TL;DR):** Koeficienty pro `ai_level` se po vrácení dovedností nezhroutily. Model je stabilní a AI prémie skutečně odráží komplexitu práce.

### C. US OLS s regionálními fixními efekty (Region FE)
* **Problém:** USA je obrovský trh a platy se drasticky liší region od regionu. V Silicon Valley (Západ) jsou platy vyšší a zároveň je tam více AI pozic. Mzdová "AI prémie" by tak mohla být jen skrytou "Silicon Valley prémií". V hlavním OLS modelu regiony nejsou, aby byl model symetrický s Německem a Indií (kde taková data chybí).
* **Řešení:** V Příloze D je spuštěn US model znovu s přidáním *4 US Census regions* (Západ, Středozápad, Jih, Severovýchod) jako fixních efektů. Tím se odfiltruje vliv drahých regionů a ukáže se očištěná "AI prémie".
* **Výsledek (TL;DR):** Mzdová prémie v USA zůstala statisticky významná i po odfiltrování těch nejdražších hubů (Západ/SV). Znamená to, že AI vynáší víc peněz celostátně, ne jen v Kalifornii.

### D. Cross-country testy (Pooled modely s interakcemi)
* **Problém:** Všechny tabulky modelují země (USA, DE, IN) odděleně. Vidíme sice, že USA má koeficient např. 0.15 a Indie 0.08, ale *nevíme, zda je ten rozdíl statisticky významný*.
* **Řešení:** Všechna data jsou v Příloze B naházena do jednoho obřího "pooled" modelu a přidají se interakční členy (např. `ai_level` × `země`). Přes tzv. Waldovy testy (testparm) Stata matematicky ověří, zda se trhy práce mezi těmito třemi zeměmi fundamentálně a signifikantně liší.
* **Výsledek (TL;DR):** Waldovy testy signifikantně zamítly nulovou hypotézu, což tvrdě (matematicky) dokazuje, že mezi americkým, německým a indickým AI trhem jsou strukturální rozdíly.

---
*Všechna výše uvedená rozhodnutí a testy jsou plně implementovány v kódu. Pokud bude potřeba argumentovat detaily, je to výslovně uvedeno v komentářích v `ai_skills_thesis_final.do` a `pipeline.py`.*
