# Poznámky k metodologii a modelům

Tento dokument slouží jako podklad pro sepsání metodologické a analytické části diplomové práce. Shrnuje, jak jsou koncipovány regresní modely, jak vznikají hlavní sledované proměnné a jakým způsobem jsou data dolována z původního datasetu.

## 1. Architektura regresních modelů (Hierarchický OLS)

Analýza mzdové prémie (wage premium) za AI dovednosti je stavěna jako hierarchická OLS regrese. Modely záměrně přidávají vysvětlující proměnné postupně po blocích. Cílem je demonstrovat čistý (izolovaný) efekt AI na mzdu po odstranění tzv. zkreslení vynechanou proměnnou (omitted variable bias). 

*   **Model A (Základní / Firemní profil):** Zahrnuje pouze charakteristiky firmy (Velikost, sektor, typ, lokalita). Řeší námitku, zda vyšší plat u AI pozic není pouhým důsledkem toho, že AI poptávají primárně velké a bohaté tech korporáty.
*   **Model B (Lidský kapitál):** Přidává zkušenosti a vzdělání. Izoluje vliv seniority – tzn. srovnává platy u stejně seniorních lidí. Obvykle způsobuje pokles zdánlivé prémie za AI, protože očišťuje zásluhy, které patřily samotné praxi.
*   **Model C (Plný model s technologiemi):** Závěrečný model. Přidává typ role a specifické dovednosti (skill clusters). Jde o test "nejtvrdšího zrna": odpovídá na otázku, zda existuje platový bonus čistě za `ai_level` i přesto, že model kontroluje přítomnost příbuzných technologií (GenAI, ML, Cloud).

### Over-controlling a Citlivostní test modelu (Sekce 6.6)
V Modelu C dochází ke specifickému designovému rozhodnutí: vstupuje do něj souběžně jak proměnná úrovně AI (`ai_level`), tak i deterministické skill clustery, které s ní silně korelují (např. `cluster_generative_ai` a `cluster_data_science__ml`). 
Tento postup mírně snižuje finální koeficient `ai_level` (působí jako mediátor), jelikož samotné technické clustery část mzdové prémie vyčerpají. Projekt na to korektně reaguje v sekci **6.6 (Citlivostní analýza)**, kde jsou tyto korelované clustery z modelu vyjmuty. Tím `ai_level` získá zpět plnou absorpční schopnost a ukazuje čistou kumulativní prémii aplikované umělé inteligence.

## 2. Hybridní strategie extrakce (AI Tier vs. Skill Clustery)

Vzhledem ke složitosti a nuancím ohledně umělé inteligence (pouhé "promptování" vs. reálné nasazování a inženýrství) nevyužívá práce pouze hloupé klíčové vyhledávání slova "AI". Zvolen byl "hybridní režim", jehož logika je implementována v `ai_skills/pipeline.py`:

*   **Skill Clustery (Tvrdá data z dictionary):** Slouží pro exaktní tech-stack pozice (zda role požaduje SQL, Python, TensorFlow atd.). Skript prohledává inzerát a deterministicky mapuje technologie s využitím standardizovaného slovníku (`ai_skills/skills_dictionary.py`). Tím je vyloučena halucinace u konkrétních nástrojů.
*   **AI Tier / ai_level (Sémantický kontext přes LLM):** Deterministický slovník ovšem nedokáže přečíst kontext (neodliší uživatele AI od vývojáře AI). Text inzerátu a název pozice je proto předložen velkému jazykovému modelu (OpenAI/GPT). Ten má za úkol analyzovat intenci inzerátu a zatřídit roli do pevných úrovní (`none`, `ai_integration`, `applied_ai`).

Syntéza obou přístupů zaručuje, že zachycujeme jak exaktní zmínky frameworků, tak hlubší HR význam inzerátu.

## 3. Původ dat z raw Glassdoor CSV

Z původních naparsovaných Glassdoor dat skript (v `ai_skills/pipeline.py`) těží ze všech možných metadat pro co největší výtěžnost informací.

Pro určení technologií a úrovní AI se využívají rovnou 4 nativní sloupce:
1.  `job_desc_text`: Hlavní hrubý text inzerátu. Čte z něj LLM pro určení AI tieru a prohledává ho dictionary engine pro zachycení skillů.
2.  `skills`: Sloupec předvyplněných štítků přímo z Glassdooru/LinkedInu. Data z něj pipeline chytře vytěžuje (pokud autor inzerátu neuvedl skill přímo v textu, ale naklikal štítky) a slučuje do obřího pole pro deterministické skill clustery (`cluster_*`).
3.  `job_title`: Extrahováno kvůli kontextu pro jazykový model, aby lépe rozeznal např. Data Scientista od Marketéra i přes podobný popis. Následně se z regexu tvoří i typový kód pozice (`job_family`).
4.  `educations`: Glassdoor metadata sloužící k čisté extrakci deterministického vzdělání (`edu_level_det`) a částečně jako background pro LLM kontext úsudek.

## 4. Poznámky k binárnímu logitu (§6A) — pro text práce

### 4.1 Linktest (misspecifikace)
Linktest M3 ukazuje signifikantní `_hatsq` (p = 0.006, koeficient = -0.052). To formálně naznačuje, že funkcionální forma modelu není plně správná — mohou chybět interakce nebo nelineární členy. Praktický dopad je však omezený: koeficient je malý a Hosmer-Lemeshow test (p = 0.496) ani AUC (0.747) nenaznačují špatný fit. **V textu práce uvést v limitacích** — např. "Linktest naznačuje mírnou misspecifikaci funkcionální formy (p = 0.006), která může odrážet chybějící interakční efekty mezi skill clustery a regionem. Hosmer-Lemeshow test (p = 0.50) a adekvátní diskriminace (AUC = 0.75) nicméně potvrzují přijatelnou prediktivní schopnost modelu."

### 4.2 Nízká senzitivita klasifikace (17%)
Klasifikační tabulka M3 ukazuje celkovou přesnost 80.5 %, ale senzitivitu jen 17.15 % (specificita 96.9 %). Důvodem je nevyváženost dat — AI pozice tvoří ~21 % vzorku, takže při cut-off 0.5 model konzervativně klasifikuje většinu jako non-AI. **V textu zdůraznit**, že cílem logitu není klasifikace, ale odhad průměrných marginálních efektů (AME). Klasifikační tabulka je reportována pro úplnost diagnostiky, nikoli jako hodnotící kritérium modelu.

### 4.3 Systematické efekty Unknown/Missing kategorií (MNAR)
V M3 jsou kategorie "Unknown" a "Missing" systematicky signifikantní:
- Unknown region: AME = +5.0 pp (p < 0.001)
- Missing education: OR = 1.35, AME = +4.3 pp (p < 0.001)
- Unknown firma size: OR = 0.77, AME = -3.6 pp (p = 0.004)

To naznačuje, že chybějící hodnoty nejsou náhodné (MNAR — Missing Not At Random). **V textu diskutovat**: např. firmy, které neuvádějí velikost, mohou být systematicky menší startupy; pozice bez uvedeného vzdělání mohou být technicky náročnější role, kde se vzdělání nepovažuje za relevantní. Missing-indicator metoda (Allison, 2001) je standardní přístup a je podpořena robustnostním testem M3 na podvzorku se známým vzděláním (6A.5), kde se koeficienty ostatních prediktorů zásadně nemění.

### 4.4 Interpretační framing logitu
Vedoucí doporučuje interpretovat logit jako "ekonometrickou analýzu odborné náročnosti" — model ukazuje, které dovednosti a charakteristiky pozic korelují s požadavkem na AI. **V textu explicitně zdůraznit**:
- Model je exploratorní/korelační, nikoliv kauzální
- Skill clustery jsou extrahovány ze stejného textu inzerátu, ze kterého LLM přidělil `has_ai` — proto nejde hovořit o kauzalitě
- Hlavní přínos modelu: kvantifikace, které technologické profily jsou asociovány s AI požadavky (modern cloud/web/data = pozitivní korelace, legacy/scripting/enterprise = negativní)
- Toto doplňuje deskriptivní analýzu odborné náročnosti (§5.1) o formální ekonometrický test

### 4.5 Nesignifikance typu firmy (type_cat)
Typ firmy (Private/Public/Nonprofit/Gov) je ve všech třech modelech (M1-M3) nesignifikantní. V textu zdůvodnit ponechání v modelu: proměnná je zahrnuta jako kontrolní faktor na základě teoretické motivace (vlastnická struktura může ovlivňovat investice do AI), její nesignifikance je sama o sobě analytický výsledek — rozhoduje "co se na pozici dělá", nikoliv typ zaměstnavatele.
