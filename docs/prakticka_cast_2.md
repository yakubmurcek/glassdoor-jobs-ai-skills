# 5 Výsledky empirické analýzy

Tato kapitola shrnuje empirická zjištění o poptávce po AI dovednostech v IT pracovních inzerátech ve třech zemích — Spojených státech amerických, Německu a Indii. Výklad postupuje od deskriptivního výskytu AI požadavků (kapitola 5.1) k modelům, které identifikují, s jakými charakteristikami pozic a firem jsou tyto požadavky asociovány (kapitoly 5.2 a 5.3), a uzavírá se analýzou vztahu mezi AI úrovní a inzerovanou mzdou (kapitola 5.4). Všechny logistické a multinomiální modely reportují průměrné marginální efekty (AME) s robustními standardními chybami clusterovanými na firmu; mzdové modely pracují s přirozeným logaritmem mzdy ln(mzda). Metodické detaily (filtrace vzorku, definice proměnných, diagnostika modelů) jsou popsány samostatně v metodické kapitole a zde se již neopakují.

## 5.1 Výskyt AI požadavků

První empirická otázka práce je deskriptivní: jak časté jsou AI požadavky v IT inzerátech a jak se jejich prevalence liší mezi třemi zkoumanými trhy? V souladu s taxonomií použitou v celé práci rozlišujeme tři vzájemně vylučující úrovně — žádný AI požadavek (*None*), **AI Integration** (pozice, která integruje hotové AI nástroje, generativní AI, API služby a prompt engineering do běžné práce) a **Applied/Core AI** (pozice, která se přímo podílí na vývoji a nasazování AI systémů, strojového učení nebo modelů). Tabulka 1 shrnuje absolutní počty a sloupcové podíly těchto úrovní v rámci každé země.

**Tabulka 1 — Výskyt úrovní AI v IT inzerátech podle země**

| Úroveň AI | USA N (%) | Německo N (%) | Indie N (%) |
|---|---:|---:|---:|
| None | 14 181 (79,45 %) | 5 232 (81,72 %) | 13 294 (93,71 %) |
| AI Integration | 2 354 (13,19 %) | 622 (9,72 %) | 394 (2,78 %) |
| Applied/Core AI | 1 313 (7,36 %) | 548 (8,56 %) | 498 (3,51 %) |
| **Celkem** | **17 848 (100,00 %)** | **6 402 (100,00 %)** | **14 186 (100,00 %)** |

*Zdroj: vlastní výpočet na základě scrapovaných inzerátů Glassdoor, Stata výstup z finálního běhu 16. 4. 2026 (Tabulka_1_Vyskyt_AI.rtf). N = počet inzerátů v dané úrovni, % = sloupcové procento — součet v rámci země = 100 %.*

Souhrnně vyžaduje nějakou úroveň AI dovedností přibližně **20,6 % amerických**, **18,3 % německých** a pouze **6,3 % indických** inzerátů. Hlavní příběh tabulky tedy není binární rozdíl "USA vs. zbytek", nýbrž jednoznačná dvouúrovňová struktura: rozvinuté západní trhy (USA, Německo) se pohybují ve velmi podobném pásmu celkové AI prevalence okolo jedné pětiny inzerátů, zatímco indický trh je zhruba **třikrát nižší**. Jakékoli další srovnání mezi zeměmi má smysl interpretovat právě na tomto pozadí — ne jako spojité rozdíly, ale jako výrazný strukturální zlom mezi dvěma skupinami trhů.

Při pohledu na kompozici AI požadavků se však USA a Německo přece jen znatelně liší. V USA dominuje **AI Integration** (13,2 %), tedy pozice, které s AI pracují spíše na aplikační úrovni — integrace generativních AI nástrojů, využívání hotových modelů přes API, prompt engineering. Applied/Core AI je sice stále významná kategorie (7,4 %), ale tvoří pouze přibližně třetinu všech AI inzerátů. V Německu je poměr zjevně vyrovnanější: AI Integration (9,7 %) a Applied/Core AI (8,6 %) jsou si téměř rovny. Jinými slovy, pokud se německý inzerát vůbec dotkne AI, je o něco pravděpodobnější, že jde o pozici s hlubším vývojářským či výzkumným zaměřením, než je tomu v USA. Tento vzorec je konzistentní s průmyslovým profilem německého trhu (inženýrsky orientované obory, kde má AI přesah do výzkumu a vývoje), zatímco americký trh odráží širší a rychlejší adopci komerčních AI nástrojů napříč běžnými IT rolemi. Jakékoli silnější kauzální tvrzení by však přesahovalo rámec deskriptivního zjištění a je zde formulováno pouze jako pracovní interpretační hypotéza.

Indie se odlišuje od obou předchozích trhů řádově. Celkový podíl AI požadavků 6,3 % je výrazně pod hodnotou obou rozvinutých trhů, přičemž největší rozdíl se objevuje v kategorii **AI Integration** (2,8 % v IN proti 13,2 % v US a 9,7 % v DE). Kategorie Applied/Core AI je v Indii relativně silnější vůči AI Integration (3,5 % proti 2,8 %), což je opačná kompoziční struktura než v USA. Jednou z pravděpodobných rovin vysvětlení je profil indického IT sektoru v našem vzorku, který je v Glassdoor datech silně zastoupen servisními, outsourcingovými a implementačními pozicemi, kde se AI nástroje v textech inzerátů objevují pomaleji než u vývojově-produktových firem. Opět však ponecháváme tuto úvahu v poloze opatrné interpretace — detailnější dekompozici podle profesní skupiny (*job family*) a firemních charakteristik poskytují až logistické modely v následující podkapitole.

Deskriptivní zjištění této podkapitoly lze shrnout do tří pozorování. Zaprvé, AI požadavky zůstávají i na nejpokročilejších trzích menšinovým jevem — **čtyři z pěti IT inzerátů v USA i Německu se žádné AI dovednosti explicitně nedotknou**. Zadruhé, rozdíl mezi USA a Německem není v celkové prevalenci, ale v kompozici — americký trh tíhne k integračnímu (aplikačnímu) využití AI, německý trh ke specializovanějším vývojovým rolím. Zatřetí, indický trh je v rozsahu AI poptávky řádově níže než oba rozvinuté trhy a propad je zvlášť výrazný u AI Integration.

Samotný deskriptivní výskyt však neříká, **s čím** AI požadavky korelují — zda se koncentrují v určitých profesních skupinách, zda je lze vysvětlit technologickým profilem pozice (skill clustery) nebo zda jsou spjaty s firemními charakteristikami (velikost, sektor). Právě tyto otázky adresují logistické modely v kapitole 5.2.

## 5.2 Determinanty AI požadavku — binární logit

Pro identifikaci faktorů asociovaných s přítomností AI požadavku (binární proměnná `has_ai` = 1 pro AI Integration nebo Applied/Core AI, 0 jinak) jsou odhadnuty dva **plné, vzájemně komplementární logit modely** pro každou ze tří zemí — celkem tedy šest sloupců jedné souhrnné tabulky.

<!-- TODO: Po merge v `ai_skills_thesis_final.do` (duben 2026) je "QA & Testing" sloučeno do kategorie "Other" kvůli sparse cells v DE/IN mlogit. Před odevzdáním zaměnit "QA & Testing" za jinou kategorii, která ZŮSTALA separátní — např. "Management" nebo "DevOps & Cloud" již v seznamu je, takže stačí smazat ", QA & Testing". Nové kategorie job_family: Data & AI, DevOps & Cloud, Management, Software Developer, Software Engineer, Sr+ Software Engineer, Other. -->
- **Model A — Profesní pohled:** `has_ai` ~ job family + kontroly. Cílem je zachytit, jak se AI poptávka liší mezi profesními rodinami (např. Data & AI, Software Engineer, DevOps & Cloud, QA & Testing).
- **Model B — Dovednostní pohled:** `has_ai` ~ skill clustery + stejné kontroly (bez job family). Cílem je posoudit, které technologické dovednosti (cloud, webové frameworky, enterprise systémy, databáze atd.) s AI pozicemi koreluji.

V obou modelech jsou jako kontroly zahrnuty: velikost firmy, typ organizace, sektor (NACE), remote pracovní režim, **vzdělání ANO/NE** (dichotomizace: alespoň bakalář / nižší nebo neuvedeno) a **praxe ANO/NE** (dichotomizace: vyžadována seniornější praxe / ne). Kontrolní proměnné nejsou předmětem substantivní interpretace, jejich role je pouze ošetřit confounding. Všechny modely jsou plné (žádné inkrementální vrstvení) a reportují **průměrné marginální efekty (AME)** v procentních bodech, s robustními standardními chybami clusterovanými na úrovni firmy.

Job family je detailně prozkoumána právě v Modelu A; následné modely v práci se zaměřují na skill clustery, které jsou pro výzkumnou otázku diplomové práce podstatnější (práce zkoumá AI *dovednosti*, nikoli *profese*).

<!-- TABULKA 2 — Binární logit has_ai ~ (job family | skill clustery) + kontroly, 6 sloupců (3 země × 2 modely), AME; doplnit po novém Stata runu. -->

## 5.3 Determinanty úrovně AI — multinomiální logit

Binární logit v kapitole 5.2 pracuje s dichotomickou proměnnou `has_ai`, a slévá tak dvě kvalitativně odlišné úrovně AI do jedné. Multinomiální logistický model v této podkapitole tento rozpor rozepisuje zpět: vysvětlovanou proměnnou je tříhodnotová `ai_level` (None / AI Integration / Applied/Core AI), baseline tvoří kategorie *None*. Prediktory jsou **skill clustery** (bez job family, která již byla detailně rozebrána v kapitole 5.2) plus tytéž kontroly ANO/NE pro vzdělání a praxi a firemní charakteristiky.

Souhrnná tabulka obsahuje devět sloupců: **3 země × 3 AI tiery**. V rámci každé země je uveden AME skill clusteru pro pravděpodobnost, že inzerát skončí v dané AI úrovni (oproti referenční kategorii *None*). Tato struktura umožňuje odpovědět na tři provázané otázky současně — (i) *které dovednosti vůbec zvyšují šanci na AI požadavek*, (ii) *mění se tento profil, když místo integračních AI rolí sledujeme hlubší Applied/Core AI*, a (iii) *je tento vzorec univerzální napříč USA, Německem a Indií, nebo se liší*.

<!-- TABULKA 3 — Mlogit ai_level ~ skill clustery + kontroly, 9 sloupců (3 země × 3 tiery), AME; doplnit po novém Stata runu. -->

## 5.4 Mzdy a AI úroveň

Poslední část kapitoly se obrací k otázce, která z ekonomického hlediska z práce nejvíce vyplývá: jak je úroveň AI požadavku asociována s inzerovanou mzdou. Formulace je zde záměrně opatrná — nemluvíme o "přínosu" AI dovedností na mzdu v kauzálním smyslu, pouze o **systematické asociaci** mezi AI úrovní a pozorovanou výší inzerovaného ln(mzdy) po kontrole dalších charakteristik pozice.

### 5.4.1 Deskripce mzdy podle AI úrovně

Nejprve je prezentováno deskriptivní rozložení ln(mzdy) podle AI úrovně v každé zemi — mediány, kvartily a grafické zobrazení distribuce. Cílem této pod-sekce je zobrazit hrubý vizuální rozdíl před kontrolou ostatních faktorů; jakékoli závěry o velikosti AI "prémie" patří až do regresní sekce 5.4.2.

<!-- GRAF 1 — Kernelová hustota ln(mzdy) podle úrovně AI, po zemích (Graf_Mzda_AI_US.png, Graf_Mzda_AI_DE.png, Graf_Mzda_AI_IN.png). Doplnit po novém Stata runu. -->

### 5.4.2 Plný OLS model ln(mzdy)

Pro kvantifikaci čisté asociace mezi AI úrovní a ln(mzdou) je odhadnut **plný OLS model** pro každou zemi samostatně: ln(mzda) ~ AI úrovně (dummy AI Integration, Applied/Core AI; baseline = None) + skill clustery + kontroly (velikost firmy, typ, sektor, region, remote, vzdělání ANO/NE, praxe ANO/NE). Tabulka obsahuje tři sloupce (USA, Německo, Indie), v řádcích jsou **skills a AI tiery** jako klíčové vysvětlující proměnné; kontrolní proměnné jsou v tabulce zobrazeny pouze jako ANO/NE indikátory, bez detailní interpretace.

Formulační pravidlo pro tuto sekci: interpretujeme, zda se inzerovaná ln(mzda) vyšší úrovně AI systematicky *liší* od úrovně *None* a v jakém směru, nikoli "kolik AI dovednosti přinášejí". Srovnání mezi zeměmi je rovněž kvalitativní — sledujeme, **zda** se určitý skill cluster projeví významně v jedné zemi a ne v jiné, nikoli abychom sestrojovali kardinální žebříček "výnosnosti" dovedností.

<!-- TABULKA 4 — OLS ln(mzda) ~ AI tiery + skill clustery + kontroly, 3 sloupce (US/DE/IN); doplnit po novém Stata runu. -->
