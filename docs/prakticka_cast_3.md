# 5 Výsledky empirické analýzy — komparativní verze (§5.2 a dále)

_Poznámka: sekce 5.1 (deskriptivní statistika) zůstává beze změny z předchozí verze práce. Tento soubor obsahuje přepracované sekce 5.2 až 5.4 v komparativní podobě, založené přímo na per-country Tabulkách 2, 3 a 4 z finálního běhu analýzy._

## 5.2 Determinanty AI požadavku (binární logit)

Jedním z klíčových cílů této analýzy je zjistit, jaké dovednostní profily a charakteristiky souvisejí s požadavkem na AI dovednosti. K tomuto slouží binární logistická regrese s vysvětlovanou proměnnou has_ai, která nabývá hodnoty 1 pro inzeráty v kategorii AI Integration nebo Applied/Core AI, 0 pak pro inzeráty bez AI požadavku. Model byl odhadnut samostatně pro každou ze tří zkoumaných zemí, tedy pro USA, Německo a Indii.

Pro komparativní strukturu byl binární logit rozdělen do dvou tabulek. Tabulka 2 zahrnuje profesní skupinu (job family) spolu s kontrolními proměnnými (vzdělání, zkušenosti, NACE sektor, typ organizace, velikost firmy, region u USA a možnost remote práce). Tabulka 3 zobrazuje místo profesní skupiny 21 skill clusterů, včetně clusterů Generative AI a Data Science / Machine Learning. Tyto dva clustery byly z mzdového modelu vyřazeny kvůli cirkularitě s klasifikací ai_level (viz sekce 5.4), v binárním logitu jsou však ponechány záměrně, jelikož nejvýrazněji ilustrují, do jaké míry přítomnost konkrétních AI dovedností v textu inzerátu souvisí s AI klasifikací. Pro výsledky jsou standartně použity průměrné marginální efekty (AME), tedy změny pravděpodobností AI požadavku (v procentních bodech).

_Tabulka 2 Binární logit, profesní skupina (ref.: Software Engineer)_

| Kategorie                |          USA |      Německo |        Indie |
| ------------------------ | -----------: | -----------: | -----------: |
| Data & AI                | +27,1 \*\*\* | +59,3 \*\*\* | +34,0 \*\*\* |
| DevOps & Cloud           |  −9,8 \*\*\* |      −1,1 ns |  −5,6 \*\*\* |
| Management               |      +1,9 ns |    +6,5 \*\* |      +1,2 ns |
| Other                    | −12,6 \*\*\* |    −5,3 \*\* |  −6,2 \*\*\* |
| Software Developer       | −10,6 \*\*\* |      −1,1 ns |  −5,8 \*\*\* |
| Senior Software Engineer |      +2,0 ns |  +8,0 \*\*\* |      −1,7 ns |

_Tabulka 3 Binární logit, skill clustery (ref.: Software Engineer)_

| Cluster                  |          USA |      Německo |        Indie |
| ------------------------ | -----------: | -----------: | -----------: |
| Generative AI            | +32,8 \*\*\* | +36,2 \*\*\* | +12,3 \*\*\* |
| Data Science / ML        | +28,4 \*\*\* | +16,6 \*\*\* |  +7,1 \*\*\* |
| Dynamic Web              |  +5,6 \*\*\* |  +7,2 \*\*\* |  +2,8 \*\*\* |
| Cloud Computing          |  +4,1 \*\*\* |    +2,9 \*\* |    +1,1 \*\* |
| Data Engineering         |    +1,8 \*\* |  +4,6 \*\*\* |      +0,7 \* |
| BI & Analytics           |  +2,5 \*\*\* |  +3,7 \*\*\* |    +0,9 \*\* |
| Frontend Development     |  +2,8 \*\*\* |      +0,7 ns |      +0,5 ns |
| Enterprise Platforms     |      +2,8 \* |      +2,1 ns |      +0,3 ns |
| DevOps & Containers      |      +0,6 ns |    +2,5 \*\* |      +0,0 ns |
| Backend Development      |      +0,9 ns |      +2,3 \* |      −0,1 ns |
| Systems Programming      |    +1,9 \*\* |      +1,4 ns |      +0,2 ns |
| Architecture & Methods   |      +0,6 ns |      +0,5 ns |      +0,1 ns |
| Security & Identity      |      −1,2 ns |      −2,4 \* |      +0,6 ns |
| Mobile & Desktop         |      −0,3 ns |      −0,6 ns |    −1,8 \*\* |
| Testing / QA & Debugging |      −1,0 \* |      −1,4 ns |      −0,1 ns |
| Databases & Storage      |    −1,7 \*\* |      −1,9 \* |      −0,6 ns |
| Networking               |      −0,5 ns |    −4,1 \*\* |      −0,6 ns |
| OS & Embedded            |    −2,4 \*\* |  −4,4 \*\*\* |  −2,0 \*\*\* |
| Certifications           |      −3,0 \* |      −5,4 \* |      −0,6 ns |
| Scripting / Shell        |  −3,2 \*\*\* |    −5,2 \*\* |      −1,6 \* |
| Enterprise / Managed     |  −4,5 \*\*\* |    −2,6 \*\* |      −0,3 ns |
| N                        |       17 848 |        6 402 |       14 186 |
| Pseudo R²                |        0,360 |        0,383 |        0,551 |

_Signifikance: \* p < 0,05; \*\* p < 0,01; \*\*\* p < 0,001; ns = nesignifikantní_

Jak Tabulka 3 napovídá, přítomnost Generative AI (GPT, LLM, Copilot), Data Science či Machine Learning dovedností (TensorFlow, PyTorch, scikit-learn) je ve všech třech zemích nejsilnějším pozitivním prediktorem AI požadavku. To je v souladu s očekáváním, jelikož tyto dva clustery jsou úzce spjaty se samotnou AI klasifikací a prakticky každý inzerát, který je zmiňuje, je zároveň klasifikován jako inzerát s AI požadavkem. Zajímavější a analyticky nosnější jsou proto výsledky u ostatních, necirkulárních skill clusterů jako jsou Cloud Computing, Data Engineering, Dynamic Web a BI / Analytics, které predikují AI požadavek konzistentně ve všech třech zemích. Jedná se o dovednosti spjaté s moderní cloudově-webovou infrastrukturou a s prací s daty. Tyto čtyři clustery lze považovat za technologické jádro, kolem kterého se AI poptávka ve všech třech zemích soustřeďuje. Naopak negativními prediktory jsou Scripting / Shell a OS / Embedded, tedy dovednosti spíše spjaté s údržbou infrastruktury. Na americkém trhu lze k negativním prediktorům přidat ještě cluster Enterprise / Managed, Certifications a Databases / Storage.

Řada skill clusterů se však mezi zeměmi chová odlišně, a právě tyto rozdíly přinášejí zajímavé poznatky o struktuře jednotlivých trhů. Frontend Development je silným pozitivním prediktorem AI požadavku v USA (+2,8 p. b.), v Německu ani v Indii však statisticky významný vliv nemá. Americký trh tedy častěji kombinuje webové UI dovednosti s AI rolemi, což souvisí s nasazováním AI funkcí do spotřebitelských webových aplikací typických pro americké firmy (chatboti, doporučovací systémy, personalizace). V Německu a Indii tyto dva světy zjevně tolik nesplývají. Cluster Mobile & Desktop poukazuje na negativní efekt pouze v Indii (−1,8 p. b.), což značí, že mobilní vývoj v Indii zůstává výrazně soustředěn spíše v tradičních rolích bez AI a funguje jako samostatná část trhu.

Cluster Data Engineering má v Německu podstatně silnější pozitivní efekt (+4,6 p. b.) než v USA (+1,8 p. b.) nebo v Indii (+0,7 p. b.). To naznačuje, že německý trh silněji propojuje datovou infrastrukturu s AI rolemi, pravděpodobně v důsledku vyšší koncentrace průmyslových a enterprise aplikací, které vyžadují robustní řešení datové infrastruktury. Podobný vzorec vykazuje cluster DevOps & Containers, který je pozitivně signifikantní pouze v Německu (+2,5 p. b.), u ostatních zemí je statisticky neutrální. Zajímavý je poměrně silný negativní efekt pro cluster Cluster Enterprise / Managed v USA (−4,5 p. b.) a Německu (−2,6 p. b.), v Indii jsou však statisticky neutrální.

Pozoruhodný je také celkový vzorec intenzit koeficientů napříč zeměmi. Koeficienty skill clusterů jsou v Německu typicky silnější než v USA a Indii. To může odrážet skutečnost, že AI pozice v Německu jsou koncentrovány v užším okruhu specializovaných rolí s velmi jasně vymezenými dovednostními požadavky, zatímco v USA AI prostupuje trhem rovnoměrněji. Indické koeficienty jsou naopak obecně slabší, což odpovídá jak nižšímu celkovému výskytu AI požadavků (6,3 %), tak i charakteru indického IT trhu, který je v průměru méně specializovaný a obsahuje vyšší podíl generalistů. Pseudo R² modelu dosahuje ve všech třech zemích hodnot mezi 0,36 a 0,55, přičemž nejvyšší je v Indii, což je dáno nízkou četností AI požadavků.

Na Tabulce 2 lze spatřit, že profesní skupina Data & AI je silně pozitivním prediktorem AI požadavku ve všech třech zemích, což samo o sobě není překvapivé. Síla tohoto efektu se však mezi zeměmi výrazně liší. Nejsilnější je v Německu (+59,3 p. b.), následuje Indie (+34,0 p. b.), a nejslabší je USA (+27,1 p. b.). To naznačuje, že na německém trhu existuje jasnější oddělení mezi datovými AI rolemi a zbytkem profesí. V USA je AI rozprostřena rovnoměrněji napříč profesemi, a to včetně běžných softwarových inženýrů. Kategorie Management a Senior Software Engineer jsou signifikantně pozitivní pouze v Německu (+6,5 p. b., resp. +8,0 p. b.). Profesní skupiny Other, Software Developer a DevOps & Cloud jsou s různou intenzitou ve všech třech zemích spojeny s nižší pravděpodobností AI požadavku než referenční profese Software Engineer.

Waldův test společné signifikance interakcí job_family × country v pooled logit modelu (Příloha B) formálně potvrzuje, že se profesní efekty mezi zeměmi statisticky liší (χ²(12) = 182,7; p < 0,001). Podobně Waldův test pro interakce klíčových skill clusterů (Cloud Computing, Data Science / ML, Backend Development) s proměnnou country potvrzuje signifikantní cross-country heterogenitu dovednostních efektů (χ²(6) = 62,75; p < 0,001). Obě zjištění tedy formálně doplňují kvalitativní pozorování popsaná výše.

## 5.3 Používání vs. vývoj AI (multinomický logit)

Binární logit v sekci 5.2 odpovídá na otázku, které profily souvisejí s AI požadavkem obecně. Stěžejním přínosem této práce je však rozlišení mezi pozicemi, které AI pouze integrují do svých procesů (AI Integration), a pozicemi, kde se AI přímo vyvíjí (Applied/Core AI). K tomuto účelu byl pro každou zemi odhadnut multinomický logistický model s třemi kategoriemi výsledné proměnné (None jako referenční, AI Integration, Applied/Core AI).

Specifikace multinomického logitu se od binárního logitu liší. Vzdělání (edu_logit) bylo z modelu vyřazeno kvůli nedostatečnému počtu pozorování v kombinaci HS / Associate × Applied/Core AI (v Indii 1 pozorování, v Německu 6), které porušuje pravidlo minimálně 50 pozorování na buňku pro stabilní MLE odhad. Profesní skupina byla z modelu rovněž vyřazena, jelikož u Applied/Core AI v Německu a Indii by kombinace s některými kategoriemi (Frontend & Design, QA & Testing, Security) opět způsobovala quasi-complete separation. Vliv vzdělání a profesní skupiny na AI požadavek obecně je však kontrolován v binárním logitu (sekce 5.2) a v mzdovém modelu (sekce 5.4). Multinomický model obsahuje všech 21 skill clusterů včetně Generative AI a Data Science / ML (důvod stejný jako u Tabulky 2) a kontrolní proměnné (zkušenosti, NACE sektor, typ organizace, velikost firmy, region u USA, remote práce).

**Tabulka 3 — Multinomický logit ai_level, AME v p. b. per země a tier**

| Cluster                  |  USA AI Int. |     USA App. | Německo AI Int. | Německo App. | Indie AI Int. |  Indie App. |
| ------------------------ | -----------: | -----------: | --------------: | -----------: | ------------: | ----------: |
| Generative AI            | +23,3 \*\*\* |  +9,9 \*\*\* |    +22,5 \*\*\* | +14,0 \*\*\* |   +7,0 \*\*\* | +5,2 \*\*\* |
| Data Science / ML        | +14,2 \*\*\* | +12,6 \*\*\* |       +2,4 \*\* | +12,3 \*\*\* |       +0,6 \* | +5,8 \*\*\* |
| Dynamic Web              |  +2,6 \*\*\* |  +3,0 \*\*\* |         +0,8 ns |  +6,1 \*\*\* |       +0,5 ns | +2,2 \*\*\* |
| Cloud Computing          |  +2,8 \*\*\* |  +1,4 \*\*\* |       +2,5 \*\* |      +0,6 ns |       +0,5 ns |     +0,6 ns |
| Data Engineering         |      −1,4 \* |  +3,2 \*\*\* |         +0,1 ns |  +4,4 \*\*\* |       −0,3 ns | +0,9 \*\*\* |
| BI & Analytics           |    +1,6 \*\* |      +0,8 \* |         +2,2 \* |      +1,6 \* |       +0,2 ns |     +0,5 \* |
| Frontend Development     |  +4,3 \*\*\* |    −1,1 \*\* |     +4,4 \*\*\* |  −3,2 \*\*\* |   +1,5 \*\*\* |     −0,7 \* |
| Enterprise Platforms     |  +4,4 \*\*\* |      −1,6 \* |         +3,0 ns |      −1,0 ns |       +0,2 ns |     +0,2 ns |
| DevOps & Containers      |      +0,8 ns |      −0,1 ns |         +0,4 ns |    +2,0 \*\* |       −0,3 ns |     +0,3 ns |
| Backend Development      |      +1,5 \* |      −0,5 ns |     +3,3 \*\*\* |      −0,8 ns |       +0,7 \* |     −0,7 \* |
| Systems Programming      |    −2,0 \*\* |  +3,6 \*\*\* |         −1,9 ns |  +3,1 \*\*\* |       −0,6 ns |     +0,7 ns |
| Architecture & Methods   |      +1,1 ns |      −0,6 ns |         +0,6 ns |      +0,2 ns |       +0,5 ns |     −0,4 ns |
| Security & Identity      |      −0,2 ns |      −1,0 \* |         −0,4 ns |      −1,7 ns |     +0,8 \*\* |     −0,2 ns |
| Mobile & Desktop         |      +0,1 ns |      −0,4 ns |         +0,8 ns |      −1,9 ns |       −0,1 ns |   −1,9 \*\* |
| Testing / QA & Debugging |      −0,3 ns |      −0,8 \* |         +1,1 ns |  −2,5 \*\*\* |       +0,3 ns |     −0,4 ns |
| Databases & Storage      |    −1,8 \*\* |      −0,0 ns |         −1,2 ns |      −0,9 ns |       −0,5 ns |     −0,3 ns |
| Networking               |      −0,1 ns |      −0,4 ns |         −2,8 \* |      −1,3 ns |       −0,4 ns |     −0,2 ns |
| OS & Embedded            |      −1,3 ns |      −1,1 \* |         −2,0 ns |      −2,0 \* |     −1,4 \*\* |     −0,5 ns |
| Certifications           |      −1,5 ns |    −2,1 \*\* |         −3,0 ns |      −2,7 ns |       +0,0 ns |     −0,7 ns |
| Scripting / Shell        |      −1,6 ns |      −1,6 \* |         −2,8 ns |      −2,6 \* |       +0,9 ns | −2,6 \*\*\* |
| Enterprise / Managed     |  −3,2 \*\*\* |  −1,6 \*\*\* |         −1,7 \* |      −1,2 ns |       −0,0 ns |     −0,3 ns |
| N                        |       17 848 |       17 848 |           6 402 |        6 402 |        14 186 |      14 186 |
| Pseudo R²                |        0,323 |        0,323 |           0,361 |        0,361 |         0,518 |       0,518 |

_AI Int. = AI Integration; App. = Applied/Core AI. Signifikance: \* p < 0,05; \*\* p < 0,01; \*\*\* p < 0,001; ns = nesignifikantní._

**Společné prediktory obou AI tierů**

Některé skill clustery zvyšují pravděpodobnost obou typů AI pozic ve všech třech zemích, avšak s různou intenzitou. Dynamic Web je konzistentně pozitivní pro Applied/Core AI ve všech zemích a pro AI Integration v USA, což potvrzuje, že moderní webové technologie jsou nedílnou součástí jak vývoje, tak nasazení AI aplikací. Cloud Computing predikuje AI Integration v USA a Německu, ale pro Applied/Core AI je pozitivní pouze v USA. To naznačuje, že cloudová infrastruktura je primárně prostředkem nasazení hotových AI služeb, méně samotného vývoje AI systémů. Cluster Enterprise / Managed a Scripting / Shell oba konzistentně snižují pravděpodobnost AI pozic ve všech zemích a u obou tierů, potvrzujíc, že tradiční legacy technologie jsou s AI nekompatibilní bez ohledu na úroveň AI zapojení.

**Diskriminátory mezi AI Integration a Applied/Core AI**

Analyticky nejnosnější jsou skill clustery, jejichž efekt se mezi AI tiery kvalitativně liší, případně směřuje na opačné strany. Tyto clustery umožňují rozlišit mezi používáním AI a samotným vývojem AI a představují hlavní analytický přínos multinomického modelu.

Cluster Data Engineering je napříč zeměmi nejčistším diskriminátorem ve prospěch Applied/Core AI. V USA (+3,2 p. b.), Německu (+4,4 p. b.) i Indii (+0,9 p. b.) je signifikantně pozitivní pro Applied/Core AI, zatímco pro AI Integration je ve všech zemích buď slabě negativní nebo statisticky neutrální. Správa datových pipeline, ETL procesů a datová architektura jsou tedy výhradně doménou AI vývoje, nikoliv integrace. Pozice využívající hotové AI API tuto dovednost nepotřebují, zatímco pozice vyvíjející AI modely ji vyžadují pro přípravu trénovacích dat.

Cluster Systems Programming, tedy nízkoúrovňové programování v jazycích jako C, C++ nebo Rust, vykazuje v USA statisticky významný opačný efekt mezi AI tiery. Snižuje pravděpodobnost AI Integration (−2,0 p. b.) a zároveň zvyšuje pravděpodobnost Applied/Core AI (+3,6 p. b.). V Německu je tento vzorec velmi podobný (Applied/Core AI +3,1 p. b.), v Indii však cluster ztrácí signifikanci u obou kategorií. Interpretace je konzistentní s americkou analýzou: nízkoúrovňové programování je klíčovou kompetencí pro vývoj AI infrastruktury (optimalizaci pipeline, GPU computing, nasazení AI modelů na zařízeních), zatímco s pouhou integrací AI nástrojů nejde dohromady.

Cluster Frontend Development je naopak napříč zeměmi konzistentně pozitivním prediktorem AI Integration (USA +4,3 p. b., Německo +4,4 p. b., Indie +1,5 p. b.) a nulovým nebo mírně negativním pro Applied/Core AI. Jedná se tedy o aplikační dovednost typickou pro pozice, kde se AI nasazuje do existujících uživatelských rozhraní. Podobný vzorec vykazuje Enterprise Platforms v USA (+4,4 p. b. pro AI Integration, −1,6 p. b. pro Applied/Core AI), v Německu a Indii však tento cluster ztrácí signifikanci u obou kategorií. Cluster Backend Development je slabším prediktorem AI Integration v USA a Indii, což odráží roli backendových vývojářů v integraci AI služeb přes API rozhraní.

**Robustnost třístupňového rozlišení AI napříč zeměmi**

Kvalitativně odlišné dovednostní profily AI Integration a Applied/Core AI se napříč zeměmi strukturálně opakují, ačkoliv s různou intenzitou koeficientů. Applied/Core AI pozice ve všech třech zemích vyžadují klíčové inženýrské dovednosti v datové infrastruktuře a systémovém programování, zatímco AI Integration pozice se vyznačují aplikačními dovednostmi ve frontendu, enterprise platformách a cloudovém nasazení. Tento kvalitativní rozdíl tedy není artefaktem amerického trhu ani LLM klasifikace, nýbrž obecně platným vzorcem, který je empiricky zachytitelný i v mezinárodním srovnání. Tím se potvrzuje, že třístupňové rozlišení AI požadavků přijaté v této práci není pouhou teoretickou typologií, ale má smysluplnou a robustní dovednostní strukturu.

Určitou výjimku představuje cluster Security & Identity, který v Indii vykazuje slabě pozitivní efekt pro AI Integration (+0,8 p. b., p < 0,01), zatímco v USA a Německu je statisticky neutrální nebo negativní. Lze to interpretovat jako rychlejší integraci AI do oblasti kybernetické bezpečnosti v indickém IT sektoru, pravděpodobně v kontextu outsourcingových služeb pro západní klienty, kde AI nástroje pro detekci hrozeb nacházejí rychlejší uplatnění.

**Limity multinomického logitu**

Základním předpokladem multinomického logitu je nezávislost irelevantních alternativ (IIA), jež se standardně ověřuje Hausmanovým testem. V modelech pro všechny tři země však tento test vrací zápornou testovou statistiku (pro USA χ²(37) = −102,1), což Stata explicitně označuje jako selhání asymptotických předpokladů testu. Důvodem je, že variančně-kovarianční matice rozdílů mezi dvěma srovnávanými modely není pozitivně definitní, což je známá slabina Hausmanova testu u multinomických modelů s větším počtem prediktorů (Long & Freese, 2014; Cheng & Long, 2007). Test tedy nelze interpretovat jako potvrzení ani zamítnutí IIA. Multinomický logit je z tohoto důvodu v této práci použit jako explorativní nástroj pro identifikaci kvalitativních dovednostních profilů. Vzory průměrných marginálních efektů (směr a relativní velikost efektů) jsou spolehlivější než přesné bodové odhady koeficientů, a právě na těchto vzorech stojí hlavní zjištění této kapitoly.

## 5.4 Mzdová prémie za AI (OLS)

Předchozí sekce ukázaly, jaký dovednostní a profesní profil s AI požadavky souvisí. Navazující otázka zní, jak se tyto AI kompetence promítají do mezd. K odpovědi byl odhadnut OLS regresní model vysvětlující přirozený logaritmus roční mzdy (ln_salary), v souladu s Mincerovou mzdovou rovnicí. Stejně jako v předchozích sekcích byl model odhadnut samostatně pro každou zemi. Specifikace obsahuje úrovně AI požadavku (ai_level), 19 skill clusterů bez clusterů Generative AI a Data Science / Machine Learning (tyto dva clustery jsou z OLS vyřazeny kvůli cirkularitě s ai_level), kategorické vzdělání (edu_ols, 5 úrovní), zkušenosti (exp_category, 4 úrovně), NACE sektor, typ organizace, velikost firmy a možnost remote práce. U amerického modelu je navíc zahrnuta proměnná region, která je pro Německo a Indii z modelu vypuštěna, jelikož odpovídá pouze americké administrativní struktuře.

Dopad vyřazení cirkulárních clusterů z hlavního modelu byl ověřen samostatnou citlivostní analýzou v Příloze C, která prezentuje OLS s plnou sadou skill clusterů. Výsledky ukazují, že zahrnutí Generative AI a Data Science / ML má na odhady AI prémie jen mírný dopad. V USA se koeficient AI Integration posouvá z +9,6 % na +7,6 % a Applied/Core AI z +14,0 % na +9,7 %. Oba koeficienty zůstávají vysoce statisticky významné (p < 0,001). Hlavní Model bez cirkulárních clusterů tedy dává čistší odhad mzdové hodnoty AI role jako takové, neboť nepřejímá mechanicky část variability, která pramení přímo z klasifikace ai_level. Rozdíl R² mezi oběma specifikacemi je jen 0,05 procentního bodu, což ukazuje, že hlavní závěr o existenci a velikosti AI prémie je robustní v obou variantách.

**Tabulka 4 — OLS ln(plat), mzdová prémie za AI per země**

| Proměnná        |          USA | Německo |      Indie |
| --------------- | -----------: | ------: | ---------: |
| AI Integration  |  +9,6 \*\*\* | +3,1 ns |    +9,0 \* |
| Applied/Core AI | +14,0 \*\*\* | +7,4 ns | +11,7 \*\* |
| N               |       14 642 |     514 |      9 735 |
| R²              |        0,375 |   0,272 |      0,148 |
| Mean VIF        |         1,40 |    1,82 |       1,59 |

_Hodnoty v %, koeficienty logaritmického modelu interpretovány jako přibližné procentuální změny mzdy. Signifikance: \* p < 0,05; \*\* p < 0,01; \*\*\* p < 0,001; ns = nesignifikantní. Model obsahuje všechny skill clustery kromě Generative AI a Data Science / ML, dále kategorické vzdělání, zkušenosti a firemní kontroly. U USA je navíc zahrnut region._

V USA činí AI prémie po kontrole všech pozorovatelných faktorů +9,6 % pro AI Integration a +14,0 % pro Applied/Core AI, přičemž oba koeficienty jsou vysoce statisticky významné. V Indii jsou odhady velmi podobné (AI Integration +9,0 %, Applied/Core AI +11,7 %), oba rovněž statisticky významné, ačkoliv s mírně nižší silou. Německo vykazuje bodově nižší hodnoty (AI Integration +3,1 %, Applied/Core AI +7,4 %), oba koeficienty jsou však statisticky nesignifikantní. Nižší signifikance německých odhadů souvisí s velmi malým vzorkem inzerátů s uvedenou mzdou (N = 514, pouhých 8 % z německého vzorku). Německý trh práce obecně inzeruje mzdu mnohem méně často než americký nebo indický, což limituje statistickou sílu mzdového modelu.

Pooled OLS model s interakcemi ai_level × country (Příloha B) formálně ověřuje, zda se AI prémie mezi zeměmi statisticky liší. Waldův test těchto interakcí (F(4; 12 686) = 2,56; p = 0,037) zamítá hypotézu o homogenitě AI prémie napříč zeměmi. Rozdíl je však tažen primárně Německem, jehož AI Integration prémie je oproti USA o 11,1 procentního bodu nižší (p = 0,005). USA a Indie jsou si naproti tomu v úrovni AI prémie statisticky rovny (interakce ai_level × IN nejsou signifikantní). Heterogenita AI prémie je tedy spíše slabší než silná a projevuje se především v rozdílu mezi německým a americko-indickým pásem.

**Robustnost vůči selekčnímu zkreslení**

Nízké pokrytí platu v Německu představuje potenciální zdroj selekčního zkreslení, jelikož firmy, které mzdu zveřejňují, mohou být systematicky odlišné od těch, které ji neuvádějí. Tento problém byl ověřen Heckmanovým selekčním modelem (Příloha A), který odhaduje pravděpodobnost zveřejnění mzdy v první fázi a koriguje mzdovou rovnici ve druhé fázi inverzním Millsovým poměrem. Waldův test nulové korelace mezi selekční a mzdovou rovnicí (H₀: ρ = 0) dává pro Německo výsledek χ²(1) = 0,49; p = 0,484. Selekční bias v německém vzorku tedy není statisticky významný, což znamená, že firmy zveřejňující mzdu v Německu se strukturálně neliší od těch, které ji neuvádějí. Německý odhad AI prémie tedy není systematicky vychýlen, má pouze sníženou statistickou sílu kvůli malému vzorku.

V Indii je naopak selekční bias marginálně signifikantní (χ²(1) = 5,17; p = 0,023), což naznačuje mírné zkreslení OLS odhadů. Bodový odhad indické AI prémie se však po Heckmanově korekci výrazně nemění, takže hlavní závěr zůstává v platnosti. V USA je selekční bias statisticky neutrální (χ²(1) = 2,20; p = 0,138). Celkově lze tedy AI prémii v USA i v Indii považovat za důvěryhodnou, zatímco v Německu je třeba odhady interpretovat s vědomím omezené statistické síly, nikoliv však jako vychýlené.

**Mean VIF** se ve všech třech zemích pohybuje mezi 1,40 a 1,82, tedy hluboko pod kritickou hranicí 5 (případně 10 podle přísnějšího kritéria). Multikolinearita v OLS modelech tedy nepředstavuje problém. Maximální individuální hodnoty VIF nepřesahují v žádné zemi hranici 3,0, což je v kontextu ekonometrické literatury považováno za bezpečné.
