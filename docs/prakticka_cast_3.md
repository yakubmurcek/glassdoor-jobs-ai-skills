# 5 Výsledky empirické analýzy

Tato kapitola představuje výsledky empirické analýzy poptávky po AI dovednostech v IT inzerátech ve třech zemích — USA, Německu a Indii. Nejprve je popsán zkoumaný vzorek a jeho základní charakteristiky (§5.1). Následně jsou ve třech paralelních modelech odhadnuty determinanty AI požadavku (§5.2), kvalitativní profily dvou úrovní AI pozic (§5.3) a mzdová prémie za AI (§5.4). Kapitolu uzavírá hlubší inkrementální a mediační analýza na americkém vzorku (§5.5), která těží z jeho velikosti a ilustruje mechanismy stojící za komparativními zjištěními.

## 5.1 Deskriptivní statistika

Finální analytický vzorek obsahuje 38 436 inzerátů ze tří zemí, z toho 17 848 z USA (46,4 %), 14 186 z Indie (36,9 %) a 6 402 z Německa (16,7 %). Americký vzorek je největší, a umožňuje proto v §5.5 hlubší inkrementální dekompozici, na kterou by DE a IN vzorek neměl dostatečnou statistickou sílu. Všechny tři vzorky prošly stejným pipeline extrakce AI dovedností (deterministický slovník + LLM analýza, sjednoceno přes union), takže jsou přímo srovnatelné.

**Výskyt AI požadavků.** AI adopce se mezi zeměmi výrazně liší (viz Tabulka 1). V USA vyžaduje alespoň nějakou úroveň AI dovedností 20,6 % inzerátů (13,2 % AI Integration, 7,4 % Applied/Core AI). Německo má podobný celkový podíl 18,3 % (9,7 % AI Integration, 8,6 % Applied/Core AI) — pozoruhodné, že podíl pokročilých Applied/Core rolí v Německu dokonce mírně převyšuje podíl v USA. Indie má výrazně nižší celkový podíl AI inzerátů, pouze 6,3 % (2,8 % AI Integration, 3,5 % Applied/Core AI). Nižší AI podíl v IN pravděpodobně odráží odlišnou strukturu trhu práce — větší zastoupení outsourcingových a mezinárodních IT service rolí (Infosys, Wipro, TCS, Accenture) s nižším průměrným technologickým profilem pozic.

**Tabulka 1** _Rozložení úrovní AI v IT inzerátech podle země_

| AI úroveň | USA (N) | USA (%) | DE (N) | DE (%) | IN (N) | IN (%) |
|---|---:|---:|---:|---:|---:|---:|
| None | 14 181 | 79,5 | 5 232 | 81,7 | 13 294 | 93,7 |
| AI Integration | 2 354 | 13,2 | 622 | 9,7 | 394 | 2,8 |
| Applied/Core AI | 1 313 | 7,4 | 548 | 8,6 | 498 | 3,5 |
| **Celkem** | **17 848** | **100,0** | **6 402** | **100,0** | **14 186** | **100,0** |

_Poznámka: AI úrovně odvozeny pouze z LLM klasifikace desc\_tier\_llm (bez post-hoc override ze skill clusterů). Detailní definice úrovní viz metodická kapitola 4._

**Pokrytí informace o mzdě.** Inzerovanou mzdu obsahuje jen část vzorku a pokrytí se mezi zeměmi dramaticky liší: v USA 82,0 % inzerátů (14 642 z 17 848), v Indii 68,6 % (9 735 z 14 186) a v Německu pouze 8,0 % (514 z 6 402). Malá velikost německého mzdového vzorku vede k očekávaně sníženě statistické síle v §5.4 a je klíčovým metodologickým omezením komparativních OLS modelů. Systematičnost výběru (zda inzeráty s uvedeným platem jsou reprezentativní) ověřuje Heckmanův test v Příloze C a diskuse v §5.4.

**Mzdové hladiny podle úrovně AI.** Mzdy jsou v celé analýze konvertovány na roční ekvivalent v USD (pro DE přes EUR→USD kurz, pro IN přes INR→USD a přepočet měsíc/rok). Obrázek 1 ukazuje rozložení ročních mezd podle úrovně AI pro každou zemi zvlášť; následující přehled shrnuje klíčové statistiky.

[Obrázek 1 — Graf rozdělení ročních mezd dle úrovně AI, facet per země (US / DE / IN)]

_Mzdové hladiny (roční USD) podle úrovně AI a země:_

| Země | AI úroveň | N | Průměr | Medián | Hrubá prémie vs. None |
|---|---|---:|---:|---:|---:|
| USA | None | 11 601 | 119 524 | 114 000 | — |
| USA | AI Integration | 1 947 | 140 424 | 135 000 | +20 900 (+17,5 %) |
| USA | Applied/Core AI | 1 094 | 150 498 | 148 495 | +30 975 (+25,9 %) |
| DE | None | 396 | 80 389 | 78 638 | — |
| DE | AI Integration | 72 | 84 210 | 84 171 | +3 821 (+4,8 %) |
| DE | Applied/Core AI | 46 | 93 027 | 91 744 | +12 638 (+15,7 %) |
| IN | None | 9 104 | 7 803 | 6 343 | — |
| IN | AI Integration | 267 | 9 964 | 7 077 | +2 161 (+27,7 %) |
| IN | Applied/Core AI | 364 | 11 565 | 7 065 | +3 762 (+48,2 %) |

_Poznámka: Hrubá prémie je nominální rozdíl průměrných mezd oproti kategorii None, bez kontroly vzdělání, zkušeností, profesní skupiny ani dalších faktorů. Čistou AI prémii po kontrole pozorovatelných kvantifikuje §5.4. Pro DE platí nízké N (zvláště u Applied/Core AI), kvůli kterému má německý odhad menší přesnost._

Hrubé mzdové rozdíly potvrzují základní hypotézu, že AI pozice vyžadují vyšší kompenzaci, a to ve všech třech zemích. V USA činí hrubá prémie u Applied/Core AI přibližně 31 000 USD ročně, v Německu zhruba 13 000 USD a v Indii něco přes 3 700 USD. Relativní prémie (v procentech) je naopak nejvyšší v Indii (+48 % u Applied/Core AI), což souvisí s nízkou mzdovou základnou non-AI pozic — absolutní rozdíl je malý, ale mediánová nonAI mzda ~6 300 USD tvoří nízký jmenovatel. Důležité je, že tyto hrubé rozdíly zahrnují i vliv vzdělání, zkušeností, profesní skupiny a technologického profilu pozice. Kolik z nich představuje čistou AI prémii po kontrole pozorovatelných faktorů kvantifikuje §5.4.

**Další deskriptivní zjištění.** Pro výzkumnou otázku jsou podstatné tři vzory, které jsou konzistentní napříč zeměmi a později figurují v komparativních modelech §5.2–§5.4. Za prvé, AI pozice nabízejí častěji práci na dálku než pozice bez AI — rozdíl v podílu remote inzerátů je signifikantní ve všech třech zemích a nejvýraznější je v USA (37,4 % AI vs. 26,1 % non-AI). Za druhé, AI požadavky jsou silně koncentrované v profesní skupině Data & AI, přičemž stejný vzor se projevuje ve všech třech zemích (koeficient Data & AI v binárním logitu §5.2 je vysoce signifikantní a největší v absolutní hodnotě — Tabulka 2, §5.2). Za třetí, Indie vykazuje oproti USA a Německu o poznání nižší celkový tlak na AI kompetence napříč všemi profesními skupinami s výjimkou Data & AI, což naznačuje, že indický IT trh rozvíjí AI kompetence primárně v dedikovaných datových/AI rolích, nikoli jako průřezovou dovednost, která by prostupovala všemi typy inženýrských pozic.

Tyto tři deskriptivní vzory motivují strukturu následující analýzy: binární logit (§5.2) identifikuje, co predikuje AI požadavek; multinomický logit (§5.3) rozlišuje mezi povrchovou integrací a hlubokou AI expertízou; OLS regrese (§5.4) kvantifikuje mzdovou prémii po kontrole pozorovatelných faktorů. Regresní tabulky prezentuji vždy v paralelní podobě pro všechny tři země, aby byl rozdíl mezi trhy přímo viditelný v každém řádku.

## 5.2 Determinanty AI požadavku (binární logit)

Jedním z klíčových cílů této analýzy je zjistit, jaké dovednostní profily a charakteristiky souvisejí s požadavkem na AI dovednosti. K tomuto slouží binární logistická regrese s vysvětlovanou proměnnou has_ai, která nabývá hodnoty 1 pro inzeráty v kategorii AI Integration nebo Applied/Core AI, 0 pak pro inzeráty bez AI požadavku. Model byl odhadnut samostatně pro každou ze tří zkoumaných zemí, tedy pro USA, Německo a Indii.

Pro komparativní strukturu byl binární logit rozdělen do dvou tabulek. Tabulka 2 zahrnuje profesní skupinu (job family) spolu s kontrolními proměnnými (vzdělání, zkušenosti, NACE sektor, typ organizace, velikost firmy a možnost remote práce). Tabulka 3 zobrazuje místo profesní skupiny 21 skill clusterů, včetně clusterů Generative AI a Data Science / Machine Learning. Tyto dva clustery byly z mzdového modelu vyřazeny kvůli cirkularitě s klasifikací ai_level (viz sekce 5.4), v binárním logitu jsou však ponechány záměrně, jelikož nejvýrazněji ilustrují, do jaké míry přítomnost konkrétních AI dovedností v textu inzerátu souvisí s AI klasifikací. Pro výsledky jsou standardně použity průměrné marginální efekty (AME), tedy změny pravděpodobností AI požadavku (v procentních bodech).

Ještě krátká poznámka k interpretaci. Všechny modely v této kapitole identifikují podmíněné asociace mezi proměnnými, nikoliv kauzální efekty. V textu se pro stručnost občas objevují výrazy jako „efekt" nebo „prémie", vždy jde ale o popis vztahů pozorovaných v datech při kontrole ostatních faktorů, ne o tvrzení o příčinné souvislosti. Kauzální interpretace by vyžadovala instrumentální proměnnou nebo kvazi-experimentální design, které v práci nejsou použity.

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

Jak Tabulka 3 napovídá, přítomnost Generative AI (GPT, LLM, Copilot), Data Science či Machine Learning dovedností (TensorFlow, PyTorch, scikit-learn) je ve všech třech zemích nejsilnějším pozitivním prediktorem AI požadavku. To je v souladu s očekáváním, jelikož tyto dva clustery jsou úzce spjaty se samotnou AI klasifikací a prakticky každý inzerát, který je zmiňuje, je zároveň klasifikován jako inzerát s AI požadavkem. Zajímavější a analyticky nosnější jsou proto výsledky u ostatních, necirkulárních skill clusterů jako jsou Cloud Computing, Data Engineering, Dynamic Web a BI / Analytics, které predikují AI požadavek konzistentně ve všech třech zemích. Jedná se o dovednosti spjaté s moderní cloudově-webovou infrastrukturou a s prací s daty. Tyto čtyři clustery lze považovat za technologické jádro, kolem kterého se AI poptávka ve všech třech zemích soustřeďuje. Naopak negativními prediktory jsou Scripting / Shell a OS / Embedded, tedy dovednosti spíše spjaté s údržbou infrastruktury. V USA a Německu se k negativním prediktorům přidávají ještě clustery Enterprise / Managed, Certifications a Databases / Storage, v Indii jsou ale tyto tři clustery statisticky neutrální.

Řada skill clusterů se však mezi zeměmi chová odlišně, a právě tyto rozdíly přinášejí zajímavé poznatky o struktuře jednotlivých trhů. Frontend Development je silným pozitivním prediktorem AI požadavku v USA (+2,8 p. b.), v Německu ani v Indii však statisticky významný vliv nemá. Americký trh tedy častěji kombinuje webové UI dovednosti s AI rolemi, což souvisí s nasazováním AI funkcí do spotřebitelských webových aplikací typických pro americké firmy (chatboti, doporučovací systémy, personalizace). V Německu a Indii tyto dva světy zjevně tolik nesplývají. Cluster Mobile & Desktop poukazuje na negativní efekt pouze v Indii (−1,8 p. b.), což značí, že mobilní vývoj v Indii zůstává výrazně soustředěn spíše v tradičních rolích bez AI a funguje jako samostatná část trhu.

Cluster Data Engineering má v Německu podstatně silnější pozitivní efekt (+4,6 p. b.) než v USA (+1,8 p. b.) nebo v Indii (+0,7 p. b.). To naznačuje, že německý trh silněji propojuje datovou infrastrukturu s AI rolemi, pravděpodobně v důsledku vyšší koncentrace průmyslových a enterprise aplikací, které vyžadují robustní řešení datové infrastruktury. Podobný vzorec vykazuje cluster DevOps & Containers, který je pozitivně signifikantní pouze v Německu (+2,5 p. b.), u ostatních zemí je statisticky neutrální. Zajímavý je poměrně silný negativní efekt pro cluster Enterprise / Managed v USA (−4,5 p. b.) a Německu (−2,6 p. b.), v Indii je však statisticky neutrální.

Pozoruhodný je také celkový vzorec intenzit koeficientů napříč zeměmi. Koeficienty skill clusterů jsou v Německu typicky silnější než v USA a Indii. To může odrážet skutečnost, že AI pozice v Německu jsou koncentrovány v užším okruhu specializovaných rolí s velmi jasně vymezenými dovednostními požadavky, zatímco v USA AI prostupuje trhem rovnoměrněji. Indické koeficienty jsou naopak obecně slabší, což odpovídá jak nižšímu celkovému výskytu AI požadavků (6,3 %), tak i charakteru indického IT trhu, který je v průměru méně specializovaný a obsahuje vyšší podíl generalistů. Pseudo R² modelu dosahuje ve všech třech zemích hodnot mezi 0,36 a 0,55, přičemž nejvyšší je v Indii, což je dáno nízkou četností AI požadavků.

Na Tabulce 2 lze spatřit, že profesní skupina Data & AI je silně pozitivním prediktorem AI požadavku ve všech třech zemích, což samo o sobě není překvapivé. Síla tohoto efektu se však mezi zeměmi výrazně liší. Nejsilnější je v Německu (+59,3 p. b.), následuje Indie (+34,0 p. b.), a nejslabší je USA (+27,1 p. b.). To naznačuje, že na německém trhu existuje jasnější oddělení mezi datovými AI rolemi a zbytkem profesí. V USA je AI rozprostřena rovnoměrněji napříč profesemi, a to včetně běžných softwarových inženýrů. Kategorie Management a Senior Software Engineer jsou signifikantně pozitivní pouze v Německu (+6,5 p. b., resp. +8,0 p. b.). Profesní skupina Other vykazuje signifikantně nižší pravděpodobnost AI požadavku než Software Engineer ve všech třech zemích, zatímco Software Developer a DevOps & Cloud mají signifikantně negativní efekt pouze v USA a Indii. V Německu jsou obě tyto kategorie statisticky neutrální, což naznačuje, že na německém trhu Software Developer a DevOps & Cloud pozice nejsou výrazně odlišné od referenčního Software Engineer v pravděpodobnosti AI požadavku.

Při interpretaci koeficientů profesní skupiny je vhodné zmínit ještě jednu metodologickou vsuvku. Protože proměnná job_family byla odvozena ze stejného LLM extrakčního pipelinu jako ai_level, koeficient Data & AI v Tabulce 2 částečně odráží vnitřní konzistenci klasifikace, a je proto namístě jej interpretovat s opatrností. Dovednostně založený pohled v Tabulce 3 (skill clustery bez job_family) nabízí nezávislejší optiku, jelikož clustery jsou získávány primárně deterministickou extrakcí klíčových slov a role LLM je v nich pouze doplňková.

Waldův test společné signifikance interakcí job_family × country v pooled logit modelu (Příloha B) formálně potvrzuje, že se profesní efekty mezi zeměmi statisticky liší (χ²(12) = 182,7; p < 0,001). Podobně Waldův test pro interakce klíčových skill clusterů (Cloud Computing, Data Science / ML, Backend Development) s proměnnou country potvrzuje signifikantní cross-country heterogenitu dovednostních efektů (χ²(6) = 62,75; p < 0,001). Obě zjištění tedy formálně doplňují kvalitativní pozorování popsaná výše.

## 5.3 Používání vs. vývoj AI (multinomický logit)

Binární logit v sekci 5.2 odpovídá na otázku, které profily souvisejí s AI požadavkem obecně. Jedním z přínosů a zaměření této práce je však samotné rozlišení na pozice, které AI pouze integrují do svých stávajících procesů (AI Integration), a dále ty pozice, kde se AI přímo vyvíjí či je k tomu blízko (Applied/Core AI). K tomuto účelu byl pro každou zemi odhadnut multinomický logistický model s třemi kategoriemi výsledné proměnné (None jako referenční, AI Integration, Applied/Core AI).

Specifikace multinomického logitu se od binárního logitu mírně liší. Vzdělání (edu_logit) bylo z modelu vyřazeno kvůli nedostatečnému počtu pozorování. Profesní skupina byla z modelu již vyřazena, jelikož práce se primárně zaměřuje na zkoumání vlivu dovedností. Následující multinomický model obsahuje všech 21 skill clusterů včetně Generative AI a Data Science / ML a kontrolní proměnné (zkušenosti, NACE sektor, typ organizace, velikost firmy, remote práce).

Vzhledem k tomu, že úrovně ai_level lze vnímat jako narůstající hloubku AI adopce (None → AI Integration → Applied/Core AI), by jako alternativa přicházel v úvahu i ordered logit, který by ušetřil několik stupňů volnosti. Tato specifikace ale vyžaduje předpoklad proporcionality šancí napříč úrovněmi, což data zjevně nesplňují. Některé skill clustery totiž na dvě úrovně AI působí v opačných směrech. Například Systems Programming v USA snižuje pravděpodobnost AI Integration (−2,0 p. b.), ale současně zvyšuje pravděpodobnost Applied/Core AI (+3,6 p. b.); Frontend Development je zase silný pozitivní prediktor AI Integration (+4,3 p. b.), zatímco na Applied/Core AI má mírně negativní vliv (−1,1 p. b.). Pokud stejný prediktor působí na dvě sousední úrovně opačně, proporcionalita šancí je porušena a ordered logit by tyto kvalitativní rozdíly v dovednostních profilech překryl. Multinomický logit, který každou kategorii modeluje samostatně, pro danou strukturu dat lépe odpovídá.

**Tabulka 4 — Multinomický logit ai_level, AME v p. b. per země a tier**

| Skill Cluster            |  USA AI Int. |     USA App. | Německo AI Int. | Německo App. | Indie AI Int. |  Indie App. |
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

Některé skill clustery zvyšují pravděpodobnost obou úrovní požadavků AI (AI Integration a Applied/Core AI) ve všech třech zemích, avšak s různou intenzitou. Dynamic Web je konzistentně pozitivní pro Applied/Core AI ve všech zemích a pro AI Integration v USA, což potvrzuje, že moderní webové technologie jsou nedílnou součástí jak vývoje, tak nasazení AI aplikací. Cloud Computing predikuje AI Integration v USA a Německu, pro Applied/Core AI je pozitivní pouze v USA, a v Indii je překvapivě nesignifikantní. S výjimkou Indie tyto výsledky naznačují, že cloudová infrastruktura je primárně prostředkem nasazení hotových AI služeb, méně už samotného vývoje AI systémů. U tradičních technologií je výsledek méně jednotný. Scripting / Shell vykazuje signifikantně negativní efekt na Applied/Core AI ve všech třech zemích, avšak na AI Integration není statisticky významný v žádné. Enterprise / Managed má silný negativní efekt v USA u obou úrovní AI a v Německu pouze u AI Integration, v Indii je však statisticky neutrální na obou úrovních. Tyto dovednostní clustery tedy nejsou s AI vyloženě nekompatibilní a každý z nich vykazuje vlastní odlišný vzorec. Scripting / Shell se projevuje jako negativní primárně u Applied/Core AI, zatímco u AI Integration nemá ve všech třech zemích signifikantní vliv. Enterprise / Managed má zase v USA silnější negativní efekt u AI Integration než u Applied/Core AI. Celkově jsou negativní efekty těchto zastávajících (legacy) technologií silnější na západních trzích než v Indii. To může být teoreticky dáno vyšším zastoupením těchto starších systémů vyžadujících údržbu vzhledem k indickém outsourcingu IT služeb.

Cluster Data Engineering je napříč zeměmi nejčistším ukazatelem ve prospěch Applied/Core AI v USA (+3,2 p. b.), Německu (+4,4 p. b.) i Indii, i když tam již méně (+0,9 p. b.). Na nižší úrovni (AI Integration) je v těchto zemích buď slabě negativní nebo statisticky neutrální. Správa datových pipeline, ETL procesů a datová architektura jsou tedy výhradně doménou skutečného AI vývoje, nikoliv integrace. Pozice využívající hotová AI API (rozhraní pro komunikaci aplikací) tuto dovednost nepotřebují, zatímco pozice vyvíjející AI modely ji vyžadují pro přípravu trénovacích dat.

Cluster Systems Programming, tedy nízkoúrovňové programování v jazycích jako C, C++ nebo Rust, vykazuje v USA statisticky významný opačný efekt mezi úrovněmi AI. Snižuje pravděpodobnost AI Integration (−2,0 p. b.) a zároveň zvyšuje pravděpodobnost Applied/Core AI (+3,6 p. b.). V Německu je vzorec podobný, ale signifikantní je pouze efekt na vyšší úroveň Applied/Core AI (+3,1 p. b.). V Indii tento cluster ztrácí signifikanci u obou kategorií. Interpretace na amerických datech tak dávají největší logiku, jelikož nízkoúrovňové programování je klíčovou kompetencí pro vývoj infrastruktury pro AI (optimalizaci pipeline, GPU computing, nasazení AI modelů na zařízeních), tedy vyšší úrovně práce s AI, oproti pouhé integraci, kde se tyto dovednosti takto nevyužijí.

Co se týče clusteru Frontend Development, je naopak napříč zeměmi konzistentně pozitivním prediktorem AI Integration (USA +4,3 p. b., Německo +4,4 p. b., Indie +1,5 p. b.) a až mírně negativním pro Applied/Core AI. Jedná se tedy o dovednost typickou pro pozice, kde se AI nasazuje a aplikuje do existujících programů, služeb či uživatelských rozhraní. Tento podobný vzorec lze v USA nalézt u Enterprise Platforms. Za zmínku stojí ještě dovednosti spojené s Backend Development, které jsou pozitivním prediktorem AI Integration ve všech třech zemích (USA +1,5 p. b., Německo +3,3 p. b., Indie +0,7 p. b.), přičemž nejsilnější efekt je v Německu. To odráží roli backendových vývojářů v integraci AI služeb na backendové úrovni.

Odlišné dovednostní profily AI Integration a Applied/Core AI se napříč zeměmi většinou strukturálně opakují, ačkoliv se to děje s různou intenzitou. Applied/Core AI pozice ve všech třech zemích vyžadují klíčové inženýrské dovednosti v datové infrastruktuře a systémovém programování, zatímco AI Integration pozice se vyznačují aplikačními dovednostmi (Frontend, Backend), enterprise platformách a cloudovém nasazení. Tento rozdíl tedy nevypadá jako pouhý artefakt amerického trhu nebo klasifikace LLM, spíše se jeví jako obecně platný vzorec, jež je zachytitelný i v tomto mezinárodním srovnání. Tím se také potvrzuje, že přijaté třístupňové rozlišení AI požadavků zvolené pro tuto práci, má smysluplnou dovednostní strukturu a interpretaci.

Výjimku představují například cluster dovedností Security & Identity, který v Indii vykazuje slabě pozitivní efekt pro AI Integration (+0,8 p. b., p < 0,01), zatímco v USA a Německu je statisticky neutrální nebo negativní. Tento jev by mohl být interpretován jako integrace AI do oblasti bezpečnosti v indickém IT sektoru, pravděpodobně v kontextu outsourcingových služeb (pro západní klienty), kde AI nástroje pro detekci hrozeb nacházejí rychlejší uplatnění.

Základním předpokladem multinomického logitu je nezávislost irelevantních alternativ (IIA), jež se standardně ověřuje Hausmanovým testem. V modelech pro všechny tři země však tento test vrací zápornou testovou statistiku (pro USA χ²(37) = −102,1), což Stata explicitně označuje jako selhání asymptotických předpokladů testu. Důvodem je, že variančně-kovarianční matice rozdílů mezi dvěma srovnávanými modely není pozitivně definitní, což je známá slabina Hausmanova testu u multinomických modelů s větším počtem prediktorů (Long & Freese, 2014; Cheng & Long, 2007). Test tedy nelze interpretovat jako potvrzení ani zamítnutí IIA. Multinomický logit je z tohoto důvodu v této práci použit jako explorativní nástroj pro identifikaci kvalitativních dovednostních profilů. Vzory průměrných marginálních efektů (směr a relativní velikost efektů) jsou spolehlivější než přesné bodové odhady koeficientů, a právě na těchto vzorech stojí hlavní zjištění této kapitoly.

## 5.4 Mzdová prémie za AI (OLS)

Předchozí podkapitoly ukázaly, jaký dovednostní a profesní profil s AI požadavky souvisí. Nabízí se však také otázka, jak se tyto AI kompetence promítají do samotných mezd pozic. Jako odpověď byl v práci odhadnut OLS regresní model, vysvětlující přirozený logaritmus roční mzdy (ln_salary) v souladu s Mincerovou mzdovou rovnicí. Stejně jako v předchozích sekcích byl model odhadnut samostatně pro každou zemi. Specifikace obsahuje úrovně AI požadavku (ai_level), 19 skill clusterů bez Generative AI a Data Science / Machine Learning. Tyto dva clustery byly z OLS záměrně vyřazeny kvůli cirkularitě s hlavní proměnnou ai_level určující úroveň AI v dané pozici. Model obsahuje proměnné kategorického vzdělání (edu_ols, 5 úrovní), zkušeností (exp_category, 4 úrovně), NACE sektoru, typu organizace, velikosti firmy a možnosti remote práce. Specifikace je symetrická napříč zeměmi bez regionálních fixních efektů, jelikož německá ani indická data nemají ekvivalentní harmonizovanou regionální proměnnou. Robustnost hlavního amerického odhadu vůči zavedení US Census regions jako fixních efektů je ověřena v Příloze D.

Dopad vyřazení cirkulárních clusterů z hlavního modelu byl ověřen samostatnou citlivostní analýzou v Příloze C, která prezentuje OLS s plnou sadou skill clusterů. Výsledky ukazují, že zahrnutí Generative AI a Data Science / ML má na odhady AI prémie jen mírný dopad. V USA se koeficient AI Integration posouvá z +11,5 % na +10,4 % a Applied/Core AI z +16,2 % na +14,0 %. Oba koeficienty zůstávají vysoce statisticky významné (p < 0,001). Hlavní model bez cirkulárních clusterů tedy dává čistší odhad mzdové hodnoty AI role jako takové, neboť nepřejímá mechanicky část variability, která pramení přímo z klasifikace ai_level. Rozdíl R² mezi oběma specifikacemi je prakticky nulový (0,278 vs. 0,278), což ukazuje, že hlavní závěr o existenci a velikosti AI prémie je robustní v obou variantách.

**Tabulka 5 — OLS ln(plat), mzdová prémie za AI per země**

| Proměnná        |          USA | Německo |      Indie |
| --------------- | -----------: | ------: | ---------: |
| AI Integration  | +11,5 \*\*\* | +3,1 ns |    +9,0 \* |
| Applied/Core AI | +16,2 \*\*\* | +7,4 ns | +11,7 \*\* |
| N               |       14 642 |     514 |      9 735 |
| R²              |        0,278 |   0,272 |      0,148 |
| Mean VIF        |         1,37 |    1,82 |       1,59 |

_Hodnoty v %, koeficienty logaritmického modelu interpretovány jako přibližné procentuální změny mzdy. Signifikance: \* p < 0,05; \*\* p < 0,01; \*\*\* p < 0,001; ns = nesignifikantní. Model obsahuje všechny skill clustery kromě Generative AI a Data Science / ML, dále kategorické vzdělání, zkušenosti a firemní kontroly. Specifikace je symetrická napříč zeměmi bez regionálních fixních efektů._

V USA činí AI prémie po kontrole všech pozorovatelných faktorů +11,5 % pro AI Integration a +16,2 % pro Applied/Core AI, přičemž oba koeficienty jsou vysoce statisticky významné. V Indii jsou odhady bodově nižší (AI Integration +9,0 %, Applied/Core AI +11,7 %), oba rovněž statisticky významné, ačkoliv s mírně nižší silou. Německo vykazuje bodově nejnižší hodnoty (AI Integration +3,1 %, Applied/Core AI +7,4 %), avšak ani jeden z koeficientů není statisticky signifikantní. Nižší signifikance německých odhadů souvisí s velmi malým vzorkem inzerátů s uvedenou mzdou (N = 514, pouhých 8 % z německého vzorku). Německý trh práce obecně inzeruje mzdu mnohem méně často než americký nebo indický, to představuje limitaci statistické síly mzdového modelu.

Pooled OLS model s interakcemi ai_level × country (Příloha B) formálně ověřuje, zda se AI prémie mezi zeměmi statisticky liší. Waldův test těchto interakcí (F(4; 12 686) = 2,56; p = 0,037) zamítá hypotézu o homogenitě AI prémie napříč zeměmi. Rozdíl je však tažen primárně Německem, jehož AI Integration prémie je oproti USA o 11,1 procentního bodu nižší (p = 0,005). USA a Indie jsou si naproti tomu v úrovni AI prémie statisticky rovny (interakce ai_level × IN nejsou signifikantní). Heterogenita AI prémie je tedy spíše slabší než silná a projevuje se především v rozdílu mezi německým a americko-indickým pásem.

**Robustnost vůči selekčnímu zkreslení**

Nízké pokrytí platu v Německu představuje potenciální zdroj selekčního zkreslení, jelikož firmy, které mzdu zveřejňují, mohou být systematicky odlišné od těch, které ji neuvádějí. Tento problém byl ověřen Heckmanovým selekčním modelem (Příloha A), který odhaduje pravděpodobnost zveřejnění mzdy v první fázi a koriguje mzdovou rovnici ve druhé fázi inverzním Millsovým poměrem.

Je třeba uvést jednu metodologickou upřesňující poznámku. V této specifikaci slouží Heckmanův model spíše jako robustnostní test vůči funkční formě než jako plnohodnotná korekce selekčního zkreslení. Selekční i mzdová rovnice totiž obsahují stejnou sadu regresorů, jelikož formální exclusion restriction (tedy proměnná, která by ovlivňovala zveřejnění mzdy, ale ne její výši) není v datech k dispozici. Nulová hypotéza ρ = 0 je tak identifikována pouze nelinearitou inverzního Millsova poměru, a výsledky je proto vhodné interpretovat jako test citlivosti OLS odhadů vůči funkční formě, nikoli jako čistou korekci selekce.

Waldův test nulové korelace mezi selekční a mzdovou rovnicí (H₀: ρ = 0) dává pro Německo výsledek χ²(1) = 0,49; p = 0,484. V Německu tedy Heckmanův model žádný signifikantní rozdíl oproti OLS nezachytil — německý odhad AI prémie má pouze sníženou statistickou sílu kvůli malému vzorku, nikoliv systematické zkreslení. V USA je test ρ = 0 rovněž statisticky neutrální (χ²(1) = 2,20; p = 0,138). V Indii vychází ρ signifikantní (χ²(1) = 5,17; p = 0,023), vzhledem k absenci exclusion restriction v této specifikaci však nelze spolehlivě rozhodnout, zda jde o skutečnou selekci, nebo o mírné porušení předpokladu bivariate-normálního rozdělení chyb. Bodový odhad indické AI prémie se nicméně po Heckmanově korekci zásadně nemění, takže hlavní závěr o existenci a přibližné velikosti AI prémie zůstává v platnosti ve všech třech zemích. V Německu je třeba odhady interpretovat s vědomím omezené statistické síly, nikoliv jako vychýlené.

**Mean VIF** se ve všech třech zemích pohybuje mezi 1,37 a 1,82, tedy hluboko pod kritickou hranicí 5 (případně 10 podle přísnějšího kritéria). Multikolinearita mezi hlavními substantivními prediktory (skill clustery a úrovně AI požadavku) tedy nepředstavuje problém, jelikož jejich individuální hodnoty VIF v žádné zemi nepřesahují 1,9. Vyšší individuální hodnoty se objevují pouze u dummy proměnných pro velikost firmy (max VIF 2,5 v USA, 7,4 v Německu, 5,6 v Indii), což je běžný důsledek vzájemné korelace mezi kategoriemi jedné kategorické proměnné a nemá dopad na odhady AI prémie ani koeficientů skill clusterů.

## 5.5 Hlubší dekompozice na americkém vzorku

Zatímco předchozí sekce §5.2 až §5.4 srovnávaly tři trhy v symetrické specifikaci, tato sekce využívá americký vzorek k hlubší inkrementální a mediační analýze. USA je s N = 17 848 inzeráty (z toho 14 642 s inzerovanou mzdou) zdaleka největším vzorkem této práce a umožňuje dekompozice, které by na německém nebo indickém vzorku trpěly sníženou statistickou silou a sparse cells. Cílem není duplikovat komparativní pohled, nýbrž doplnit ho hlubším porozuměním mechanismům stojícím za AI prémií. Modely v této sekci zahrnují US Census regions jako fixní efekty, protože region je v USA silný mzdový prediktor a jeho zahrnutí odpovídá klasické Mincerově mzdové rovnici, která tradičně regionální kontroly uvádí. Komparativní Tabulka 5 naopak region FE vynechává, aby byla specifikace symetrická napříč třemi zeměmi. Kvantifikace, o kolik procentních bodů region FE posouvá americké koeficienty AI prémie, je předmětem Přílohy D.

### Determinanty AI požadavku — inkrementální binární logit

Komparativní §5.2 ukázala vzorce binárního logit AI požadavku napříč zeměmi. Tato subsekce doplňuje obraz postupnou inkrementální dekompozicí na americkém vzorku, která ukazuje, jakou prediktivní sílu nese firemní kontext a profil role samostatně. Modely jsou odhadovány ve třech krocích. Model M1 obsahuje výhradně firemní charakteristiky (NACE sektor, typ organizace, velikost firmy a region) a dosahuje Pseudo R² pouhých 2,4 %. Samotné firemní charakteristiky mají tedy jako prediktor AI požadavku velmi slabou výpovědní hodnotu. Model M2 je naopak nahrazuje profilem role, který tvoří technologické skill clustery, profesní skupina, vzdělání a zkušenosti, a Pseudo R² se posouvá na 11,7 %. Plný Model M3 oba bloky spojuje, přináší však už jen marginální zlepšení na 12,9 %. Profil role tedy vysvětluje téměř pětkrát více variability AI požadavku než firemní charakteristiky a přidání firemních charakteristik k profilu role model dále prakticky nezlepší. O tom, zda pozice vyžaduje AI, rozhoduje spíše obsah práce než identita zaměstnavatele.

Diagnostika plného Modelu M3 potvrzuje dobrou kvalitu odhadu. Hosmer-Lemeshowův test shody je nesignifikantní (χ²(8) = 12,29; p = 0,139), AUC = 0,746 odpovídá přijatelné schopnosti modelu odlišit AI a non-AI pozice a hodnoty VIF nepoukazují na problém s multikolinearitou. Linktest ukázal drobnou nepřesnost ve specifikaci modelu (_hatsq p = 0,003), která pravděpodobně souvisí s absencí interakcí mezi proměnnými. Efekt cloudových dovedností by mohl být silnější v tech hubech, efekt webových technologií se může lišit podle velikosti firmy. Tyto interakce v hlavním modelu nejsou, protože by komplikovaly interpretaci marginálních efektů. Klasifikační tabulka vykazuje celkovou přesnost 80,5 % při senzitivitě 16,9 % a specificitě 96,9 %. Nízká senzitivita je běžným důsledkem toho, že AI pozice tvoří jen asi 21 % amerického vzorku, model je tedy opatrný v tom, co označí jako AI. Cílem analýzy však není klasifikace, nýbrž odhad marginálních efektů.

**Technologická propast na IT trhu**

Výsledky Modelu M3, interpretované pomocí průměrných marginálních efektů (AME), odhalují výraznou technologickou propast na americkém IT trhu. Na jedné straně stojí moderní, cloudově orientovaný technologický stack, jehož přítomnost v inzerátu výrazně zvyšuje pravděpodobnost AI požadavku. Na druhé straně tradiční podnikové a infrastrukturní technologie, které s AI pozicemi korelují negativně. Nejsilnějším pozitivním prediktorem je skill cluster Dynamic Web (AME = +9,4 p. b.; p < 0,001). Přítomnost dovedností typu React, Node.js nebo moderních webových frameworků zvyšuje pravděpodobnost AI požadavku o téměř 10 procentních bodů. Tento výsledek odráží reálný tržní trend, jelikož AI funkce (chatboti, doporučovací systémy, personalizace) jsou nejčastěji nasazovány právě prostřednictvím webových aplikací. Následují Cloud Computing (+6,6 p. b.), Enterprise Platforms (+5,0 p. b.), Data Engineering (+4,5 p. b.) a Frontend Development (+3,6 p. b.). Doplňují je BI / Analytics (+3,5 p. b.), Systems Programming (+3,3 p. b.), DevOps / Containers (+2,6 p. b.) a Backend Development (+2,3 p. b.).

Na opačné straně jsou technologie typické pro tradiční IT role. Enterprise / Managed systémy (AME = −6,5 p. b.; p < 0,001) a Scripting / Shell (−5,6 p. b.; p < 0,001) mají nejsilnější negativní vliv, následují Certifications (−3,8 p. b.), Databases / Storage (−3,4 p. b.) a OS / Embedded (−3,3 p. b.). AI požadavky se tedy koncentrují v pozicích s moderním cloudově-webovým profilem, zatímco tradiční podnikové a infrastrukturní role AI zatím výrazněji neadoptovaly. Tato propast má jeden důležitý praktický důsledek. Přechod k AI neznamená jen osvojit si samotné AI nástroje, nýbrž modernizovat celé technologické zázemí. Firmy a pracovníci zakotvení v legacy technologiích mají k AI strukturálně dál.

Zajímavý je také výsledek pro firemní charakteristiky a senioritu. Typ organizace (Private, Public, Unknown/Other/Gov) nemá statisticky významný vliv v žádné specifikaci. Stejně tak seniorita po kontrole skill clusterů a job family ztrácí signifikanci (Junior p = 0,262; Senior+ p = 0,233). AI požadavky tedy nejsou výsadou seniorních pozic, pronikají na všechny úrovně seniority a napříč různými typy zaměstnavatelů, pokud má pozice odpovídající technologický profil.

**Test mediace: dovednosti vs. profesní skupina**

Profesní skupina (job family) a úroveň seniority mohou v modelu částečně absorbovat efekt technologických dovedností na pravděpodobnost AI požadavku. Pro zachycení tohoto posunu byly odhadnuty tři vnořené modely: M3 (plný; Pseudo R² = 0,129), M3-nojf (bez job_family; Pseudo R² = 0,102) a M3-nojf-noexp (bez job_family i seniority; Pseudo R² = 0,101). Po vyřazení profesní skupiny z modelu se AME klíčových skill clusterů zřetelně posunuly. Cloud Computing vzrostl z +6,6 na +8,5 p. b. (+29 %), Data Engineering z +4,5 na +7,3 p. b. (+62 %) a negativní efekt Scripting / Shell se prohloubil z −5,6 na −7,8 p. b. (+39 %). Nově signifikantními se staly Security / Identity (p = 0,013) a Testing / QA (p = 0,038), tedy clustery, které byly v plném modelu profesní skupinou absorbovány. Je však vhodné tato čísla chápat jako účetní dekompozici posunu AME mezi vnořenými specifikacemi, nikoli jako formální mediaci v Baron-Kennyho smyslu. V nelineárních modelech typu logit totiž část posunu může pramenit z přeškálování latentního indexu, a procentuální vyjádření („+29 %", „+62 %") je proto třeba číst jako orientační heuristiku, nikoli jako exaktní mediační podíl. Jasný je nicméně kvalitativní závěr: technologické dovednosti predikují jak AI požadavek, tak profesní skupinu, a v plném modelu profesní skupina jejich samostatný efekt zeslabuje. Seniorita naopak v tomto smyslu mediátorem není. Srovnání M3-nojf a M3-nojf-noexp ukazuje, že odebrání seniority téměř nic nemění (ΔPseudo R² = 0,001; ΔLL ≈ 8,2) a koeficienty skill clusterů zůstávají prakticky stejné. AI požadavky tedy nejsou vázány na úroveň zkušeností, jelikož juniorní pozice s moderním tech stackem mají stejnou šanci na AI požadavek jako pozice seniorní.

### Inkrementální OLS mzdové prémie (Model A → B → C)

Komparativní §5.4 odhadovala čistou AI prémii per země. Tato subsekce dekomponuje hrubou AI prémii v USA postupným přidáváním kontrol firemního profilu, lidského kapitálu a technologického profilu pozice a ukazuje, která vrstva kontrol AI prémii vysvětluje a která ne. Byly sestaveny tři inkrementální OLS modely vysvětlující přirozený logaritmus roční mzdy (ln_salary) v souladu s Mincerovou mzdovou rovnicí. Díky logaritmické transformaci závislé proměnné lze odhadnuté koeficienty interpretovat jako přibližné procentuální změny mzdy.

Model A představuje výchozí specifikaci obsahující pouze firemní a kontextové proměnné, tedy úrovně AI požadavku, NACE sektor, region, možnost remote práce, typ a velikost organizace. Model vysvětluje 19,9 % variability mezd (R² = 0,199; N = 14 642; standardní chyby klastrované na firmu). V tomto nejjednodušším modelu dosahuje koeficient pro AI Integration hodnoty +0,118 (p < 0,001), což odpovídá přibližně 11,8% mzdové prémii. Pro Applied/Core AI činí koeficient +0,177 (p < 0,001), odpovídající 17,7 % prémii. Tyto odhady představují hrubou AI prémii kontrolovanou jen za firemní charakteristiky a stále v sobě nesou vliv osobních charakteristik uchazeče i technologického profilu pozice.

Model B přidává proměnné lidského kapitálu, tedy podrobnější vzdělání (High School, Associate, Bachelor, Master+) a kategorii zkušeností. Vysvětlující síla modelu roste na R² = 0,305, tedy o 10,7 procentního bodu oproti Modelu A. Společnou signifikanci přidaného bloku potvrzuje Waldův F-test (F(7; 7 361) = 145,26; p < 0,001). Po kontrole vzdělání a zkušeností se AI prémie očekávaně snížila. AI Integration klesla na +10,6 % (koeficient 0,106; p < 0,001), Applied/Core AI na +16,4 % (koeficient 0,164; p < 0,001). Pokles oproti Modelu A ukazuje, že asi jeden procentní bod hrubé AI prémie lze přičíst tomu, že AI pozice systematicky vyžadují vzdělanější uchazeče s delší praxí. Většina AI prémie ale zůstává nevysvětlena samotným lidským kapitálem.

Model C je plnou specifikací, která k Modelu B přidává 19 binárních proměnných technologických skill clusterů (bez cirkulárních clusterů Generative AI a Data Science / ML) a profesní skupinu. Vysvětlující síla roste na R² = 0,375 (+7,0 p. b. oproti Modelu B). Společnou signifikanci bloku skill clusterů a job family potvrzuje Waldův F-test (F(25; 7 361) = 46,03; p < 0,001). Konečné odhady čisté AI prémie v Modelu C činí u AI Integration +8,6 % (koeficient 0,086; p < 0,001) a u Applied/Core AI +11,7 % (koeficient 0,117; p < 0,001). Oproti Modelu B je pokles výrazný (z 10,6 % na 8,6 %, respektive z 16,4 % na 11,7 %), což ukazuje, že podstatná část AI prémie je vysvětlena specifickým technologickým profilem pozice, jelikož AI pozice vyžadují širší a hodnotnější portfolio technických dovedností. I po této kontrole však zůstává čistá AI prémie statisticky vysoce významná a ekonomicky podstatná. Diagnostika multikolinearity (průměrný VIF 1,48; maximální 3,02) nepotvrzuje žádný problém.

Postupná dekompozice napříč modely A, B a C ukazuje, jak se AI prémie mění s přidáváním kontrol. Z původních 11,8 % / 17,7 % (Model A) klesá přes 10,6 % / 16,4 % po kontrole lidského kapitálu (Model B) na finálních 8,6 % / 11,7 % po kontrole všech faktorů (Model C). Celkově je tedy přibližně 27 % hrubé AI Integration prémie a 34 % hrubé Applied/Core AI prémie vysvětleno měřitelnými charakteristikami pozice a uchazeče. Podstatná většina, zejména u AI Integration, zůstává jako nezávislý efekt AI kompetencí, který tradiční determinanty mzdy nevysvětlují.

### Determinanty mzdy v Modelu C

AI prémie je reálná a statisticky významná, ale u platu není nejsilnějším prediktorem. Na základě koeficientů Modelu C lze sestavit hierarchii vlivů na mzdu. Nejsilnější efekt má geografická lokace. Pozice v regionu West (Silicon Valley, Seattle) nesou prémii +16,8 % oproti referenční kategorii South. Zkušenost 6 a více let přidává +12,1 % oproti střední kategorii 3–5 let. Applied/Core AI s +11,7 % se řadí hned za seniorní zkušenosti a předstihuje i prémii seniorního softwarového inženýra (+10,3 % oproti referenční profesní skupině Software Engineer). AI Integration přidává +8,6 %, magisterský a vyšší titul +8,4 %, nízkoúrovňové programování v C/C++/Rust (Systems Programming) +7,3 % a prémie za práci ve velmi velkých firmách (10 000+ zaměstnanců) činí +6,8 %.

Na opačné straně jsou juniorní pozice (penalizace −17,5 %), profesní kategorie Software Developer a Other (−12,5 % a −11,1 % oproti Software Engineer), Associate degree (−8,1 %) a Midwest region (−5,5 %). Klíčovým zjištěním je, že čistá Applied/Core AI prémie po kontrole všech pozorovatelných faktorů převyšuje i prémii za seniorní softwarové inženýrství. Hlubší AI kompetence jsou tak na americkém IT trhu oceněny srovnatelně nebo silněji než tradiční seniorní postup. Zároveň AI prémie tradiční faktory lidského kapitálu nenahrazuje, jelikož vzdělání, praxe i regionální lokace zůstávají nezávisle silnými prediktory. AI prémie tedy funguje jako bonus navrch, nikoli jako alternativa k tradičním mzdovým determinantám.

Protože výše uvedená hierarchie je postavená na nestandardizovaných koeficientech, které zachycují pouze velikost procentuálního efektu bez ohledu na to, jak častá a variabilní je daná kategorie v datech, byl Model C odhadnut i se standardizovanými (beta) koeficienty prostřednictvím volby `, beta` v příkazu `regress`. Standardizované koeficienty převádějí závislou i nezávislé proměnné na jednotky směrodatných odchylek a umožňují tak srovnání skutečné relativní důležitosti jednotlivých prediktorů pro variabilitu mezd, tedy s kontrolou za to, jak rozšířená každá kategorie ve vzorku je. Tato kontrola je důležitá zejména u dummy proměnných s nízkou četností (např. region West), jejichž vysoký nestandardizovaný koeficient může nadhodnocovat jejich celkový přínos k vysvětlení variability mezd.

<!-- TODO (po spuštění `reg ln_salary ..., beta` v analysis/stata): doplnit 2–3 věty s pořadím pěti nejsilnějších prediktorů podle standardizovaného koeficientu (poslední sloupec výstupu) a explicitně porovnat s hierarchií výše. Pokud pořadí drží → stačí potvrzení robustnosti. Pokud se pořadí liší (typicky: West jako malá kategorie klesne, častější proměnné jako seniorní zkušenost nebo job family mohou stoupnout) → upravit formulace v odstavci výše a zde vysvětlit, proč se pořadí mění. -->

### Alternativní specifikace se spojitou požadovanou praxí a mediace přes profesní skupinu

Vedle inkrementální dekompozice stojí za ověření i alternativní funkční forma zkušeností a role profesní skupiny jako mediátoru mzdové prémie.

**Spojitá forma zkušeností**

Jako robustnostní test funkční formy byl odhadnut Model C-Mincer, inspirovaný Mincerovou mzdovou rovnicí, kde kategorickou proměnnou zkušeností nahradila kontinuální proměnná minimálně požadovaných let praxe a její kvadratický člen. Je třeba zdůraznit, že proměnná měří praxi **požadovanou zaměstnavatelem v inzerátu**, nikoliv skutečnou praxi pracovníka, bodové odhady tedy není korektní interpretovat jako návratnost lidského kapitálu v Mincerově smyslu. Spíše popisují, jak se očekávaná náročnost pozice (v letech praxe) promítá do nabízené mzdy. Výsledky nicméně vykreslují vzorec konzistentní s klasickou Mincerovou křivkou: každý další požadovaný rok praxe přidává +6,8 % ke mzdě, ale s klesající intenzitou (kvadratický člen −0,2 %). Hlavní odhad AI prémie zůstal konzistentní (AI Integration +8,5 %, Applied/Core AI +11,6 %). Model dosahuje R² = 0,420 na menším podvzorku N = 12 608 (pozorování s chybějícími údaji o zkušenostech jsou automaticky vyřazena). AI prémie je tedy stabilní i při přechodu z kategorické na kontinuální funkční formu zkušeností.

**Mediace přes profesní skupinu**

Profesní skupina může vysvětlovat část vztahu mezi AI dovednostmi a mzdou, protože AI pozice se koncentrují v lépe placených profesních skupinách (Data & AI, Sr+ Software Engineer). Pro kvantifikaci tohoto podílu byl odhadnut Model C bez proměnné job_family (R² = 0,337). V tomto modelu vzrostla AI prémie na +9,6 % (AI Integration) a +14,0 % (Applied/Core AI). Profesní skupina tedy v lineární OLS specifikaci absorbuje přibližně 1 procentní bod AI Integration efektu a asi 2,3 procentního bodu Applied/Core AI efektu, což lze interpretovat jako částečnou mediaci. Typ pozice je jedním z kanálů, kterými se AI dovednosti promítají do vyšších mezd, a tento kanál je výraznější u hlubších Applied/Core AI rolí než u pozic pouze integrujících AI nástroje.

Srovnání s dekompozicí v logitovém modelu (sekce 5.5.1) ukazuje zajímavý kontrast. V logitu se po vyřazení profesní skupiny AME klíčových dovednostních clusterů zřetelně posílily (například Cloud Computing a Data Engineering o 29 a 62 % původní hodnoty), zatímco v OLS profesní skupina absorbuje jen malou část mzdové prémie za AI (přibližně 1 až 2 procentní body z celkových 8,6 a 11,7 %). Profesní role tedy silněji souvisí s tím, zda uchazeč AI pozici získá, než s tím, kolik za ni dostane zaplaceno.

Pro plnou interpretaci je vhodné připomenout, že komparativní Tabulka 5 v §5.4 ukazuje US koeficienty bez job_family a zároveň bez region FE (+11,5 % AI Integration; +16,2 % Applied/Core AI). Model C-nojf v této subsekci ukazuje US bez job_family, ale s region FE (+9,6 %; +14,0 %). Rozdíl mezi těmito dvěma specifikacemi (přibližně 2 procentní body u obou tierů) odpovídá právě tomu, kolik regionální heterogenita (zejména prémie regionu West) absorbuje z čisté AI prémie. Kvantifikaci tohoto posunu side-by-side uvádí Příloha D.

Další robustnostní kontrola hlavního OLS modelu, konkrétně zahrnutí cirkulárních clusterů Generative AI a Data Science / ML zpět do RHS, je prezentována souhrnně pro všechny tři země v §5.4 a Příloze C. Pro americký vzorek potvrzuje stejný vzorec jako v komparativní analýze: AI prémie klesá mírně (AI Integration z 8,6 % na 7,6 %; Applied/Core AI z 11,7 % na 9,7 %), zůstává vysoce signifikantní a hlavní závěr o existenci čisté AI prémie platí v obou variantách.
