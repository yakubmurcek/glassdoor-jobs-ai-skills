# 5 Výsledky empirické analýzy

Tato kapitola představuje výsledky empirické analýzy poptávky po AI dovednostech v IT pracovních inzerátech. Analýza postupuje od deskriptivního popisu vzorku přes inferenční statistické testy až po vícerozměrné regresní modely. Nejprve je prezentována analýza amerického trhu práce (N = 17 848), která slouží jako hlavní referenční vzorek, následně je rozšířena o komparativní perspektivu tří zemí (USA, Německo, Indie; N = 38 432). Všechny výpočty byly provedeny v softwaru Stata 18 s robustními standardními chybami.

## 5.1 Deskriptivní statistika

### 5.1.1 Struktura vzorku a zastoupení AI požadavků

Výchozí dataset amerických IT pracovních inzerátů z portálu Glassdoor obsahoval 18 464 pozorování. Po aplikaci filtru důvěryhodnosti LLM klasifikace (confidence ≥ 0,7) bylo vyřazeno 607 pozorování (3,3 %) a dalších 9 pozorování pocházelo z roku 2023 a starších. Finální analytický vzorek tak čítá 17 848 pozorování.

Ze zkoumaného vzorku vyžadovalo alespoň nějakou úroveň AI dovedností přibližně 20,6 % inzerátů. Konkrétně 13,2 % pozic představuje kategorii AI Integration, tedy pozice, kde je AI integrována do běžných pracovních procesů (například využití generativních AI nástrojů, API integrace, prompt engineering). Dalších 7,4 % pozic spadá do kategorie Applied/Core AI, tedy pozic zaměřených na přímý vývoj a budování AI systémů (strojové učení, trénování modelů, vývoj algoritmů). Zbývajících 79,5 % inzerátů nepožadovalo žádné explicitní AI dovednosti. Je třeba uvést, že kategorie Core AI byla sloučena s Applied AI z důvodu pouhých 6 pozorování, což je pod hranicí 50 pozorování na buňku potřebnou pro stabilní odhady v multinomiálních modelech. Toto sloučení představuje metodologickou limitaci, neboť skutečné AI výzkumné pozice nelze v tomto datasetu oddělit od pozic aplikovaného AI vývoje.

### 5.1.2 Mzdová distribuce

Mzdové údaje byly k dispozici pro 14 640 pozorování (82,0 % vzorku). Mediánový roční plat ve vzorku činil 117 885 USD. Exploratorní srovnání mzdových hladin podle úrovně AI požadavků odhalilo výrazné nominální rozdíly. Pozice bez AI požadavků vykazovaly průměrný roční plat 119 532 USD (medián 114 000 USD). U pozic kategorie AI Integration dosahoval průměrný plat 140 487 USD (medián 135 000 USD), což představuje nominální nárůst přibližně o 21 000 USD. Nejvyšší ohodnocení vykazovaly pozice Applied/Core AI s průměrem 150 498 USD (medián 148 495 USD), tedy nominální prémii přibližně 31 000 USD oproti pozicím bez AI. Tyto hrubé nominální rozdíly však zahrnují vliv zavádějících proměnných (confounders), jako jsou vzdělání, zkušenosti, region a typ pozice. Čistá AI prémie po kontrole těchto faktorů je kvantifikována v sekci 5.3.

### 5.1.3 Vzdělání a zkušenosti

Z hlediska požadovaného vzdělání vyžadovalo 60,4 % inzerátů ve vzorku titul bakalář nebo vyšší. V granulárním rozlišení (používaném v OLS modelech): 58,5 % pozic požadovalo bakalářský titul, 1,9 % magisterský nebo vyšší, 1,9 % asociovaný titul, 2,1 % středoškolské vzdělání a 35,7 % inzerátů vzdělání nespecifikovalo.

Při srovnání podle AI statusu je důležité všimnout si asymetrie v *chybějících* údajích o vzdělání: u non-AI inzerátů chybí vzdělání u 33,6 % pozic, u AI inzerátů je tento podíl 43,6 %. Po vyloučení pozic bez uvedeného vzdělání vyžaduje bakalářský titul nebo vyšší přibližně 93,2 % non-AI inzerátů (8 772 / 9 417), oproti 96,7 % AI inzerátů (1 999 / 2 067). Zdánlivě „nižší" formální požadavky AI pozic v hrubých součtech jsou tedy artefaktem vyšší missingness, nikoli reálným signálem nižších kvalifikačních nároků. Z tohoto důvodu je v multinomiálním modelu (sekce 5.4) kategorie *Missing* ponechána jako samostatná úroveň proměnné `edu_logit` (Missing / HS+Associate / Bachelor+) a jako referenční kategorie je použita Bachelor+ (`ib2.edu_logit`). Chí-kvadrát test tří úrovní vzdělání proti AI flagu dává χ²(2) = 161,3; p < 0,001.

Požadované zkušenosti se ve vzorku rozložily následovně: 47,1 % pozic požadovalo 3–5 let praxe (kategorie Mid), 20,4 % Senior+ (6 a více let), 18,0 % Junior (0–2 roky) a 14,5 % pozic zkušenosti nespecifikovalo. Průměrný požadavek na minimální zkušenost činil 4,5 roku (medián 5 let).

### 5.1.4 Pracovní režim a AI

Celkový podíl pozic nabízejících práci na dálku (remote) činil 28,4 %. Zajímavý je však rozdíl v rozložení podle AI požadavků: mezi pozicemi s AI požadavky nabízelo vzdálenou práci 37,4 % inzerátů, zatímco u pozic bez AI to bylo pouze 26,1 %. Tento rozdíl je statisticky vysoce významný (Pearsonův χ² = 182,7; p < 0,001). AI pozice jsou tedy signifikantně flexibilnější z hlediska místa výkonu práce, což může reflektovat jak povahu AI práce (digitální, nezávislá na fyzické lokaci), tak i konkurenční strategii zaměstnavatelů při náboru vzácných AI talentů.

### 5.1.5 Sektorové a profesní rozložení

Sektorově dominují ve vzorku informační technologie (42,3 % ze známých sektorů), následované výrobou (8,5 %) a finančními službami (7,5 %). Koncentrace AI požadavků se však dramaticky liší podle profesní skupiny (job family). Nejvyšší podíl AI požadavků vykazuje kategorie Data & AI, kde AI dovednosti vyžaduje 55,7 % pozic. Následují seniorní softwaroví inženýři (33,3 %) a softwaroví inženýři (26,8 %). Naopak Software Developer vykazuje podstatně nižší podíl (15,9 %) a DevOps & Cloud 17,8 %. Tyto rozdíly jsou statisticky významné (χ² = 1 200; p < 0,001).

Podíl AI požadavků rovněž roste se senioritou: u juniorních pozic (0–2 roky) vyžaduje AI 17,2 % inzerátů, u středně zkušených (3–5 let) 22,1 % a u seniorních (6+ let) 22,4 %. Při rozlišení na podkategorie je patrné, že podíl Applied/Core AI roste výrazněji: z 6,2 % u juniorních pozic přes 7,4 % u středních až po 8,7 % u seniorních. To naznačuje, že hlubší AI expertíza je častěji požadována u zkušenějších pracovníků.

### 5.1.6 Geografická koncentrace AI pozic

AI pozice vykazují výraznou geografickou koncentraci na západě USA (West: 28,8 % AI pozic vs. 21,1 % non-AI) a v kategorii celostátních/remote pozic (Unknown: 32,4 % AI vs. 24,1 % non-AI). Naopak regiony Midwest (7,7 % AI vs. 11,6 % non-AI) a South (20,0 % AI vs. 31,1 % non-AI) jsou podreprezentovány. Tyto rozdíly jsou statisticky významné (χ² = 321,7; p < 0,001) a odrážejí známou koncentraci technologického průmyslu v Silicon Valley, San Franciscu a Seattlu.

## 5.2 Statistické testy

Před konstrukcí regresních modelů byly provedeny inferenční testy k ověření, zda pozorované deskriptivní rozdíly nejsou dílem náhody.

### 5.2.1 Test mzdových rozdílů

Dvouvýběrový t-test prokázal statisticky vysoce významný rozdíl v průměrných ročních platech mezi AI a non-AI pozicemi (t = −29,44; p < 0,001). Hrubý mzdový rozdíl činil 24 557 USD ročně ve prospěch AI pozic. Cohenovo d nabylo hodnoty 0,600, což indikuje středně silný praktický efekt. AI mzdová prémie tedy není pouze statistickým artefaktem velkého vzorku, ale představuje věcně významný rozdíl.

Robustnost tohoto závěru byla ověřena neparametrickým Mann-Whitneyovým U testem, který nevyžaduje předpoklad normálního rozdělení mezd. Výsledek (z = −27,84; p < 0,001) plně potvrdil závěr parametrického testu.

### 5.2.2 ANOVA s Bonferroniho korekcí

Pro ověření, zda mzdové rozdíly existují i mezi jednotlivými úrovněmi AI (None vs. AI Integration vs. Applied/Core AI), byla provedena analýza rozptylu (ANOVA) s post-hoc Bonferroniho korekcí pro mnohočetná srovnání. Všechny tři páry se vzájemně statisticky signifikantně lišily (p < 0,001 u všech porovnání): Applied AI vs. AI Integration (+10 010 USD), Non-AI vs. AI Integration (−20 955 USD), Non-AI vs. Applied AI (−30 965 USD). Mzdová prémie tedy roste monotónně s hloubkou AI expertízy, což validuje rozlišení úrovní AI přijaté v této práci.

Robustní neparametrickou alternativou (Kruskal-Wallisův test) byl tento výsledek opětovně potvrzen.

### 5.2.3 Chi-kvadrát testy profilových charakteristik

Distribuce požadovaných zkušeností se signifikantně liší mezi AI a non-AI pozicemi (χ² = 73,24; p < 0,001). AI pozice vykazují vyšší podíl požadavků na střední a seniorní úroveň zkušeností, zatímco juniorní a nespecifikované pozice jsou u AI relativně méně zastoupeny. Toto zjištění potvrzuje, že AI pozice vyžadují od uchazečů objemnější profil lidského kapitálu.

## 5.3 AI prémie na platu (OLS regresní analýza)

Centrální otázkou této práce je kvantifikace čisté mzdové prémie za AI dovednosti po kontrole zavádějících proměnných. K tomuto účelu byly zkonstruovány OLS regresní modely vysvětlující přirozený logaritmus roční mzdy (ln_salary), v souladu s Mincerovou mzdovou rovnicí. Procentuální interpretace semi-logaritmických koeficientů je v celé práci vyjádřena standardní aproximací β × 100 %, která je v ekonometrii běžně používána pro koeficienty s absolutní hodnotou do přibližně 0,15 (kde se aproximační chyba pohybuje do 1 procentního bodu). Pro koeficienty s velkou absolutní hodnotou (např. country dummies v komparativní analýze) je použit přesný výpočet (exp(β) − 1) × 100 %.

### 5.3.1 Model A — Firemní profil

Model A představuje výchozí specifikaci obsahující pouze firemní a kontextové proměnné: úroveň AI požadavku (ai_level: 0 = žádné AI, 1 = AI Integration, 2 = Applied/Core AI), NACE sektor, region, možnost remote práce, typ a velikost organizace. Model vysvětluje 20,5 % variability mezd (R² = 0,205, N = 14 640, robustní standardní chyby).

V tomto nejjednodušším modelu dosahuje koeficient pro AI Integration hodnoty +0,117 (p < 0,001), což odpovídá přibližně 11,7% mzdové prémii. Pro Applied/Core AI činí koeficient +0,176 (p < 0,001), odpovídající 17,6% prémii. Tyto odhady představují hrubou AI prémii kontrolovanou pouze za firemní charakteristiky a stále zahrnují vliv osobních charakteristik uchazeče (vzdělání, zkušenosti) i technologického profilu pozice, které mohou být korelovány s AI požadavky.

### 5.3.2 Model B — Lidský kapitál

Model B rozšiřuje specifikaci o proměnné lidského kapitálu: granulární vzdělání ve čtyřech úrovních (High School, Associate, Bachelor, Master+) a kategorii zkušeností (Missing, Junior 0–2, Mid 3–5, Senior+ 6+). Přidání těchto proměnných zvýšilo vysvětlující sílu modelu na R² = 0,310, tedy nárůst o 10,5 procentního bodu oproti Modelu A. Společnou signifikanci přidaného bloku lidského kapitálu potvrzuje Waldův F-test: F(7; 14 582) = 271,68; p < 0,001. Tento test je preferován před LR testem, protože nepředpokládá homoskedasticitu — LR test by byl pro robustní specifikaci nekonzistentní.

Po kontrole vzdělání a zkušeností se AI prémie očekávaně snížila: AI Integration +10,5 % (koeficient 0,105; p < 0,001) a Applied/Core AI +16,3 % (koeficient 0,163; p < 0,001). Pokles oproti Modelu A (z 11,7 % na 10,5 % u Integration, z 17,6 % na 16,3 % u Applied/Core) ukazuje, že přibližně 7–10 % hrubé AI prémie bylo vysvětleno tím, že AI pozice systematicky vyžadují vzdělanější uchazeče s delší praxí. Většina AI prémie však zůstává nevysvětlena samotným lidským kapitálem.

### 5.3.3 Model C — Plný model (technické dovednosti a profesní profil)

Model C představuje plnou specifikaci, která k Modelu B přidává 21 binárních proměnných technologických skill clusterů a profesní skupinu (job family). Přidání těchto proměnných zvýšilo vysvětlující sílu na R² = 0,380 (+7,0 p. b. oproti Modelu B). Společnou signifikanci přidaného bloku (cluster_* + job_family) potvrzuje Waldův F-test: F(27; 14 582) = 70,00; p < 0,001.

Konečné odhady čisté AI prémie v Modelu C činí: AI Integration +7,5 % (koeficient 0,075; p < 0,001) a Applied/Core AI +9,6 % (koeficient 0,096; p < 0,001). Oproti Modelu B došlo k výraznému poklesu (z 10,5 % na 7,5 % u Integration, z 16,3 % na 9,6 % u Applied/Core), což ukazuje, že podstatná část AI prémie byla vysvětlena specifickým technologickým profilem pozice — AI pozice vyžadují širší a hodnotnější portfolio technických dovedností. I po této kontrole však zůstává statisticky vysoce významná a ekonomicky relevantní čistá AI prémie.

Hierarchická struktura modelů A → B → C tak ukazuje postupnou dekompozici hrubé AI prémie: z původních 11,7/17,6 % (Model A) přes 10,5/16,3 % po kontrole lidského kapitálu (Model B) až na finálních 7,5/9,6 % po kontrole všech faktorů (Model C). Celkově tedy přibližně 36 % hrubé AI Integration prémie a 45 % hrubé Applied/Core AI prémie je vysvětleno měřitelnými charakteristikami pozice a uchazeče.

Diagnostika multikolinearity pomocí VIF (Variance Inflation Factor) potvrdila absenci problematické kolinearity. Průměrný VIF Modelu C činil 1,92, přičemž maximální individuální VIF dosáhl 6,53 pro jednu z kategorií profesní skupiny (job family) — pod kritickou hranicí 10. Mírně zvýšené VIF u kategorických proměnných s mnoha úrovněmi (job family, type) jsou konstrukční povahy a nepředstavují problém pro odhady.

### 5.3.4 Hierarchie faktorů určujících plat

AI prémie je sice reálná a statisticky významná, avšak není nejsilnějším prediktorem platu. Klíčové je zasadit ji do kontextu ostatních faktorů. Na základě koeficientů Modelu C lze vytvořit hierarchii vlivů na mzdu. Nejsilnější efekt vykazuje geografická lokace — pozice v regionu West (Silicon Valley, Seattle) nesou prémii +22,4 % oproti referenční kategorii. Seniorní softwarový inženýr (Sr+ Software Engineer) má prémii +17,3 % oproti referenční profesní skupině. Zkušenost 6 a více let přináší +12,0 % oproti střední kategorii 3–5 let. Applied/Core AI s +9,6 % se řadí na úroveň manažerské prémie (+9,4 %). AI Integration s +7,5 % je srovnatelná s prémií za dovednosti v oblasti systems programming (+7,3 %) a DevOps/Cloud (+7,0 %).

Na opačném konci spektra: juniorní pozice (0–2 roky) nesou penalizaci −17,5 %, Associate degree −X,X % a High School −X,X % oproti referenční kategorii Bachelor. Toto uspořádání ukazuje, že AI prémie je aditivním bonusem nad rámec tradičních faktorů lidského kapitálu, nikoliv jejich náhradou.
<!-- TODO: Doplnit přesné koeficienty edu_ols po novém Stata runu (baseline = Bachelor) -->

### 5.3.5 Test mediace přes profesní skupinu

Profesní skupina (job family) může být mediátorem vztahu mezi AI dovednostmi a mzdou, neboť AI pozice se koncentrují v lépe placených profesních skupinách (Data & AI, Sr+ Software Engineer). Pro ověření byl odhadnut Model C bez proměnné job_family (R² = 0,342). V tomto modelu vzrostla AI prémie na +8,4 % (AI Integration) a +11,8 % (Applied/Core AI), což potvrzuje částečnou mediaci. Job family absorbuje přibližně 1–2 procentní body AI efektu, což naznačuje, že typ pozice je jedním z kanálů, kterým se AI dovednosti promítají do vyšších mezd.

### 5.3.6 Mincerova specifikace (kontinuální zkušenosti)

Pro soulad s klasickou Mincerovou mzdovou rovnicí byl odhadnut Model C-Mincer, kde kategorická proměnná zkušeností byla nahrazena kontinuální proměnnou (roky minimální požadované zkušenosti) a jejím kvadratickým členem. Výsledky potvrdily klesající výnosy ze zkušeností: každý další rok praxe přidává +6,8 %, ale s klesající intenzitou (kvadratický člen −0,2 %). AI prémie zůstala konzistentní: AI Integration +7,7 %, Applied/Core AI +9,9 %. R² = 0,423, nicméně tento model běží na menším vzorku (N = 12 606), neboť pozorování s chybějícími údaji o zkušenostech jsou automaticky vyřazena, na rozdíl od kategorického modelu, kde jsou kódována jako explicitní kategorie.

## 5.4 Determinanty AI požadavku (logistické modely)

Mzdové modely (sekce 5.3) ukázaly, kolik si za AI dovednosti zaměstnavatelé připlatí. Stejně podstatná je však otázka opačného směru: *které pozice AI vůbec vyžadují a co je pro ně charakteristické?* Logistická regrese s binární vysvětlovanou proměnnou `has_ai` umožňuje identifikovat dovednostní profily a organizační charakteristiky, které jsou s AI požadavky asociovány. Výsledkem je ekonometrická analýza odborné náročnosti — formální kvantifikace toho, jaké technologické zázemí a jaký typ role zvyšují pravděpodobnost, že zaměstnavatel po uchazeči žádá práci s umělou inteligencí.

### 5.4.1 Inkrementální struktura modelů

Klíčové designové rozhodnutí předcházelo samotnému odhadu: z prediktivní sady byly a priori vyřazeny skill clustery Generative AI a Data Science / ML. Důvod je logický — přítomnost dovedností typu GPT, LLM, TensorFlow či PyTorch v inzerátu přímo *implikuje* AI požadavek, nikoliv jej predikuje. Jejich zahrnutí by model činilo cirkulárním. Kvantifikace dopadu tohoto rozhodnutí je předmětem citlivostní analýzy (sekce 5.5.2).

Modely byly odhadovány inkrementálně ve třech krocích a již první výsledek přináší zásadní zjištění. Model M1 (Profil firmy), zahrnující pouze NACE sektor, typ organizace, velikost firmy a region, dosáhl Pseudo R² pouhých 2,5 %. **Vlastnosti zaměstnavatele tedy vysvětlují jen nepatrný zlomek toho, zda pozice vyžaduje AI.** Jinými slovy: AI prostupuje IT trhem průřezově — není koncentrována ve velkých korporacích, v konkrétním sektoru ani v konkrétním regionu. Toto je jeden z ústředních empirických výsledků práce.

Teprve Model M2 (Profil role), zahrnující technologické skill clustery, profesní skupinu, vzdělání a zkušenosti, posunul Pseudo R² na 11,7 %. Přidání firemních proměnných v plném Modelu M3 přineslo marginální zlepšení na 13,0 %. Poměr přínosů je výmluvný: rolové charakteristiky vysvětlují téměř pětkrát více variability než firemní profil. **O tom, zda pozice vyžaduje AI, rozhoduje „co se tam dělá", nikoliv „kdo je zaměstnavatel".**

Všechny modely konvergovaly bez problémů (4–6 iterací, žádné „not concave" ani „perfect separation" chyby). Diagnostika Modelu M3 potvrdila adekvátní kvalitu odhadu: Hosmer-Lemeshowův test dobré shody je nesignifikantní (χ²(8) = 7,38; p = 0,496), AUC = 0,747 odpovídá přijatelné diskriminační schopnosti a průměrný VIF (OLS proxy) = 1,82 s maximem 6,17 vylučuje problematickou multikolinearitu. Linktest funkcionální formy odhalil mírnou misspecifikaci (_hatsq p = 0,006), která po přidání dvou teoreticky motivovaných interakcí (region × cloud a size × dynamic_web) vymizela (p = 0,053) — efekt cloudových dovedností se liší podle regionu (silnější v tech hubech) a efekt webových technologií podle velikosti firmy. Tyto interakce nejsou v hlavním modelu zahrnuty kvůli komplikaci interpretace AME; jejich absence představuje mírnou limitaci kompenzovanou příznivou HL diagnostikou. Klasifikační tabulka vykazuje celkovou přesnost 80,5 % (senzitivita 17,2 %, specificita 96,9 %), kde nízká senzitivita je standardním důsledkem nevyváženosti dat (~21 % AI pozic). Cílem analýzy však není klasifikace, nýbrž odhad marginálních efektů, které jsou na volbu klasifikačního prahu nezávislé.

### 5.4.2 Technologická propast: moderní stack versus legacy systémy

Výsledky Modelu M3, reportované ve formě průměrných marginálních efektů (AME), odhalují výraznou technologickou propast na IT trhu práce. Na jedné straně stojí moderní, cloudově orientovaný technologický stack, jehož přítomnost v inzerátu výrazně zvyšuje pravděpodobnost AI požadavku. Na druhé straně tradiční podnikové a infrastrukturní technologie, které s AI pozicemi korelují negativně.

Nejsilnějším pozitivním prediktorem je skill cluster **Dynamic Web** (AME = +9,4 p. b.; p < 0,001): přítomnost dovedností typu React, Node.js nebo moderních webových frameworků zvyšuje pravděpodobnost AI požadavku o téměř 10 procentních bodů. Tento výsledek odráží reálný tržní trend — AI funkce (chatboty, doporučovací systémy, personalizace) jsou nejčastěji nasazovány právě prostřednictvím webových aplikací. Následují **Cloud Computing** (+6,6 p. b.), **Enterprise Platforms** (+5,0 p. b.), **Data Engineering** (+4,5 p. b.) a **Frontend Development** (+3,6 p. b.). Doplňují je BI / Analytics (+3,5 p. b.), Systems Programming (+3,3 p. b.), DevOps / Containers (+2,6 p. b.) a Backend Development (+2,4 p. b.).

Opačný pól tvoří technologie asociované s tradičními IT rolemi. **Enterprise / Managed systémy** (AME = −6,5 p. b.; p < 0,001) a **Scripting / Shell** (−5,6 p. b.; p < 0,001) vykazují nejsilnější negativní asociaci, následovány certifikacemi (−3,9 p. b.), Databases / Storage (−3,4 p. b.) a OS / Embedded (−3,3 p. b.).

Výsledný obraz je jednoznačný: **AI požadavky se na trhu práce koncentrují v pozicích s moderním, cloudově-webovým technologickým profilem, zatímco tradiční podnikové, infrastrukturní a legacy role AI zatím výrazněji neadoptovaly.** Tato propast má implikace pro vzdělávací politiku i individuální kariérní strategii — přechod k AI nepředpokládá pouze osvojení samotných AI nástrojů, ale celkovou modernizaci technologického zázemí.

Významný je také výsledek pro organizační proměnné. Typ organizace (Private, Public, Nonprofit/Gov) nemá statisticky významný vliv v žádné specifikaci (p > 0,27). Stejně tak seniorita po kontrole skill clusterů a job family ztrácí signifikanci (AME Junior p = 0,220; Senior+ p = 0,184). AI požadavky tedy *nejsou* výsadou seniorních pozic — pronikají na všechny úrovně seniority a napříč všemi typy zaměstnavatelů, pokud má pozice odpovídající technologický profil.

### 5.4.3 Interpretační rámec a limity kauzální inference

Model je třeba interpretovat v kontextu jeho designu. Skill clustery jsou extrahovány ze stejného textu inzerátu, ze kterého LLM klasifikátor přidělil AI status, proto nelze hovořit o kauzalitě v tradičním smyslu. Přínos modelu je jiný: poskytuje *formální ekonometrickou kvantifikaci* toho, jaký technologický profil je s AI pozicemi asociován, a doplňuje tak deskriptivní analýzu odborné náročnosti (sekce 5.1) o statisticky testovatelný rámec. Zjištění o technologické propasti (moderní stack vs. legacy), průřezovém charakteru AI adopce a irelevanci organizačního typu jsou robustní korelační výsledky, i když kauzální mechanismy za nimi vyžadují další výzkum.

### 5.4.4 Test mediace: proč je důležité rozlišovat „typ role" od „dovedností"

Profesní skupina (job family) a úroveň seniority mohou představovat mediátory vztahu mezi technologickými dovednostmi a pravděpodobností AI požadavku. Pokud je tomu tak, plný model jejich vliv absorbuje a skutečný efekt dovedností je v něm podhodnocen. Toto rozlišení je pro interpretaci výsledků zásadní: chceme vědět, zda AI požadavek determinuje konkrétní pozice (tj. „jsi ML Engineer, proto děláš AI"), nebo konkrétní dovednostní profil (tj. „umíš cloud a moderní web, proto je pravděpodobné, že budeš dělat AI").

Pro identifikaci mediačních efektů ve smyslu Baron & Kenny (1986) byly odhadnuty tři vnořené specifikace: M3 (plný model; Pseudo R² = 0,130), M3-nojf (bez job_family; Pseudo R² = 0,103) a M3-nojf-noexp (bez job_family i seniority; Pseudo R² = 0,102).

**Profesní skupina je silný mediátor.** Po vyřazení job_family se efekty klíčových skill clusterů výrazně posílily: Cloud Computing vzrostl z +6,6 na +8,5 p. b. (+29 %), Data Engineering z +4,5 na +7,3 p. b. (+62 %) a negativní efekt Scripting/Shell se prohloubil z −5,6 na −7,8 p. b. (+39 %). Nově signifikantními se staly clustery Mobile/Desktop (p = 0,036), Security/Identity (p = 0,003) a Testing/QA (p = 0,023), které byly v plném modelu profesní skupinou absorovány. Mediace je konzistentní s podmínkami Baron & Kenny (1986): technologické dovednosti predikují jak AI požadavek, tak profesní skupinu, a efekt dovedností se po zahrnutí mediátoru (job_family) signifikantně zeslabí.

**Seniorita mediátorem není.** Porovnání M3-nojf vs. M3-nojf-noexp ukazuje, že odebrání seniority téměř nic nemění (ΔPseudo R² = 0,001; ΔLL = 8,4). Koeficienty skill clusterů se mezi oběma specifikacemi prakticky neliší. Tento výsledek má přímou implikaci pro trh práce: **AI požadavky nejsou vázány na úroveň zkušeností — juniorní pozice s moderním tech stackem mají stejnou pravděpodobnost AI požadavku jako seniorní.**

Rozlišení mediátorů (job family) od nezávislých prediktorů (skill clustery) je klíčové pro správnou interpretaci. Plný model M3 ukazuje „čistý" efekt dovedností po kontrole typu role; model bez job_family ukazuje kumulativní efekt dovedností včetně jejich nepřímého působení skrze profesní zařazení. Obě perspektivy jsou analyticky hodnotné, avšak pro porozumění tomu, *které dovednosti přitahují AI*, je informativnější model bez mediátoru.

### 5.4.5 Chybějící hodnoty: náhodné, nebo systematické?

Kategorie „Unknown" a „Missing" vykazují v Modelu M3 systematicky signifikantní efekty, což naznačuje, že chybějící hodnoty nejsou v datech náhodné (MNAR — Missing Not At Random). Tyto efekty však nejsou statistickým artefaktem — mají smysluplnou interpretaci v kontextu IT trhu práce.

Unknown region (AME = +5,0 p. b.; p < 0,001) pravděpodobně zachycuje celostátní a plně remote pozice, které jsou s AI asociovány silněji (viz sekce 5.1.4). Missing education (AME = +4,3 p. b.; p < 0,001) odpovídá technicky náročným rolím, kde zaměstnavatel upřednostňuje portfolio dovedností před formálním vzděláním — právě tento přístup je typický pro AI-orientované firmy. Unknown velikost firmy (AME = −3,6 p. b.; p = 0,004) může reflektovat menší startupy, které údaje na Glassdoor nezveřejňují a mají nižší adopci AI.

Robustnost výsledků vůči chybějícím hodnotám byla ověřena na podvzorku se známým vzděláním (N = 11 484): koeficienty ostatních prediktorů zůstaly konzistentní (Pseudo R² = 0,118 vs. 0,130 na plném vzorku), což potvrzuje, že missing-indicator metoda (Allison, 2001) nezkresluje odhady klíčových proměnných ani při 35,7 % chybějících hodnot vzdělání.

## 5.5 Dovednostní profil AI pozic

### 5.5.1 Kvantitativní složitost AI pozic

AI pozice vyžadují objemnější portfolio technických dovedností. Průměrný počet požadovaných hard skills u pozic bez AI činil přibližně 16,0 (SD = 10,1), u AI Integration pozic 19,5 (SD = 10,4) a u Applied/Core AI pozic 20,6 (SD = 10,8). Rozdíl přibližně 4 dovedností mezi non-AI a AI pozicemi je statisticky vysoce významný (t = −20,0; p < 0,001) a reflektuje komplexnější povahu AI rolí, kde je od uchazečů očekáváno zvládání širšího arzenálu technologií.

### 5.5.2 Citlivostní analýza: dopad vyřazení cirkulárních clusterů

Hlavní logistické modely (sekce 5.4) a priori vyřazují clustery Generative AI a Data Science / ML jako tautologické s vysvětlovanou proměnnou. Pro kvantifikaci dopadu tohoto designového rozhodnutí byla provedena citlivostní analýza, kde byly oba clustery do Modelu M3 vráceny.

Po zařazení cirkulárních clusterů vzrostlo Pseudo R² z 13,0 % na přibližně 34 %. Generative AI se stala naprosto dominantním prediktorem (AME ≈ +32 p. b.) a Data Science / ML druhým nejsilnějším (AME ≈ +26 p. b.). Tyto dva clustery tedy vysvětlovaly přibližně 21 procentních bodů variability, což představuje zhruba 62 % celkové prediktivní schopnosti rozšířeného modelu.

Přítomnost cirkulárních clusterů zároveň potlačila efekty ostatních prediktorů. Ve srovnání s hlavním modelem (bez GenAI/DS-ML) klesly AME zbývajících clusterů přibližně o polovinu: Cloud Computing z +6,6 na ≈ +3 p. b., Dynamic Web z +9,4 na ≈ +5 p. b., Data Engineering z +4,5 na ≈ +1 p. b. Některé clustery (DevOps/Containers, Systems Programming) ztratily signifikanci zcela. Tento vzorec potvrzuje správnost vyřazení: při ponechání cirkulárních clusterů v modelu by efekty ostatních technologických dovedností byly podceněny, neboť GenAI a DS/ML absorbují variabilitu, která ve skutečnosti pramění z širšího technologického profilu pozice.

<!-- TODO: Čísla citlivostní analýzy (Pseudo R² ≈ 34 %, AME GenAI/DS-ML) pocházejí z dřívějšího referenčního runu. Pro finální verzi práce zvážit přidání explicitní logit specifikace s GenAI/DS-ML do do-filu jako formální citlivostní test. -->

### 5.5.3 Multinomiální logit — „používání AI" versus „vývoj AI"

Stěžejní analytickou přidanou hodnotou je rozlišení mezi pozicemi, které AI pouze integrují do svých procesů (AI Integration), a pozicemi, které AI přímo vyvíjejí (Applied/Core AI). K tomuto účelu byl odhadnut multinomiální logistický model (mlogit) se třemi kategoriemi výsledné proměnné (None, AI Integration, Applied/Core AI) a referenční kategorií None. Shodně s binárním logitem (sekce 5.4) byly cirkulární clustery Generative AI a Data Science / ML a priori vyřazeny; vzdělání (edu_logit) bylo rovněž vyřazeno kvůli nedostatečné buňkové frekvenci (HS/Associate × Applied AI = 23 obs < 50). Modely byly odhadovány inkrementálně (M1–M3) se shodnou strukturou jako v sekci 5.4; níže jsou reportovány výsledky plného Modelu M3 (Pseudo R² = 0,125).

Výsledky AME z multinomiálního Modelu M3 odhalují kvalitativně odlišné dovednostní profily obou kategorií — a právě v *rozdílech* mezi nimi spočívá hlavní přínos multinomiálního přístupu oproti binárnímu logitu.

#### Společné prediktory: co zvyšuje pravděpodobnost AI požadavku obecně

Některé skill clustery zvyšují pravděpodobnost obou typů AI pozic, avšak s rozdílnou intenzitou. **Dynamic Web** je nejsilnějším pozitivním prediktorem u obou kategorií, přičemž efekt je silnější pro Applied/Core AI (+5,1 p. b.; p < 0,001) než pro AI Integration (+4,4 p. b.; p < 0,001). Podobně **Cloud Computing** predikuje obě kategorie, ale téměř dvojnásobně silněji AI Integration (+4,3 p. b.; p < 0,001) oproti Applied/Core AI (+2,4 p. b.; p < 0,001). Toto rozložení dává ekonomický smysl: cloudová infrastruktura je primárně *prostředkem nasazení* AI služeb (integrace), zatímco moderní webové technologie jsou nedílnou součástí jak nasazení, tak vývoje AI aplikací. Dalšími společnými pozitivními prediktory jsou BI / Analytics (Integration: +2,0 p. b.; Applied: +1,3 p. b.) a DevOps / Containers (Integration: +1,6 p. b.; Applied: +1,0 p. b.).

Na negativním pólu stojí **Enterprise / Managed systémy** (Integration: −4,2 p. b.; Applied: −2,7 p. b.) a **Scripting / Shell** (Integration: −3,1 p. b.; Applied: −2,5 p. b.), které konzistentně snižují pravděpodobnost obou typů AI pozic — tradiční podnikové a infrastrukturní technologie s AI nesouvisejí bez ohledu na úroveň AI zapojení.

#### Diskriminující prediktory: kde se „používání" a „vývoj" AI zásadně liší

Nejcennějším analytickým výstupem multinomiálního modelu jsou clustery, jejichž efekt se mezi kategoriemi kvalitativně liší — včetně případů, kdy směřují na opačné strany.

**Systems Programming** vykazuje statisticky významný *opačný efekt* mezi kategoriemi: snižuje pravděpodobnost AI Integration o 1,7 p. b. (p = 0,034), avšak zvyšuje pravděpodobnost Applied/Core AI o 4,2 p. b. (p < 0,001). Jinými slovy, nízkoúrovňové programování (C, C++, Rust, embedded systémy) je aktivně *neslučitelné* s pouhou integrací AI nástrojů, ale představuje klíčovou kompetenci pro vývoj AI infrastruktury — optimalizace inferenčních pipeline, GPU computing, deployment AI modelů na edge zařízeních.

**Data Engineering** představuje nejčistší diskriminátor: nemá žádný statisticky významný vliv na P(AI Integration) (+0,2 p. b.; p = 0,703), avšak je jedním z nejsilnějších prediktorů P(Applied/Core AI) (+4,2 p. b.; p < 0,001). Správa datových pipeline, ETL procesy a datová architektura jsou tedy výhradně doménou AI vývoje, nikoliv integrace — Applied/Core AI pozice vyžadují schopnost zpracovat a připravit data pro trénink modelů, zatímco pozice využívající hotové AI API tuto dovednost nepotřebují.

**Frontend Development** a **Enterprise Platforms** vykazují opačný vzorec: silně zvyšují P(AI Integration) (Frontend: +4,9 p. b., p < 0,001; Enterprise Platforms: +5,8 p. b., p < 0,001), ale nemají statisticky významný vliv na P(Applied/Core AI) (Frontend: −0,7 p. b., ns; Enterprise Platforms: −0,7 p. b., ns). Tyto aplikační dovednosti odpovídají pozicím, kde se AI nasazuje do existujících uživatelských rozhraní a podnikových systémů — chatboty v e-commerce, AI-poháněné dashboardy, doporučovací systémy na webu.

**Backend Development** se rovněž profiluje jako prediktor Integration (+2,2 p. b.; p < 0,001) spíše než Applied AI (+0,4 p. b.; ns), což odráží roli backendových vývojářů v integraci AI služeb přes API.

#### Souhrnná interpretace

Výsledné dovednostní profily obou kategorií jsou konzistentní s teoretickým rozlišením úrovní AI adopce. Applied/Core AI pozice vyžadují *fundamentální inženýrské dovednosti* (systémové programování, data engineering, nízkoúrovňová optimalizace), zatímco AI Integration pozice se vyznačují *aplikačními dovednostmi* (frontend, enterprise platformy, cloudové nasazení). Toto zjištění přímo validuje třístupňové rozlišení AI požadavků, které je ústředním konceptem této práce: binární logit odpoví na otázku „co predikuje AI?", multinomiální logit na otázku „co odlišuje používání AI od vývoje AI?".

#### Profesní skupiny a seniorita

Z hlediska profesních skupin je asymetrie mezi kategoriemi markantní. Oproti referenční kategorii Data & AI mají všechny ostatní job families dramaticky nižší pravděpodobnost Applied/Core AI: Other (−20,9 p. b.; p < 0,001), DevOps & Cloud (−19,5 p. b.; p < 0,001), Software Developer (−19,6 p. b.; p < 0,001), Software Engineer (−15,2 p. b.; p < 0,001) a Sr+ Software Engineer (−14,6 p. b.; p < 0,001). U AI Integration jsou efekty mnohonásobně slabší a většinou statisticky nevýznamné (Software Developer: −7,0 p. b.; Sr+ Software Engineer: −2,1 p. b., ns). Integrace AI tedy proniká relativně rovnoměrně napříč profesemi, zatímco vývoj AI zůstává silně soustředěn v datových a AI specializacích.

Seniorita vykazuje odlišný vzorec pro obě kategorie. Pro AI Integration mají všechny úrovně zkušeností signifikantně nižší pravděpodobnost oproti referenční kategorii Mid (3–5 let): Junior −1,5 p. b. (p = 0,040), Senior+ −1,6 p. b. (p = 0,016), Missing −2,7 p. b. (p < 0,001). Pro Applied/Core AI jsou však efekty seniority statisticky nevýznamné (Junior: −0,1 p. b., p = 0,851; Senior+: +0,6 p. b., p = 0,193). Vývoj AI tedy není vázán na konkrétní úroveň zkušeností — rozhodující jsou technické dovednosti, nikoliv délka praxe.

#### Test mediace v multinomiálním rámci

Mediační analýza (modely M3a bez job_family a M3b bez job_family i seniority) potvrzuje vzorce zjištěné v binárním logitu (sekce 5.4.4), avšak s důležitými nuancemi specifickými pro jednotlivé AI kategorie. Po vyřazení job_family z modelu vzrostl efekt Data Engineering na P(Applied AI) z +4,2 na +6,4 p. b. (+52 %), což potvrzuje, že podstatná část efektu Data Engineering je mediována přes profesní skupinu. Současně se v modelu bez job_family stávají nově signifikantními Security / Identity (−1,8 p. b.; p < 0,001), Testing / QA (−1,2 p. b.; p = 0,003) a Frontend Development (−1,5 p. b.; p = 0,001) pro Applied AI — tyto efekty jsou v plném modelu absorovány profesní skupinou. Pokles log-likelihood (M3: −10 020 → M3a: −10 356) potvrzuje, že job_family je silný mediátor; seniorita jím prakticky není (M3a → M3b: ΔLL = 12).

#### Předpoklady a limity multinomiálního logitu

Hausmanovým testem byla ověřena nezávislost irelevantních alternativ (IIA), základní předpoklad multinomiálního logitu. Test formálně zamítá IIA (χ²(49) = 92,16; p = 0,0002), avšak variančně-kovarianční matice V_b − V_B není pozitivně definitní, což činí výsledek nespolehlivým — jedná se o známou limitaci Hausmanova testu u multinomiálních modelů s mnoha prediktory (Long & Freese, 2014; Cheng & Long, 2007). Multinomiální logit je v této práci použit jako exploratorní nástroj pro identifikaci kvalitativních dovednostních profilů; AME vzory (směr a relativní velikost efektů) jsou robustnější než přesné bodové odhady koeficientů. Tato limitace je diskutována v sekci 5.7.

## 5.6 Komparativní analýza zemí (USA, Německo, Indie)

Komparativní analýza rozšiřuje záběr o tři ekonomicky, institucionálně a kulturně odlišné trhy práce s použitím **odlišného datasetu** než sekce 5.1–5.5. Zatímco předchozí analýza byla zaměřena na americký trh práce (USA-only), komparativní část kombinuje data ze tří zemí. Celkový sdružený dataset čítá 44 832 pozorování (USA: 18 464, Indie: 17 114, Německo: 9 254). Po filtraci (confidence ≥ 0,7 a vyloučení starších inzerátů) zůstalo 38 432 pozorování. Mzdy byly převedeny na USD pevnými kurzy za období scrapingu (EUR/USD = 1,165; INR/USD = 88) a anualizovány s použitím country-specific odpracovaných hodin ročně (USA: 2 080 h, DE: 1 607 h dle OECD 2024, IN: 1 920 h). Odlehlé mzdové hodnoty mimo rozsah 3 000–500 000 USD (855 pozorování) byly vyřazeny.

### 5.6.1 Podíl AI pracovních míst podle země

Podíl AI pozic se dramaticky liší mezi zeměmi. USA vykazují nejvyšší podíl AI požadavků (19,5 %), následovány Německem (15,7 %) a výrazně zaostávající Indií (6,2 %). Tyto rozdíly jsou vysoce statisticky významné (Pearsonův χ² = 1 200; p < 0,001). Procenta se počítají z filtrovaného datasetu (po aplikaci filtru confidence ≥ 0,7; N = 38 432).

Nízký podíl AI pozic v Indii může reflektovat několik strukturálních faktorů: odlišnou maturitu IT trhu s větším podílem outsourcingových a podpůrných rolí, nižší penetraci platformy Glassdoor (s možným zkreslením směrem k méně technologicky pokročilým firmám) a obecně jinou strukturu poptávaných IT služeb.

### 5.6.2 Mzdové hladiny a pokrytí mzdových dat

Mediánové roční platy (v USD) se řádově liší: USA 117 869 USD, Německo 79 104 USD, Indie 6 535 USD. Tyto rozdíly primárně odrážejí cenovou hladinu a ekonomickou úroveň jednotlivých zemí, nikoliv produktivitu per se.

Klíčovým omezením komparativní analýzy je výrazně odlišné pokrytí mzdových dat. V USA byly mzdové údaje k dispozici u 82,0 % pozorování, v Indii u 64,9 %, avšak v Německu pouze u 8,0 % (514 pozorování). Extrémně nízké pokrytí v Německu je strukturální povahy a odráží německou tradici mzdové mlčenlivosti (Gehaltsgeheimnis), absenci zákonné povinnosti zveřejňovat mzdy v inzerátech (na rozdíl od řady amerických států) a nízkou penetraci Glassdooru v Německu. Z tohoto důvodu je OLS mzdová regrese fakticky porovnáním USA vs. Indie — Německo přispívá pouze 2,1 % mzdového vzorku a koeficienty pro Německo je nutno interpretovat se značnou opatrností.

### 5.6.3 Vzdělávací požadavky

Vzdělávací požadavky se mezi zeměmi významně liší (χ² = 1 400; p < 0,001). USA nejčastěji požadují vysokoškolský titul (60,4 %), Indie v 52,0 % případů a Německo pouze ve 33,3 %. Nízký podíl v Německu koresponduje s tradicí duálního vzdělávacího systému, kde profesní kvalifikace získaná učňovským vzděláváním (Ausbildung) je zaměstnavateli uznávána jako plnohodnotná alternativa k univerzitnímu titulu.

### 5.6.4 Sdružený OLS model s country fixed effects

Pro testování, zda se AI mzdová prémie liší mezi zeměmi, byl odhadnut sdružený (pooled) Model B s fixními efekty na zemi (dummy proměnné pro Indii a USA, referenční kategorie = Německo). Model dosáhl R² = 0,936 (N = 23 897), přičemž vysoká hodnota R² je dána tím, že dummy proměnné zemí absorbují řádové mzdové rozdíly mezi ekonomikami.

Klíčové koeficienty sdruženého modelu: AI Integration +8,4 % (koeficient 0,084; p < 0,001), Applied/Core AI +11,6 % (koeficient 0,116; p < 0,001). Indie vykazuje platy přibližně o 90 % nižší než Německo (koeficient −2,269; exp(β) − 1 = −0,897), zatímco USA přibližně o 93 % vyšší (koeficient +0,656; exp(β) − 1 = 0,927). Možnost remote práce nese prémii +10,9 % (koeficient 0,109; p < 0,001).

### 5.6.5 Interakční model — homogenita AI prémie napříč zeměmi

Klíčovým testem komparativní analýzy je otázka, zda se AI mzdová prémie statisticky významně liší mezi zeměmi. K tomuto účelu byl odhadnut interakční model s plnou interakcí country × ai_level.

Interakční koeficienty (Indie × AI Integration: −0,103, p = 0,476; Indie × Applied AI: −0,030, p = 0,840; USA × AI Integration: −0,095, p = 0,494; USA × Applied AI: −0,037, p = 0,797) nejsou ani jednotlivě, ani společně statisticky významné. Společný F-test všech čtyř interakčních členů dává F(4; 23 833) = 0,15; p = 0,965.

Tento nulový výsledek je důležitým zjištěním: mezi zeměmi neexistuje statisticky významný rozdíl v AI mzdové prémii. Rozdíly mezi zeměmi se projevují ve frekvenci AI pracovních míst (viz deskriptivní statistika a multinomiální logit), nikoliv v jejich mzdovém ohodnocení relativně k non-AI pozicím. Je však nutné interpretovat tento výsledek s přiměřenou opatrností. Německo přispívá do mzdového vzorku pouhými 514 pozorováními (2,1 % celku), což výrazně omezuje statistickou sílu (power) testu pro detekci případných německých odchylek. Interakční model tak fakticky testuje homogenitu AI prémie mezi USA a Indií s vysokou silou, zatímco pro Německo je schopnost testu odhalit i středně velké rozdíly omezená. Přesto je vysoká p-hodnota společného F-testu (0,965) natolik vzdálena hranici významnosti, že závěr o aproximativní homogenitě AI prémie je oprávněný — alespoň pro dvojici USA–Indie, která tvoří 97,9 % mzdového vzorku.

### 5.6.6 Multinomiální logit — pravděpodobnost AI pozice podle země

Komparativní multinomiální logit (N = 32 279; Pseudo R² = 0,405) potvrzuje výrazné strukturální rozdíly v pravděpodobnosti AI pozice. Indie je oproti Německu přibližně o 83 % méně pravděpodobná co do vystavování AI pozic na úrovni AI Integration (RRR = 0,171; p < 0,001) i Applied AI (RRR = 0,161; p < 0,001).

Průměrné marginální efekty ukazují, že příslušnost k Indii snižuje pravděpodobnost AI Integration o 5,8 p. b. a Applied AI o 3,3 p. b. oproti Německu. USA mají oproti Německu mírně vyšší pravděpodobnost AI Integration (+3,4 p. b.; p = 0,029), avšak u Applied/Core AI je rozdíl statisticky nevýznamný (−0,4 p. b.; p = 0,734). To naznačuje, že po kontrole skill clusterů jsou USA a Německo si v hloubce AI adopce podobné — surové rozdíly v podílu AI pozic mezi USA a Německem vznikají kompozičním efektem (odlišnou strukturou pozic a dovedností), nikoliv systematicky odlišným přístupem k AI.

Nejsilnějšími prediktory AI klasifikace v komparativním modelu zůstávají Generative AI (RRR = 47,3× pro AI Integration, 69,5× pro Applied AI) a Data Science / ML (9,0× resp. 46,9×), což plně potvrzuje výsledky z amerického vzorku.

## 5.7 Diskuse výsledků a limitace

### 5.7.1 Shrnutí klíčových zjištění

Analýza přináší pět hlavních zjištění.

Za prvé, AI prémie je reálná a měřitelná. Čistá mzdová prémie za AI dovednosti činí 7,5–9,6 % po kontrole všech relevantních faktorů (vzdělání, zkušenosti, region, sektor, typ a velikost firmy, profesní skupina). Prémie roste monotónně s hloubkou AI expertízy, jak potvrzuje ANOVA s Bonferroniho korekcí.

Za druhé, AI prémie existuje, ale není dominantním mzdovým faktorem. Geografie (West +22,4 %), typ pozice (Sr+ Engineer +17,3 %) a zkušenosti (Senior+ +12,0 %) mají na plat větší absolutní vliv. AI prémie je aditivním bonusem nad rámec těchto tradičních faktorů lidského kapitálu. Toto zjištění je v souladu s Mincerovou teorií, kde vzdělání a zkušenosti zůstávají fundamentálními determinanty mezd — AI dovednosti doplňují, ale nenahrazují tento základ.

Za třetí, o AI požadavku rozhoduje role, nikoliv firma. Firemní profil predikuje AI požadavek slabě (Pseudo R² = 2,5 %), zatímco typ dovedností a role podstatně lépe (Pseudo R² = 11,7 %; resp. 13,0 % v plném modelu). AI prostupuje IT trhem průřezově — není omezena na konkrétní sektory, velikosti firem nebo typy organizací.

Za čtvrté, existuje kvalitativní rozdíl mezi „používáním AI" a „vývojem AI". Multinomiální logit prokázal, že některé skill clustery vykazují statisticky významné *opačné efekty* mezi kategoriemi: Systems Programming snižuje P(AI Integration), ale zvyšuje P(Applied/Core AI); naopak Frontend Development a Enterprise Platforms predikují výhradně Integration. Data Engineering je nejčistším diskriminátorem — predikuje Applied/Core AI, ale nemá vliv na Integration. Tyto vzorce validují třístupňové rozlišení AI požadavků jako empiricky smysluplný koncept.

Za páté, AI mzdová prémie nevykazuje statisticky významné rozdíly mezi zeměmi. Rozdíly mezi USA, Německem a Indií se projevují ve frekvenci AI pracovních míst, nikoliv v relativní mzdové prémii (F-test interakcí: p = 0,965). Tento závěr je nejrobustnější pro dvojici USA–Indie; pro Německo je statistická síla testu omezena malým vzorkem (n = 514).

### 5.7.2 Zasazení do kontextu literatury

Zjištěná AI prémie 7,5–9,6 % je konzistentní s dosavadní literaturou. Bone et al. (2025) reportují signifikantní mzdovou prémii za AI dovednosti ve Velké Británii. Výsledky rovněž korespondují s konceptem skill-biased technological change (SBTC), neboť AI dovednosti zvyšují relativní mzdu kvalifikovaných pracovníků. Zjištění, že AI proniká průřezově napříč sektory a velikostmi firem, koresponduje s charakteristikou AI jako general purpose technology (GPT), tedy technologie s širokým aplikačním záběrem (Engberg et al., 2025).

Rozlišení mezi AI Integration a Applied/Core AI přispívá k debatě o heterogenitě AI dopadů na trhu práce. Zatímco Engberg et al. (2025) sledují změny ve struktuře úkolů uvnitř povolání, tato práce ukazuje, že AI se promítá i do požadovaných dovednostních profilů, a to kvalitativně odlišným způsobem pro různé úrovně AI práce.

Komparativní zjištění o absenci statisticky významných rozdílů v AI prémii naznačuje, že globální trh AI dovedností funguje relativně integrovaně co do relativního ocenění těchto dovedností — alespoň mezi USA a Indií, pro které je statistická síla testu dostatečná. Absolutní mzdové rozdíly mezi zeměmi jsou samozřejmě obrovské a odrážejí rozdílné cenové hladiny a produktivitu ekonomik, avšak procentuální prémie za AI se mezi zkoumanými zeměmi statisticky významně neliší.

### 5.7.3 Limitace

Práce podléhá několika omezením, která je nutné při interpretaci výsledků zohlednit.

**Kauzalita versus korelace.** Regresní modely identifikují asociace, nikoliv kauzální efekty. Není vyloučeno, že jedinci s AI dovednostmi mají i další neměřené schopnosti, které zvyšují jejich produktivitu a mzdu (selection bias / omitted variable bias). Pro kauzální inferenci by byl potřeba kvazi-experimentální design (např. instrumentální proměnné nebo difference-in-differences), což přesahuje rozsah této práce.

**Datová limitace Glassdooru.** Data pocházejí z jednoho portálu a mohou být zkreslena směrem k větším firmám a technologickým hub městům. Menší firmy a tradiční podniky mohou být podreprezentovány. Zároveň platí, že 18 % pozic v americkém vzorku nemá mzdové údaje, což představuje potenciální selection bias u OLS modelů. Pro formalizaci této obavy byla v Stata skriptu zařazena dvouúrovňová diagnostika missingness platu (sekce 4.15a–c v logu). Za prvé, kontingenční test ukazuje, že pravděpodobnost uvedeného platu se napříč AI úrovněmi statisticky významně neliší (χ² = 2,40; p = 0,121) — missingness tedy není systematicky spojena s AI statusem. Za druhé, logit `has_salary` na širší sadě observables (region, NACE sektor, velikost firmy, typ, remote, profesní skupina, vzdělání, zkušenosti) s Waldovým testem společné signifikance prokazuje, že data nejsou MCAR: χ²(32) = 2 926,2; p < 0,001. Missingness systematicky souvisí s observables (zejména s regionem a velikostí firmy). OLS odhady AI prémie je proto nutno interpretovat jako podmíněné na vzorku inzerátů se zveřejněnými platy; formulaci „absence selection bias" se v textu explicitně vyhýbáme. Vzhledem k tomu, že missingness nekovaruje s AI tierem, je však pravděpodobné, že potenciální zkreslení působí spíše na absolutní úroveň mezd než na relativní AI prémii, která je předmětem této práce.

**Německé mzdové pokrytí.** S pouhými 514 mzdovými pozorováními pro Německo (8 % vzorku) je OLS komparativní analýza fakticky porovnáním USA vs. Indie. Koeficienty pro Německo je nutno interpretovat se značnou opatrností.

**Selection bias a Heckmanova korekce.** Heckmanova korekce selekčního zkreslení potvrdila statisticky významnou lambdu (ρ = −0,053; p = 0,004), což indikuje existenci selection bias v OLS modelech. Praktický dopad je však zanedbatelný — AI koeficienty se mezi OLS a Heckmanem liší minimálně (AI Integration: 0,075 vs 0,075; Applied/Core AI: 0,096 vs 0,095). Heckmanův model je však identifikován pouze z funkcionální formy probit modelu (bez exkluzní restrikce), což je metodologicky slabá identifikační strategie (Puhani, 2000). Tento model proto slouží jako robustnostní kontrola potvrzující stabilitu OLS odhadů AI prémie, nikoliv jako preferovaná specifikace.

**Předpoklad IIA v multinomiálním logitu.** Multinomiální logit předpokládá nezávislost irelevantních alternativ (IIA). Hausman test tohoto předpokladu vrátil statisticky významný výsledek (χ²(49) = 92,16; p < 0,001), avšak variační matice nebyla pozitivně definitní, což činí test nespolehlivým (Long & Freese, 2014). IIA tedy nelze v tomto datasetu formálně ověřit ani zamítnout. Multinomiální logit je v této práci používán exploratorně — klíčové závěry se opírají o vzory v průměrných marginálních efektech (AME), které jsou vůči porušení IIA robustnější než bodové koeficienty.

**Časový průřez.** Jde o průřezový snapshot (2024–2025), nikoliv o časovou řadu. Nelze proto říci, zda AI prémie roste, klesá nebo stagnuje. Longitudinální analýza je směrem pro budoucí výzkum.

**Sloučení Core AI.** Pouhých 6 pozorování v kategorii Core AI neumožňuje oddělit skutečné AI výzkumné pozice od aplikovaného AI vývoje.

**Klasifikace LLM.** Absence inter-rater reliability studie (Cohen's kappa s manuálním kódováním vzorku) představuje metodologické omezení. Klasifikace byla validována hybridním přístupem (deterministický slovník + LLM), confidence threshold ≥ 0,7 a požadavkem na intersekci AI tier + specifické AI dovednosti po odfiltrování buzzwords. Plná validační studie je legitimním rozšířením pro budoucí výzkum.
