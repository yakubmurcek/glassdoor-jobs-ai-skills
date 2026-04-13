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

Zatímco OLS modely kvantifikují mzdovou prémii za AI dovednosti, logistické modely odpovídají na komplementární otázku: jaké charakteristiky predikují, zda pozice vůbec vyžaduje AI? Tato analýza má exploratorní charakter a slouží k identifikaci dovednostních profilů asociovaných s AI pozicemi.

### 5.4.1 Inkrementální struktura modelů

Modely byly odhadovány inkrementálně ve třech krocích, aby bylo možné rozlišit přínos firemních a rolových charakteristik.

Model M1 (Profil firmy) zahrnoval pouze firemní proměnné: NACE sektor, typ organizace, velikost firmy a region. Pseudo R² tohoto modelu dosáhlo pouhých 2,1 %. To znamená, že o tom, zda pozice vyžaduje AI, rozhoduje primárně obsah a charakter role, nikoliv agregované vlastnosti zaměstnavatele. AI prostupuje IT trhem průřezově napříč sektory, velikostmi firem i typy organizací.

Model M2 (Profil role) zahrnoval technologické skill clustery, profesní skupinu, vzdělání a zkušenosti. Pseudo R² dramaticky vzrostlo na 33,3 %. Přidání firemních proměnných v Modelu M3 (Kompletní) přineslo pouze marginální zlepšení na 34,0 % (+0,7 procentního bodu). Typ a dovednostní obsah pozice je tedy zcela rozhodujícím prediktorem AI požadavku.

Všechny modely konvergovaly bez problémů (4–6 iterací, žádné „not concave" ani „perfect separation" chyby), což svědčí o strukturální čistotě dat.

### 5.4.2 Klíčové prediktory AI požadavku (Average Marginal Effects)

Výsledky jsou reportovány ve formě průměrných marginálních efektů (AME), které udávají změnu pravděpodobnosti AI požadavku v procentních bodech při přítomnosti dané charakteristiky. AME jsou ekonomicky srozumitelnější než odds ratios, neboť přímo vyjadřují absolutní změnu pravděpodobnosti průměrované přes celou distribuci pozorování.

Naprosto dominantním prediktorem je skill cluster Generative AI (AME = +31,9 p. b.; p < 0,001). Přítomnost dovedností v oblasti generativní AI (GPT, LLM, prompt engineering) zvyšuje pravděpodobnost, že pozice bude klasifikována jako AI pozice, o téměř 32 procentních bodů. Druhým nejsilnějším prediktorem je cluster Data Science / ML (AME = +25,9 p. b.; p < 0,001), zahrnující dovednosti jako TensorFlow, PyTorch, scikit-learn a strojové učení.

Mezi další signifikantní pozitivní prediktory patří: Dynamic Web (+5,2 p. b.), Cloud Computing (+3,5 p. b.), Frontend Development (+3,3 p. b.) a BI / Analytics (+2,3 p. b.). Negativně asociovány s AI požadavkem jsou naopak Enterprise / Managed systémy (−4,1 p. b.), certifikace (−3,3 p. b.) a Scripting / Shell (−2,4 p. b.), což naznačuje, že tradiční podnikové a infrastrukturní role AI zatím výrazněji neadoptovaly.

Typ organizace (Private, Public, Nonprofit/Gov) nemá statisticky významný vliv na pravděpodobnost AI požadavku v žádné specifikaci (p > 0,40 u všech kategorií). Stejně tak úroveň seniority se v logistickém modelu M3 neukazuje jako signifikantní prediktor (Junior p = 0,314; Senior+ p = 0,104), což může působit překvapivě vzhledem k deskriptivním rozdílům, avšak po kontrole skill clusterů a job family je variabilita seniority absorbována.

### 5.4.3 Poznámka k potenciální cirkularitě

Skill clustery Generative AI a Data Science / ML obsahují dovednosti (GPT, LLM, TensorFlow, PyTorch), které přímo implikují AI požadavek. Proto je třeba interpretovat logistické modely jako exploratorní nástroj pro identifikaci dovednostních profilů, nikoliv jako kauzální odhad. Pro adresování této metodologické otázky byla provedena citlivostní analýza (viz sekce 5.5.2). Pro obezřetnou interpretaci proto upřednostňujeme *de-confounded* specifikaci ze sekce 5.5.2 (bez Generative AI a Data Science / ML clusterů) jako primární a plný Model M3 interpretujeme jako horní mez prediktivní síly, nikoli jako kauzální odhad.

## 5.5 Dovednostní profil AI pozic

### 5.5.1 Kvantitativní složitost AI pozic

AI pozice vyžadují objemnější portfolio technických dovedností. Průměrný počet požadovaných hard skills u pozic bez AI činil přibližně 16,0 (SD = 10,1), u AI Integration pozic 19,5 (SD = 10,4) a u Applied/Core AI pozic 20,6 (SD = 10,8). Rozdíl přibližně 4 dovedností mezi non-AI a AI pozicemi je statisticky vysoce významný (t = −20,0; p < 0,001) a reflektuje komplexnější povahu AI rolí, kde je od uchazečů očekáváno zvládání širšího arzenálu technologií.

### 5.5.2 Citlivostní analýza (bez cirkulárních clusterů)

Pro adresování potenciální cirkularity mezi skill clustery a AI klasifikací byla provedena citlivostní analýza logistického modelu M3, z něhož byly vyřazeny oba potenciálně cirkulární clustery (Generative AI a Data Science / ML).

Pseudo R² dramaticky pokleslo z 34,0 % na 12,8 %. To znamená, že tyto dva clustery vysvětlovaly přibližně 21 procentních bodů variability, což představuje zhruba 62 % celkové prediktivní schopnosti modelu. Zároveň je však toto zjištění důležité: ztráta prediktivní síly ukazuje, že GenAI a DS/ML clustery nesou unikátní informaci, která není redundantní s ostatními prediktory.

Po jejich vyřazení přebraly vysvětlující sílu zbývající clustery. Cloud Computing vzrostl z +3,2 na +6,2 p. b. (zdvojnásobení), Dynamic Web z +4,8 na +8,8 p. b., Data Engineering z +1,3 na +4,8 p. b. (ztrojnásobení). Nově signifikantními se staly DevOps/Containers (+2,4 p. b.) a Systems Programming (+3,6 p. b.). Tento vzorec naznačuje, že AI požadavky korelují se širším technologickým profilem pozice, nejen s přímými AI dovednostmi.

### 5.5.3 Multinomiální logit — „používání AI" versus „vývoj AI"

Stěžejní analytickou přidanou hodnotou je rozlišení mezi pozicemi, které AI pouze integrují do svých procesů (AI Integration), a pozicemi, které AI přímo vyvíjejí (Applied/Core AI). K tomuto účelu byl odhadnut multinomiální logistický model (mlogit) se třemi kategoriemi výsledné proměnné (None, AI Integration, Applied/Core AI).

Výsledky AME z multinomiálního modelu M2 odhalují kvalitativně odlišné dovednostní profily obou kategorií.

Pro AI Integration jsou nejsilnějšími prediktory: Generative AI (+23,4 p. b.), Data Science / ML (+14,3 p. b.), Frontend Development (+4,4 p. b.), Enterprise Platforms (+4,2 p. b.) a Cloud Computing (+2,8 p. b.). Tento profil odpovídá pozicím, kde se AI nástroje aplikují v rámci existujících softwarových systémů — webové aplikace obohacené o AI funkce, podnikové platformy s integrovanou AI, cloudové nasazení AI služeb.

Pro Applied/Core AI jsou nejsilnějšími prediktory: Data Science / ML (+11,8 p. b.), Generative AI (+9,2 p. b.), Systems Programming (+3,2 p. b.), Dynamic Web (+2,7 p. b.) a Data Engineering (+2,3 p. b.). Zásadní je zde přítomnost Systems Programming a Data Engineering, které jsou u AI Integration nevýznamné nebo dokonce negativní. Tyto fundamentální inženýrské dovednosti — nízkoúrovňové programování, správa datových pipeline — odlišují skutečné AI vývojáře od „pouhých" uživatelů AI.

Klíčovým rozlišujícím faktorem je tedy to, že Applied/Core AI pozice vyžadují fundamentální dovednosti (systémové programování, data engineering, klasická ML/statistika), zatímco AI Integration pozice se vyznačují aplikačními dovednostmi (frontend, enterprise platformy, generativní AI nástroje). Toto zjištění přímo validuje rozlišení mezi „rozumět AI" a „používat AI", které je ústředním konceptem této práce.

Z hlediska profesních skupin mají oproti kategorii Data & AI všechny ostatní job families výrazně nižší pravděpodobnost Applied/Core AI (DevOps & Cloud: −6,6 p. b.; Software Developer: −6,9 p. b.; Other: −8,1 p. b.). U AI Integration jsou rozdíly menší a statisticky méně významné, což ukazuje, že integrace AI proniká rovnoměrněji napříč profesemi, zatímco vývoj AI zůstává soustředěn v datových a AI specializacích.

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

Za třetí, o AI požadavku rozhoduje role, nikoliv firma. Firemní profil predikuje AI požadavek slabě (Pseudo R² = 2,1 %), zatímco typ dovedností a role velmi dobře (Pseudo R² = 33,3 %). AI prostupuje IT trhem průřezově — není omezena na konkrétní sektory, velikosti firem nebo typy organizací.

Za čtvrté, existuje kvalitativní rozdíl mezi „používáním AI" a „vývojem AI". Multinomiální logit prokázal, že Applied/Core AI pozice se od AI Integration liší požadavkem na fundamentální dovednosti (systémové programování, data engineering), zatímco Integration pozice se vyznačují aplikačními dovednostmi (frontend, enterprise platformy, generativní AI nástroje).

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
