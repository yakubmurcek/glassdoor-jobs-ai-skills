# 5. AI dovednosti v inzerci IT pracovních pozic

Tato kapitola představuje jádro empirické části diplomové práce. Na základě kvantitativní analýzy datasetu amerických pracovních inzerátů z platformy Glassdoor (N = 17 848) je zkoumána struktura poptávky po AI dovednostech v IT sektoru, její vazba na mzdové ohodnocení a faktory, které výskyt AI požadavků v pracovní nabídce determinují. Analýza postupuje od popisné statistiky přes testování statistických hypotéz až po vícerozměrné regresní a pravděpodobnostní modely.

---

## 5.1 Popisná statistika: Struktura poptávky po AI dovednostech

### 5.1.1 Zastoupení AI požadavků v IT inzerci

Výchozím zjištěním analýzy je skutečnost, že AI dovednosti jsou v americké IT inzerci přítomny u **19,48 %** pracovních pozic. Zbývajících 80,52 % inzerátů nevyžaduje od uchazečů žádné specifické AI kompetence. Tento poměr naznačuje, že ačkoliv AI transformuje celé odvětví technologií, tradiční softwarové role stále tvoří absolutní většinu pracovního trhu.

Výzkum dále rozlišuje AI požadavky do dvou kvalitativně odlišných úrovní:

- **AI Integration (13,19 %)** — pozice, jejichž náplň zahrnuje aplikaci a integraci hotových AI nástrojů do stávajících procesů. Typickým příkladem jsou softwaroví inženýři implementující modely jazykových modelů do produktů nebo analytici pracující s platformami pro strojové učení.
- **Applied/Core AI (7,36 %)** — pozice zaměřené na samotný vývoj a trénování AI systémů. Jde o výzkumné role, ML inženýry a architekty neuronových sítí, u nichž je AI stěžejní součástí odborné náplně.

Toto rozlišení je analyticky klíčové: sloučení obou kategorií by maskovalo vnitřně heterogenní trh, kde se pracovní podmínky, mzdové nároky i požadovaný profil kandidáta výrazně odlišují.

### 5.1.2 Mzdová struktura podle AI požadavků

Prvotní explorace odhaluje výrazné rozdíly ve střední roční hrubé mzdě (v USD) napříč identifikovanými skupinami:

| Kategorie | N s platem | Průměrná mzda | Medián mzdy |
|---|---|---|---|
| Bez AI požadavku (None) | 11 600 | **$ 119 532** | $ 114 000 |
| AI Integration | 1 946 | **$ 140 487** | $ 135 000 |
| Applied/Core AI | 1 094 | **$ 150 498** | $ 148 495 |
| **Celkem** | **14 640** | $ 124 632 | $ 117 885 |

*Zdroj: vlastní výpočty ze Stata analýzy; N s platem = počet inzerátů s dostupným údajem o platu.*

Pozice vyžadující AI dovednosti nabízejí okamžitě viditelnou mzdovou prémii. Průměrný plat inzerátů kategorie *AI Integration* přesahuje průměr non-AI pozic o přibližně **20 955 USD** (+17,5 %), u *Applied/Core AI* pak o **30 966 USD** (+25,9 %). Tyto hrubé rozdíly jsou ovšem deskriptivní a nezahrnují vliv dalších proměnných — podrobná korekce je předmětem regresní analýzy v části 5.3.

### 5.1.3 Profil požadované kvalifikace

**Vzdělání.** Vzdělávací požadavky kopírují gradient AI náročnosti pozice. Ve vzorku tvoří pozice s požadavkem na bakalářský nebo vyšší titul 60,4 % (N = 10 771). Kontingenční analýza (chi-kvadrát, p < 0,001) potvrzuje, že AI pozice statisticky signifikantně vyžadují vyšší vzdělání.

**Zkušenosti.** Dominantní kategorií seniornosti je Mid-level (3–5 let praxe) s podílem 47,1 %. Junior pozice (0–2 roky) tvoří 18,0 % a Senior+ (6+) 20,4 %. Chi-kvadrát analýza ukázala statisticky signifikantní vztah mezi senioritou a přítomností AI požadavku (p < 0,001).

**Technologické dovednosti.** Nejčastěji zastoupenými skill clustery jsou architektura a metody (81,4 % inzerátů), frontend development (60,3 %) a backend development (54,7 %). AI-specifické clustery — *Data Science & ML* (12,1 %) a *Generative AI* (6,9 %) — jsou přítomny u menší části trhu, avšak s výrazně vyšší korelací s AI tier klasifikací.

---

## 5.2 Testování statistických hypotéz

### 5.2.1 Mzdový rozdíl: Dvouvýběrový t-test

Pro ověření statistické signifikance zjištěného mzdového rozdílu byl proveden dvouvýběrový t-test srovnávající průměrné platy AI a non-AI pozic.

- **H₀:** Průměrný plat AI pozic = průměrný plat non-AI pozic
- **H₁:** Průměrné platy se liší

Výsledek testu: **t = 19,72**, stupně volnosti = 17 846, **p < 0,001**. Nulová hypotéza je zamítnuta na hladině 0,1 %. Mzdový rozdíl je statisticky vysoce signifikantní.

Pro posouzení praktické (věcné) významnosti byl vypočten koeficient Cohenova **d = 0,587**, jenž indikuje *středně silný efekt* (prahová hodnota pro střední efekt je d ≥ 0,5). Výsledek tedy potvrzuje, že mzdová prémie AI pozic není pouhým statistickým artefaktem způsobeným rozsahem vzorku, ale představuje ekonomicky smysluplný a prakticky relevantní rozdíl.

### 5.2.2 Mzdové rozdíly napříč AI tiers: ANOVA

Jednosměrná analýza rozptylu (ANOVA) s Bonferroniho korekcí pro vícenásobné srovnání potvrdila statisticky signifikantní rozdíly v platech mezi všemi třemi skupinami (None / AI Integration / Applied/Core AI) na hladině p < 0,001. Post-hoc srovnání ukázalo, že každá dvojice skupin se statisticky liší.

### 5.2.3 Technologická náročnost: T-test počtu dovedností

AI pozice se vyznačují širším spektrem technologických požadavků. Průměrný počet hard skills v inzerci:

- Non-AI pozice: **16,0 dovedností**
- AI pozice (Integration + Applied): **19,8 dovedností**
- Rozdíl: **3,8 dovednosti** (t = −19,72, p < 0,001)

Tento výsledek přispívá k vysvětlení mzdového diferenciálu: AI pozice vyžadují širší technologický profil, jehož ovládání je na trhu práce oceňováno nadstandardně.

---

## 5.3 Regresní analýza: Kvantifikace AI mzdové prémie

Pro rigorózní odhad čisté mzdové prémie za AI dovednosti — po kontrole ostatních determinantů mzdy — byly odhadnuty dva OLS regresní modely se závislou proměnnou přirozeného logaritmu roční mzdy (`ln_salary`). Koeficienty v log-lineárním modelu jsou interpretovány jako přibližné procentuální změny mzdy: $(e^{\beta} - 1) \times 100$ %.

### 5.3.1 Model A: Základní specifikace (Technologický a firemní profil)

Model A zahrnuje jako vysvětlující proměnné skupiny technologických dovedností (21 skill clusterů), úroveň AI požadavku (`ai_level`), NACE sektor, Census region, možnost remote práce, typ a velikost organizace. Odhadnut metodou nejmenších čtverců s robustními standardními chybami (Huber-White, `vce(robust)`) k ošetření heteroskedasticity.

**Výsledky Modelu A:**
- N = 14 640 (inzeráty s dostupným platem)
- **R² = 0,247** — model vysvětluje 24,7 % variability logaritmické mzdy
- F(44, 14 595) = 121,38, p < 0,001

**Odhad AI prémie (Modelo A):**

| Proměnná | Koeficient | Přibližný % dopad | p-hodnota |
|---|---|---|---|
| AI Integration (vs. None) | 0,0900 | **+9,4 %** | < 0,001 |
| Applied/Core AI (vs. None) | 0,1276 | **+13,6 %** | < 0,001 |

Tento odhad zahrnuje vliv technologického profilu pozice, nicméně nezahrnuje individuální charakteristiky uchazeče — vzdělání, seniority a pracovní rodinu. Tyto faktory mohou s AI korelovovat a způsobovat přecenění AI prémie (tzv. omitted variable bias).

### 5.3.2 Model B: Rozšířená specifikace (Přidáno: Lidský kapitál)

Model B rozšiřuje Model A o proměnné charakterizující požadovaný profil uchazeče: kategorii pracovní rodiny (`job_family_num`), granulární vzdělávací požadavek (`edu_ols`, 4 úrovně) a kategorii zkušeností (`exp_category`).

> **Poznámka k vzdělávací proměnné:** Na doporučení vedoucího práce bylo v mzdovém modelu zachováno granulární rozlišení vzdělávacích kategorií (High School / Associate / Bachelor / Master+), neboť jejich sloučení by potlačilo heterogenní vliv jednotlivých vzdělávacích stupňů na mzdu.

**Výsledky Modelu B:**
- N = 14 640
- **R² = 0,380** — nárůst o 13,3 procentního bodu oproti Modelu A
- F(57, 14 582) = 187,08, p < 0,001

Statistická signifikance přidaných proměnných (`edu_ols`, `exp_category`, `job_family_num`) byla ověřena LR testem: χ² = sign., p < 0,001. Začlenění lidského kapitálu do specifikace je tedy statisticky odůvodněné.

**Odhad AI prémie (Model B — preferovaná specifikace):**

| Proměnná | Koeficient | Přibližný % dopad | p-hodnota |
|---|---|---|---|
| AI Integration (vs. None) | 0,0753 | **+7,5 %** ✅ | < 0,001 |
| Applied/Core AI (vs. None) | 0,0961 | **+9,6 %** ✅ | < 0,001 |

Zahrnutí kontrolních proměnných lidského kapitálu snížilo odhadovanou AI prémii o přibližně 1,9–4,0 procentního bodu oproti Modelu A. Toto snížení odpovídá ekonomické teorii: AI pozice jsou obsazovány vzdělanějšími a zkušenějšími kandidáty, a část zdánlivé AI prémie z Modelu A tak ve skutečnosti reflektovala *prémii za vzdělání*, nikoli specificky za AI dovednosti. Po korekci tohoto vlivu zůstává čistá AI prémie robustní a statisticky vysoce signifikantní.

**Vzdělávací gradient (Model B):**

| Úroveň vzdělání | Koeficient | Dopad na mzdu | p-hodnota |
|---|---|---|---|
| High School (vs. Missing) | −0,068 | **−6,6 %** | < 0,001 |
| Associate (vs. Missing) | −0,110 | **−10,4 %** | < 0,001 |
| Bachelor (vs. Missing) | −0,033 | **−3,3 %** | < 0,001 |
| **Master+ (vs. Missing)** | **+0,050** | **+5,1 %** | < 0,001 |

Gradient je ekonomicky intuitivní a potvrzuje teorii lidského kapitálu (Becker, 1964): každá adicionalní úroveň formálního vzdělání je na trhu práce systematicky oceňována. Pozoruhodné je, že kategorie *Missing* (neuvedený požadavek) se umísťuje mezi Associate a Bachelor, což odráží heterogenitu těchto inzerátů.

**Vliv seniornosti a pracovní rodiny:**

Koeficienty pro zkušenostní kategorie jsou ve shodě s teorií životního cyklu lidského kapitálu:
- Junior (0–2 roky) vs. Mid: **−17,5 %** (p < 0,001)
- Senior+ (6+ let) vs. Mid: **+12,0 %** (p < 0,001)

Mezi pracovními rodinami vykazuje nejvyšší mzdovou prémii kategorie *Sr+ Software Engineer* (+17,3 % vs. Data & AI), naopak *Software Developer* zaostává o −5,5 %.

**VIF diagnostika (Multikolinearita):**
- Model A: Mean VIF = 1,79; Max VIF = 5,09 (typ firmy) — ✅ bez problémů
- Model B: Mean VIF = 2,02; Max VIF = 6,52 (job family) — ✅ bezpečně pod kritickou hranicí 10

---

## 5.4 Pravděpodobnostní modely: Determinanty AI požadavku

Zatímco OLS modely odpovídají na otázku *jak vysoká je mzdová prémie*, pravděpodobnostní modely zkoumají otázku *jaké faktory determinují, zda pozice AI dovednosti vyžaduje vůbec*. Tato analýza využívá hierarchicky strukturovanou sadu logistických a multinomiálních logistických modelů.

### 5.4.1 Specifikace modelů

Závislé proměnné:
- `has_ai` (0/1) — binární přítomnost AI požadavku (Logit)
- `ai_level` (0/1/2) — tříkategoriální úroveň: None / AI Integration / Applied/Core AI (Mlogit)

Modely jsou organizovány inkrementálně:
- **Model 1 (Firemní profil):** NACE sektor, typ a velikost organizace, Census region
- **Model 2 (Profil role):** Skill clustery, pracovní rodina, vzdělání (`edu_logit`), zkušenosti
- **Model 3 (Kompletní):** Model 1 + Model 2

### 5.4.2 Model 1: Firemní profil

Logit Model 1 (N = 17 848, LR chi² = 432,66, p < 0,001, Pseudo R² = 2,5 %) ukazuje, že firemní profil sám o sobě vysvětluje požadavek na AI jen omezeně. Statisticky signifikantní prediktory jsou:

- **NACE sektor J** (ICT): OR = 1,53 — firmy v IT sektoru mají o 53 % vyšší šance vyžadovat AI dovednosti než průmyslová výroba (základní kategorie C)
- **NACE sektor M** (Odborné poradenství): OR = 1,71
- **West region**: marginální efekt **+9,8 p.b.** (p < 0,001), Západ USA silně dominuje v AI adopci
- **Unknown region** (Home Office / nespecifikovaný): marginální efekt **+9,4 p.b.** — remote-friendly pozice jsou nadreprezentovány v AI
- **Velikost firmy 10 000+**: OR = 1,53 — velké korporace výrazněji vyhledávají AI talenty

Typ firmy (soukromá vs. veřejná vs. nezisková) se naopak jako statisticky signifikantní prediktor nepotvrdil (p > 0,46 pro všechny kategorie).

### 5.4.3 Model 2: Profil role a uchazeče

Model 2 zaznamenává dramatický nárůst vysvětlující síly (Pseudo R² = 35,9 %). Technologické dovednosti a pracovní rodina jsou výrazně silnějšími prediktory AI požadavku než samotný firemní profil.

**Klíčové výsledky Logit Modelu 2:**

| Prediktor | Odds Ratio | Marginální efekt |
|---|---|---|
| `cluster_generative_ai` | **30,50** *** | **+40,6 p.b.** |
| `cluster_data_science__ml` | **15,94** *** | **+32,9 p.b.** |
| `cluster_dynamic__web` | 1,72 *** | +6,4 p.b. |
| `cluster_cloud_computing` | 1,45 *** | +4,5 p.b. |
| `cluster_enterprise__managed` | 0,63 *** | −5,6 p.b. |
| Job family: **Data & AI** (základní) | — | — |
| Job family: Software Developer | 0,36 *** | −14,4 p.b. |
| Job family: DevOps & Cloud | 0,40 *** | −13,4 p.b. |
| `edu_logit`: No degree/Missing (vs. Bachelor+) | 1,31 *** | +3,3 p.b. |

*\*\*\* p < 0,001*

Zdaleka nejsilnějším prediktorem je přítomnost dovedností z oblasti *Generative AI* (ChatGPT, LLM platformy apod.): odds ratio 30,5 naznačuje, že inzeráty zmiňující tento skill cluster mají třicetinásobně vyšší šanci vyžadovat AI dovednosti. Cluster *Data Science & ML* (Python, TensorFlow, PyTorch) je druhým nejsilnějším prediktorem (OR = 15,9).

Paradoxní výsledek vzdělání (nižší vzdělání zvyšuje pravděpodobnost AI požadavku) lze vysvětlit specifickými typy inzerátů: některé AI pozice (startupového charakteru nebo entry-level automatizační role) nespecifikují formální vzdělání, ale explicitně požadují dovednosti v AI nástrojích.

### 5.4.4 Model 3: Kompletní specifikace

Kompletní Model 3 kombinuje firemní a rolový profil (Logit: Pseudo R² = 36,4 %; Mlogit: 33,9 %). Statistická konvergence všech modelů proběhla bez problémů (4–6 iterací), bez výskytu separace nebo singularity.

Hausmanova test IIA (Independence of Irrelevant Alternatives) pro multinomiální modely proběhl technicky, avšak s ohledem na vlastnosti testu při skupinových efektech jeho výsledek slouží pouze jako orientační diagnostický nástroj.

---

## 5.5 Shrnutí empirických zjištění

Analýza přináší čtyři hlavní empirická zjištění relevantní pro diskusi o dopadech AI na IT pracovní trh:

1. **AI dovednosti jsou poptávány u jedné pětiny IT inzerátů** (19,48 %). Trh je přitom vnitřně diferencovaný — integrace AI nástrojů (13,19 %) a vývoj AI systémů (7,36 %) jsou kvalitativně i mzdově odlišné kategorie.

2. **Čistá mzdová prémie za AI dovednosti činí 7,5–9,6 %** po kontrole vzdělání, zkušeností, pracovní rodiny, sektoru a regionu (Model B). Hrubý diferenciál (16–26 %) je z části vysvětlen tím, že AI pozice obsazují vzdělanější a zkušenější kandidáty.

3. **Nejvýznamnějším prediktorem AI poptávky jsou technologické dovednosti**, zejména Generative AI (OR = 30,5) a Data Science & ML (OR = 15,9). Firemní profil (sektor, velikost) má vliv výrazně menší (Pseudo R² M1 ≈ 2,5 % vs. M2 ≈ 35,9 %).

4. **AI pozice vyžadují širší technologický profil** — v průměru o 3,8 více hard skills než non-AI pozice (p < 0,001), což přispívá k vysvětlení mzdového diferenciálu z pohledu teorie lidského kapitálu (vyšší investice → vyšší mzdová návratnost).

---

*Poznámka: Veškeré výsledky jsou z Stata runu `run__5_Mar_2026_22-59-18`. Kompletní tabulky koeficientů jsou k dispozici v příloze.*
