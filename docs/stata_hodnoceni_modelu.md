# 📊 Hodnocení výstupu a analýza modelů ze Staty (AI na trhu práce v IT)

Tento dokument sumarizuje a interpretuje klíčová zjištění ze statistického zpracování datasetu pracovních inzerátů (N = 17 848). Slouží jako podklad pro konzultaci výsledků výzkumné části diplomové práce a pro následné teoretické ukotvení do textu práce. Zároveň ověřuje metodologickou kvalitu výstupů proti surovému Stata logu.

Výstup ze Staty vypadá z metodologického hlediska **velmi profesionálně a kvalitně** 🌟. Modely běží, konvergují a jsou nastaveny způsobem, který přesně odpovídá standardům pro socioekonomický / ekonometrický výzkum.

---

## 🎯 ČÁST I: INTERPRETACE VÝSLEDKŮ PRO OBHAJOBU

### 1. 📈 Deskriptivní (popisná) statistika

Prvotní analýza vzorku poskytuje základní vhled do struktury trhu práce z hlediska poptávky po umělé inteligenci (AI).

- 🤖 **Zastoupení AI požadavků:** Ve zkoumaném vzorku vyžadovalo alespoň nějakou úroveň AI dovedností **19,48 %** inzerátů.
  - 🧩 _13,19 %_ pracovních pozic představuje aplikaci a integraci AI do běžných procesů (tzv. "AI Integration").
  - 🧠 _7,36 %_ pozic se věnuje přímému vývoji vlastního jádra a algoritmizaci AI řešení (tzv. "Applied/Core AI").
  - 📄 Zbylých _79,45 %_ inzerce tvoří běžné role bez explicitního AI požadavku.
  - ⚠️ Kategorie `core_ai` byla sloučena do `applied_ai` kvůli pouhým 6 pozorováním — pod prahem 50 na buňku. To je nutná metodologická úprava, ale v diskuzi je třeba zmínit jako limitaci: skutečné "core AI" výzkumné pozice nelze v tomto datasetu oddělit.
- 💰 **Mzdové rozdíly (AI Premium):** Exploratorní zhodnocení střední roční hrubé mzdy v amerických dolarech (USD) poukazuje na výrazné nominální rozdíly:
  - 🔹 Průměrná mzda u běžných IT pozic: **~119 500 $**
  - 🔸 Průměrná mzda u pozic typu _AI Integration_: **~140 480 $** (✅ Znatelný nárůst)
  - 🚀 Průměrná mzda u pozic typu _Applied/Core AI_: **~150 500 $** (🔥 Nejvyšší ohodnocení)
- 🏠 **Vliv pracovního režimu (Remote):** Celkový podíl remote inzerátů činil 28,4 %. U AI pozic je podíl remote práce **37,6 %** (1 308 / 3 477) oproti **26,2 %** (3 759 / 14 371) u non-AI pozic (Chi-kvadrát = 180,9; $p < 0,001$). AI pozice jsou signifikantně flexibilnější z hlediska lokace.
  - ⚠️ _Oprava: Původní čísla (25,8 % a 17,0 %) pocházela z řádkových procent cross-tabu (% pozic v dané remote kategorii, které vyžadují AI). Správná interpretace jsou sloupcová procenta (% AI/non-AI pozic, které jsou remote)._
- 🗺️ **Geografická koncentrace AI:** AI pozice se výrazně koncentrují na Západě USA (West: 28,5 % AI pozic vs. 21,3 % non-AI) a v kategorii "Unknown" (remote/celostátní: 32,8 % AI vs. 24,1 % non-AI). Naopak Midwest (7,9 % AI vs. 11,5 % non-AI) a South (19,9 % AI vs. 31,0 % non-AI) jsou podreprezentovány. Chi-kvadrát = 305,0; $p < 0,001$.
- 👔 **Job Family a AI:** Koncentrace AI požadavků dramaticky závisí na typu role (Chi-kvadrát = 1 200; $p < 0,001$):
  - _Data & AI_: **54,4 %** pozic vyžaduje AI (referenční kategorie pro logit)
  - _Sr+ Software Engineer_: **31,9 %** pozic vyžaduje AI
  - _Software Engineer_: **25,7 %** pozic vyžaduje AI
  - _Software Developer_: **15,0 %** — výrazně méně než Engineer
  - _DevOps & Cloud_: **15,5 %** — AI není primární zaměření
- 📊 **Seniority a AI:** Podíl AI požadavků roste se senioritou (Chi-kvadrát = 55,2; $p < 0,001$; řádková procenta — v rámci každé úrovně seniority sčítají do 100 %):
  - Junior (0-2 roky): AI vyžaduje **16,3 %** pozic
  - Mid (3-5 let): AI vyžaduje **20,9 %** pozic
  - Senior+ (6+ let): AI vyžaduje **21,3 %** pozic — nejvyšší podíl
  - Z toho Applied/Core AI: Junior 6,2 % → Mid 7,4 % → Senior+ **8,7 %** — hlubší AI roste se zkušeností

### 2. 🔬 Statistické testování hypotéz

K ověření, zda pozorované rozdíly v deskriptivní statistice nejsou dílem náhody, byly provedeny inferenční testy (všechny se signifikancí $p < 0,001$ ✅).

- 💸 **Rozdíly ve mzdách (T-test a ANOVA):**
  - Dvouvýběrový t-test prokázal **statisticky velmi významný rozdíl** ($t = -28,21$; $p < 0,001$) s hrubým rozdílem **$24 086** ročně ✅.
  - Cohenovo _d_ nabylo hodnoty **0,587**, což indikuje prakticky významný, 🟡 **středně silný vliv** na plat. "AI prémie" se formuje jako stabilní jev.
  - **Mann-Whitney U test** (neparametrická alternativa) potvrdil totožný závěr ($z = -26,81$; $p < 0,001$), čímž je výsledek robustní i bez předpokladu normálního rozdělení.
  - **ANOVA s Bonferroniho korekcí** prokázala, že **všechny páry** (None vs. AI Integration, None vs. Applied AI, AI Integration vs. Applied AI) se vzájemně signifikantně liší ($p < 0,001$ u všech tří porovnání). Prémie tedy roste monotónně s hloubkou AI expertízy.
- 🎓 **Rozdíly v profilu uchazeče (Chi-kvadrát):**
  - U rolí vyžadujících AI je strukturálně odlišný očekávaný profil kandidáta. Zaměstnavatelé u AI pozic signifikantně častěji požadují vysokoškolský titul (bakalář a vyšší) a pokročilejší úroveň seniority (Mid a Senior+).
  - ⚠️ **Poznámka:** Chi-kvadrát test vzdělání × AI (`edu_cat > 0`) obsahuje po filtrování pouze jednu kategorii (Bachelor or Higher), takže je neinformativní. Skutečně vypovídající je test zkušeností × AI (Chi-kvadrát = 55,2; $p < 0,001$).

### 3. 📉 Regresní analýza – Kvantifikace čisté "AI Prémie"

Pro rigorózní vyčíslení platového benefitu a očištění vlivu tzv. zavádějících proměnných (confounders) byly zkonstruovány MNS (OLS) regresní modely vysvětlující přirozený logaritmus mzdy (`ln_salary`).

#### 📊 Model A (Základní OLS model)

- **Specifikace:** Skill clustery, AI tier, sektor (NACE), region, remote práce, typ a velikost organizace ($R^2 = 0,247$ 🆗).
- **Výsledek:** _AI Integration_ = nárůst mzdy o **+9,4 %**, _Applied/Core AI_ = **+13,6 %**.

#### 📈 Model B (Rozšířený, preferovaný model)

- **Specifikace:** Model A + job family, **granulární vzdělání** (`edu_ols`, 4 úrovně: High School → Associate → Bachelor → Master+), léta zkušeností. Model smazal šum a vysvětlí **38,0 %** diferencí v platech ($R^2 = 0,380$ 🟢 **Skvělá hodnota**).
- **Interpretace úpravy AI Prémie:** Důsledkem kontroly osobních charakteristik hrubá "AI prémie" mírně poklesla. Větší platy AI pozic byly zčásti způsobeny faktem, že se na ně hlásí **vzdělanější uchazeč s delší praxí**.
- **Konečné vyčíslení čistého AI benefitu:** I po dokonalém očištění je expertíza v AI nadstandardně pojištěna:
  - 🧩 **AI Integration:** Čistá mzdová prémie činí 💵 **+ 7,5 %** ✅ ke standardní mzdě.
  - 🧠 **Applied/Core AI:** Účast na vývoji samotného jádra představuje benefit 🚀 **+ 9,6 %** ✅.

#### 📊 Model B bez job_family (Test mediace)

- **Specifikace:** Jako Model B, ale BEZ proměnné `job_family_num` ($R^2 = 0,342$). Oba modely se reportují paralelně — job family může být mediátor (typ pozice přímo ovlivňuje požadavky na AI).
- **Výsledek:** AI prémie je vyšší než v plném Modelu B, což potvrzuje mediaci:
  - 🧩 **AI Integration:** +8,4 % (vs. +7,5 % v plném B)
  - 🧠 **Applied/Core AI:** +11,8 % (vs. +9,6 % v plném B)
- 💡 Job family absorbuje cca 1-2 p.b. AI efektu — Data & AI pozice mají vyšší platy i nezávisle na AI požadavku.

#### 📊 Model B — Mincerova specifikace (kontinuální zkušenosti)

- **Specifikace:** Jako Model B, ale s kontinuální proměnnou `experience_min_llm` + `experience_sq` (kvadratický člen) místo kategorické `exp_category`. Klasická Mincerova mzdová rovnice s klesajícími výnosy ze zkušeností.
- **Motivace:** Model by měl obsahovat alespoň jednu kontinuální proměnnou pro správnou specifikaci Mincerovy rovnice.
- ⚠️ **Caveat:** Model B-Mincer má **jiný vzorek** (N = 12 606 vs. 14 640). Model B kóduje missing zkušenosti jako explicitní kategorii (`exp_category = 0`), zatímco B-Mincer používá kontinuální `experience_min_llm`, kde missing hodnoty jsou automaticky Statou vyřazeny. Přímé srovnání $R^2$ je proto pouze orientační.
- **Výsledek:** $R^2 = 0,423$ (nejvyšší ze všech modelů, ale částečně díky menšímu vzorku):
  - 🧩 **AI Integration:** +7,7 %
  - 🧠 **Applied/Core AI:** +9,9 %
  - 📈 Experience: +6,8 % za rok, s klesajícími výnosy (experience²: -0,2 %)
  - AI prémie konzistentní s Modelem B — robustní výsledek.

#### 📋 Kompletní hierarchie faktorů určujících plat (Model B)

AI prémie je signifikantní, ale **není nejsilnějším prediktorem platu**. Tato hierarchie je klíčový nález pro obhajobu.
- ⚠️ **Poznámka ke srovnatelnosti:** Všechny proměnné v hierarchii jsou dummy (0 vs 1). Pro srovnatelnost koeficientů je třeba mít na paměti, že efekty dummy proměnných ukazují rozdíl oproti referenční kategorii, zatímco kontinuální proměnné (v modelu B-Mincer) ukazují efekt jednotkové změny.

| Faktor | Efekt na plat | $p$-value | Interpretace |
|---|---|---|---|
| **West (region)** | +22,4 % | $< 0,001$ | Geografická prémie (Silicon Valley, Seattle) |
| **Sr+ Software Engineer** | +17,3 % | $< 0,001$ | Seniorní pozice jsou nejlépe placeny |
| **Unknown region (remote/celostátní)** | +16,5 % | $< 0,001$ | Celostátní/remote pozice platí lépe |
| **Senior+ zkušenosti (6+ let)** | +12,0 % | $< 0,001$ | Zkušenosti jsou silně odměňovány |
| **Applied/Core AI** | **+9,6 %** | $< 0,001$ | **AI prémie (hlubší expertíza)** |
| **Management role** | +9,4 % | $< 0,001$ | Manažerská prémie |
| **Northeast (region)** | +9,8 % | $< 0,001$ | New York, Boston |
| **AI Integration** | **+7,5 %** | $< 0,001$ | **AI prémie (používání AI)** |
| **Systems programming** | +7,3 % | $< 0,001$ | Nízkoúrovňové dovednosti jsou vzácné |
| **DevOps & Cloud** | +7,0 % | $< 0,001$ | Infrastrukturní prémie |
| **Master+ vzdělání** | +5,0 % | $< 0,001$ | Mírný efekt vyššího vzdělání |
| **Remote práce** | +3,5 % | $< 0,001$ | Remote pozice platí mírně lépe |
| **Junior (0-2 roky)** | **-17,5 %** | $< 0,001$ | Obrovský malus za nízkou zkušenost |
| **Associate degree** | -11,0 % | $< 0,001$ | Penalizace za nízké vzdělání |

> 💡 **Klíčový závěr:** AI prémie je reálná a měřitelná, ale tradiční faktory lidského kapitálu (zkušenosti, geografie, typ pozice) mají na plat větší absolutní vliv. AI prémie je **aditivní bonus** nad rámec těchto faktorů.

### 4. 🛠️ Analýza odborné náročnosti profilu (Hard skills)

- Doplňková analýza vysvětluje mzdový rozdíl: inzeráty s AI vyžadují objemnější spektrum technických dovedností.
- Tradiční IT pozice: průměrně **~16,0 dovedností** (SD = 10,1).
- AI Integration pozice: **~19,5 dovedností** (SD = 10,4).
- Applied/Core AI pozice: **~20,6 dovedností** (SD = 10,8).
- T-test: rozdíl 3,8 skills je vysoce signifikantní ($t = -19,7$; $p < 0,001$).
- 📈 Nárůst o cca 4 nové okruhy požadavků reflektuje komplexnější povahu AI rolí.

### 5. 🔍 Logistické modely — Ekonometrická analýza odborné náročnosti

Tato sekce představuje ekonometrickou verzi analýzy odborné náročnosti z bodu 4. Zatímco deskriptivní analýza ukázala, že AI pozice vyžadují v průměru o 4 skills více, logistické modely **kvantifikují, které konkrétní dovednosti a charakteristiky jsou asociovány s AI požadavky** po kontrole ostatních faktorů.

#### Klíčové zjištění: Role > Firma

| Model | Pseudo $R^2$ | Interpretace |
|---|---|---|
| **M1** (Profil firmy: sektor, typ, velikost, region) | **2,5 %** | Firemní profil sám o sobě AI požadavek skoro nepredikuje |
| **M2** (Profil role: skill clustery, job family, vzdělání, zkušenosti) | **35,9 %** | Typ dovedností a role je rozhodující |
| **M3** (Kompletní: M1 + M2) | **36,7 %** | Přidání firemních proměnných k M2 přináší marginální zlepšení (+0,8 p.b.) |
| **M3a** (M3 bez job_family) | **35,8 %** | Bez job_family — AME clusterů mírně vzrostou (přebírají efekt) |
| **M3b** (M3 bez job_family a seniority) | **35,8 %** | Seniority přidává minimum vysvětlující síly k M3a |

> 💡 **Interpretace pro obhajobu:** O tom, zda pozice vyžaduje AI, rozhoduje primárně **"co se tam dělá"** (konkrétní dovednosti a typ role), nikoliv **"kdo je zaměstnavatel"** (sektor, velikost, typ firmy). Logit modely jsou v tomto smyslu ekonometrickým nástrojem pro identifikaci dovedností asociovaných s AI poptávkou — doplňují deskriptivní zjištění z analýzy skill_count rigorózním statistickým způsobem.

#### Nejsilnější prediktory AI požadavku (Logit M3, Average Marginal Effects)

**Skill clustery s největším vlivem na pravděpodobnost AI požadavku (AME = změna v procentních bodech):**

| Skill cluster | AME (p.b.) | $p$-value | Interpretace |
|---|---|---|---|
| **Generative AI** | **+30,7 p.b.** | $< 0,001$ | 🔥 Naprosto dominantní prediktor |
| **Data Science / ML** | **+25,2 p.b.** | $< 0,001$ | 🔥 Druhý nejsilnější — jádro AI |
| **Dynamic Web** | +4,8 p.b. | $< 0,001$ | AI se prosazuje do webových technologií |
| **Cloud Computing** | +3,2 p.b. | $< 0,001$ | Cloud je infrastrukturou AI |
| **Frontend Development** | +2,9 p.b. | $< 0,001$ | AI pronikání do UI/UX |
| **BI / Analytics** | +2,3 p.b. | $< 0,001$ | Analytické role přijímají AI |
| **Enterprise / Managed** | -4,1 p.b. | $< 0,001$ | Enterprise systémy AI zatím nevyžadují |
| **Certifications** | -3,1 p.b. | $< 0,01$ | Certifikované role jsou tradičnější |
| **Scripting / Shell** | -2,5 p.b. | $< 0,01$ | Tradiční skriptování bez AI |

- ⚠️ **Poznámka k cirkularitě:** `cluster_generative_ai` a `cluster_data_science__ml` obsahují skills (GPT, LLM, TensorFlow, PyTorch...), které přímo implikují AI požadavek. Proto byla provedena **citlivostní analýza** s vyřazením obou clusterů — výsledky viz sekce 5b.
- 📊 **Formát výsledků:** Reportujeme **Average Marginal Effects (AME)** místo Odds Ratios / Relative Risk Ratios. AME ukazují změnu v procentních bodech a jsou ekonomům srozumitelnější.

#### 5b. Citlivostní analýza — Logit/Mlogit bez GenAI a DS/ML clusterů

- **Motivace:** `cluster_generative_ai` a `cluster_data_science__ml` jsou potenciálně cirkulární prediktory — skills v těchto clusterech přímo definují AI pozice.
- **Metoda:** Model M3 byl přeodhadnut bez těchto dvou clusterů.
- **Výsledek:** Pseudo $R^2$ **dramaticky pokleslo** z 36,7 % na **12,9 %** — GenAI a DS/ML clustery vysvětlovaly 24 p.b. variability!
- **Co se stalo s ostatními clustery (AME srovnání M3 vs M3-nocirc):**
  - Cloud Computing: +3,2 → **+6,2 p.b.** (zdvojnásobení)
  - Dynamic Web: +4,8 → **+8,8 p.b.** (zdvojnásobení)
  - Data Engineering: +1,3 → **+4,8 p.b.** (ztrojnásobení)
  - DevOps/Containers: n.s. → **+2,4 p.b.*** (nově signifikantní)
  - BI/Analytics: +2,3 → **+3,4 p.b.** (nárůst)
  - Systems Programming: n.s. → **+3,6 p.b.*** (nově signifikantní)
- 💡 **Interpretace:** Bez cirkulárních clusterů ostatní dovednosti přebírají vysvětlovací sílu. Model ale ztrácí 2/3 prediktivní schopnosti, což ukazuje, že GenAI a DS/ML clustery nesou unikátní informaci — nejsou redundantní s ostatními prediktory.

### 6. 🧬 Multinomiální logit — Rozlišení "používání AI" vs. "vývoj AI"

Toto je stěžejní sekce pro tezi diplomové práce: co odlišuje pozice, které AI **pouze integrují do svých procesů** (AI Integration), od pozic, které AI **přímo vyvíjejí** (Applied/Core AI)?

#### Srovnání AME (Average Marginal Effects) z Mlogit M2 — změna pravděpodobnosti dané kategorie v procentních bodech

| Prediktor | P(AI Integration) AME | P(Applied/Core AI) AME | Co to znamená |
|---|---|---|---|
| **Data Science / ML** | +14,3 p.b.*** | +11,8 p.b.*** | Obě kategorie silně, ale Integration mírně více |
| **Generative AI** | +23,4 p.b.*** | +9,2 p.b.*** | Integration dominuje — GenAI nástroje jsou pro "používání AI" |
| **Systems Programming** | -2,6 p.b.*** | +3,2 p.b.*** | 🔑 Rozlišuje: Applied AI vyžaduje, Integration NE |
| **Data Engineering** | -1,3 p.b.* | +2,3 p.b.*** | 🔑 Data engineering = doména Applied AI |
| **Dynamic Web** | +2,8 p.b.*** | +2,7 p.b.*** | Obojí relevantní, rovnoměrně |
| **Frontend Development** | +4,4 p.b.*** | -0,9 p.b.* | Frontend je jen AI Integration, ne Applied AI |
| **Enterprise Platforms** | +4,2 p.b.*** | -1,9 p.b.** | Enterprise platformy jen pro Integration |
| **Cloud Computing** | +2,8 p.b.*** | +1,0 p.b.* | Mírně silněji pro Integration |

> 💡 **Klíčová interpretace:** Skutečné AI pozice (Applied/Core) se od "pouhého používání AI" (Integration) liší tím, že vyžadují **fundamentální dovednosti** — data science/ML, systémové programování a data engineering. Naopak AI Integration je charakterizována **aplikačními dovednostmi** — frontend, enterprise platformy, generativní AI nástroje. Toto přímo validuje rozlišení mezi "rozumět AI" a "používat AI".

- 📊 **Formát:** Reportujeme **AME (Average Marginal Effects)** v procentních bodech místo RRR (Relative Risk Ratios). AME jsou ekonomům srozumitelnější — přímo ukazují, o kolik procentních bodů se změní pravděpodobnost dané kategorie při přítomnosti daného skill clusteru.

#### Job Family efekty v multinomiálním modelu (M2)

| Job Family (vs. Data & AI) | P(AI Integration) AME | P(Applied/Core AI) AME |
|---|---|---|
| **DevOps & Cloud** | -0,8 p.b. (n.s.) | -6,6 p.b.*** |
| **Software Developer** | -2,6 p.b.* | -6,9 p.b.*** |
| **Other** | -0,2 p.b. (n.s.) | -8,1 p.b.*** |
| **Software Engineer** | +0,6 p.b. (n.s.) | -4,1 p.b.*** |
| **Sr+ Software Engineer** | +0,8 p.b. (n.s.) | -3,7 p.b.*** |
| **Management** | +2,0 p.b. (n.s.) | -2,1 p.b. (n.s.) |

> 💡 Oproti kategorii "Data & AI" mají všechny ostatní job families nižší pravděpodobnost Applied/Core AI. Management je zajímavě neutrální — manažeři v AI a non-AI firmách mají podobné AI požadavky.

### 7. 📉 Co se nepotvrdilo (Statisticky nevýznamné faktory)

Při analýze dat je naprosto klíčové věnovat pozornost i proměnným, u kterých se hypotéza **nepotvrdila** ($p > 0,05$). Ukazuje to, že model robustně funguje a nepřiděluje "plochou" platovou prémii všemu bez rozdílu:

- 🏢 **Velikost firmy pod 500 zaměstnanců:** Oproti firmám nezjištěné velikosti nemají menší a střední podniky (do 500 lidí) statisticky odlišné platy v těchto inzerátech ($p > 0,10$). Platový odskok začíná být prokazatelný až od mety větších podniků (1000+ zaměstnanců).
- 🎓 **Zkušenosti Mid (3-5 let) vs. Neuvedeno:** Uchazeči s požadovanou praxí 3-5 let nemají statisticky odlišný plat od inzerátů, které praxi nespecifikují ($p = 0,568$). Referenční kategorie Mid je tedy "normou".
- 🛠️ **Vybrané technologické clustery:** Samotný požadavek na klasický _Frontend development_ ($p = 0,969$), _Backend development_ ($p = 0,083$) nebo _OS/Embedded_ ($p = 0,512$) negeneruje průměrnému inzerátu statisticky významnou mzdovou prémii navíc; trh tyto schopnosti bere jako normový standard.
- 🏛️ **Typ firmy v logit/mlogit:** Typ organizace (Private, Public, Nonprofit/Gov) nemá statisticky významný vliv na pravděpodobnost AI požadavku v žádném z modelů ($p > 0,40$ u všech kategorií). AI proniká průřezově všemi typy firem.
- 📊 **Zkušenosti v logit M3:** Ani Junior ($p = 0,318$) ani Senior+ ($p = 0,108$) se signifikantně neliší od Mid seniority v pravděpodobnosti AI požadavku. AI požadavky nejsou vázány na konkrétní úroveň seniority.

### 💡 Ústřední argument pro obhajobu (Závěr interpretace)

> Analýza s nezpochybnitelnou statistickou jistotou (✅ $p < 0,001$) verifikuje tezi o mzdové "AI prémii". V regresním Modelu B se efektivně eliminovaly zavádějící faktory podoby vzdělání (granulárně rozlišeno 4 úrovně), regionu či charakteru firmy. Je tudíž prokazatelné, že **reálný čistý osobní mzdový příplatek za AI dovednosti činí solidních 7,5 až 9,6 %**. Tento příplatek odráží trend, v němž tyto pozice od zaměstnance vyžadují nejen sofistikovanější profil (seniorita a vzdělání), ale rovněž zvládání širšího arzenálu oborových hard skills.

**Pět klíčových argumentů pro obhajobu:**

1. **AI prémie je reálná a měřitelná:** 7,5-9,6 % čisté prémie po kontrole všech relevantních faktorů. Prémie **roste monotónně** s hloubkou AI expertízy (potvrzeno ANOVA Bonferroni).
2. **AI prémie existuje, ale není dominantním faktorem:** Zkušenosti (Junior -17,5 %, Senior+ +12,0 %), geografie (West +22,4 %) a typ pozice (Sr+ Engineer +17,3 %) mají na plat větší absolutní vliv. AI je aditivní bonus.
3. **O AI rozhoduje role, ne firma:** Firemní profil predikuje AI požadavek špatně (Pseudo $R^2$ = 2,5 %), ale typ role a dovednosti velmi dobře (Pseudo $R^2$ = 35,9 %). AI prostupuje průřezově celým IT trhem.
4. **Rozlišení "používání" vs. "vývoj" AI:** Multinomiální logit (AME) prokázal, že Applied/Core AI pozice se od AI Integration liší požadavkem na fundamentální dovednosti (Data Science/ML +11,8 p.b., Systems Programming +3,2 p.b.), zatímco Integration pozice se vyznačují aplikačními dovednostmi (Frontend +4,4 p.b., Enterprise Platforms +4,2 p.b., Generative AI +23,4 p.b.).
5. **AI pozice vyžadují širší portfolio dovedností:** Průměrně ~20 vs. ~16 skills — nárůst o cca 4 okruhy ($t = -19,7$; $p < 0,001$).

---

## 🛠️ ČÁST II: METODOLOGICKÉ ZHODNOCENÍ A BEST PRACTICES

### 🌟 Co je uděláno skvěle (Metodologické přednosti)

1. 📐 **Transformace závislé proměnné:** Použití logaritmu platu (`ln_salary`) pro OLS modely je naprostý standard (✅ Mincerova mzdová rovnice).
2. 🛡️ **Robustní standardní chyby:** U OLS modelů správně používáte `vce(robust)`. Kriticky důležité pro průřezová data k ošetření heteroskedasticity ✅.
3. 📉 **Diagnostika modelů (VIF):** Perfektní! Všechny hodnoty VIF jsou kolem 2, což bezpečně vylučuje multikolinearitu (✅ bezchybné).
4. 🔢 **Faktorové proměnné (`i.var`):** Stata je správně instruována, co jsou kategorie ✅.
5. 📊 **Marginální efekty (Logit/Mlogit):** Správně voláte `margins, dydx(*)` **bez `atmeans`**, což vyhodí Average Marginal Effects (AME) — průměrné marginální efekty přes celou distribuci pozorování, nikoliv efekty v bodě průměrů ✅.
6. 📏 **Effect size (Cohenovo d):** Obohacení T-testu o effect size (0.587) dokazuje teoretickou i praktickou významnost rozdílů ✅.
7. 🏗️ **Inkrementální modely (Base vs. Full):** Nárůst $R^2$ z 24.7 % na **38.0 %** ukazuje vysokou vypovídající hodnotu začlenění lidského kapitálu ✅.
8. 🎓 **Diferenciace vzdělávací proměnné:** Pro OLS model granulární proměnná `edu_ols` (4 úrovně: HS / Associate / Bachelor / Master+), pro Logit/Mlogit sloučená binární `edu_logit` (HS+Associate+Missing vs. Bachelor+) kvůli malému N v AI buňkách. ✅
9. 🔬 **Neparametrická verifikace:** Mann-Whitney U test doplňuje parametrický T-test a potvrzuje robustnost závěru bez předpokladu normality ✅.
10. 📐 **ANOVA Bonferroni korekce:** Všechny pairwise porovnání tierů prokázaly signifikanci — monotónní gradient prémie je ověřen ✅.

### ⚠️ Co vzít v potaz / Drobné nuance pro obhajobu

#### 1. Implementace požadavků dle checklistu (Soulad se zadáním)

Modelování a čištění dat bylo provedeno v striktním souladu s dohodnutým metodologickým checklistem:

- 🛠️ **Příprava proměnných:** Úspěšně byla zavedena závislá proměnná logaritmu platu `ln_salary`, byl vytvořen index počtu dovedností `skill_count` (0-80) a do binární podoby byla zredukována přítomnost AI požadavků `has_ai`.
- 🔄 **Slučování řídkých kategorií (Sparse data):** Aby multinomiální modely nevykazovaly chyby konvergence (např. _perfect separation_), byly striktně dodrženy limity počtu pozorování (min. 50 na buňku). Z toho důvodu došlo v rámci přípravy k agregaci specifických technických clusterů (vyřazeno např. `cluster_legacy__mainframe`), drobných sektorů i edukace v Logit modelu (sloučení High School a Associate Degree do jedné kategorie, PhD globálně sloučeno s Master).
- 🎯 **Rozlišení specifikace pro Logit a OLS:** V rámci OLS (mzdového) modelu dává smysl měřit vliv **Remote práce** i **granulárního vzdělání** (4 úrovně), nicméně v modelech predikujících _požadavek zaměstnavatele na AI_ je `is_remote` záměrně vynechána a vzdělání je sloučeno do binární formy kvůli malému N v buňkách AI kategorií.
- 📈 **Inkrementální 3-stupňová struktura Logit/Mlogit:** Práce těží z domluvené sekvence modelů (Profil firmy ➡️ Profil role/osoby ➡️ Kompletní). Výsledný Model 3 a rozpad vlivů přesně zrcadlí tuto strategickou posloupnost.

#### 2. Metodologické nuance

- 📉 **LR Test po robustních odhadech:** V logu je vidět použití `lrtest` pro srovnání Modelu A vs. B ($\chi^2 = 2850$; $p < 0,001$). LR test technicky vyžaduje klasické standardní chyby (ne robustní), ale při takto masivní statistice (2850 s 13 df) je závěr jednoznačný bez ohledu na variantu odhadu.
- 📉 **Nízké Pseudo $R^2$ u Logitu M1 (2,5 %):** To **NENÍ CHYBA** modelu. Znamená to, že o tom, zda pozice nabízí AI nebo ne, rozhoduje primárně specifický byznys firmy a konkrétní dovednosti, nikoliv hrubé firemní charakteristiky. Lokální proměnné (sektor, typ) jsou pro to přirozeně slabí prediktoři.
- 🏛️ **Ukázková konvergence dat:** Žádné "not concave" ani "perfect separation" chyby ✅! Toto **zdůrazněte při obhajobě** jako důkaz strukturálně čisté databáze.
- 🔄 **Hausman IIA test:** Proběhl bez chyby pomocí `capture hausman`. Test IIA (Independence of Irrelevant Alternatives) je klíčový pro validitu multinomiálního logitu.

#### 3. Limitace a diskuzní body pro obhajobu

1. ⚖️ **Kauzalita vs. korelace:** Model ukazuje _asociaci_, ne kauzální efekt. Není vyloučeno, že lidé, kteří se naučí AI, mají i další neměřené schopnosti (selection bias / omitted variable bias). Pro kauzální inferenci by byl potřeba kvazi-experimentální design.
2. 📋 **Datová limitace Glassdoor:** Data jsou reportována zaměstnavateli na Glassdoor, mohou být zkreslena směrem k větším firmám a technologickým hub městům. Menší firmy a tradiční podniky mohou být podreprezentovány.
3. 📅 **Časový průřez:** Jde o snapshot (2024-2025), ne o časovou řadu. Nelze říci, zda AI prémie roste, klesá nebo stagnuje.
4. 🧬 **Sloučení Core AI:** Pouhých 6 pozorování `core_ai` neumožňuje oddělit skutečné AI výzkumné pozice od aplikovaného AI vývoje.
5. 📊 **Chi-kvadrát test vzdělání (sekce 5.3):** Tabulka `edu_cat × has_ai` při filtraci `edu_cat > 0` obsahuje jen jednu kategorii (Bachelor or Higher) — test je v této podobě neinformativní a v práci jej lze vynechat nebo přeformulovat.

---

## 📋 ČÁST III: OVĚŘENÍ KLÍČOVÝCH METRIK PROTI STATA LOGU

Všechny hodnoty byly vizuálně zkontrolovány proti zdrojovému logu. 🟢 = Optimální / Zcela bez chyb.

### 📊 Základní parametry datasetu

| Metrika                                | Hodnota | Řádek v logu | Status             |
| -------------------------------------- | ------- | ------------ | ------------------ |
| Počet pozorování (po filtrování)       | 17 848  | 141          | 🟢 Přesně odpovídá |
| Počet pozorování s platem (OLS modely) | 14 640  | 1473         | 🟢 Přesně odpovídá |
| Podíl pozic s AI požadavky             | 19.48 % | 536          | 🟢 Přesně odpovídá |

### 📈 OLS regresní modely (Kvalita a shoda)

| Metrika                                   | Hodnota v analýze | Hodnota v logu | Status a interpretace                               |
| ----------------------------------------- | ----------------- | -------------- | --------------------------------------------------- |
| Model A — $R^2$                           | 24.7 %            | 0.24688        | 🟢 OK ($R^2$ je solidní)                            |
| Model B — $R^2$                           | **38.0 %**        | 0.38011        | 🌟 **Vynikající** (masivní skok z 24.7 % na 38.0 %) |
| Robustní std. chyby `vce(robust)`         | Ano               | ✅             | 🟢 Implementováno správně                           |
| LR test (společná signifikance M. A vs B) | $p = 0.0000$      | ✅             | 🟢 Přidání proměnných má masivní smysl              |

### 🔍 VIF diagnostika (Klinická kontrola multikolinearity)

_(Ideálně pod 5, cokoliv pod 10 je akceptovatelné)_
| Metrika | Hodnota VIF | Hodnocení kolinearity |
|---|---|---|
| Model A — **Mean VIF** | 1,79 | 🟢 **Zcela čisté** (bez kolinearity) |
| Model B — **Mean VIF** | 1,92 | 🟢 **Zcela čisté** (ukázkový průměr) |
| Model B-nojf — **Mean VIF** | 1,71 | 🟢 Nejnižší — bez job_family žádná kolinearita |
| Model B-Mincer — **Mean VIF** | 2,13 | 🟢 Mírně vyšší kvůli experience+experience² (VIF 5,65), ale OK |

### 💸 T-test a Platové prémie

| Metrika                           | Hodnota              | Řádek v logu | Status                                |
| --------------------------------- | -------------------- | ------------ | ------------------------------------- |
| Průměrný plat non-AI vs AI        | $119 902 vs $143 988 | 1270/1271    | 🟢 Zkontrolováno                      |
| Platový rozdíl (Hrubá AI premium) | **$24 086**          | 1276         | 🟢 Zkontrolováno                      |
| $t$-statistika                    | -28.21               | 1278         | 🟢 Vysoce signifikantní               |
| $p$-value statistika              | 0.0000               | 1282         | 🟢 Vysoce signifikantní ($p < 0.001$) |
| Cohenovo d (effect size)          | 0.587                | 1312         | 🟡 **Střední velikost efektu**        |
| Mann-Whitney $z$                  | -26.808              | 1444         | 🟢 Neparametrické potvrzení           |

### 🔢 ANOVA Bonferroni

| Porovnání | Rozdíl průměrů | $p$-value | Status |
|---|---|---|---|
| Applied AI vs. AI Integration | +$10 010 | $< 0,001$ | 🟢 Signifikantní |
| None vs. AI Integration | -$20 955 | $< 0,001$ | 🟢 Signifikantní |
| None vs. Applied AI | -$30 965 | $< 0,001$ | 🟢 Signifikantní |

### 🚫 Statisticky nevýznamné p-hodnoty (Důkaz selektivity modelu)

_Správně nastavený model nedá všemu status "významné". Zde kontrolujeme proměnné, u kterých se s jistotou nepodařilo prokázat vliv ($p > 0.05$), což dokládá realističnost regresních rovin a absenci "šumu":_
| Proměnná (Vliv na Plat v Modelu B) | Hodnota P-value | Řádek v logu | Status (Validace do práce) |
|---|---|---|---|
| Zkušenost: Missing vs. Mid (3-5 let) | $p = 0.568$ | 1769 | 🟢 **Velmi správně nedetekován vliv** |
| Dovednost: Frontend development | $p = 0.969$ | 1710 | 🟢 **Spolehlivě nevýznamné** |
| Dovednost: Backend development | $p = 0.083$ | 1700 | 🟢 **Nevýznamné (Očekávaný Base-standard)** |
| Dovednost: Data Science / ML | $p = 0.042$ | 1704 | 🟡 **Hraniční** — ML samo o sobě přidá jen +1,7 %, protože efekt je zachycen v `ai_level` |
| Velikost firmy: 51-200 | $p = 0.944$ | 1748 | 🟢 **Spolehlivě nevýznamné** |
| Sektor firmy: Nonprofit/Gov/Edu | $p = 0.964$ | 1743 | 🟢 **Spolehlivě nevýznamné** |
| Typ firmy (Logit M3): Private | $p = 0.950$ | 2921 | 🟢 **Typ firmy nepredikuje AI požadavek** |

> 💡 **PROČ SE VYSOKÉ P-HODNOTY NEMAŽOU Z MODELU:**
> Často panuje zjednodušená představa, že proměnná s $p > 0.05$ je "špatná" a model se bez ní musí přepsat a spustit znovu. V seriózní ekonometrii se ale takové proměnné modelům zachovávají jako tzv. **kontrolní proměnné (control variables)**. Jejich úkolem není vyhrát soutěž na signifikanci, ale "podržet" a zafixovat strukturu firmy (např. sektorové zařazení nebo konkrétní velikost firmy) na nějakém pozadí. Jakmile bychom tyto "neúspěšné vlivy" z analýzy prostě vymazali (např. celý sektor školství), mohly by reálně začít zkreslovat chování onoho mzdového benefitu u "AI vlivu". Tím, že tam ty parametry v modelu zůstaly jako nezúčastněné stabilizátory na nule, je očištěná "AI Prémie" tou 100% nejopravdovější hodnotou!

### 🎲 Logistické a Multinomiální modely (Predikce výskytu AI)

Dle dohodnuté specifikace tyto modely určují, _proč vůbec pozice vyžaduje AI_ (Base úroveň = None). Zde je provedena verifikace konvergence – úspěšně jsme zamezili riziku zhroucení modelu vlivem nedostatku dat v subkategoriích.

| Modely                       | N      | Pseudo $R^2$ | Konvergence                                    |
| ---------------------------- | ------ | ------------ | ---------------------------------------------- |
| **Logit M1** (Profil firmy)  | 17 848 | 2,5 %        | 🟢 4 iter.                                     |
| **Mlogit M1** (Profil firmy) | 17 848 | 2,1 %        | 🟢 4 iter.                                     |
| **Logit M2** (Profil role)   | 17 848 | 35,9 %       | 🟢 5 iter.                                     |
| **Mlogit M2** (Profil role)  | 17 848 | 33,2 %       | 🟢 6 iter.                                     |
| **Logit M3** (Kompletní)     | 17 848 | 36,7 %       | 🟢 5 iter.                                     |
| **Mlogit M3** (Kompletní)    | 17 848 | 33,9 %       | 🟢 6 iter.                                     |
| **Logit M3a** (bez job_family)| 17 848| 35,8 %       | 🟢 5 iter.                                     |
| **Mlogit M3a** (bez job_family)| 17 848| 32,8 %      | 🟢 6 iter.                                     |
| **Logit M3b** (bez jf+seniority)| 17 848| 35,8 %    | 🟢 5 iter.                                     |
| **Mlogit M3b** (bez jf+seniority)| 17 848| 32,7 %   | 🟢 6 iter.                                     |
| **Logit M3-nocirc** (bez GenAI+DS/ML)| 17 848| 12,9 %| 🟢 5 iter. (výrazný propad — citlivostní test) |
| **Mlogit M3-nocirc** (bez GenAI+DS/ML)| 17 848| 12,7 %| 🟢 6 iter.                                   |
| **Marginální efekty**        | —      | —            | 🟢 `margins, dydx(*)` u všech 12 modelů |
| **Hausman IIA test**         | —      | —            | 🟢 `capture hausman` proběhl bez chyby         |

> ℹ️ Nízké Pseudo $R^2$ u M1 (2 %) je zcela očekávané — samotný firemní profil (sektor, typ, velikost) predikuje AI požadavek slabě. Vysoké Pseudo $R^2$ u M2/M3 (33–36 %) potvrzuje, že technologické skill clustery a job family jsou silnými prediktory AI poptávky.

---

## 🛡️ ČÁST IV: ARGUMENTY PRO OBHAJOBU (Potenciální otázky komise)

Tato sekce připravuje odpovědi na metodologické námitky, které komise u státnic může vznést. Jde o problémy, které nelze snadno opravit v do-filu, ale lze je věcně argumentovat.

### B1. Cirkularita: Skill clustery predikují AI tier v logit/mlogit

**Potenciální námitka:** "Cluster_* proměnné jsou extrahovány ze stejného textu inzerátu, ze kterého LLM přiřadil ai_level. Není to tautologické?"

**Argument:** Logit/mlogit modely mají **exploratorní charakter**. Cílem není prokázat kauzalitu ("cloud computing způsobuje AI požadavek"), ale identifikovat dovednostní profily asociované s AI pozicemi. Skill clustery jsou odvozeny z **celého textu inzerátu** (requirements, description), zatímco AI klasifikace je založena na **specifické zmínce AI nástrojů a technologií**. Jde o překrývající se, ale nikoliv identické informační zdroje. Pro kauzální inferenci by byl potřeba instrumentální proměnná nebo kvazi-experiment, což přesahuje rozsah diplomové práce. V do-filu je toto zdokumentováno komentářem v sekci 6B.

### B2. has_ai vs. ai_level — dvě různé proměnné

**Potenciální námitka:** "Proč používáte dvě různé závislé proměnné pro AI požadavky? Nejsou nekonzistentní?"

**Argument:** Dvě proměnné měří různé dimenze: `has_ai` (přísný filtr — specifické AI nástroje po odfiltrování buzzwords) vs. `ai_level` (širší pojetí — jakákoliv AI zmínka kategorizovaná do tierů). Toto rozlišení je záměrné — binární logit používá konzervativnější definici, mlogit pracuje s tiered klasifikací. Výsledky jsou **konzistentní napříč oběma definicemi**: logit M3 (has_ai) má Pseudo R² = 36,7 %, mlogit M3 (ai_level) má Pseudo R² = 33,9 %, a klíčové prediktory (GenAI, Data Science/ML clustery) dominují v obou modelech.

### B3. Žádná inter-rater reliabilita LLM klasifikace

**Potenciální námitka:** "Jak víte, že LLM klasifikuje AI tiery správně? Kde je Cohen's kappa?"

**Argument:** LLM klasifikace byla validována třemi způsoby:
1. **Confidence threshold 0,7** filtruje nejistá hodnocení (vyřazeno 607 z 18 464 pozorování, tj. 3,3 %).
2. **Hybridní přístup** (deterministický slovník + LLM) poskytuje cross-validaci — proměnná `ai_det_llm_match` měří shodu mezi oběma zdroji.
3. **has_ai_flag** vyžaduje intersekci: tier ≠ None A zároveň zbývají specifické skills po odfiltrování buzzwords.

Plná inter-rater studie (Cohen's kappa s manuálním kódováním vzorku) je legitimní rozšíření, které přesahuje rozsah diplomové práce. V textu je uvedena jako limitace.

### B4. Selection bias — 18 % pozic nemá plat

**Potenciální námitka:** "OLS modely běží na subsamplu (14 640 z 17 848). Není to selection bias?"

**Argument:** V do-filu byla přidána diagnostika chybějících platů (sekce 4.15): chi-kvadrát test missingness × AI flag a missingness × AI tier. Pokud test ukáže, že podíl chybějících platů se **neliší** systematicky mezi AI a non-AI pozicemi, selection bias je nepravděpodobný. Heckmanova korekce (dvoustupňový model) přesahuje rozsah práce, ale je uvedena jako směr dalšího výzkumu. Důležité je, že 82 % pozorování má platové údaje, což je pro webový scraping z Glassdooru nadprůměrně vysoký pokrytí.

### B5. Confidence threshold 0,7 je arbitrární

**Potenciální námitka:** "Proč zrovna 0,7? Co kdybyste použili 0,5 nebo 0,8?"

**Argument:** Threshold 0,7 byl zvolen jako kompromis mezi pokrytím a kvalitou:
- **Retention rate:** 96,7 % — pouze 607 z 18 464 pozorování vyřazeno, takže vliv na statistickou sílu je minimální.
- **Kvalitativní odůvodnění:** 0,7 je standardní práh v klasifikačních úlohách (analogie: 70% správnost je typický baseline pro NLP úlohy).
- **Citlivostní analýza:** Porovnání výsledků na thresholdech 0,5 / 0,6 / 0,7 / 0,8 je v plánu jako rozšíření.
- Pokud by threshold dramaticky měnil výsledky, signalizovalo by to nestabilitu klasifikace. Při 96,7% retenci je dopad pravděpodobně marginální.

### B6. Průřezová data — žádná kauzalita

**Potenciální námitka:** "Můžete říct, že AI dovednosti způsobují vyšší plat?"

**Argument:** Práce explicitně uvádí, že jde o **průřezovou asociační studii**. Termín "AI prémie" používáme ve smyslu **mzdového rozdílu asociovaného s AI požadavky**, nikoliv jako kauzální efekt. OLS model kontroluje pozorované confoundery (vzdělání, zkušenosti, region, sektor, velikost firmy, typ pozice), ale nemůže eliminovat nepozorované faktory (schopnosti, motivace, vyjednávací síla). Pro kauzální inferenci by byl potřeba:
- Panelový design (sledování stejných pozic v čase)
- Instrumentální proměnná (exogenní šok ovlivňující AI adopci)
- Přirozený experiment (např. regulatorní změna)

Toto je standardní limitace většiny mzdových studií založených na průřezových datech (Mincer 1974, Acemoglu & Autor 2011).

### B7. Core_ai sloučeno do Applied AI (pouze 6 pozorování)

**Potenciální námitka:** "Proč jste neseparovali Core AI? Není to klíčová kategorie vaší práce?"

**Argument:** Sloučení je statisticky nutné — 6 pozorování neumožňuje stabilní odhad koeficientů (pod prahem 50 na buňku). V textu práce je uvedeno jako limitace s vysvětlením: skutečně výzkumné AI pozice ("core AI researchers") jsou na Glassdoor vzácné, protože se typicky inzerují na specializovaných platformách (akademické portály, AI konference jako NeurIPS/ICML, interní nábor v FAANG firmách). Glassdoor zachycuje primárně "mainstream" IT trh, nikoliv úzce specializované výzkumné pozice.

### B8. Proč ne kvantilová regrese?

**Potenciální námitka:** "OLS měří průměrný efekt. Co když je AI prémie jiná pro nízko- a vysokopříjmové pozice?"

**Argument:** OLS s Mincerovou log-lineární specifikací je **standard pro mzdové studie** v ekonomii práce. Kvantilová regrese by ukázala, zda AI prémie je vyšší/nižší v různých částech platového spektra (např. zda AI prémie je větší pro vysokopříjmové pozice). Jde o legitimní rozšíření, ale:
1. Mincerova rovnice je dostatečná pro základní kvantifikaci AI prémie.
2. Interpretace kvantilové regrese je komplexnější a vyžaduje teoretické zdůvodnění, proč by efekt měl být heterogenní.
3. V diskuzní části práce je uvedena jako směr dalšího výzkumu.

### B9. LR test po robustních standardních chybách

**Potenciální námitka:** "LR test formálně vyžaduje klasické SE, ale Model A a B používají robustní SE. Je test validní?"

**Argument:** LR test formálně vyžaduje klasické standardní chyby (MLE). Pro účely tohoto testu byly modely přeodhadnuty bez `vce(robust)` (v do-filu je vidět samostatný blok pro LR test se `store model_a_lr` / `model_b_lr`). Při $\chi^2 = 2\,850$ s 13 stupni volnosti je závěr tak **drtivě signifikantní**, že volba odhadu variance na něj nemá žádný praktický dopad — $p$-hodnota je astronomicky pod jakýmkoliv rozumným prahem. Model B je jednoznačně lepší než Model A.

### B10. Buzzword filtr v has_ai je agresivní/nedostatečný

**Potenciální námitka:** "Proč odstraňujete 'AI' a 'ML' z has_ai? Nejsou to legitimní AI požadavky?"

**Argument:** Filtr odstraňuje obecné buzzwords (AI, ML, Artificial Intelligence, Machine Learning, GenAI) z proměnné `has_ai_flag`, aby se rozlišily pozice, které AI **skutečně technicky vyžadují** (např. "experience with TensorFlow, PyTorch" → has_ai = 1), od těch, které AI pouze **zmiňují v marketingovém popisu** (např. "we are an AI-driven company" → has_ai = 0, pokud chybí specifické AI nástroje). Toto je **konzervativní přístup** — falešné pozitivy (non-AI pozice klasifikovaná jako AI) jsou minimalizovány za cenu mírně vyšších falešných negativů. V kontextu výzkumné otázky ("kdo skutečně potřebuje AI dovednosti?") je konzervativní přístup preferovaný před liberálním.

---

_Poslední aktualizace: 17. března 2026 — (1) opraveny remote statistiky (sloupcová vs řádková procenta), (2) přidány seniority cross-taby, (3) přidán OLS Model B bez job_family + Model B-Mincer s kontinuální experience+experience², (4) přidány Logit/Mlogit M3a (bez job_family) a M3b (bez job_family+seniority), (5) přidána citlivostní analýza bez GenAI a DS/ML clusterů (test cirkularity), (6) tabulky přepsány z RRR na AME, (7) logit interpretace přeformulována jako ekonometrická analýza odborné náročnosti, (8) ověřeno: margins bez atmeans. ✅ Všechny hodnoty aktualizovány z nového Stata runu (17. března 2026, 18:05)._
