# 📈 Vývoj metodologie analýzy — Iterativní zlepšení modelu

Tento dokument zachycuje postup zpřesňování ekonometrické metodologie diplomové práce v průběhu konzultací s vedoucím. Slouží jako doklad systematického a vědomého přístupu ke sběru, čištění a modelování dat.

---

## 🔄 Přehled verzí analýzy

| Verze | Datum | Log soubor | Klíčová změna |
|---|---|---|---|
| **V1** (výchozí) | 5. 3. 2026 | `run__5_Mar_2026_13-50-35` | Původní binární vzdělání; jednoduché logit modely |
| **V2** (revidovaná) | 5. 3. 2026 | `run__5_Mar_2026_22-59-18` | Granulární vzdělání pro OLS; 3-modelová logit struktura; PhD→Master |

---

## 📊 Porovnání klíčových metrik: V1 → V2

### 1. OLS Modely (Mzdová regrese)

| Metrika | V1 (run 13-50-35) | V2 (run 22-59-18) | Změna |
|---|---|---|---|
| **Model A:** $R^2$ | 0.247 | 0.247 | ➡️ beze změny (model A nezměněn) |
| **Model B:** $R^2$ | 0.376 | **0.380** | ✅ **+0.4 p.b.** |
| **Model B:** N pozorování | 14 640 | 14 640 | ➡️ beze změny |
| Vzdělání v Model B | binární `edu_cat` (0/1) | granulární `edu_ols` (0–4) | ✅ **zásadní zlepšení** |
| Čistá AI prémie — AI Integration | +8.0 % | **+7.5 %** | ✅ přesnější odhad |
| Čistá AI prémie — Applied/Core AI | +10.4 % | **+9.6 %** | ✅ přesnější odhad |

> **Proč se AI prémie mírně snížila?** Přidáním granulárního vzdělání (4 úrovně místo 1) model přesněji kontroluje, že AI pozice jsou obsazovány vzdělanějšími kandidáty. Část zdánlivé „AI prémie" byla ve V1 ve skutečnosti „Master prémie". Model V2 je proto přesnější a metodologicky robustnější.

---

### 2. Nová vzdělávací proměnná: Granulární vs. Binární

Ve V2 byly vytvořeny dvě specializované vzdělávací proměnné dle doporučení vedoucího:

| Proměnná | Použití | Hodnoty | Odůvodnění |
|---|---|---|---|
| `edu_ols` | OLS modely (mzda) | 0=Missing, 1=HS, 2=Associate, 3=Bachelor, 4=Master+ | „Neslučuj — sloučením bys výsledky modelu zhoršil" |
| `edu_logit` | Logit/Mlogit modely | 0=No Degree/Missing, 1=Bachelor+ | „Musíš sloučit Associate+HS — málo obs. v AI buňce" |

**Vzdělávací gradient v Model B V2** (koeficienty oproti kategorii *Missing*):

| Úroveň vzdělání | Koeficient | Vliv na mzdu |
|---|---|---|
| High School | −0.068 | **−6.6 %** |
| Associate | −0.110 | **−10.4 %** |
| Bachelor | −0.033 | **−3.3 %** |
| **Master+** | **+0.050** | **+5.1 %** ✅ |

> V1 zobrazoval pouze jediný binární koeficient (Bachelor+ vs. ostatní), čímž ztrácel informaci o rozdítech mezi HS, Associate a Bachelor. V2 tento gradient zachycuje plně.

---

### 3. Logit / Mlogit modely (Predikce AI požadavku)

| Model | V1 Pseudo $R^2$ | V2 Pseudo $R^2$ | Počet iterací | Změna |
|---|---|---|---|---|
| Logit M1 (Profil firmy) — nový | — | **2.5 %** | 4 | ✅ nový model |
| Mlogit M1 (Profil firmy) — nový | — | **2.1 %** | 4 | ✅ nový model |
| Logit M2 / Původní Logit | ~36 % | **35.9 %** | 5 | ➡️ stabilní |
| Mlogit M2 / Původní Mlogit | ~33 % | **33.2 %** | 6 | ➡️ stabilní |
| Logit M3 (Kompletní) — nový | — | **36.4 %** | ✅ | ✅ nový model |
| Mlogit M3 (Kompletní) — nový | — | **33.9 %** | ✅ | ✅ nový model |

**Klíčová metodologická změna v Logit modelech:**
- V1: 1 Logit + 1 Mlogit model
- V2: **3 Logit + 3 Mlogit** modely — postupná (inkrementální) specifikace:
  - M1 = firemní profil (sektor, typ, velikost, region)
  - M2 = profil role a uchazeče (skill clustery, job family, vzdělání, zkušenosti)
  - M3 = kompletní (M1 + M2)

---

### 4. Globální změny při přípravě dat (obě verze)

| Úprava | V1 | V2 | Odůvodnění |
|---|---|---|---|
| Core AI (6 obs) → Applied AI | ✅ | ✅ | Méně než 50 obs., limit vedoucího |
| PhD → Master (globálně) | ❌ PhD v „Bachelor+" | ✅ **sloučeno (12 změn)** | Dle doporučení vedoucího |
| `is_remote` v Logit/Mlogit | ❌ chybně přidáno | ✅ **záměrně vynecháno** | Remote = vlastnost provozní, ne AI požadavek |
| Clustery s < 50 obs. | ✅ dropnuty 3 | ✅ dropnuty 3 | `legacy__mainframe`, `data_analysis__stats`, `tools__editors` |
| Job family: sloučení řídkých | ✅ | ✅ | `Frontend & Design`, `QA`, `Security`, `Systems` → Other |
| NACE sektory: sloučení | ✅ | ✅ | Top 5 + Unknown, ostatní → Other |

---

## 🎯 Interpretační přínos V2 pro diplomovou práci

1. **Přesnější AI prémie:** Po zavedení granulárního vzdělání klesla z 8.0 % / 10.4 % na 7.5 % / 9.6 %. Tato revize **posiluje věrohodnost** závěru — snížila se část, která byla dříve nesprávně přisuzována AI, ale ve skutečnosti reflektovala větší podíl vzdělanějších kandidátů.

2. **Vzdělávací gradient:** Nově vidíme, že Associate degree má výrazně nižší plat (−10.4 %) než prostý chybějící údaj, zatímco Bachelor je jen mírně nižší (−3.3 %). Tato informace by v binárním modelu V1 zcela zanikla.

3. **3-modelová logit struktura:** Umožňuje tvrdit, že firemní profil sám o sobě predikuje AI požadavek slabě (Pseudo $R^2$ ≈ 2 %), zatímco technologický profil role je silný prediktor (Pseudo $R^2$ ≈ 36 %). Toto je klíčový výsledek pro diskusi o tom, *kdo* AI adopci řídí.

---

*Dokument vytvořen: 5. března 2026 | Stata analýza: `ai_skills_analysis.do` | Dataset: `us_relevant_ai_stata.csv` (N = 17 848)*
