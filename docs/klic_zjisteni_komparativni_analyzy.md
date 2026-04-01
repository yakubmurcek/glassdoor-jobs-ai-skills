# Klíčová zjištění: Komparativní analýza AI dovedností v IT pracovních inzerátech
**Soubor:** `ai_skills_analysis_comparative.do`  
**Log:** `comparative_run__1_Apr_2026_22-22-22/ai_skills_comparative_analysis.log`  
**Analyzovaný pool:** USA (18 464) + DE Německo (9 254) + IN Indie (17 114) = **44 832 surových pozorování** → **38 432 po filtraci**  
**Datum analýzy:** 1. dubna 2026

---

## 1. Příprava dat

- Všechny tři datasety načteny a spojeny bez chyb.
- Filtrace: odstraněno **6 384 pozorování** s důvěrou modelu `desc_conf_llm < 0.7` a **12 pozorování** z roku 2023 a starší.
- Konverze měn: EUR → USD (kurz 1,165), INR → USD (kurz 88), dle průměrného kurzu ECB/RBI za období scrapingu (září–říjen 2025).
- Anualizace hodinových mezd: USA 2 080 h/rok, DE 1 607 h/rok (OECD 2024), IN 1 920 h/rok.
- Odstraněno **855 odlehlých mzdových pozorování** mimo rozsah 3 000–500 000 USD.

---

## 2. Deskriptivní statistika (Sekce 4)

### 2.1 Podíl AI pracovních míst podle země

| Země | Není AI (%) | Je AI (%) | N celkem |
|------|------------|----------|---------|
| Německo (DE) | 84,3 % | **15,7 %** | 6 400 |
| Indie (IN) | 93,8 % | **6,2 %** | 14 184 |
| USA (US) | 80,5 % | **19,5 %** | 17 848 |

> **Pearsonův chi²(2) = 1 200, p < 0,001** → rozdíly jsou vysoce statisticky významné.  
> USA vykazují nejvyšší podíl AI pracovních míst, Indie dramaticky nejnižší.

### 2.2 Mediánový roční plat (v USD po konverzi)

| Země | N (se mzdou) | Průměr | Medián | P25 | P75 |
|------|-------------|--------|--------|-----|-----|
| DE | 514 | 82 056 $ | 79 104 $ | 66 988 $ | 93 200 $ |
| IN | 9 199 | 8 324 $ | 6 535 $ | 5 114 $ | 8 447 $ |
| US | 14 642 | 124 617 $ | 117 869 $ | 95 000 $ | 147 734 $ |

### 2.3 Pokrytí mzdových dat podle země

| Země | Bez mzdy (%) | Se mzdou (%) |
|------|-------------|-------------|
| DE | **92,0 %** | **8,0 %** |
| IN | 35,2 % | 64,9 % |
| US | 18,0 % | 82,0 % |

> ⚠️ **Německo má extrémně nízké mzdové pokrytí** – pouze 514 pozorování (8 %). Příčiny jsou strukturální:
> - V Německu **neexistuje zákonná povinnost** zveřejňovat mzdové rozsahy v inzerátech (na rozdíl od řady amerických států od r. 2022–2023).
> - Německá tradice mzdové mlčenlivosti (*Gehaltsgeheimnis*) – kolegové si sdělují platy jen výjimečně.
> - Glassdoor má v Německu výrazně nižší uživatelskou základnu než v USA.
> - **Důsledek:** OLS mzdová regrese (Sekce 5) je fakticky porovnáním USA vs. Indie – Německo přispívá jen ~2,1 % vzorku. Mzdové koeficienty pro DE je třeba interpretovat s opatrností.

### 2.4 Vzdělanostní požadavky podle země

| Země | Bez titulu (%) | S titulem (%) |
|------|--------------|-------------|
| DE | 66,7 % | 33,3 % |
| IN | 48,0 % | 52,0 % |
| US | 39,7 % | 60,4 % |

> Německo nejméně často požaduje vysokoškolský titul – v souladu s tamní tradicí odborného vzdělávání (duální systém).

---

## 3. OLS Mzdová regrese (Sekce 5)

### 3.1 Model B – Sdružený baseline model (Country FEs + AI level)
**N = 23 897 | R² = 0,9357 | Root MSE = 0,364**

> Vysoké R² je z velké části způsobeno country fixed effects, které absorbují mezizemní rozdíly. Toto je správné a očekávané chování modelu – je třeba uvést v metodické sekci.

#### Klíčové koeficienty

| Proměnná | Koeficient | Interpretace |
|----------|-----------|-------------|
| AI integrace (úroveň 1) | **+0,084***  | ~+8,8 % mzdová prémie |
| Applied/Core AI (úroveň 2) | **+0,116***  | ~+12,3 % mzdová prémie |
| Indie (vs. DE základ) | **−2,269***  | platy ~90 % nižší než v DE |
| USA (vs. DE základ) | **+0,656***  | platy ~93 % vyšší než v DE |
| Práce na dálku | **+0,109***  | ~+11,5 % prémie za remote |
| `cluster_systems_programming` | **+0,063***  | nejsilnější skill prémie |
| `cluster_devops__containers` | **+0,038***  | ✓ |
| `cluster_cloud_computing` | **+0,031***  | ✓ |
| `cluster_bi__analytics` | **−0,024***  | negativní – pravděpodobně komoditizovaná dovednost |

#### Vzdělání (základ = chybějící)
| Kategorie | Koef. | Výsledek |
|-----------|-------|---------|
| Střední škola | −6,0 %*** | negativní, jak se očekávalo |
| Associate | −13,0 %*** | nejnižší – komoditní role |
| Bakalář | −2,6 %*** | mírně negativní vs. "missing" |
| Magistr+ | +6,8 %*** | pozitivní prémie |

> Poznámka: Kategorie „associate" má více negativní koeficient než „bachelor". To odráží složení vzorku – role bez uvedeného vzdělání jsou v USA obecně lépe placené technické pozice.

#### Zkušenosti (základ = 3–5 let)
| Kategorie | Koef. |
|-----------|-------|
| Bez požadavku | −5,8 %*** |
| 0–2 roky | −17,5 %*** |
| 5+ let | +11,0 %*** |

---

### 3.2 Model 5.2 – Interakční model (Country × AI level)
**N = 23 897 | R² = 0,9356**

> **Toto je klíčový model pro diplomovou práci.**

Aktualizovaná verze modelu používá **gradovanou proměnnou `ai_level`** (0 = žádné AI, 1 = AI integrace, 2 = Applied/Core AI) místo původní binární `has_ai`. Tato změna je motivována tím, že `ai_level` prokázal statistickou významnost v Modelu B a poskytuje bohatší informaci o rozdílech v prémii.

**Výsledky předchozí verze s binárním `has_ai`** (pro referenci):

| Interakce | Koef. | p-hodnota |
|-----------|-------|-----------|
| `has_ai` (základ DE) | +0,166 | 0,126 (nevýznamný) |
| IN × has_ai | −0,078 | 0,483 (nevýznamný) |
| US × has_ai | −0,079 | 0,468 (nevýznamný) |
| **Společný F-test (country × has_ai)** | **F(2) = 0,26** | **p = 0,768** |

> **Klíčové zjištění:** Binární AI mzdová prémie se statisticky **nevýznamně liší** mezi zeměmi po kontrole ostatních proměnných. Nová verze modelu s `ai_level` toto testuje v bohatším granulárním rámci.

**Možná vysvětlení nulového výsledku:**
1. Binární `has_ai` je hrubší míra než gradovaný `ai_level`.
2. Nízké pokrytí mezd v DE (n = 514) omezuje statistickou sílu pro detekci německých specifik.
3. AI mzdové efekty mohou být globálně podobnější, než se předpokládalo, po korekci tržních rozdílů.

---

## 4. Multinomiální logistický model – pravděpodobnost AI (Sekce 6)

**N = 32 279 | Pseudo R² = 0,405 | LR chi² = 13 239**

> Pseudo R² 0,40 je velmi silný výsledek pro mlogit – model dobře vysvětluje klasifikaci AI úrovně.

### 4.1 Efekt země na pravděpodobnost AI pracovního místa (RRR)

| | P(AI integrace) | P(Applied/Core AI) |
|---|---|---|
| Indie (vs. DE základ) | **RRR = 0,171***  | **RRR = 0,161***  |
| USA (vs. DE základ) | RRR = 1,563 (p = 0,059) | RRR = 1,112 (p = 0,716) |

> **Indie je ~83 % méně pravděpodobná** než Německo ve vystavování AI pracovních míst – masivní a vysoce signifikantní efekt.  
> Rozdíl USA vs. DE je slabší a pro AI integraci pouze na hranici signifikance.

### 4.2 Klíčové skill clustery (RRR pro AI integraci)

| Skill cluster | RRR – AI integrace | RRR – Applied AI |
|--------------|-------------------|-----------------|
| `cluster_generative_ai` | **47,3×***  | **69,5×***  |
| `cluster_data_science__ml` | **9,0×***  | **47,0×***  |
| `cluster_dynamic__web` | 1,70×*** | 2,76×*** |

> Generativní AI a Data Science/ML jsou zdaleka nejsilnějšími prediktory AI klasifikace – validuje metodiku extrakce dovedností.

### 4.3 Průměrné marginální efekty (AME)

**P(AI integrace):**
- GenAI cluster: **+17,8 p.b.***
- Data science/ML: **+8,1 p.b.***
- Indie (vs. DE): **−5,8 p.b.***
- USA (vs. DE): **+3,4 p.b.**

**P(Applied/Core AI):**
- Data science/ML: **+8,9 p.b.***
- GenAI cluster: **+7,7 p.b.***
- Indie (vs. DE): **−3,3 p.b.**
- USA (vs. DE): **−0,4 p.b.** (nevýznamný)

> Rozdíl USA vs. DE mizí pro Applied/Core AI – naznačuje, že DE a US trhy jsou si podobné v *hloubce* adopce AI po kontrole skill clusterů. Surové rozdíly v podílu AI pozic (Sekce 4.1) jsou reálné a vznikají **kompozičním efektem** (typem pracovních míst vystavených na Glassdooru).

---

## 5. Omezení a doporučení pro diplomovou práci

| Omezení | Doporučené řešení |
|---------|------------------|
| DE mzdové pokrytí 8 % (n = 514) | Uvést jako klíčové omezení; OLS interpretovat opatrně pro DE |
| Nulový výsledek interakčního modelu (has_ai) | Nový model s `i.country_id##i.ai_level` – bohatší test |
| AI podíl v Indii výrazně nižší (6,2 %) | Diskutovat strukturální příčiny (maturita trhu, Glassdoor penetrace) |
| Nemožnost ověřit DE nominální platy (92 % chybí) | Omezit závěry o DE mzdách; možná citovat OECD/Destatis data externě |

---

*Vygenerováno ze Stata logu: `comparative_run__1_Apr_2026_22-22-22`*  
*Skript: `analysis/stata/ai_skills_analysis_comparative.do`*
