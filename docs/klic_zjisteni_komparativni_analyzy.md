# Klíčová zjištění: Komparativní analýza AI dovedností v IT pracovních inzerátech
**Soubor:** `ai_skills_analysis_comparative.do`  
**Log:** `comparative_run__1_Apr_2026_22-42-42/ai_skills_comparative_analysis.log`  
**Analyzovaný pool:** USA (18 464) + DE Německo (9 254) + IN Indie (17 114) = **44 832 surových pozorování** → **38 432 po filtraci**  
**Datum analýzy:** 1. dubna 2026

---

## 1. Příprava dat

- Všechny tři datasety načteny a spojeny bez chyb.
- Filtrace: odstraněno **6 384 pozorování** s důvěrou modelu `desc_conf_llm < 0.7` a **12 pozorování** z roku 2023 a starší.
- Konverze měn: EUR → USD (kurz 1,165), INR → USD (kurz 88), dle průměrného kurzu ECB/RBI za období scrapingu (září–říjen 2025).
- Anualizace hodinových mezd: USA 2 080 h/rok, DE 1 607 h/rok (OECD 2024), IN 1 920 h/rok.
- Odstraněno **855 odlehlých mzdových pozorování** mimo rozsah 3 000–500 000 USD.

**Pokrytí kategorií zaměstnavatele (audit):**

| Proměnná | Platná pozorování | Chybějící | % chybějících |
|----------|------------------|-----------|--------------|
| `type_cat` (typ firmy) | 32 288 | 6 144 | 16,0 % |
| `size_cat` (velikost firmy) | 32 306 | 6 126 | 15,9 % |

Jde prakticky o tytéž řádky – firmy bez uvedeného typu i velikosti. Tato pozorování vypadávají z OLS mzdové regrese a je třeba je uvést jako omezení.

---

## 2. Deskriptivní statistika

### 2.1 Podíl AI pracovních míst podle země

| Země | Není AI | Je AI | AI % |
|------|--------|-------|------|
| Německo (DE) | 5 393 | 1 007 | **15,7 %** |
| Indie (IN) | 13 301 | 883 | **6,2 %** |
| USA (US) | 14 371 | 3 477 | **19,5 %** |
| **Celkem** | 33 065 | 5 367 | **14,0 %** |

> **Pearsonův chi²(2) = 1 200, p < 0,001** → rozdíly jsou vysoce statisticky významné.  
> USA mají nejvyšší podíl AI pracovních míst; Indie dramaticky nejnižší.

### 2.2 Mediánový roční plat (v USD po konverzi)

| Země | N (se mzdou) | Průměr | Medián | P25 | P75 |
|------|-------------|--------|--------|-----|-----|
| DE | 514 | 82 056 $ | 79 104 $ | 66 988 $ | 93 200 $ |
| IN | 9 199 | 8 324 $ | 6 535 $ | 5 114 $ | 8 447 $ |
| US | 14 642 | 124 617 $ | 117 869 $ | 95 000 $ | 147 734 $ |

### 2.3 Pokrytí mzdových dat podle země

| Země | Bez mzdy | Se mzdou |
|------|---------|---------|
| DE | 92,0 % | **8,0 %** |
| IN | 35,2 % | 64,9 % |
| US | 18,0 % | 82,0 % |

> ⚠️ **Německo má extrémně nízké mzdové pokrytí** – pouze 514 pozorování. Příčiny jsou strukturální:
> - Žádná zákonná povinnost zveřejňovat mzdy v inzerátech (na rozdíl od řady amerických států).
> - Německá tradice mzdové mlčenlivosti (*Gehaltsgeheimnis*).
> - Nízká penetrace Glassdooru v Německu.
>
> **Důsledek:** OLS mzdová regrese je fakticky porovnáním USA vs. Indie – DE přispívá jen ~2,1 % vzorku. Koeficienty pro DE interpretovat s opatrností.

### 2.4 Vzdělanostní požadavky podle země

| Země | Bez titulu | S titulem |
|------|-----------|---------|
| DE | 66,7 % | 33,3 % |
| IN | 48,0 % | 52,0 % |
| US | 39,7 % | 60,4 % |

> Německo nejméně často požaduje VŠ titul – v souladu s tradicí duálního vzdělávání.

---

## 3. OLS Mzdová regrese

### 3.1 Model B – Sdružený baseline model (Country FEs + AI level dummies)
**N = 23 897 | R² = 0,9357 | Root MSE = 0,364**

#### Klíčové koeficienty

| Proměnná | Koeficient | Interpretace |
|----------|-----------|-------------|
| AI integrace (`ai_level = 1`) | **+0,084\*\*\*** | ~+8,8 % mzdová prémie |
| Applied/Core AI (`ai_level = 2`) | **+0,116\*\*\*** | ~+12,3 % mzdová prémie |
| Indie (vs. DE základ) | **−2,269\*\*\*** | platy ~90 % nižší než v DE |
| USA (vs. DE základ) | **+0,656\*\*\*** | platy ~93 % vyšší než v DE |
| Práce na dálku | **+0,109\*\*\*** | ~+11,5 % prémie za remote |
| `cluster_systems_programming` | **+0,063\*\*\*** | nejsilnější skill prémie |
| `cluster_devops__containers` | **+0,038\*\*\*** | |
| `cluster_cloud_computing` | **+0,031\*\*\*** | |
| `cluster_bi__analytics` | **−0,024\*\*\*** | negativní – komoditizovaná dovednost |

#### Vzdělání (základ = chybějící)
| Kategorie | Koeficient |
|-----------|-----------|
| Střední škola | −6,0 %\*\*\* |
| Associate | −13,0 %\*\*\* |
| Bakalář | −2,6 %\*\*\* |
| Magistr+ | +6,8 %\*\*\* |

#### Zkušenosti (základ = 3–5 let)
| Kategorie | Koeficient |
|-----------|-----------|
| Bez požadavku | −5,8 %\*\*\* |
| 0–2 roky | −17,5 %\*\*\* |
| 5+ let | +11,0 %\*\*\* |

---

### 3.2 Model 5.2 – Interakční model (Country × AI level) — KLÍČOVÝ MODEL
**N = 23 897 | R² = 0,9357**

Model testuje, zda se **gradovaná AI mzdová prémie** statisticky liší podle země. Používá `i.country_id##i.ai_level` (0 = žádné AI, 1 = AI integrace, 2 = Applied/Core AI), základová kategorie = Německo bez AI.

| Koeficient | Hodnota | p-hodnota | Interpretace |
|-----------|---------|-----------|-------------|
| `ai_level = 1` (DE základ) | +0,180 | 0,195 | prémie za AI integraci v DE |
| `ai_level = 2` (DE základ) | +0,151 | 0,291 | prémie za Applied AI v DE |
| IN × AI integrace | −0,103 | 0,476 | rozdíl Indie vs. DE pro AI integraci |
| IN × Applied AI | −0,030 | 0,840 | rozdíl Indie vs. DE pro Applied AI |
| US × AI integrace | −0,095 | 0,494 | rozdíl USA vs. DE pro AI integraci |
| US × Applied AI | −0,037 | 0,797 | rozdíl USA vs. DE pro Applied AI |

**Společný F-test (všechny 4 interakce):**

> **F(4, 23 833) = 0,15 — p = 0,965**

### 🎯 Hlavní závěr

**Mezi zeměmi neexistuje statisticky významný rozdíl v AI mzdové prémii** — ani pro binární, ani pro gradovanou specifikaci. Výsledek je robustní:

| Model | Specifikace | F-test | p-hodnota |
|-------|-----------|--------|-----------|
| Starší (srovnání) | `country × has_ai` (2 interakce) | F(2) = 0,26 | 0,768 |
| **Aktuální** | **`country × ai_level` (4 interakce)** | **F(4) = 0,15** | **0,965** |

**Interpretace pro diplomku:** Rozdíly mezi zeměmi jsou ve *frekvenci* AI pracovních míst (viz mlogit), nikoliv v jejich *mzdové prémii*. Globální trh AI dovedností je co do odměňování relativně homogenní po kontrole tržních rozdílů.

---

## 4. Multinomiální logistický model (Sekce 6)

**N = 32 279 | Pseudo R² = 0,405 | LR chi² = 13 239**

### 4.1 Efekt země na pravděpodobnost AI pracovního místa (RRR)

| | P(AI integrace) | P(Applied/Core AI) |
|---|---|---|
| Indie (vs. DE základ) | **RRR = 0,171\*\*\*** | **RRR = 0,161\*\*\*** |
| USA (vs. DE základ) | RRR = 1,563 (p = 0,059) | RRR = 1,112 (p = 0,716) |

> **Indie je ~83 % méně pravděpodobná** než Německo ve vystavování AI pozic – masivní a vysoce signifikantní.

### 4.2 Klíčové skill clustery (RRR)

| Skill cluster | RRR – AI integrace | RRR – Applied AI |
|--------------|-------------------|-----------------|
| `cluster_generative_ai` | **47,3×\*\*\*** | **69,5×\*\*\*** |
| `cluster_data_science__ml` | **9,0×\*\*\*** | **46,9×\*\*\*** |
| `cluster_dynamic__web` | 1,70×\*\*\* | 2,76×\*\*\* |

> GenAI a Data Science/ML jsou zdaleka nejsilnějšími prediktory AI klasifikace – validuje metodiku extrakce.

### 4.3 Průměrné marginální efekty (AME)

**P(AI integrace):**
- GenAI cluster: **+17,8 p.b.\*\*\***
- Data science/ML: **+8,1 p.b.\*\*\***
- Indie (vs. DE): **−5,8 p.b.\*\*\***
- USA (vs. DE): **+3,4 p.b.\***

**P(Applied/Core AI):**
- Data science/ML: **+8,9 p.b.\*\*\***
- GenAI cluster: **+7,7 p.b.\*\*\***
- Indie (vs. DE): **−3,3 p.b.\*\***
- USA (vs. DE): −0,4 p.b. (nevýznamný)

> Rozdíl USA vs. DE mizí pro Applied/Core AI – DE a US trhy jsou si podobné v *hloubce* AI adopce po kontrole skill clusterů. Surové rozdíly v podílu AI pozic jsou reálné a vznikají **kompozičním efektem**.

---

## 5. Omezení a doporučení pro diplomovou práci

| Omezení | Doporučené řešení |
|---------|------------------|
| DE mzdové pokrytí 8 % (n = 514) | Uvést jako klíčové omezení; OLS pro DE interpretovat opatrně |
| ~16 % obs bez `type_cat`/`size_cat` | Uvést v metodice; pozorování vypadají z OLS (N = 23 897 vs. 38 432) |
| Nulový interakční výsledek (robustní) | Prezentovat jako věcné zjištění – homogenita AI prémie napříč trhy |
| AI podíl v Indii výrazně nižší (6,2 %) | Diskutovat strukturální příčiny (maturita trhu, Glassdoor penetrace) |

---

*Vygenerováno ze Stata logu: `comparative_run__1_Apr_2026_22-42-42`*  
*Skript: `analysis/stata/ai_skills_analysis_comparative.do`*
