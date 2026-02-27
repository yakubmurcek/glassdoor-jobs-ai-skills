# Hodnotící zpráva pipeline

**Vygenerováno**: 2025-12-26T09:41:33.476532
**Základní verze (Baseline)**: `us_relevant_30_ai.csv`
**Kandidátní verze (Candidate)**: `us_relevant_30_ai_v1.csv`

## Shrnutí

| Metrika                          | Hodnota            |
| -------------------------------- | ------------------ |
| Celkem pozic                     | 30                 |
| Míra shody                       | **90,0 %** (27/30) |
| Změna jistoty (Confidence)       | -0,005             |
| Změna shody (Agreement)          | +3,3 %             |
| Průměr tvrdých dovedností/pozice | 4,7                |
| Průměr měkkých dovedností/pozice | 2,1                |

## Distribuce úrovní (Tier)

| Úroveň           | Baseline | Kandidát | Změna |
| ---------------- | -------- | -------- | ----- |
| `none`           | 24       | 25       | +1    |
| `ai_integration` | 2        | 3        | +1    |
| `applied_ai`     | 3        | 1        | -2    |
| `core_ai`        | 1        | 1        | 0     |

## Změny v klasifikaci (3)

### 1. Pozice 4: Softwarový inženýr II - Inovace

**Změna**: `ai_integration` → `none`

|               | Baseline | Kandidát |
| ------------- | -------- | -------- |
| Jistota       | 0,80     | 0,90     |
| AI dovednosti | AIML     | nan      |

**Úryvek popisu pozice**:

> KOHO HLEDÁME
> Hledáme mid-level inženýra, který dokáže nezávisle dodávat funkce, mentorovat juniorní inženýry a přispívat k architektonickým rozhodnutím. Jste silný komunikátor, spolupracovník napříč funkcemi a někdo, komu se daří v dynamickém prostředí.
> NA ČEM BUDETE PRACOVAT
> Navrhovat a implementovat škálovatelné služby a uživatelská rozhraní
> Vést vývoj funkcí pomocí React...

**Odůvodnění (Baseline)**:

> Pozice zmiňuje znalost AIML, ale nenaznačuje přímé zapojení do vývoje AI modelů.

**Odůvodnění (Kandidát)**:

> Pozice zahrnuje full-stack vývoj bez jakýchkoliv úloh spojených s AI.

---

### 2. Pozice 8: Fullstack inženýr

**Změna**: `applied_ai` → `ai_integration`

|               | Baseline | Kandidát |
| ------------- | -------- | -------- |
| Jistota       | 0,75     | 0,80     |
| AI dovednosti | AI       | nan      |

**Úryvek popisu pozice**:

> O společnosti Diffit
> Diffit je platforma pro výukové materiály poháněná umělou inteligencí, navržená tak, aby pomohla učitelům dělat jejich nejlepší práci udržitelněji. Učitelé používají Diffit k získání „přesně těch pravých“ aktivit pro třídu, což šetří čas a pomáhá všem studentům získat přístup k obsahu na úrovni jejich ročníku. Připojte se k rostoucí, oborově zaměřené AI edtech společnosti fungující plně remote a budujte kvalitní, bezpečné a dostupné vzdělávací zdroje pro učitele...

**Odůvodnění (Baseline)**:

> Role zahrnuje budování funkcí pro platformu poháněnou AI, což naznačuje určitou úroveň práce s aplikovanou AI.

**Odůvodnění (Kandidát)**:

> Role je ve společnosti zaměřené na AI, ale nezahrnuje vývoj AI modelů; zaměřuje se na full-stack vývoj.

---

### 3. Pozice 25: Full Stack softwarový inženýr

**Změna**: `applied_ai` → `ai_integration`

|               | Baseline | Kandidát                  |
| ------------- | -------- | ------------------------- |
| Jistota       | 0,85     | 0,75                      |
| AI dovednosti | AI, NLP  | AI-powered marketing, NLP |

**Úryvek popisu pozice**:

> O nás
> Tildei je marketingová platforma poháněná AI, která vytváří inteligentní brand agenty pro obchodní a marketingové konverzace. Budujeme komplexní, vlastní Znalostní grafy značek z produktových katalogů, marketingových materiálů, FAQ a pravidel značky. Následně nasazujeme agenty napříč sociálními a digitálními kanály pro zapojení zákazníků 24/7 v jakémkoli jazyce. Naši agenti přinášejí marketingové a obchodní výsledky...

**Odůvodnění (Baseline)**:

> Role zahrnuje budování marketingových agentů poháněných AI, což naznačuje praktickou aplikaci AI technologií.

**Odůvodnění (Kandidát)**:

> Role zahrnuje budování funkcí pro platformu poháněnou AI, ale nezahrnuje přímý vývoj AI modelů.

---
