# Hodnotící zpráva pipeline

**Vygenerováno**: 2026-01-04T13:12:47.130790
**Základní verze (Baseline)**: `us_relevant_50_gpt5nano_jan4.csv`
**Kandidátní verze (Candidate)**: `us_relevant_50_gpt5mini_jan4.csv`

## Shrnutí

| Metrika                          | Hodnota            |
| -------------------------------- | ------------------ |
| Celkem pozic                     | 50                 |
| Míra shody                       | **94,0 %** (47/50) |
| Změna jistoty (Confidence)       | -0,003             |
| Změna shody (Agreement)          | +0,0 %             |
| Průměr tvrdých dovedností/pozice | 20,8               |
| Průměr měkkých dovedností/pozice | 5,9                |

## Distribuce úrovní (Tier)

| Úroveň           | Baseline | Kandidát | Změna |
| ---------------- | -------- | -------- | ----- |
| `none`           | 34       | 34       | 0     |
| `ai_integration` | 12       | 13       | +1    |
| `applied_ai`     | 4        | 3        | -1    |
| `core_ai`        | 0        | 0        | 0     |

## Změny v klasifikaci (3)

### 1. Pozice 29: Senior softwarový inženýr - Fullstack

**Změna**: `ai_integration` → `applied_ai`

|               | Baseline                                                                                                                      | Kandidát                                                                                                                                                                                             |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Jistota       | 0,95                                                                                                                          | 0,90                                                                                                                                                                                                 |
| AI dovednosti | GenAI, AgenticAI, ChatGPT, Vertex AI, AWS AI, Claude, Perplexity, CrewAI, Anthropic, Jasper, Cohere, Hugging Face, OpenAI, AI | GenAI, AgenticAI, GPT-like modely, ChatGPT, Vertex AI, AWS AI, Claude, Perplexity, CrewAI, Anthropic, Jasper, Cohere, Hugging Face, machine learning, trénování modelů, fine-tuning, nasazení modelů |

**Úryvek popisu pozice**:

> :
> Kdo je SimSpace
> Společnost SimSpace byla spuštěna v roce 2015 s jediným účelem –
> . Organizace po celém světě, na které se denně spoléháme, že udrží naše blízké v bezpečí. Naše zdravotnická zařízení, školy, finanční instituce, tranzitní centra, obchody s potravinami a pracoviště, abychom jmenovali alespoň některé. K dosažení globální odolnosti poskytujeme elitní platformu pro kybernetické testování, abychom zajistili nenapadnutelné kybernetické prostředí...

**Odůvodnění (Baseline)**:

> nan

**Odůvodnění (Kandidát)**:

> nan

---

### 2. Pozice 33: (Na volné noze) AI Automation inženýr fullstack (Code & No-Code)

**Změna**: `applied_ai` → `ai_integration`

|               | Baseline   | Kandidát                                                 |
| ------------- | ---------- | -------------------------------------------------------- |
| Jistota       | 0,82       | 0,90                                                     |
| AI dovednosti | HeyGen, AI | AI nástroje, python, APIs, heygen, no-code, automatizace |

**Úryvek popisu pozice**:

> Shrnutí
> Hledáme talentovaného technologického specialistu se znalostí jak programování, tak no-code platforem. Ideální kandidát by měl mít hluboké porozumění AI technologiím a schopnost je efektivně implementovat do obchodních procesů. Ochota učit se a přizpůsobovat se novým nástrojům a trendům je nezbytná. Tato role je perfektní pro někoho, kdo chce zlepšovat své dovednosti v dynamickém...

**Odůvodnění (Baseline)**:

> nan

**Odůvodnění (Kandidát)**:

> nan

---

### 3. Pozice 37: Full-Stack Inženýr

**Změna**: `applied_ai` → `ai_integration`

|               | Baseline                                                    | Kandidát                                                                                |
| ------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Jistota       | 0,82                                                        | 0,70                                                                                    |
| AI dovednosti | GenAI, LLM, machine learning, zpracování přirozeného jazyka | generativní AI, GenAI, machine learning, zpracování přirozeného jazyka, ML, NLP, Python |

**Úryvek popisu pozice**:

> Přehled:
> Odpovědnosti:
> Kvalifikace:
> Připojte se k naší špičkové platformě generativní AI (GenAI), LIGER™, vytvořené technologickým studiem LMI Forge. LIGER™ využívá sílu pokročilých technologií, datové analytiky a nejnovějších poznatků z machine learningu a zpracování přirozeného jazyka k poskytování bezpečných, privátních a důvěryhodných řešení GenAI pro vládu.
> LMI je nový druh poskytovatele digitálních řešení...

**Odůvodnění (Baseline)**:

> nan

**Odůvodnění (Kandidát)**:

> Společnost/produkt je výslovně platforma GenAI, ale úkoly role se zaměřují na backendový vývoj v Pythonu a zmiňují pouze integraci ML nástrojů (nikoli trénování modelů nebo vlastnictví ML pipeline), takže se zdá, že jde o integrační práci, nikoliv o roli aplikovaného ML inženýra.

---
