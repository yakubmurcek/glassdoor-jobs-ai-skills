# Zpráva o kvalitě výstupů LLM

**Dataset:** Analýza pracovních inzerátů v USA  
**Analyzované záznamy:** 1 500 (z celkových 18 464 – zpracování zastaveno na ~8 %)  
**Datum:** 4. ledna 2026  
**Model:** GPT-5-mini (OpenAI Flex Processing Tier)

---

## Shrnutí

LLM pipeline úspěšně zpracoval 1 500 pracovních inzerátů z datasetu `us_relevant.csv`. Manuální a automatická validace ukazuje, že kvalita výstupů je **akceptovatelná pro účely analýzy**, s vysokou mírou kompletnosti napříč většinou sloupců a přiměřenou přesností klasifikace.

**Klíčová zjištění:**

- ✅ 100 % záznamů má AI tier klasifikaci
- ✅ 99,8 % záznamů má extrahované skills
- ✅ 83,7 % klasifikací s vysokou confidence (≥0,9)

---

## 1. Kompletnost dat

| Sloupec              | Účel                           | Kompletnost | Poznámky                                  |
| -------------------- | ------------------------------ | ----------- | ----------------------------------------- |
| `desc_tier_llm`      | AI klasifikace pozice          | 100 %       | none/ai_integration/applied_ai            |
| `ai_confidence`      | Confidence klasifikace         | 100 %       | Průměr: 0,89                              |
| `desc_ai_llm`        | Extrahované AI skills          | 30,4 %      | Vyplněno pouze pro AI pozice (očekávané)  |
| `desc_rationale_llm` | Vysvětlení nízké confidence    | 4,5 %       | Pouze pro nejisté klasifikace (očekávané) |
| `hardskills`         | Extrahované technické skills   | 99,8 %      | Výborné pokrytí                           |
| `softskills`         | Extrahované soft skills        | 94,9 %      | Dobré pokrytí                             |
| `skill_cluster`      | Skills seskupené dle kategorie | 99,8 %      | (např. "Programming: python, java")       |
| `edulevel_llm`       | Požadované vzdělání            | 52,2 %      | Mnoho pozic nespecifikuje požadavky       |
| `experience_min_llm` | Roky praxe                     | 83,5 %      | Dobře extrahováno                         |

---

## 2. Přesnost AI Tier klasifikace

### Distribuce

| Tier             | Počet | Procento | Popis                                  |
| ---------------- | ----- | -------- | -------------------------------------- |
| `none`           | 1 137 | 75,8 %   | Žádné AI/ML odpovědnosti               |
| `ai_integration` | 269   | 17,9 %   | Používá AI nástroje/API (Copilot, GPT) |
| `applied_ai`     | 94    | 6,3 %    | Vytváří/trénuje ML modely              |

### Validace (manuální kontrola 50 vzorků)

Provedl jsem manuální kontrolu stratifikovaného vzorku 50 řádků (20 none, 15 ai_integration, 15 applied_ai). Každý popis pozice jsem přečetl a vyhodnotil, zda LLM klasifikace odpovídá skutečnému obsahu.

**Výsledky manuální validace:**

| Tier             | Správně   | Chybně | Přesnost  |
| ---------------- | --------- | ------ | --------- |
| `none`           | 20/20     | 0      | **100 %** |
| `ai_integration` | 14/15     | 1      | **93 %**  |
| `applied_ai`     | 15/15     | 0      | **100 %** |
| **Celkem**       | **49/50** | **1**  | **98 %**  |

**Detailní hodnocení vzorků:**

#### Tier `none` (20/20 správně ✅)

Všechny pozice v tomto vzorku byly skutečně standardní vývojářské role bez AI komponent:

- ID 6 (Putnam Recruiting): Healthtech konzultant, full-stack bez AI
- ID 115 (Trellus): Same-day delivery platforma, standardní web development
- ID 155 (Printify): E-commerce platforma, React/Node.js bez ML
- ID 439 (Classavo): EdTech platforma, transformace učebnic – žádné AI
- ID 525 (Index Analytics): RESTful APIs a AWS, čistě data engineering
- ID 541 (Seek Now): Insurance claims software, full-stack bez AI
- ID 972-1458: Security pozice (IAM, cybersecurity) – správně označeny jako non-AI

#### Tier `ai_integration` (14/15 správně, 1 sporná)

Většina pozic správně identifikována jako "využívá AI nástroje":

- ID 91 (Firefly Lab): "data science foundation to train doctors" – ✅ správně ai_integration (používají data science nástroje, netrénují modely)
- ID 218 (Mattermost): "Claude Code, Cursor, GitHub Copilot, AI Tools" – ✅ jasně používá AI nástroje
- ID 445 (Navigate AI): "AI/ML, AR, CV, computer vision" – ✅ integrace AI do produktu
- ID 648 (XBOW): "AI-powered system... autonomously discovers vulnerabilities" – ✅ používá AI, nestaví ji
- ID 656 (Lyra Health): "AI, data science" v kontextu mental health – ✅ používá AI služby
- ID 723 (Noonlight): "github copilot, chatgpt, ai-assisted coding tools" – ✅ jasně nástroje

**Sporný případ (ID 293, Runpod):** Firma je "AI and machine learning cloud infrastructure" – popis říká, že staví infrastrukturu PRO AI, ale sami nepracují s ML modely. LLM dal ai_integration, což je hraničně správné (mohlo by být i none).

#### Tier `applied_ai` (15/15 správně ✅)

Všechny pozice skutečně zahrnují práci s ML modely:

- ID 29 (SimSpace): "GenAI, AgenticAI, model fine-tuning, machine learning" – ✅ jasně applied_ai
- ID 72 (ProFocus): "NLP, generative AI, TensorFlow, PyTorch" – ✅ trénuje modely
- ID 236 (Knowmadics): "machine learning models, ML model inputs/outputs" – ✅ práce s ML modely
- ID 273 (FinOps Blueprint): "LLMs, Azure OpenAI, LangChain, vector search, embeddings" – ✅ staví AI-native platformu
- ID 452 (SchoolAI): "machine learning, ML models, data science" – ✅ EdTech s vlastními ML modely
- ID 480 (Alex AI): "ml, dl, fine-tune, inference at scale, llms" – ✅ AI recruiter s vlastními modely
- ID 489 (Gather): "mcp, agents, llm provider apis, claude code" – ✅ staví AI produkt Grapevine
- ID 524 (Archer): "generative AI, GenAI, LLMs, RAG, vector databases, prompt engineering" – ✅
- ID 531 (Traba): "ML, AI agents, multi-agent AI workflows" – ✅ autonomous AI staffing
- ID 722 (Chalk): "machine learning, applied machine learning, data scientist" – ✅ ML platforma
- ID 737 (Orum): "speech recognition, machine learning, ai-driven" – ✅ vlastní AI pro sales
- ID 741 (Aleph): "ai-native, intelligent agents, machine learning" – ✅ FP&A s AI agenty
- ID 1396 (Oteemo): "ai/ml, llms, vulnerability prioritization" – ✅ AI pro cybersecurity

### Závěr validace

LLM klasifikace je **vysoce přesná (98 %)** na základě manuální kontroly 50 vzorků. Jediný sporný případ (ID 293) je na hranici mezi dvěma kategoriemi, což není chyba, ale odraz skutečné nejasnosti v popisu pozice

---

## 3. Kvalita extrakce skills

Pipeline extrahuje skills jak z původního sloupce `skills` (ze zdroje dat), tak z plného textu popisu pozice.

### Porovnání příkladů

**ID 100: Programmer III - Full Stack (JT4)**

Původní sloupec skills:

> Jira, Rust, Go, Waterfall, .NET Core, C#, MongoDB, DoD experience, SQL, Docker...

Extrahované `hardskills`:

> agile, angular, asp.net, c#, c++, ci/cd, confluence, continuous delivery, continuous integration, devsecops, django, docker, dotnet, frontend development, full stack, gitlab, golang, graphql, javascript, jira, kubernetes, mongodb, nosql, postgresql, python, react, restful api, rust, sql, test management, ui, vue, waterfall

**Pozorování:** LLM extrahoval **výrazně více skills** z popisu pozice než bylo v původním sloupci – toto je přidaná hodnota.

### Skills jsou správně kategorizovány v `skill_cluster`:

Příklad výstupu:

```
Programming: c#, python, javascript, react, typescript
Data & Cloud: aws, docker, kubernetes
Integration: restful api, graphql, microservices
Security: cybersecurity, firewall, networking
```

---

## 4. Extrakce vzdělání a praxe

### Distribuce úrovně vzdělání

| Úroveň          | Počet | %      |
| --------------- | ----- | ------ |
| Bachelor's      | 718   | 47,9 % |
| Nespecifikováno | 717   | 47,8 % |
| High School     | 33    | 2,2 %  |
| Associate       | 23    | 1,5 %  |
| Master's        | 8     | 0,5 %  |
| PhD             | 1     | 0,1 %  |

### Požadavky na praxi

- Záznamy se specifikovanou praxí: 1 252/1 500 (83,5 %)
- Průměr: 4,3 roků
- Medián: 4,0 roků
- Rozsah: 0 až 20 let

**Manuální validace (vzorek):**

| ID   | Název pozice                    | LLM říká | Skutečně v popisu pozice   |
| ---- | ------------------------------- | -------- | -------------------------- |
| 1386 | Sr. Software Engineer, Security | 6 let    | "6 years of experience" ✅ |
| 1233 | Sr. Cloud Security Engineer     | 8 let    | "8+ years" v popisu ✅     |
| 25   | Full Stack Software Engineer    | 5 let    | "5+ years" zmíněno ✅      |

---

## 5. Kalibrace confidence

LLM poskytuje confidence score pro své AI tier klasifikace:

| Rozsah confidence | Počet | %      |
| ----------------- | ----- | ------ |
| ≥ 0,9 (Vysoká)    | 1 256 | 83,7 % |
| 0,8 – 0,89        | 177   | 11,8 % |
| < 0,8 (Nízká)     | 67    | 4,5 %  |

**Případy s nízkou confidence obsahují užitečná vysvětlení:**

> ID 4 (Nike): _"The posting lists 'Proficiency in AIML' but gives no concrete ML duties (training/fine-tuning/MLOps). It's ambiguous whether hands-on model work is required, so I assign ai_integration with moderate confidence."_

> ID 1233 (Finch): _"Company name includes 'AI' but the listed duties focus on cloud security/automation and do not mention working with AI/ML models or services."_

Toto je přesně ten typ nuancovaného uvažování, který chceme.

## 6. Statistická analýza

### 6.1 Přesnost klasifikace – interval spolehlivosti

Na základě manuální validace 50 vzorků:

| Metrika             | Hodnota          |
| ------------------- | ---------------- |
| Pozorovaná přesnost | 98,0 % (49/50)   |
| 95% Wilson Score CI | [89,5 %, 99,6 %] |

**Binomický test vs náhodná klasifikace (H₀: accuracy = 33,3 %):**

- p-value = 1,34 × 10⁻²²
- **Závěr:** Zamítáme H₀ při α = 0,05. Klasifikace je statisticky významně lepší než náhodná.

### 6.2 Shoda mezi LLM a deterministickým klasifikátorem

Porovnání LLM klasifikace s keyword-based deterministickým detektorem na celém datasetu (n = 1 500):

**Confusion Matrix:**

|                 | LLM: none | LLM: AI |
| --------------- | --------- | ------- |
| **Det: non-AI** | 1 114     | 101     |
| **Det: AI**     | 23        | 262     |

| Metrika            | Hodnota | Interpretace          |
| ------------------ | ------- | --------------------- |
| Celková shoda      | 91,7 %  | -                     |
| Cohen's κ (Kappa)  | 0,757   | Substantial agreement |
| Cramér's V         | 0,764   | Large effect size     |
| Phi koeficient (φ) | 0,766   | Strong association    |

**McNemar's test (jsou klasifikátory systematicky odlišné?):**

- χ² = 47,81, p-value < 0,0001
- LLM našel AI tam, kde Det ne: 101 případů
- Det našel AI tam, kde LLM ne: 23 případů
- **Závěr:** LLM je systematicky citlivější na AI-related pozice než keyword-based detektor.

### 6.3 Distribuce confidence score

| Metrika                 | Hodnota        |
| ----------------------- | -------------- |
| Průměr (μ)              | 0,888          |
| Směrodatná odchylka (σ) | 0,049          |
| Medián                  | 0,900          |
| IQR                     | [0,900, 0,900] |

**Shapiro-Wilk test normality (n = 500):**

- W = 0,561, p-value < 0,0001
- **Závěr:** Distribuce confidence není normální (koncentrovaná kolem 0,9).

### 6.4 Extrakce let praxe

| Metrika             | Hodnota              |
| ------------------- | -------------------- |
| Platné extrakce     | 1 252/1 500 (83,5 %) |
| Průměr              | 4,28 let             |
| Směrodatná odchylka | 2,34 let             |
| 95% CI pro průměr   | [4,15, 4,41] let     |

### 6.5 Distribuce AI tier – Chi-square test

**Pozorované frekvence:**

- none: 1 137 (75,8 %)
- ai_integration: 269 (17,9 %)
- applied_ai: 94 (6,3 %)

**Chi-square goodness of fit test vs uniformní distribuce:**

- χ² = 1 247,93, p-value ≈ 0
- **Závěr:** Distribuce je signifikantně neuniformní (p < 0,001), což odpovídá očekávání – většina pozic není AI-related.

---

## 7. Závěr

**Kvalita výstupu LLM je dostatečná pro analýzu v rámci diplomové práce.** Klíčové body:

1. **Integrita dat ověřena** – Všech 1 500 řádků odpovídá zdroji podle ID, názvu pozice a firmy
2. **Vysoká kompletnost** – Klíčové sloupce mají 95–100% míru vyplnění
3. **Přesnost klasifikace je rozumná** – LLM dělá nuancovaná rozhodnutí a poskytuje vysvětlení pro nejisté případy
4. **Extrakce skills přidává hodnotu** – Extrahuje více skills z textu popisu než původní zdroj dat poskytoval
5. **Kalibrace confidence funguje** – Případy s nízkou confidence jsou skutečně nejednoznačné

**Doporučení:** Pokračovat v analýze. Manuální kontrola 50 vzorků ukazuje **98% přesnost klasifikace**.

---

## Příloha: Vzorek validovaných řádků

### Applied AI pozice (skutečná AI práce)

| ID  | Firma       | Název pozice                         | Detekované AI skills                                                  |
| --- | ----------- | ------------------------------------ | --------------------------------------------------------------------- |
| 29  | SimSpace    | Senior Software Engineer - Fullstack | GenAI, AgenticAI, ChatGPT, Vertex AI, Hugging Face, model fine-tuning |
| 32  | ManTech     | Full Stack Developer                 | machine learning, model deployment, PySpark, Docker, Kubernetes       |
| 400 | Fusion Risk | Full Stack Engineer                  | LLMs, RAG, prompt engineering, Cursor, ChatGPT Pro, Azure OpenAI      |
| 85  | Zoom        | Software Engineer – Java             | Deep learning, model training, TensorFlow                             |

### AI Integration pozice (používání AI nástrojů)

| ID  | Firma         | Název pozice                         | Detekované AI skills                        |
| --- | ------------- | ------------------------------------ | ------------------------------------------- |
| 2   | PrePass       | Software Engineer                    | GitHub Copilot, Cursor, AI pair programmers |
| 8   | Diffit        | Fullstack Engineer                   | AI-powered platform, Python, Flask          |
| 27  | Waltz Health  | Full Stack Developer                 | AI-driven, Azure Cognitive Services, OpenAI |
| 600 | Monarch Money | Software Engineer, Internal Tools/AI | openai, langchain                           |

### Non-AI pozice (správně klasifikované)

| ID  | Firma     | Název pozice                 | Tech Stack                    |
| --- | --------- | ---------------------------- | ----------------------------- |
| 1   | Treinetic | Full Stack Software Engineer | Angular, Spring Boot, GraphQL |
| 100 | JT4       | Programmer III               | C#, .NET, Docker, Kubernetes  |
| 300 | Agilant   | Sr. Full Stack Developer     | React, PHP, AWS, Laravel      |
