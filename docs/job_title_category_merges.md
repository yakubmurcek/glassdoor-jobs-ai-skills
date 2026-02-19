# Normalizace job titles: Sloučení malých kategorií

> **Účel**: Dokumentuje, jak bylo 138 normalizovaných kategorií job titles sloučeno
> do 50 širších kategorií, aby každá měla ≥ 50 pozorování pro statistickou analýzu.
> Sloučení je aplikováno přímo v `job_title_normalizer.py`.

## Souhrn

| Metrika            | Před        | Po                       |
| ------------------ | ----------- | ------------------------ |
| Celkem kategorií   | 138         | **50**                   |
| Kategorií pod 50   | 89          | **0**                    |
| Nejmenší kategorie | CTO (1)     | iOS Developer (50)       |
| Other              | 308 (1.67%) | 308 (1.67%) — beze změny |

## Proč slučovat?

Původní normalizer produkoval 138 kategorií, z toho **89 mělo méně než 50 pozorování**.
Mnoho z nich byly blízce příbuzné varianty (např. `Python Developer`, `PHP Developer`,
`Go Developer` jsou všechno jazykově specifické varianty software developera).
Pro regresní analýzu ve Statě jsou kategorie s velmi malým N problematické — produkují
nespolehlivé odhady a znepřehledňují výstup.

**Přístup**: Malé kategorie sloučit do nejbližší sémanticky příbuzné větší kategorie.
Žádná informace se neztrácí — původní raw sloupec `job_title` je stále k dispozici
pro detailní analýzu.

## Tabulky sloučení

### Varianty Software Developer

Jazykově a doménově specifičtí vývojáři sloučeni do **Software Developer**.

| Malá kategorie       | Počet | Sloučeno do        |
| -------------------- | ----- | ------------------ |
| Python Developer     | 35    | Software Developer |
| PHP Developer        | 15    | Software Developer |
| C++ Developer        | 13    | Software Developer |
| Go Developer         | 8     | Software Developer |
| Rust Developer       | 3     | Software Developer |
| Ruby Developer       | 2     | Software Developer |
| Scala Developer      | 1     | Software Developer |
| Blockchain Developer | 2     | Software Developer |
| API Developer        | 32    | Software Developer |
| CRM Developer        | 26    | Software Developer |
| BI Developer         | 18    | Software Developer |
| ETL Developer        | 6     | Software Developer |

---

### Varianty Engineer

Specializované inženýrské role bez dostatečně velké samostatné kategorie, sloučeny do **Engineer**.

| Malá kategorie            | Počet | Sloučeno do |
| ------------------------- | ----- | ----------- |
| Application Engineer      | 39    | Engineer    |
| Integration Engineer      | 46    | Engineer    |
| Design Engineer           | 32    | Engineer    |
| Solutions Engineer        | 49    | Engineer    |
| Product Engineer          | 21    | Engineer    |
| Performance Engineer      | 7     | Engineer    |
| Forward Deployed Engineer | 11    | Engineer    |
| Founding Engineer         | 9     | Engineer    |
| Graphics Engineer         | 9     | Engineer    |
| Hardware Engineer         | 4     | Engineer    |
| Computer Vision Engineer  | 1     | Engineer    |

---

### Security → Security Engineer / Security Analyst

Security role rozděleny do dvou skupin: inženýrské → **Security Engineer**,
analytické/operační → **Security Analyst**.

| Malá kategorie                | Počet | Sloučeno do       |
| ----------------------------- | ----- | ----------------- |
| Cloud Security Engineer       | 43    | Security Engineer |
| Application Security Engineer | 4     | Security Engineer |
| Information Security Engineer | 2     | Security Engineer |
| Security Architect            | 4     | Security Engineer |
| SOC Analyst                   | 45    | Security Analyst  |
| Security Operations Analyst   | 28    | Security Analyst  |
| Vulnerability Analyst         | 14    | Security Analyst  |
| Threat Intelligence Analyst   | 18    | Security Analyst  |
| Information Assurance Analyst | 9     | Security Analyst  |
| GRC Analyst                   | 9     | Security Analyst  |
| Security Compliance Analyst   | 4     | Security Analyst  |
| IAM Analyst                   | 5     | Security Analyst  |
| RMF Analyst                   | 1     | Security Analyst  |
| Penetration Tester            | 13    | Security Analyst  |

---

### QA & Testing

| Malá kategorie | Počet | Sloučeno do |
| -------------- | ----- | ----------- |
| QA Analyst     | 49    | QA Engineer |

---

### Architecture → Solutions Architect

Všechny architektonické varianty sloučeny do **Solutions Architect** (největší architektonická kategorie).

| Malá kategorie        | Počet | Sloučeno do         |
| --------------------- | ----- | ------------------- |
| Cloud Architect       | 35    | Solutions Architect |
| Enterprise Architect  | 16    | Solutions Architect |
| Systems Architect     | 13    | Solutions Architect |
| Application Architect | 12    | Solutions Architect |
| Technical Architect   | 12    | Solutions Architect |
| Data Architect        | 17    | Solutions Architect |
| Information Architect | 3     | Solutions Architect |
| Network Architect     | 3     | Solutions Architect |
| Mobile Architect      | 3     | Solutions Architect |
| AI Architect          | 25    | Solutions Architect |
| AEM Architect         | 4     | Solutions Architect |
| DevOps Architect      | 5     | Solutions Architect |

---

### UI/UX & Design → UI/UX Designer

Všechny designové role konsolidovány do **UI/UX Designer**.

| Malá kategorie   | Počet | Sloučeno do    |
| ---------------- | ----- | -------------- |
| UI Developer     | 48    | UI/UX Designer |
| UI/UX Developer  | 38    | UI/UX Designer |
| UX Designer      | 26    | UI/UX Designer |
| UI Designer      | 17    | UI/UX Designer |
| Product Designer | 14    | UI/UX Designer |
| Web Designer     | 35    | UI/UX Designer |

---

### Game & Creative → Game Development (nová kategorie)

Všechny herní a kreativní role sloučeny do nové kategorie **Game Development**.

| Malá kategorie     | Počet | Sloučeno do      |
| ------------------ | ----- | ---------------- |
| Game Designer      | 42    | Game Development |
| Game Developer     | 23    | Game Development |
| 3D Artist          | 30    | Game Development |
| Game Artist        | 3     | Game Development |
| Game Producer      | 10    | Game Development |
| Technical Artist   | 16    | Game Development |
| Animator           | 10    | Game Development |
| Level Designer     | 7     | Game Development |
| Technical Producer | 15    | Game Development |

---

### Data & AI

| Malá kategorie     | Počet | Sloučeno do    |
| ------------------ | ----- | -------------- |
| Data Scientist     | 7     | AI/ML Engineer |
| MLOps Engineer     | 6     | AI/ML Engineer |
| Analytics Engineer | 4     | AI/ML Engineer |
| Data Analyst       | 35    | Data Engineer  |

---

### IT, podpora & consulting → IT Analyst

Obecné IT role, analytici a konzultanti sloučeni do **IT Analyst**.

| Malá kategorie           | Počet | Sloučeno do |
| ------------------------ | ----- | ----------- |
| IT Specialist            | 40    | IT Analyst  |
| Technical Support        | 7     | IT Analyst  |
| Cloud Support Engineer   | 5     | IT Analyst  |
| Systems Analyst          | 42    | IT Analyst  |
| Business Analyst         | 47    | IT Analyst  |
| Business Systems Analyst | 20    | IT Analyst  |
| Technical Consultant     | 27    | IT Analyst  |
| SAP Consultant           | 5     | IT Analyst  |
| Systems Administrator    | 13    | IT Analyst  |

---

### Management & Leadership → Engineering Manager

Všechny manažerské a vedoucí role sloučeny do **Engineering Manager**.

| Malá kategorie               | Počet | Sloučeno do         |
| ---------------------------- | ----- | ------------------- |
| Software Development Manager | 12    | Engineering Manager |
| Director of Engineering      | 8     | Engineering Manager |
| Head of Engineering          | 1     | Engineering Manager |
| CTO                          | 1     | Engineering Manager |
| CIO                          | 1     | Engineering Manager |
| Project Manager              | 21    | Engineering Manager |
| Product Manager              | 18    | Engineering Manager |
| Product Owner                | 20    | Engineering Manager |
| Scrum Master                 | 6     | Engineering Manager |
| Program Manager              | 5     | Engineering Manager |
| Delivery Manager             | 3     | Engineering Manager |

---

## Ponechané kategorie (≥ 50, bez sloučení)

Tyto kategorie již měly dostatečný počet pozorování a **nebyly modifikovány**:

| Kategorie                     | Počet              |
| ----------------------------- | ------------------ |
| Software Engineer             | 2182               |
| Software Developer            | 1858 (po sloučení) |
| Senior Software Engineer      | 1258               |
| Full Stack Developer          | 1143               |
| Engineer                      | 936 (po sloučení)  |
| Security Engineer             | 707 (po sloučení)  |
| DevOps Engineer               | 677                |
| Data Engineer                 | 654 (po sloučení)  |
| Programmer                    | 592                |
| Systems Engineer              | 534                |
| Frontend Developer            | 450                |
| Cybersecurity Analyst         | 446                |
| Site Reliability Engineer     | 425                |
| QA Engineer                   | 402 (po sloučení)  |
| Security Analyst              | 394 (po sloučení)  |
| Solutions Architect           | 383 (po sloučení)  |
| Test Engineer                 | 344                |
| Automation Engineer           | 314                |
| Other                         | 308                |
| Embedded Software Engineer    | 305                |
| IT Analyst                    | 285 (po sloučení)  |
| Web Developer                 | 278                |
| UI/UX Designer                | 261 (po sloučení)  |
| Cloud Engineer                | 259                |
| Backend Developer             | 255                |
| QA Tester                     | 247                |
| Staff Software Engineer       | 201                |
| Java Developer                | 177                |
| Engineering Manager           | 161 (po sloučení)  |
| Firmware Engineer             | 159                |
| Software Development Engineer | 158                |
| Game Development              | 156 (nová)         |
| Software Architect            | 143                |
| Salesforce Admin/Developer    | 137                |
| Database Developer            | 131                |
| Principal Software Engineer   | 124                |
| Mobile Developer              | 105                |
| Quality Engineer              | 100                |
| SDET                          | 97                 |
| Junior Software Engineer      | 85                 |
| .NET Developer                | 85                 |
| Lead Software Engineer        | 75                 |
| AI/ML Engineer                | 70 (po sloučení)   |
| Database Administrator        | 64                 |
| Platform Engineer             | 63                 |
| Tech Lead                     | 62                 |
| DevSecOps Engineer            | 56                 |
| Android Developer             | 55                 |
| Infrastructure Engineer       | 53                 |
| iOS Developer                 | 50                 |
