# Kontrola požadavků vedoucího -- checklist

## 1) Binární proměnná

-   Vytvoř **binární proměnnou (0/1)**:
    -   `0 = None`
    -   `1 = cokoliv s AI`
-   Slouží **pro robustnost modelu**.

------------------------------------------------------------------------

## 2) Count proměnná pro skills

-   Kromě binární proměnné vytvoř také **count proměnnou**.
-   Ta bude **sčítat počet dovedností v dané oblasti**.
-   Rozsah např.:\
    **0 až 29 skills**.

------------------------------------------------------------------------

## 3) Sloučení malých clusterů

-   Projdi **skupiny skillů (clusters)**.
-   Pokud má cluster **méně než 30--50 pozorování**, sluč ho s jiným.
-   Příklad:
    -   `3D artist + Game designer`
-   Cílem je vytvořit **smysluplné skupiny s dostatečným počtem
    pozorování**.

------------------------------------------------------------------------

## 4) Dořešení pozic „Other"

-   Projdi kategorii **"Other" v názvech pozic** (cca **900+
    pozorování**).
-   Zkus využít **AI** pro:
    -   přiřazení části z nich do existujících kategorií
    -   nebo jejich **další redukci/sloučení**.

------------------------------------------------------------------------

# 5) Mzda (Salary)

### Typ mzdy

Vytvoř **pomocnou proměnnou**, která uloží typ původní mzdy:

-   `Annual`
-   `Monthly`
-   `Hourly`

### Přepočet

-   **Annual** → ponechat beze změny\
-   **Monthly** → vynásobit **12**\
-   **Hourly** → přepočítat na **roční mzdu** podle zjištěných norem

Pokud bude přepočet příliš riskantní:

-   **hodinové mzdy raději vyřadit**
-   ale **nejdříve zkus přepočet**

### Logaritmus

-   Připrav proměnnou:

`log(salary)`

-   Ta se použije **v regresním modelu**.

------------------------------------------------------------------------

# 6) Vzdělání a praxe

### PhD

-   **Sloučit PhD s Master**

### Associate Degree

Pro **pravděpodobnostní model (AI occurrence)**: - sloučit:

`Associate + High School`

(důvod: **málo pozorování \< 50**)

Pro **mzdovou regresi**: - slučování **není nutné**

### Praxe (Experience)

-   **Opravit nuly**
-   `0` má být pouze tam, kde je:
    -   **"entry level"**
    -   **"no experience"**

Pokud údaj **chybí**: - musí být **Missing** - **ne nula**

### Seniorita

-   Sloučit:

`Expert → Senior`

------------------------------------------------------------------------

# 7) Ostatní proměnné

### Sektory

-   Sloučit sektory s **méně než 50 pozorováními**\
    (např. hotely, restaurace, food).

-   `Media` můžeš **případně vyhodit**.

### Lokace

-   Do datasetu je potřeba **namapovat státy USA**.
-   Aktuálně jsou tam:
    -   jen **města**
    -   nebo **remote**

### Firma

Přidat proměnné:

-   **Size** (velikost firmy)
-   **Type** (např. Private/Public)

### Rok založení (Founded)

-   Zkontroluj **kolik dat chybí**.
-   Použije se spíše jako **záchranná proměnná**, pokud by **Size
    nefungovala**.

------------------------------------------------------------------------

# 8) Příprava na odeslání

### Do-file a seznam proměnných

Po vyčištění dat:

-   sepiš **názvy finálních proměnných**
-   vlož je do:
    -   **samostatného souboru**, nebo
    -   **přímo do Do-file**

Cíl:\
vedoucí musí vidět **s jakými proměnnými se bude pracovat**.

### Termín odeslání

-   Ideálně **po 13:00**
-   nebo **jakmile to bude hotové**

Vedoucí si s daty:

-   zkusí **pohrát**
-   a **navrhne finální modely**

------------------------------------------------------------------------

# 9) Shrnutí strategie modelů

Data připravuješ pro **tři úrovně modelů**:

### 1. Základ

Proměnné:

-   Sektor
-   Typ firmy
-   Velikost firmy
-   Lokace

### 2. Rozšířený model

Základ +

-   Vzdělání
-   Praxe
-   Pozice (normalizované)

### 3. Kompletní model

Rozšířený model +

-   Skill clusters
-   Pozice

------------------------------------------------------------------------

# Poznámky

# 1. Mzdový model (lineární regrese)

### Cíl

Zjistit:

**jak různé úrovně AI dovedností ovlivňují výši platu**

### Závislá proměnná (Y)

-   **Logaritmus mzdy**

### Hlavní sledovaná proměnná

-   **AI Tier**

Ukáže:

o kolik se **zvýší plat oproti kategorii "None"**

------------------------------------------------------------------------

## Model A (základní)

Proměnné:

-   AI Tier
-   Sektor
-   Typ firmy (Private/Public)
-   Velikost firmy
-   Lokalita
-   Remote (zda je to práce z domova)

------------------------------------------------------------------------

## Model B (s lidským kapitálem)

Model A +

-   Vzdělání
-   Praxe

------------------------------------------------------------------------

## Model C (kompletní)

Model B +

-   Skill clusters
-   Normalizované názvy pozic

Poznámka:

-   U **pozic a skillů** se budou testovat:
    -   verze **s nimi**
    -   i **bez nich**

Cíl:\
zjistit **jak ovlivňují model**.

------------------------------------------------------------------------

# 2. Pravděpodobnostní model (Multinominální probit nebo logit)

### Cíl

Zjistit:

**jaké firmy a na jaké pozice nejčastěji vyžadují AI dovednosti**

------------------------------------------------------------------------

### Závislá proměnná (Y)

**AI Tier**

Kategorie:

-   `None`
-   `Integrated`
-   `Applied/Core`

Referenční kategorie:

`None`

Model tedy počítá pravděpodobnost:

-   `Integrated vs None`
-   `Core/Applied vs None`

Ve výsledné tabulce se zobrazí **dva sloupce**.

------------------------------------------------------------------------

# Modely pravděpodobnosti

## 1. Základní model (firemní profil)

Proměnné:

-   Sektor
-   Typ firmy (Private/Public atd.)
-   Velikost firmy (Size)
-   Lokalita (Stát)

### Důležité upozornění

Do tohoto modelu **nedávej proměnnou Remote**.

Důvod:

-   práce z domova **nevysvětluje**, proč by pozice měla obsahovat AI.

### Cíl

Zjistit:

-   jaký **typ firmy**
-   jaká **velikost firmy**
-   v jakých **lokalitách**

AI **nejčastěji požadují**.

------------------------------------------------------------------------

## 2. Model rozšířený o dovednosti (Skills)

Proměnné:

-   vše ze základního modelu
-   **Skill clusters**

### Cíl

Zjistit:

zda inzeráty požadující určité **hard skills**\
zároveň **častěji požadují AI**.

------------------------------------------------------------------------

## 3. Kompletní model

Proměnné:

-   vše z modelu 2
-   Vzdělání
-   Praxe
-   Normalizované názvy pozic (Job titles)

Modely si pak můžete rozkouskovat a zkoušet přidávat praxi a pozice
postupně, abyste viděli, jak se výsledky mění.

------------------------------------------------------------------------

# Důležité upozornění pro multinominální model

Je nutné kontrolovat **kontingenční tabulky**.

Na rozdíl od mzdové regrese:

-   zde musí mít **každá podkategorie dost pozorování**.

Ideálně:

**minimálně 50**.

Týká se hlavně proměnné **Vzdělání**.

Proto bude nutné:

-   sloučit některé kategorie\
    např.:

`High School + Associate Degree`

Jinak může **Stata házet chyby**.
