# Úprava kategoriálních proměnných pro zachování robustnosti modelu

Tento dokument shrnuje problémy s distribucí dat v analyzovaném datasetu pracovních inzerátů z oblasti IT a navrhuje statisticky nezbytné úpravy. 

## 1. Identifikace problému (Řídká data / Sparse Data Problem)
Naším cílem je modelovat proměnnou `ai_level` (úroveň AI zapojení), která má 3 kategorie:
* **None** (Žádné AI, n = 14 190)
* **AI Integration** (Používá AI nástroje jako koncový uživatel, n = 2 354)
* **Applied/Core AI** (Aplikuje, trénuje a buduje AI modely, n = 1 313)

Pro zajištění statistické spolehlivosti a konvergence v odhadech modelu multinomiální logistické regrese (`mlogit`) je v ekonometrii vyžadováno, aby **každá sledovaná podkategorie měla alespoň 50 pozorování v rámci všech sledovaných skupin závislé proměnné**.

Při pohledu na nejmenší nezávislou skupinu zkoumaných dat (**Applied/Core AI**, pouhých 1 313 inzerátů) zjistíme, že některé kategorie pracovních inzerátů nesplňují tuto hranici. Vznikají zde buňky s jednotkami pozorování, což by vedlo ke statisticky pofidérním výsledkům (vysoké směrodatné odchylky atd.).  Z tohoto důvodu je nutné proměnné účelně a promyšleně agregovat.

Níže jsou uvedeny konkrétní problémy a návrhy na jejich řešení, rozdělené dle daných proměnných:

---

## 2. Navrhované úpravy jednotlivých proměnných

### A) Požadované vzdělání (`edu_cat`)
* **Původní stav v Applied/Core AI:** 
  * Neuvedeno/Missing: 177
  * Středoškolské (Highschool): **16**
  * Vyšší odborné (Associate): **7**
  * Bakalářské (Bachelor): 712
  * Magisterské a vyšší (Master+): **44**
* **Odůvodnění:** Převážná většina trhu buď vzdělání v inzerátu specificky neuvádí, nebo plošně požaduje vysokoškolský titul (minimálně bakalářský, což zahrnuje mj. přes 10 400 inzerátů v celém datasetu). LLM (umělá inteligence extrahující data) byla záměrně instruována k přísnosti: *"Nevyvozuj 'Bakaláře' jen proto, že jde o IT pozici."* V důsledku tohoto objektivního zpracování skončilo mnoho inzerátů s hodnotou "neuvedeno" a inzeráty deklarující menší než bakalářské, ale i čistě vyhrazené pro magisterské tituly, jsou u vzorku úzce profilovaných Applied AI pozic na trhu natolik minoritní, že nedosahují hranice 50 pozorování.
* **Provedené řešení:** Původních 5 kategorií se agreguje do **binární proměnné (dummy variable)**:
  * `0 = Bez VŠ titulu / Neuvedeno` (Zahrnující Missing, Highschool, Associate)
  * `1 = Vysokoškolský titul: Bakalář a vyšší` (Zahrnující Bachelor, Master, PhD)

### B) Požadovaná délka praxe / Seniorita (`exp_category`)
* **Původní stav v Applied/Core AI:** 
  * Entry-level (0 let): **46**
  * Expert (10 a více let): **25**
* **Provedené řešení:** Dojde ke sjednocení hraničních, málo početných kategorií s jejich nejbližšími logickými sousedy.
  *Kategorie "Entry" (46 poz.) se sloučí do kategorie "Junior (0-2 roky)".*
  *Kategorie "Expert" (25 poz.) se sloučí do kategorie "Senior (6-10 let)".*
* **Nový (agregovaný) stav:** Junior (0-2), Mid (3-5) a Senior+ (6+). Tento krok zachová rozdíly v senioritě a současně bezpečně splní limit 50 pozorování.

### C) Rodina pozice - Odbornost (`job_family`)
* **Původní stav v Applied/Core AI:**
  * Frontend & Design: **9**
  * QA & Testing: **30**
  * Security: **42**
  * Systems & Embedded: **40**
* **Odůvodnění:** Applied/Core AI pozice silně akcentují role typu *Data & AI* (341 obs) nebo *Software Engineer* (219 obs). Na druhé straně barikády však inženýři zaměření ryze na Frontend design, testování, nebo zabezpečení prakticky neexistují (ve vztahu k budování a tréninku AI modelů).
* **Provedené řešení:** Sloučení těchto řídce obsazených specifických odvětví do předexistující společné kategorie `Other` (Ostatní). Mezi samostatnými proměnnými stále zůstávají nosné kategorie softwarových vývojářů, DevOps, datových analytiků atd.

### D) NACE Sektor Odvětví (`sector_nace`)
* **Původní stav v Applied/Core AI:** Ze 16 zastoupených průmyslových odvětví celých **11 odvětví nevyhovuje** pravidlu n=50 (mj. Zemědělství, Stavebnictví, Velkoobchod/Maloobchod, Logistika a další doprovodné průmyslové sekce, kde si AI modely zatím nikdo lokálně nestaví).
* **Provedené řešení:** Ponechat jen 5 nejdůležitějších a statisticky významných odvětví a ty nedisponující velkým výskytem agregovat. Zachováme odvětví J (IT & Telekomunikace), K (Finance/Pojišťovnictví), C (Zpracovatelský průmysl/Výroba), M (Vědecké/Profesní obory) a Q (Zdravotnictví) a samostatnou podskupinu Unknown. Všech 11 zbylých nevýznamných průmyslových sekcí bude sjednoceno pod kategorii `Other` (Ostatní odvětví).

### E) Typ společnosti podle vlastnictví (`type_cat`)
* **Původní stav v Applied/Core AI:** `Subsidiary / Pobočky` (40), a `Other / Zbytek` (26). 
* **Provedené řešení:** Množina kategoriálních hodnot se agreguje takto: Kategorie přidružených dceřiných společností (`Subsidiary`) bude včleněna do logicky vyššího a podobného nad-celku `Private` (Soukromé společnosti). Hodnota `Other` bude následně překlasifikována do odpovídající zbytkové kategorie `Unknown`.

### F) Vyřazení neplatných vlastností a dovedností (Skill Clusters)
V datové sadě byl zachytáván u každého inzerátu i výskyt desítek technologických parametrů / dovedností (proměnné `cluster_*`). Rozřazením do naší nejmenší skupiny "Applied/Core AI" (1 313 inzerátů) se bohužel ukázalo, že tři specifické dovednosti mají natolik zanedbatelný výskyt, že je nelze použít pro modelování multinomiální logistické regrese bez narušení předpokladů (a rizika tzv. *perfect prediction* erroru).

Jedná se o:
* `cluster_legacy__mainframe` (Tato zastaralá technologie se objevila u pouhých 49 inzerátů v celém datasetu 18 000 pozic).
* `cluster_data_analysis__stats` (Pouhých 29 pozorování v rámci Applied/Core AI pozic).
* `cluster_tools__editors` (Pouhých 47 pozorování v rámci Applied/Core AI pozic).

* **Provedené řešení:** Tyto tři konkrétní proměnné musí být kompletně vyřazeny z navrhovaných matematických modelů (nelze je agregovat, tvoří binární Ano/Ne indikátor přítomnosti v textu inzerátu). Ostatních více než 20 zachycených skill-clusterů bezpečně limit 50 pozorování splňuje.

---

Všechny výše zmíněné operace a postupy jsou naprosto standardním způsobem (v rámci machine-learningu i thesí pracujících s regresními modely chybějících / řídkých dat), jak zamezit jevu umělých odchylek s ohromnou variancí a získat čistější a validnější predikční model v programovém prostředí Stata.
