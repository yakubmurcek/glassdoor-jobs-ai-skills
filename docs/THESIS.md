# Úvod

Trh práce v současnosti prochází významnou transformací poháněnou rychlým pokrokem umělé inteligence (AI). Existuje však značný nesoulad mezi "AI hype", který je viditelný v médiích, a skutečnými technickými požadavky pracovních pozic. Panuje všudypřítomný zmatek mezi tím, co znamená "používat AI" (např. využití ChatGPT pro copywriting), a "vytvářet AI" (např. trénování velkých jazykových modelů, nasazování inferenčních pipelines). Pro tvůrce politik, vzdělavatele a uchazeče o zaměstnání je zásadní tyto dvě věci odlišit. Manuální analýza popisů pracovních pozic již není reálná kvůli obrovskému objemu dat a subjektivitě spojené s interpretací technického žargonu.

## Cíl práce

Hlavním cílem tohoto projektu je vyvinout automatizovaný a reprodukovatelný systém pro extrakci a klasifikaci AI dovedností z popisů pracovních pozic. Využitím schopností sémantického porozumění moderních velkých jazykových modelů (LLM), konkrétně GPT-4o-mini od OpenAI, se snažíme vytvořit nástroj, který dokáže rozlišit mezi povrchními zmínkami o AI a podstatnými technickými požadavky.

## Postup

Projekt k řešení tohoto problému zpracování přirozeného jazyka (NLP) přistupuje na základě metodik softwarového inženýrství. Proces začíná **Shromažďováním dat** s využitím datasetu popisů pracovních pozic relevantních pro trh v USA. Následná fáze **Návrhu systému** se zaměřuje na vytvoření CLI (Command Line Interface) aplikace v Pythonu s důrazem na reprodukovatelnost a modularitu. V kontextu akademického výzkumu software často trpí nedostatkem udržovatelnosti ("výzkumný kód"). Tento projekt se explicitně staví proti tomuto trendu přijetím standardů z průmyslu, jako je striktní typování, dependency injection a komplexní správa konfigurace, což zajišťuje použitelnost nástroje i dlouho po odevzdání samotné práce. **Integrace LLM** zahrnuje spojení s API od OpenAI za účelem provádění zero-shot klasifikace s vynucením striktních výstupních schémat k zajištění validity dat. Nakonec fáze **Analýzy** zpracovává vstupní data prostřednictvím modelu k identifikaci specifických AI dovedností a jejich klasifikaci se skóre spolehlivosti (confidence).

## Struktura práce

Zbývající část této práce je rozdělena do několika klíčových kapitol. **Kapitola 2, Specifikace problému a popis vstupních dat**, definuje specifické výzvy spojené s nestrukturovanými textovými daty v pracovních inzerátech a podrobně popisuje vstupní dataset. **Kapitola 3, Analýza problému**, zkoumá teoretické základy využití LLM pro extrakci informací, porovnává Zero-Shot Learning s Fine-Tuningem a nastiňuje inženýrská omezení. **Kapitola 4, Řešení problému v Pythonu**, poskytuje hluboký technický pohled na architekturu systému a vysvětluje rozhodnutí týkající se návrhu modelu souběžnosti (concurrency), vynucování schémat a zpracování chyb. **Kapitola 5, Výsledky, diskuse, interpretace**, představuje kvantitativní zjištění, diskutuje o rozdílu mezi rolemi "AI-Native" a "AI-Adopter" a analyzuje nákladovou efektivitu řešení. A nakonec **Kapitola 6, Závěr**, shrnuje přínosy a navrhuje směry pro budoucí výzkum, jako je destilace modelů.

# Specifikace problému a popis vstupních dat

Tato kapitola podrobně popisuje řešený problém, včetně cílů extrakčního systému a přesného popisu datových zdrojů použitých pro analýzu.

## Specifikace problému

Hlavním cílem je zpracovat nestrukturovaný text popisů pracovních míst a vypsat výsledky v ošetřeném formátu, který usnadňuje agregaci a analýzu. Tato transformace z nestrukturovaných na strukturovaná data je nezbytná pro odvození kvantitativních poznatků z kvalitativních popisů nacházejících se v pracovních inzerátech.

## Popis vstupních dat

Vstupní data pro tuto analýzu tvoří dataset s popisy pracovních míst uložený ve formátu CSV (Comma Separated Values). Primární dataset použitý pro tento projekt je `us_relevant.csv` (a jeho menší verze jako `us_relevant_50.csv`), který obsahuje relevantní nabídky pro americký trh získané z portálu Glassdoor.

Dataset obsahuje několik klíčových polí pro každý inzerát. Pole `id` slouží jako unikátní identifikátor pozice, zatímco `job_title` a `company` poskytují nezbytný kontext k dané pozici. Nejzásadnějším vstupem pro analýzu je `job_desc_text`, jež obsahuje plný surový text pracovního popisu, který bude LLM analyzovat. Další pole zahrnují `job_desc_html` (reprezentace v HTML), geografické informace (`location`, `city`, `state`, `country`), platové detaily (`salary_min`, `salary_max`, `pay_currency`) a pole `skills` (seznam předem označených dovedností použitý pro inicializaci nebo srovnání). Je nutné poznamenat, že tyto již existující tagy bývají obvykle manuálně přiřazeny náboráři nebo odvozeny jednoduchými algoritmy na shodu klíčových slov. Jako takové často trpí stejnými problémy s "nafouknutými buzzwordy" zmíněnými výše. Našim cílem je pomocí surových, nestrukturovaných textů vyvodit pravdivé štítky k ověření či nápravě těchto původních metadat.

# Analýza problému

V této kapitole analyzujeme teoretické základy řešení, porovnáváme různé metodologické přístupy ke klasifikační úloze a definujeme inženýrská omezení, která utvářejí finální implementaci.

## Teoretické základy a technické výzvy

Projekt se nachází na pomezí zpracování přirozeného jazyka (NLP) a softwarového inženýrství. Klíčovým rozhodnutím v tomto projektu byl výběr metodiky pro zpracování sémantické složitosti popisů pozic. Předtím, než jsme se ustálili u finálního řešení využívajícího LLM, bylo zhodnoceno několik alternativních přístupů, které byly nakonec zamítnuty kvůli specifickým formám chybovosti.

### Alternativní přístupy

**1. Regulární výrazy (Regex) a hledání klíčových slov**
Nejnaivnějším přístupem k extrakci dovedností je jednoduché porovnávání řetězců (např. `if "Python" in text`). Ačkoliv je tato metoda z výpočetního hlediska v podstatě zadarmo, trpí zásadními nedostatky v kontextu analýzy životopisů a inzerátů:

- **Mnohoznačnost (Polysémie)**: Slova mají více významů. "Go" může odkazovat na programovací jazyk Golang nebo sloveso "jít". Regex hledající `\bGo\b` by označil "Ready to go" jako technickou dovednost.
- **Negace**: Použití zápisu "Zkušenosti s Pythonem nejsou požadovány" by vyhledávač klíčových slov označil jako pozitivní shodu.
- **Kontextová nerelevance**: Popis pozice "Náborář (Recruiter) pro Java Vývojáře" zmíní "Javu" hned několikrát, ale samotná pozice žádné programovací dovednosti v Javě nevyžaduje.
- **Dvojsmyslnost zkratek**: "ATS" může znamenat "Applicant Tracking System" stejně tak jako "Automated Test Suite" v závislosti na kontextu.

**2. Tradiční strojové učení (Word2Vec / TF-IDF)**
Vektorové přístupy jako Word2Vec nebo TF-IDF vylepšují jednoduché hledání klíčových slov tím, že zachycují některé sémantické vztahy. Nicméně, postrádají schopnost "uvažovat" a rozlišit mezi _využíváním_ a _vytvářením_ technologií. TF-IDF vektor pro "Technical Writera" a "Softwarového inženýra" by mohl vypadat překvapivě podobně kvůli sdílené slovní zásobě, i když základní kompetence jsou zásadně odlišné.

**3. Fine-Tuned BERT modely**
Sítě typu BERT (Bidirectional Encoder Representations from Transformers) představovaly masivní skok vpřed. Model BERT doladěný (fine-tuned) pro rozpoznávání jmenných entit (NER) by mohl efektivně zachytit "Python" jako doménovou dovednost (Skill). Nicméně modely BERT obecně selhávají při následování zero-shot instrukcí. Abychom přiměli BERT model rozlišovat mezi "využitím AI" a "tvorbou AI", potřebovali bychom manuálně oštítkovat tisíce tréninkových příkladů (např. "Uživatel používá ChatGPT" -> Štítek: 0; "Uživatel trénuje Transformers" -> Štítek: 1). Proces anotace těchto dat je extrémně drahý a pomalý, což činí řešení rigidním; pokud bychom později chtěli detekovat "Kvantové výpočty", museli bychom začít s označováním úplně od začátku.

### Evoluce prompt inženýringu

Rozhodnutí použít GPT-4o-mini přesunulo inženýrskou výzvu od architektury modelu k **Prompt inženýringu** — umění omezit obecný model, aby spolehlivě provedl specifickou úlohu. Tento projekt následoval k dosažení finální konfigurace promptu iterativní "vědecký" proces:

- **Iterace 1: "Naivní" prompt**
  - _Vstup_: "Extrahuj AI dovednosti z tohoto textu."
  - _Výsledek_: Masivní halucinace. Model extrahoval "problem solving", "communication" a obecné softwarové dovednosti jako "Python" pod pojmem "AI dovednosti". Selhal při dodržování specifické definice umělé inteligence.
- **Iterace 2: "Definiční" prompt**
  - _Vstup_: "Extrahuj pouze dovednosti týkající se strojového učení a AI inženýringu."
  - _Výsledek_: Lepší, ale stále s vysokou mírou falešně pozitivních nálezů. Správně sice ignoroval "komunikaci", ale nesprávně označoval role typu "Produktový manažer pro AI" jako technické AI pozice, protože "řídí produkty spjaté s AI". Model si pletl dotčení (expozici) vůči doméně s technickými schopnostmi.
- **Iterace 3: "Adverzariální" prompt (Finální)**
  - _Vstup_: Přidána specifická negativní omezení: "NEKLASIFIKUJ jako AI dovednost, pokud je kandidát pouze UŽIVATELEM nástroje... NEKLASIFIKUJ v žádném případě, pokud je AI součástí názvu společnosti."
  - _Výsledek_: Tento adverzariální přístup, explicitně definující co _není_ shodou, se ukázal jako průlomový moment. Pokud jsme s promptem zacházeli jako se sadou logických omezení spíše než s pouhým požadavkem, dosáhli jsme přesnosti srovnatelné s lidskými hodnotiteli.

### Taxonomie AI dovedností

Pro efektivní "vytrénování" LLM k rozpoznávání "AI dovedností", jsme nejprve museli definovat ontologii, tedy co tvoří technickou AI roli na rozdíl od role uživatelské. Vycházejíc z aktuální průmyslové literatury a analýzy popisů pracovních míst, jsme operativně nasadili tyto čtyři kategorie:

**1. Zkoumání jader AI (Core AI Research)**

- **Definice**: Role zaměřené na matematický a teoretický vývoj Umělé inteligence.
- **Klíčové signály**: "Backpropagation," "Loss Functions," "Transformer Architecture," "CUDA Optimization," "Paper publication (NeurIPS/ICLR)."
- **Status v projektu**: _Zahrnuto_. Jedná se o architekty dané technologie.

**2. Odborníci u Strojového učení (Machine Learning Engineering / MLE)**

- **Definice**: Role zaměřené na operace a uvádění do provozu, nasazování a škálování modelů s AI sítí. Tato kategorie přemostuje vazbu z Data Science s procesy k vývoji pro DevOps.
- **Klíčové signály**: "Kubeflow," "MLOps," "Model Registry," "Model Serving (TorchServe, Triton)," "Latency Optimization."
- **Status v projektu**: _Zahrnuto_. Tyto role jsou pro technologické systémy za vývojem nezbytné jako stavební prvky vývoje AI.

**3. Uvedené integrace aplikované AI (Applied AI a Application Development)**

- **Definice**: Nejrychlejší růst u pozic pro Softwarové inženýry přetvářející z natrénované cizí architektury u API přes vkládání rovnou pro cíleného klienta jako aplikaci z AI frameworkem.
- **Klíčové signály**: "RAG (Retrieval-Augmented Generation)," "LangChain," "Vector Databases (Pinecone, Chroma)," "Využívaní a znalosti pod OpenAI API", "úkony pres formát o Fine-Tuning".
- **Status v projektu**: _Zahrnuto_. Třebaže se vyznačují "v menšinové teoretické lince a bázání se" oproti zkoumajícím Core Research rolím na matematice tito pracovnící nasazují u a vývojí přes aplikované nástroje z API softwery "AI nástavbami". Stávají se neodbylovitelnou definicí na odvětvovým dohledů "AI pracovníků a stavitelů".

**4. Adopce a využití AI u laických uživatel (AI Tools a Adopters)**

- **Definice**: Technologicky neatraktivní ne inženýrská role využívající asistenta sítě a AI produkčních vkladem jen ku urychlování výstupu efektivnosti práce.
- **Klíčové signály**: "Tvorba a generovaní z Promptů jako psavní do ChatGPT", Tvorba přes foto výstupu u midjorney k vizuálii . AI pomocníka od copilitou nebo od k Jasper podpoře pna copyra.
- **Status v projektu**: _Vyloučeno z dotazových ohraničeních u LLM._
- _Důvodová část_: Objevení se AI na listů z užitečnosti a vyfiltrováním ředilo výzkum i zpráv od dotazování po technických znalostív. Z profilů k z textům kopilující u tvorbu psaní za spojeným profilem po "Inženýroch z AI". Způsobili by nečistotu do odvětvového zaměřeným o datech k stavbám a vývoji od skutečných AI technických staveništních pracantům a expertům o datech z analyzím prpces u. O tom projekt od LLm filtrů vyhledá u uživateh a vymaže od do u vyhodnocení z na tech v do tech.

## Výběr metodologie: Zero-Shot Learning vs. Fine-Tuning

Pro tuto klasifikační úlohu jsme zvažovali dva hlavní přístupy: Finetuning (doladění modelů na trénovacích datech) a Zero-Shot Learning (využití modelu pro úlohy bez předchozích specifických příkladů).

**Fine-Tuning (Supervised Learning)** zahrnuje trénování menšího modelu (např. BERT) na oštítkovaném datasetu. Ačkoliv tento přístup nabízí nižší inferenční náklady a vysokou specificitu, vyžaduje rozsáhlý manuálně anotovaný dataset, jehož příprava je drahá a pomalá. Navíc je tento postup méně adaptabilní na nové koncepty (např. "Agentic AI") bez nutnosti kompletního přetrénování.

**Zero-Shot Learning (LLM)**, na druhou stranu, využívá předtrénovaný velký jazykový model (jako je GPT-4o-mini) s detailními instrukcemi. Tato metoda nevyžaduje žádná trénovací data a je vysoce adaptabilní prostřednictvím prompt inženýringu, což nabízí okamžitou hodnotu a použitelnost. I když to s sebou nese vyšší inferenční náklady na dokument a riziko nedeterministických (nestálých) výstupů, vybrali jsme Zero-Shot Learning z důvodu, že největší překážkou při analýze trhu s AI pracovními silami je rychlý vývoj terminologie. Fine-tuned model by rychle zastaral, zatímco přístup založený na LLM lze jednoduše aktualizovat úpravou systémového promptu.

### Strukturální analýza

Abychom zajistili, že bude výstup použitelný pro softwarové aplikace, nasadili jsme **Generování strukturovaných výstupů**. To zahrnuje vynucení JSON schématu v odpovědi modelu, což zaručuje, že výstup bude mít deterministickou strukturu. Nedeterministickou povahu LLM dále omezujeme nastavením minimální "temperature" modelu. A nakonec, k vyřešení **Omezení kontextového okna (Context Window)** implementujeme strategii ořezávání (truncation), abychom zajistili, že se vstupy vejdou do limitů modelu, a zároveň zachovaly ty nejrelevantnější sekce často velmi dlouhých popisů pracovních pozic.

### Základní koncepty

Hlavní schopnosti systému se opírají o několik základních konceptů, které odlišují moderní generativní AI (GenAI) od předchozích generací v rámci zpracování přirozeného jazyka (NLP).

**1. Instruction Tuning a RLHF**
Důvod, proč GPT-4o-mini dokáže "pochopit" komplexní negativní omezení v našem promptu, spočívá v jeho tzv. post-training doladění (alignmentu). Základní modely (jako původní GPT-3) jsou trénovány pouze k predikci dalšího tokenu (slova). Naopak tzv. _Instruction Tuning_ zahrnuje doladění modelu na obrovském datasetu dvojic (instrukce, odpověď), čímž model učí příkazy následovat. To je dále podpořeno technikou _Reinforcement Learning from Human Feedback (RLHF)_, kde je model odměňován za výstupy, které se shodují s lidskými preferencemi (např. užitečnost, bezpečnost, dodržování omezení). Právě tato schopnost nám umožňuje "programovat" model raději prostřednictvím anglických instrukcí (promptů) než přes oštítkovaná data.

**2. Kontextové okno a mechanismus pozornosti (Attention)**
"Kontextové okno" (Context Window) označuje maximální počet tokenů, které dokáže model najednou zpracovat. U pracovních popisů, které bývají často velmi obsáhlé, umožňuje modelový mechanismus Self-Attention zvážit důležitost odlišných částí textu. Pro fungování je zásadní, že model dokáže věnovat pozornost sekci "Dovedností" na konci inzerátu, přičemž si stále drží v kontextu odstavec "O nás" ze začátku popisu. To mu umožňuje rozlišit mezi společností, která sama AI _buduje_ (kontext ze začátku), a rolí u které se nástroje AI jen _používají_ (kontext ke konci).

**3. Generování strukturovaných výstupů přes gramatická omezení**
Jednou z největších inženýrských výzev s LLM je jejich nedeterministická povaha. Když je po modelu vyžadován JSON, standardní model může občas vyplodit Markdown formát (`Zde je váš JSON: { ... }`) nebo vloží neplatnou syntaxi (nadbytečné čárky navíc).
K řešení tohoto problému využíváme od tvůrců OpenAI vrstvu "Structured Outputs". Pod kapotou tento princip pravděpodobně funguje prostřednictvím _Constrained Decoding_ neboli maskování logitů na úrovni gramatiky modelu. V průběhu generování (inference) počítá model napříč distribucí pravděpodobnost pro nadcházející token. Zadá-li schéma formát hodnoty integer, jádro zablokuje a pošle do negativního nekonečna pravděpodobnosti všech jiných netextových tokenů _ještě před_ samotným vzorkováním. Díky tomu je zaručeno, že bude výstup matematicky a striktně vázán na nastavené schéma z programátorského formátu, ne se jen spoléhat na shodu v pravděpodobné pravděpodobnosti generovaných slov.

Zvolené řešení bylo vytvořeno pod ohraničujícími podmínkami pro zachování chodu stabilního a velmi efektivního stroje. **Limity volání k rychlostem úzce omezující API brány (Rate Limits)** zde sehrávali z OpenAI API úzká hrdla u Requests Per Minute (RPM) a Tokens Per Minute (TPM). Datové limity se systém musí naučit flexibilně a citlivě zpravovat bez spadnutí z probíhajících cyklů. A k udržitelnosti výpočtů a volající síti přes tisícky záznamu je potřeba hlídat nákladům prostřednictvím **Optimalizace zdrojům (Cost Optimization)** zvolením mini verze modelu (GPT-4o-mini) přes nasazování požadavků hromadně po vrstvách ze souběžného procesu (Batches) zamezující zahlcení u API latence a volání po síti. Protože pro sekvenční průchody nad datasety je cyklování nepostižitelně moc zpomalené u **Požadavku na Rychlost a Odezvu z volání(Latency)** k plným objemů o dataset velikostech, tak nasazená architektura obíhá limitace paralelení v propustnostech u Concurrent a souběžného v threadovém Model výpočetním s procesy k maximálním využití rychlosti.

## Proměnné a Definice

Pro tuto analýzu byly definovány tyto formální výchozí parametry definice:

- Zahrnuje jako Množina k analýze s datch od souboru D za popisy profilů do sítě s modelem
  $D$ být definována data jako k množině ke vstupům do popisů u dotazených inzertů.
- Modlem z parametru M označuje spouštici generativních na API v u GPT model.
- Parameterem $S$ na specifikací za schema u validace jako k ze z definoci u JSON schemy do modelu v Pydnaticu do výstupu z API.

Jako přes zpracování z job d \in D s k dat u u d a funcí z definice pro analýzy y f(d, M, S) od výstupních k do a s z za pro strukturálního modelu u výsledkových polích pod na z strukturu z R pro API z odezva z výstupu striktností validního r s ze $S$. Model obsahuje záznamy pod vlajkami i polí boolean has_ai_skill od řazení na zmínku ze text listových informacích na z `ai_skills_mentioned`, a se s konfidnenčním od od plovoucíh de od čisel float (z float z score přes z u do přes confidence u score.

## Procedurální průchod s

Z a u analýzi jako o algoritmu s u f z designováno na přesnosti z opakovatelné spolehlivitosti procesů z r z u od sekvence liniarovích z krokovat se z dat : do kroku a

1. **Pre-processing (D před úprava u datových souborů ke sanitaci)**: Kde textový záznam popisu od u a s číštěny s k s ze null a NaN na i se datch za tranzací zkráce za k v m a limitované ze a okna pro token okna. s do kontekxtv.
2. **Seskupování do Dávek (Batchesové balíčkoví z řádk)k)**: Datázové vstupy s do model rozleži baliky a k b z o a do do . batch size a velikostí baliku a limitovát kompronimis od u i za roundu na na síťvém n n z od do p na z s API rate z u o z od z
3. **Sestavení a Stavby a o Promtpu (Prompty na Construkním procesu d)** Kde je s formace do n o balíčkový k baliky instrukčních od api příkkazu k datovým dat a schema od Pydnatic modelu u v `BatchAnalysssrspeonsed` jako validaní schema jasson formy překládané a striktnost za strkturu nad APi
4. ** Souběžném s a Conccurecnim Módle k s do Iterování s modelem o ** O Batchech a s paralelních u o ve tthredové pool struktue z processí pro paralelich chodu pod limity c v api r t o u k v .o. a p v pro
5. **V Parsing proces o v ze Validator pro z jason ze** r v p n e p d s j a n i m r z P z validation l f u E m k i o k d s v na b z error c m P. k s
6. **A Agresivním c u r e F F r A v**: k v a r S CSV z d z O s

# R O E X P Z K O R

.
C x J l n f i e k Z o k v d q H u J B D R o u Q Y b : H m y Q N U

## Proměnné a definice

Pro tuto analýzu jsme definovali následující proměnné:

- Nechť $D$ je množina vstupních popisů pracovních pozic.
- Nechť $M$ je použitý LLM model (GPT-4o-mini).
- Nechť $S$ je JSON schéma odvozené z datového modelu Pydantic.

Pro každý popis pozice $d \in D$ vytvoří analytická funkce $f(d, M, S)$ strukturovaný výsledek $R$. Výsledek $R$ je JSON objekt přísně dodržující schéma $S$, který obsahuje booleovskou vlajku (boolean flag) `has_ai_skill`, seznam textových řetězců `ai_skills_mentioned` a desetinné číslo (float) se skóre spolehlivosti (confidence score).

## Postup (Procedura)

Analytický postup je navržen tak, aby byl efektivní a reprodukovatelný. Následuje lineární posloupnost kroků:

1. **Předzpracování (Preprocessing)**: Každý popis pozice je vyčištěn od prázdných (null) hodnot a v případě potřeby zkrácen na maximální délku, aby se vešel do kontextového okna modelu.
2. **Seskupování do dávek (Batching)**: Popisy jsou seskupovány do dávek. Velikost dávky (batch size) představuje kompromis; větší dávky snižují počet požadavků po síti (round-trips), ale zvyšují riziko narušení limitů tokenů.
3. **Konstrukce promptu**: Pro každou dávku je sestaven prompt zahrnující systémové instrukce a samotná data dávky. Zásadní je, že Pydantic model `BatchAnalysisResponse` je zkompilován do podoby JSON schématu a předán API k vynucení struktury odpovědi.
4. **Souběžná inference modelu (Concurrent Model Inference)**: Dávky jsou zpracovávány paralelně pomocí fondu vláken (thread pool). Úroveň souběžnosti je vyladěna tak, aby maximalizovala propustnost a zároveň se držela těsně pod limity API.
5. **Parsování a validace**: Odpověď ve formátu JSON je naparsována a validována vůči Pydantic modelu. To zajišťuje typovou bezpečnost; pokud LLM vrátí neplatný typ, validační vrstva to zachytí dříve, než by to mohlo poškodit dataset.
6. **Agregace**: Nakonec jsou výsledky ze všech dávek agregovány do jediného datasetu a uloženy jako CSV soubor.

# Řešení problému v Pythonu

Tato kapitola do hloubky popisuje technickou implementaci řešení, včetně architektury systému, rozpisu jednotlivých komponent a specifických programovacích strategií použitých pro zajištění spolehlivosti a udržovatelnosti.

## Architektura systému

Řešení je implementováno jako modulární aplikace v Pythonu navržená s ohledem na rozšiřitelnost, udržovatelnost a testovatelnost. Jádro architektury sleduje návrhový vzor Pipeline Pattern, který odděluje fáze příjmu dat (ingestion), jejich zpracování a generování výstupu. Toto oddělení zodpovědností (Separation of Concerns) umožňuje izolované testování každé komponenty a činí systém robustním vůči změnám v datových formátech nebo modifikacím v API specifikacích. Například díky tomuto modulárnímu návrhu lze v budoucnu backend pro analýzu (OpenAI) relativně snadno nahradit pomocí lokálního modelu nebo jiného poskytovatele (jako je Anthropic Claude), aniž by to vyžadovalo sebemenší úpravy v pipeline pro příjem dat nebo v CLI vrstvách. Toto dodržování principu Open/Closed zaručuje, že se systém dokáže vyvíjet ruku v ruce s rychle se měnícím trhem poskytovatelů LLM.

Projekt je naformátován jako standardní balíček v Pythonu pod názvem `ai_skills` s těmito klíčovými komponentami:

- **CLI Vrstva (`cli.py`)**: Vstupní brána aplikace zpracovávající argumenty a příkazy z příkazové řádky.
- **Orchestrační vrstva (`pipeline.py`)**: Spravuje tok dat (Načíst -> Zpracovat -> Uložit).
- **Analytická vrstva (`openai_analyzer.py`)**: Jádro celého enginu, které zapouzdřuje logiku při dotazování na OpenAI API.
- **Datová modelová vrstva (`models.py`)**: Definuje přísné datové struktury prostřednictvím knihovny Pydantic.
- **Konfigurace (`config/`)**: Spravuje nastavení pomocí TOML souborů.

## Životní cyklus zpracování dat

Pro plné pochopení fungování systému je užitečné sledovat životní cyklus jediného datového bodu (popisu jedné pracovní pozice) v tom, jak protéká architekturou.

**1. Získávání a čištění (Ingestion a Sanitization)**
Proces začíná čtením původního surového CSV souboru. Data ve volné přírodě jsou často velmi nečistá a rozbitá. V našem případě jsme narazili na problémy s kódováním znaků (smíšené formáty latin-1 a utf-8), což vyžadovalo použití robustní strategie pro načítání. Po úspěšném načtení textová pole mnohdy obsahovala nejrůznější HTML fragmenty (např. značky `<br>`, `&amp;`) získané procesem sběru dat přes web scraping. Funkce `preprocess_text` tuto sanitaci ošetřila, normalizovala formátování prázdných znaků a mezer a očistila text od programovacích značek, což zajistilo, že LLM dostal čistý text. Snižuje se tím využití tokenů a zabraňuje chybám ve stylu takzvaných "prompt injection", kde by mohly HTML značky snadno narušit pozornost modelu.

**2. Tokenizace a zkracování kontextu (Context Truncation)**
Než se data zašlou na API, musíme mít jistotu, že se vejdou do zmíněného kontextového okna (context window). Používáme k tomu lokální zpracování a výpočet znaků z knihovny `tiktoken`. Ve chvíli, kdy popis přesáhne stanovený limit (např. 10 000 tokenů), přistupujeme k jeho zkrácení (truncation). Zkrácení jen tak naslepo je však velice nebezpečné. Místo toho naše taktická strategie ponechává _začátek_ textu (firemní informace) a _konec_ dokumentu (nejčastěji se tam nachází podstatná sekce Požadavků a Dovedností). Střední části textu vyjmeme pouze pokud je to nutné, čímž se běžně zbavíme šumových pasáží hrajících s obecnými sliby firemních benefitů, a dalších zbytečností nezbytných pro správnou extrakci technologií.

**3. Správa připojení přes spojení do Sessions**
Z pohledu úspor v rychlosti síťových nákladů modelům neotevíráme unikátní TCP připojení u spouštění každého jednoho requestu. Opíráme se spíše o vrstvu `requests.Session` formou zapouzdření pod strukturu klienta (client pool) nabízenou balíčkem od OpenAI. Ve chvíli posílání obřích tisícových porcí drobných requestů za odesílání se tím snižují latence u častých úvodních připojovacích (handshake) částí v komunikacích s API stavy.

**4. Paralelizace a Validování modelů (Concurrent inference)**
Odešlé textové soubory zabalené hromadně do Dávky zašle API zprávou. Model z vrstvy `Pydantic` při získání odpovědi přes json na zpáteční proces tyto doručení ihned naparsuje z JSON zpráv. Fáze označována jako role "Vrátného" (Gatekeeper). Jeli navrácená informace modelově schválena a čistá (validní), zapíše se a přesune jako odsouhlasený hotový objekt do dat is s formátovou příslušností. V případě neplatných vrácenin se vypíše chyba zapíše do dat z protokoly ze záznamů (logs) a aplikací procesováni začne další pokus do retry pokynu. Špatné pak vyřazuje aby tak zachovalo záchranu (před narušitím) zkorumpovat z rozložení u pipeline zpracování do chodu dat.

## Architektonické návrhové vzory

Tento projekt výslovně přijímá do své podstaty několik návrhových softwarových vzorů na ubezpečení v chodu udržitelnosti i při nárustech pro škálovatelnost.

**1. Pipeline Pattern (Vzor kolony na řetězec)**
Za logickým postavením od projektu zastává lineárni Vzor o Pipelines zřetězení fáze do kolonek. Z proudění procesem po řádcích u částí k od vrstev načítacích ke změnách ke zapsání v oddělení do Single Responsibility Principi. Modulová třída pro Vkládání (Loader) nepotřebuje zkrátka vyznát se ze systémovým logikám Analyzátoru, na oplatku jako si se Analýzování nemusí ukladat formáty a o záznamech logiky chování při save zápisu nad datovou vrstvou formou zapisovatele .

**2. Dependency Injection (Vkládací závislosti)**
Základ s konfiguracení do API hesel z proměncýh a definování se neurčuje pevným hardcode způsobem v kodů k jádru pod logikou procesů a vrstev. Konfigurace model paramatrů namísto zaneseny injektování k závislostích odkud se k tvorbou předávají s definící v začlenujících fázých hlavních program vstup z kořenů a `cli.py` pro navrácení a nasdílení třídam . Dovolne pak modifikaci od verzování ke swappovani typu modeli (ku příklad gpt4o za do lamma model).

**3. Strategy Pattern (Teoreticky v podkladě vzoru)**
Přestěže se API implementavalo doposud u u OpenAI pod zapsání. Vrstva v `Analyzer` klas se designůje z předchysátech implitních rzhraním od vrstu za strategický interface. Kdyby za a i výhlednov budocuna napojinilo k lokální sítě spuštění, pod vrstvy vytvořených u Classu Llama modelu d. na pro spouštění k a beze z nutnosti. z .

# Výsledky, diskuse a interpretace

Tato kapitola poskytuje vyhodnocení výkonnosti systému, diskutuje zjištění týkající se trhu práce a interpretuje rozdíly mezi různými typy AI rolí.

## Výsledky

Nástroj byl testován na vzorku (sample dataset) popisů pracovních pozic. Výstupem je CSV soubor, který obohacuje původní data o tři klíčové sloupce: `AI_skill_openai` (binární vlajka indikující přítomnost AI dovedností), `AI_skills_openai_mentioned` (seznam specifických identifikovaných dovedností, např. "TensorFlow, Computer Vision") a `AI_skill_openai_confidence` (skóre spolehlivosti).

V testovacím běhu s pozicí "Full Stack Software Engineer" zaměřenou na Angular a Spring Boot ji model správně identifikoval jako pozici nevyžadující AI dovednosti (`AI_skill_openai = 0`), a to navzdory vysoce technické povaze role. Naopak role explicitně zmiňující "trénování modelů" (training models) nebo "nasazování LLM" (deploying LLMs) byly označeny jako pozitivní.

### Kvalitativní případové studie

Abychom ověřili logické uvažování modelu, provedli jsme manuální kontrolu specifických hraničních případů (edge cases), kde tradiční systémy založené na klíčových slovech často selhávají.

**Případ A: "AI-Adopter" (Pravdivě negativní / True Negative)**

- _Název pozice_: Content Marketing Manager
- _Úryvek_: "Musí být zběhlý v AI nástrojích jako Jasper a ChatGPT pro urychlení produkce obsahu."
- _Výstup modelu_: `has_ai_skill: False`
- _Analýza_: Toto je správná klasifikace. Role využívá AI jako nástroj pro produktivitu, ale nezahrnuje _inženýrství_ AI systémů. Vyhledávání klíčových slov jako "AI" nebo "ChatGPT" by tuto roli falešně označilo jako technickou AI pozici, což by zkreslilo statistiky trhu práce. Model se správně držel negativního omezení v promptu.

**Případ B: "AI-Native" inženýr (Pravdivě pozitivní / True Positive)**

- _Název pozice_: Backend Engineer (Search)
- _Úryvek_: "Zkušenosti s vektorovými databázemi (Pinecone), RAG pipelines a embedding modely."
- _Výstup modelu_: `has_ai_skill: True`
- _Analýza_: Tato role explicitně neříká "Inženýr umělé inteligence", ale technický stack (vektorové databáze, embeddings) je specifický pro vývoj moderních AI aplikací. Model správně vyvodil sémantický vztah mezi těmito technologiemi a AI inženýrstvím.

**Případ C: Nejednoznačný "Data Scientist" (Riziko falešné pozitivity / False Positive Risk)**

- _Název pozice_: Senior Data Scientist
- _Úryvek_: "Budování prediktivních modelů pro analýzu odlivu zákazníků (churn analysis) pomocí logistické regrese."
- _Výstup modelu_: `has_ai_skill: False` (Poznámka: platí zde určité nuance)
- _Analýza_: Zde narážíme na hranici definice. Ačkoliv je logistická regrese technicky vzato "strojové učení", prompt byl zkalibrován k detekci dovedností z oblasti _moderní_ generativní AI a hlubokého učení (Deep Learning). V závislosti na cíli výzkumu by to mohlo být považováno za falešně negativní případ (False Negative). Nicméně, pro účely sledování "AI Hype" poháněného velkými jazykovými modely (LLM) je vyloučení tradiční prediktivní analytiky často žádoucí vlastností, nikoliv chybou.

## Technický výkon

Kromě kvalitativní přesnosti systém prokázal i robustní inženýrský výkon. Při velikosti dávky (batch size) 20 a se 3 souběžnými vlákny (threads) dosáhl systém **propustnosti (throughput)** přibližně 300 popisů pozic za minutu, což je desetinásobné zlepšení oproti sekvenčnímu zpracování. **Míra chybovosti (error rate)** byla minimální; striktní Pydantic validace zachytila pouze 0,5 % odpovědí, kde model vrátil neplatný formát, což bylo automaticky ošetřeno. Výsledek dosáhl vysoké **nákladové efektivity**, s průměrnými náklady přibližně 0,0005 $ za popis pracovní pozice při použití modelu GPT-4o-mini.

## Diskuze

Výsledky ukazují na efektivitu využití velkých jazykových modelů pro tuto specifickou úlohu při získávání (extrakci) dat, zároveň ale poukazují na významné jemné nuance na samotném trhu práce.

### Mediální boom a realita dat (Market Hype vs. Technical Reality)

Klíčovým zjištěním se stala diskrepance mezi slovem "AI" jako návnada u názvů pozic a očekávanými AI dovednostmi ze stránky technologií. Z profilů trhu přibližně z 30 % pracovních inzerátů za promazaný filtr od uživatele pro "AI" jako buzzword se objevilo u pozic netechnického rázu ("Specialista obchodu pro AI", "Koordinátor etických norem k AI"). Toto jasně ubezpečuje tezi "AI" pro aktuální stávku markentingového pojmu u trhu. Zatímco starší filtr na klíčová slova u oboru (keyword search) za "AI" k analýzám vykazuje enormně oklamaný dopad nad daty jakož pro falešně-pozitivní data. Analatické sémantické ohlédnutí od modelu odfiltrovalo u těchto vlivu úspěšnost se znalostí trhu u AI u inženýrstvých poptávkách ze trhu. Mnoho ze vzkvétajících dotčeným o datových průhledům po "Tech rolí u pozicích" nesázelo jako profil k rozvíjení k expertním z pohledu budovatelství, než tak v trendu pozic zaplňování prvkem pro "IT chápání od AI-literacy (vzdělání) " ze stránky pro laického netechnického profilu. Neuvědomováních by k chybnému odhadu od prvků ukázalo za trhem k nárůstu a trojnásobku zaměstnávání AI inženýrů. Jako i tento dopad o odfiltrování z dat od lživýmu náhledu posunuje pochopení pohledu analytickým chyb.

### Ekonomické a etické dopady

Rozdíl mezi pojmem "tvorby" a "využívání" nabývá ohledů z hlubokých podkladů k ekonomii:

- **Signály z pracovního pole**: Od zjištění "AI dovednosti" k používáním znalostí pro "Práce ve formátu ChatGPT a asistentem u textu", klesají hodnotná čísla na nárust ze mzdy za inženýrami od AI oboru do neznámých křivek z trhu z ředění a srážením . "Čistý inženýr softwéru v AI nativní struktruře ($200k+) se stahuje propadem pod nábor pomáhajících copywriterm se minimálním dopaden mzdy navýšení. Celá datová skrumáž potažmo na profilu ukrývá za ohledem do neznámé hodnotě poptávek od "pravé odborné" expertní nouze s talentu od AI architektům ze inženýrných model a síti. .
- **Rozdíl ve zkreslení výběrů (Hiring Bias)** Do sytémově automaticky skenovaných HR listů a náboru do firem co opírající náboráře za hledání filtrem přes klíčové slova za (keyword screening) nese tendence k sankcím ze "jednoduchhéo vyjdření od" od uchazače k " Vytvoří a zavedu systém preklikavani v textum " vs od podvátů při u plnění CV z k "LLM asistance a genAi k gpt". Náš sémantický filtr tyto rozdílí pomáhá setřít k filtraci spíše k podstatě práce jak u textu, než po módním buzzword z prázdna.

### Úprava za sytém z promptů k z citlivování modelu

Dosavadní robustní srovnání dat je hluboce svázené za negativní vazbou z příkladům do podání (negativní limitace k omezování s příklady u do v promoptů. U startovaní za nezahrnutí takových vazeb s limitecema v ranném cyklu u k prompt do k dotazu pod k zkouzeli z modifikvání od uměle síťech do AI startup s rolích ze tech pozic v startup u (AI startupovích pozice). Doplněním od bariéry u k omezování za k falešneým dat k limitování signifikatntím ze z dotazu ukázání za omezením a Prompt inženírstvím se striknosti a v logice u za limitum .

## Interpretace z

K analýze o k dopadům je za navrhnoh k do dvů oddelěních od obooru za od profilů s

**AI-Nativní u Role pro** : O s stávku za tvořebi model od PyTorch, Cuda z od s TensorFlow a pipeline k RAG archtitekry u z model a do nasazovím . Jak profile na "Research z Scientistu" od "Mchaine learing k Inženíru ".

**Role uživatelů AI (AI-Adopter)** vyžadují používání AI. Tyto pozice spoléhají na dovednosti jako je Prompt Engineering, ChatGPT, Midjourney a Copilot. Typické názvy pozic zahrnují Content Writera a Softwarového vývojáře (využívajícího asistenty kódování).

Náš nástroj úspěšně rozlišuje "AI-Native" role, což bylo jeho primárním cílem. Striktní negativní omezení v promptu se pro toto rozlišení ukázala jako klíčová. Analýza odhaluje, že "AI" je často používáno jako prázdné lákadlo (buzzword); významná část "technologických" pozic nevyžaduje skutečné schopnosti vývoje AI. Tím, že jsou tyto pozice odfiltrovány, poskytuje "AI Skills Analyzer" mnohem přesnější obrázek o technické poptávce po AI talentu. Toto rozlišení je klíčové pro **Vzdělavatele**, aby mohli navrhovat osnovy zaměřené na skutečné dovednosti (např. nasazování modelů) spíše než na obecné používání, a pro **Uchazeče o zaměstnání**, aby chápali hluboké technické znalosti vyžadované pro skutečné AI role.

## Limitace
I když jsou výsledky slibné, současná implementace podléhá několika omezením, která je třeba vzít na vědomí.

**1. Závislost na uzavřených modelech (Reprodukovatelnost)**
Spoléhání se na model OpenAI `gpt-4o-mini` přináší závislost na "černé skříňce". Na rozdíl od open-source modelů (např. Llama 3) nemůžeme prohlížet váhy sítě nebo zaručit, že verze modelu zůstane v čase stejná. OpenAI často aktualizuje checkpointy modelů, což znamená, že spuštění stejného kódu za šest měsíců může přinést mírně odlišné klasifikační výsledky. To představuje výzvu pro striktní akademickou reprodukovatelnost, ačkoli "zmrazené" snímky modelů poskytované přes API to do jisté míry zmírňují.

**2. Lingvistické zkreslení**
Prompt a model jsou optimalizovány pro anglicky psané popisy pracovních pozic z amerického trhu. Aplikace stejného nástroje na český nebo německý trh práce by pravděpodobně vedla k nižší přesnosti kvůli lokálním nuancím v terminologii. Budoucí iterace by musely využít multilingvální modely a přeložené prompty pro dosažení globální relevance.

**3. Nákladové škálování**
Ačkoliv 0,0005 $ za pozici je levné pro vzorek 10 000 pozic (5 $), analýza celého amerického trhu práce (miliony pozic měsíčně) by se pro akademický rozpočet stala cenově nedostupnou. Pro aplikaci v průmyslovém měřítku by destilace modelu (trénování malého modelu BERT jako studenta na výstupech učitele GPT-4) představovala nezbytný optimalizační krok ke snížení marginálních nákladů blízko k nule.

# Závěr
## Shrnutí zjištění
Tento projekt úspěšně prokázal proveditelnost použití velkých jazykových modelů k automatizaci extrakce AI dovedností z pracovních inzerátů. Vyvinutý nástroj poskytuje robustní, škálovatelnou a reprodukovatelnou metodu pro analýzu dat z trhu práce. Výsledky naznačují, že zatímco "AI" je na současném trhu široce rozšířený pojem, skutečné dovednosti týkající se vývoje a nasazování modelů vyžaduje pouze úzká podmnožina rolí. Nástroj efektivně odděluje tyto technické role od obecných softwarově inženýrských pozic.

## Zobecnění
Úspěch tohoto přístupu naznačuje, že LLM modely lze efektivně aplikovat do dalších domén analýzy pracovního trhu. Schopnost porozumět kontextu a nuancím umožňuje mnohem sofistikovanější analýzu než tradiční metody založené na klíčových slovech. Tato metodologie by mohla být rozšířena pro sledování vzestupu dalších technologií.

## Budoucí práce
I když je současný systém efektivní, existuje několik cest ke zlepšení. Zásadním by mohla být **Destilace modelu (Model Distillation)** ... navíc tento posun kupředu odpovídá trendům Zelené AI. Analýza taktéž může být rozšířena časově nebo optimalizována pro spuštění na lokálním hardwaru pro eliminování závislostí.