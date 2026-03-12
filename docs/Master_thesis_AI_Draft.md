# Návrhy kapitol pro úpravu diplomové práce (AI Focus)

Tento dokument obsahuje přepsané a rozšířené sekce z původní osnovy (viz `Master_thesis.docx`), které mnohem více odrážejí aktuální stav umělé inteligence (LLM, RAG) a její reálné dopady na trh práce podle tvého výzkumu. Texty můžeš rovnou upravit či zkopírovat do Wordu.

---

## 2.2 Dovednosti jako složka lidského kapitálu (Doplnění o AI kontext)

_Současný text o Beckerovi a rozdělení na general vs. specific training je skvělý. Navrhuji za něj přidat tento odstavec aplikující teorii přímo na zjištění ohledně umělé inteligence:_

V kontextu dnešního prudkého rozvoje umělé inteligence lze Beckerovo rozdělení dovedností pozorovat ve zcela nové dynamice. Schopnost využívat nástroje generativní umělé inteligence pro osobní produktivitu, jakým je například psaní efektivních příkazů (prompt engineering) do jazykových modelů typu ChatGPT či GitHub Copilot, se na trhu práce stává novou formou **obecného lidského kapitálu (general human capital)**. Tyto dovednosti jsou naprosto agnostické vůči pracovnímu prostředí a zaměstnanec si je plně přenáší mezi různými zaměstnavateli. Z toho plyne i neochota firem do tohoto druhu vzdělání aktivně investovat; očekávají, že si pracovník tyto dovednosti osvojí sám, případně si koupí licenci na nástroj a ihned z něj těží (tzv. profese _AI-Adopters_).

Oproti tomu tvorba a správa složitých podnikových LLM architektur (např. Retrieval-Augmented Generation nad uzavřenými interními dokumenty firmy, či nasazování a fine-tuning lokálních open-source modelů kvůli datové bezpečnosti) představuje moderní **specifický lidský kapitál (firm-specific human capital)**. Tato znalost se pevně váže na konkrétní datovou infrastrukturu a doménové znalosti podniku (profese _AI-Natives_ / inženýři). Zároveň se zde vytváří enormní mzdová prémie, protože nahradit zaměstnance, který rozumí interní AI a datové struktuře podniku, je pro firmu vysoce nákladné.

---

## 2.4 Technologické změny a trh práce (Oživení o LLM a polarizaci)

_K existující teorii (Frey & Osborne, Acemoglu) doporučuji přidat dovětek, jak přesně do toho zapadá generativní AI._

Nejnovější posuny vyvolané generativní umělou inteligencí (Large Language Models) nabourávají původní předpoklady modelu ALM (Autor, Levy & Murnane, 2003). Zatímco dřívější vlny digitalizace a výpočetní techniky nahrazovaly převážně rutinní úkoly, moderní generativní AI prokazuje schopnost vykonávat středně pokročilé kognitivní a analytické úkoly, tedy činnosti doposud domněle chráněné před automatizací (např. psaní kódu, sumarizace komplexních dokumentů, či tvorba marketingového obsahu).

Tento vývoj ještě více urychluje polarizaci trhu práce predikovanou Goos & Manningem (2007) neboli tzv. hollowing out z pohledu inženýrských dovedností. AI dnes tlačí dolů poptávku po „běžných“ juniorských pozicích (tradiční kódeři nebo copywriteři), protože jejich činnost zastane efektivně AI asistent. Zároveň ale raketově narůstá poptávka po malé, avšak úzkoprofilové skupině _AI Integrátorů_ a architektů, kteří tyto agenty umí stavět a nasazovat do produkce. Tyto technologické změny vychýlené ve prospěch technicky nejschopnější (Skill-Biased Technological Change) tak aktuálně zažívají na IT trhu svůj historický vrchol, což predikuje i vznik masivních mzdových rozdílů (wage premium) mezi obyčejným "uživatelem" a skutečným "tvůrcem" umělé inteligence.

---

## 3.1 Umělá inteligence (Zkrácení historie a přesun pozornosti k GenAI)

_Původně hodně historická pasáž. Navrhuji zachovat z historie 1-2 věty a rovnou skočit k podstatě dnešního trhu, na který reagují tvé inzeráty z Glassdooru._

Snahy o vytvoření umělé inteligence a její první koncepty (symbolická AI) sahají již do padesátých let minulého století, k osobnostem jako Alan Turing či John McCarthy. Skutečný bod zlomu pro trh práce ovšem nenastal v oblasti teoretického výzkumu, nýbrž až o několik desetiletí později se vznikem architektury Transformer (2017) a následným nástupem generativní umělé inteligence a velkých jazykových modelů (Large Language Models, LLM) na konci roku 2022 v podobě produktu ChatGPT.

Zatímco klasické strojové učení (Machine Learning), jež trhu dominovalo zhruba do roku 2020, bylo převážně analytické a prediktivní (např. shlukování dat, klasifikace obrázků, odhad cen či předpovědi prodeje), **Generativní AI** přinesla fundamentální posun v užitečnosti. Modely dnes prokazují sémantické porozumění nestrukturovanému textu a dokáží tvořit nový, koherentní obsah (kód, řešení problémů, texty, obrázky) na úrovni interagujícího agenta. Tato vlastnost odstartovala raketové investice a vznik zcela nových technologií (tzv. LLM ekosystému), načež firmám vznikla akutní potřeba najímat lidský kapitál, jenž tyto moderní sítě dokáže ovládat a začlenit do firemních produktů. Znalost neuronových sítí založených na architektuře Transformer dnes proto reprezentuje odlišný segment trhu práce oproti tradičním datovým analytikům.

---

## 3.4 AI dovednosti – taxonomie a kategorizace (Reálný odraz trhu z tvých dat)

_Zde popisuji to tvé fantastické jádro: rozdělení z Bottom-up přístupu._

Při hodnocení pracovních inzerátů a identifikaci skutečného podílu úkolů (tzv. task-based approach) se ukazuje rozdělení pozic pouze na "IT bez AI" a "IT s AI" jako hrubě nedostačující. Na základě Bottom-up přístupu ke kategorizaci AI pozic, kdy inzerát hodnotíme dle explicitních tvrdých dovedností (hard skills), je nutné AI dovednosti na dnešním pracovním trhu rozdělovat minimálně do tří kvalitativně odlišných kategorií:

**1. Core AI (Vývoj samotných modelů a výzkum)**
Jedná se o tvůrce základních modelů. Jde převážně o hluboce expertní a vědecké pozice, vyžadující vysokoškolskou specializaci (Masters / PhD) v matematice nebo computer science.
_Klíčové dovednosti (signály z inzerátů):_ PyTorch, TensorFlow, vývoj LLM modelů, úpravy architektury neuronových sítí (Transformers), optimalizace výpočtů (CUDA).

**2. Applied AI / AI Integration (Integrace a orchestrování)**  
Toto je nejrychleji rostoucí segment současného trhu práce v IT. Tyto role netrénují od nuly nové modely za miliony dolarů. Úkolem těchto inženýrů je integrovat API (např. od OpenAI či Anthropic) do produktů vlastní firmy a budovat tzv. AI infrastrukturu za účelem poskytnutí hodnoty koncovým uživatelům.
_Klíčové dovednosti (signály z inzerátů):_ RAG pipelines (Retrieval-Augmented Generation), vektorové databáze (Pinecone, Chroma, Milvus), nasazování přes LangChain, MLOps, a fine-tuning open-source modelů.

**3. AI-Adopters (Uživatelé AI pro zvýšení produktivity)**
Nejedná se o inženýry umělé inteligence v pravém slova smyslu, ačkoliv inzeráty na tyto pozice často nesou štítek nebo buzzword slova "AI". Tito pracovníci konzumují výstupy AI k tomu, aby naplnili rutinní nebo nerutinní dovednosti ve své běžné profesi.
_Klíčové dovednosti (signály z inzerátů):_ Udržování povědomí o ChatGPT, GitHub Copilot, používání nástrojů Midjourney nebo Jasper, "Prompt Engineering" a aplikování promptů přes chatové rozhraní.

Analýza jasně prokazuje, že tyto tři kategorie čelí naprosto odlišným mzdovým prémiím, křivkám poptávky a riziku substituce automatizací. Ignorování této taxonomie vede ke zkreslení makroekonomických statistik o poptávce po AI.

---

## 3.5 Trendy v poptávce a fenomén "AI Washing" a 3.8 Dopady AI

_Pojďme zužitkovat ten tvůj poznatek z extrakce inzerátů o Thezi marketingu._

S rostoucím mediálním nadšením ("hype") ohledně umělé inteligence po roce 2023 lze sledovat nový trend poptávky, který lze označit termínem **"AI Washing"**. Tento fenomén popisuje stav, kdy organizace záměrně obohacují názvy pracovních pozic (job titles) nebo úvodní odstavce inzerátů o klíčové slovo "AI", aby působily inovativně na trhu práce k přilákání špičkových kandidátů (tzv. signaling efekt) nebo k potěšení investorů. Bližší sémantická analýza obsahu (tasks a hard skills) v těchto inzerátech nicméně často odhalí, že profil pozice nevyžaduje žádné kompetence z _Core AI_ či _Applied AI_, a pracovní očekávání se omezuje maximálně na úroveň _AI-Adopter_ (např. psaní lepších textů za pomoci chatbota).

Tato diskrepance způsobuje metodologický problém při klasickém vyhledávání klíčových slov (keyword matching). Pokud trh práce zkoumáme pouhým hledáním zkratky "AI" v popisech, dojde k masivnímu naředění zjištěných dat takzvanými falešně pozitivními (false-positive) rolemi z netechnického rázu. Skutečná, hluboká poptávka po IT inženýrech, jež umí spravovat strojové modely, je trhem skryta právě v těchto mračnech marketingově upravených inzerátů. O to zásadnější význam má na data využít pokročilou filtraci skrze jazykové modely, které dokáží kontextuálně vyloučit "uživatele" od inženýrských tvůrců.
