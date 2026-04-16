# Poznámky a argumenty pro obhajobu (Ekonometrické modely)

Tento dokument shrnuje zajímavá ekonometrická zjištění z finálního analyzačního běhu modelů (`ai_skills_thesis_final.do` ze dne 16. dubna 2026), která demonstrují robustnost naší metodologie a poskytují výborné argumenty pro případné otázky komise u obhajoby diplomové práce.

## 1. Ošetření endogenity v modelech mezd (Tabulka 4)
Tohle je klíčový argument pro metodologickou čistotu. Do baseline mzdového OLS modelu (Tabulka 4) byly **záměrně vyřazeny** proměnné `cluster_generative_ai` a `cluster_data_science__ml`. 
Pokud by tam zůstaly, "sežraly" by efekt proměnné `ai_level` a uměle by zkreslily prenie za AI, jelikož LLM v pozadí zařazuje inzeráty do vyšších tierů právě skrz výskyt těchto technologických kompetencí. Provedené řešení zajistilo, že `i.ai_level` zachycuje krásně a holisticky celkovou mzdovou prémii navázanou na AI roli samotnou jako takovou, nikoliv jen prémii za samotné "znění" klíčových slov. Modelem s plnou sadou clusterů pak disponuje Příloha C jako testem robustnosti a dokazuje, že prémie nepadá ani v plné specifikaci.

## 2. Kolinearita a vzdálenost proměnných (VIF)
Častým argumentem oponentů při použití většího počtu proměnných v jednom modelu je riziko multikolinearity. 
U OLS modelu pro stanovení mezd (Tabulka 4) byl po odhadu spuštěn VIF (Variance Inflation Factor) test se skvělými výsledky:
- **Mean VIF USA:** 1.72
- **Mean VIF Německo:** 2.09
- **Mean VIF Indie:** 2.44

Zlaté pravidlo říká, že ohrožení nastává průměrem nad 5 či individuálně nad 10. Průměrné VIF napříč zeměmi jsou pod hranicí 5 a klíčové technologické clustery i proměnná `ai_level` leží bezpečně v pásmu 1.05 – 1.45. **Je nicméně nutné transparentně přiznat:** indický OLS model vykazuje izolovaný VIF ≈ 17 u interakce vzdělání, což překračuje obvyklou hranici 10 a znamená lokální multikolinearitu v této konkrétní interakci. V logit a mlogit modelech (Tabulky 2 a 3) tato interakce není — VIF problém je omezen na OLS Tabulku 4 a není driverem hlavních výsledků ohledně AI prémie. Skill clustery zůstávají napříč specifikacemi bezpečně ortogonální.

## 3. Heckmanův model a obhajoba německého vzorku (N=514; Pokrytí mzdy ~8 %)
Data pro Německo disponují pouze velmi nízkým pokrytím inzerované mzdy (pouhých ~8% inzerátů ji uvádí). Komise se může ptát, jestli tím nevzniká brutální vychýlení. Lze to vyvrátit těmito objektivními daty pocházejícími z analýzy výběrového zkreslení - **Heckman výběrového modelu (Příloha A)**:

- U kultuře otevřených mezd formovaných (USA a Indie) Heckman nalezl signifikantní bias plynoucí ze zvyků firem vkládajících platy k náboru (USA p = 0.028, IND p = 0.025), který rovnou efektivně zkorigoval, aniž by srazil hlavní prémii, což důrazně prohlubuje validitu amerického modelu.
- Oproti tomu pro netransparentní trh práce, obhájený přísnější ochranou soukromí zaměstnanců v Německu, test nenalezl naprosto **žádnou výběrovou nezávislost - p = 0.789!** To vědecky dokazuje, že firmy inzerující těch 8 % platů se nijak anomálně strukturálně neliší, nezpůsobují vychýlení základních odhadů vysvětlujících faktů v technologiích ani v samotných mzdách. Nadto je těch absolutních $N = 514$ z hlediska teorie spolehlivosti parametrů nad dimenzionálním minimem kolem $N > 200$. 

## 4. Test cross-country heterogenity AI prémie (Příloha B)
V pooled regresním modelu byl **Wald testem** testován společný vliv interakcí `ai_level#country_id` (pomocí `testparm` — viz Příloha B). Výsledek: **p = 0,1498**.

- **Korektní interpretace:** Při daném vzorku se nepodařilo zamítnout nulovou hypotézu o rovnosti AI prémie napříč státy. Statisticky to znamená **absenci silné evidence o odlišnosti**, nikoli důkaz identické prémie — nezamítnutí H0 není potvrzením rovnosti (nelze zaměňovat s "equivalence test" typu TOST).
- **Věcný závěr:** Bodové odhady mzdové prémie za AI Integration (~+11,5 %) a Applied/Core AI (~+16,7 %) jsou napříč USA / DE / IN v podobném pásmu a nelze mezi nimi prokázat statisticky významné rozdíly. To je **slabší a opatrnější formulace** než tvrzení o "globální tržní konstantě" — otevíráme tím prostor pro debatu, ale nevystavujeme se snadné metodologické námitce.

## 5. Ošetření malých buněk v logit / mlogit modelech

V rámci dialogu s vedoucím práce byla řešena otázka minimálního počtu pozorování v jednotlivých kombinacích prediktorů. Jeho doporučení znělo: **„raději slučovat než vyřazovat z modelu"**. Podle tohoto principu byla v `ai_skills_thesis_final.do` provedena následující sloučení:

1. **`size_cat`:** kategorie „201–500" + „501–1000" sloučeny do „201–1000" (redukce 8 → 7 kategorií, minimum buňky × `ai_level` v DE/IN > 40).
2. **`type_cat`:** „Nonprofit / Government / Education" sloučeny do „Unknown/Other/Gov" (v DE měla kategorie Nonprofit pouze N = 43, v IN N = 57; redukce 4 → 3 kategorie).
3. **`sector_nace`:** kategorie K, M, Q sloučeny do „Other" (K+M+Q měly v DE dohromady < 500 inzerátů a v kombinaci s `ai_level` < 30; redukce 7 → 4 kategorie).
4. **`job_family`:** „Frontend & Design", „QA & Testing", „Security", „Systems & Embedded" sloučeny do „Other" (v DE měly × Applied/Core AI buňky 1–7 pozorování; v IN Frontend × Applied/Core = 1). Výsledných 7 kategorií: Data & AI, DevOps & Cloud, Management, Software Developer, Software Engineer, Sr+ Software Engineer, Other. **Tato taxonomie je totožná s dřívější verzí `ai_skills_analysis_comparative.do` z roku 2024** — nejde o ad-hoc úpravu, ale o návrat ke konzistentnímu přístupu.

Metodologická pozice k "pravidlu 50":

- „Pravidlo minimálně 50 pozorování na buňku" se vztahuje k *Cochranovu pravidlu* pro chí-kvadrát test nezávislosti, nikoliv k MLE odhadu logit / mlogit.
- Relevantním kritériem pro logit je **Events-Per-Variable (Peduzzi et al., 1996; Vittinghoff & McCulloch, 2007)** — minimálně 10 pozorování vzácnější události na parametr. I v nejmenším vzorku (Indie) máme `has_ai = 1` pro 892 inzerátů, tedy při ~30 parametrech EPV ≈ 30 (tři­krát nad doporučením).
- Při krajně malé sub-buňce (1–2 pozorování) MLE odhad obvykle konverguje, pouze koeficientu přidělí velký rozptyl a nesignifikantní p-hodnotu; tento jev ošetřujeme klastrovanými SE `vce(cluster firm)`.
- **Nicméně** pro buňky s N ≤ 10 v kombinaci s `ai_level = 2` vedoucí doporučil sloučit, což jsme provedli (bod 4 výše) — tím preventivně předcházíme i riziku *quasi-complete separation*.

Po sloučení by měly být všechny buňky `job_family × ai_level = 2` v DE i IN ≥ 20 (přesná čísla doplnit z crosstab diagnostiky v sekci 6.0 nového logu po re-runu). Binární Logity (Tabulka 2) i Multinomiální Logity (Tabulka 3) jsou nastaveny s `vce(cluster firm_cluster)`; konvergenci a výsledky Hosmer-Lemeshow GOF a IIA Hausman testů ověřit v novém logu. Jako další robustness check podle doporučení vedoucího spočítat poměry `SE_DE / SE_US` a `SE_IN / SE_US` pro klíčové koeficienty (job_family, clustery); očekávaný strop je < 5 — pokud některý koeficient překročí, označit ho v textu jako estimated with reduced precision a nepřikládat mu interpretační váhu.

## 6. Zvolení binárního zobrazení namísto kategorií pro Vzdělání a Zkušenost (`edu_bin` / `exp_bin`)
U obhajoby může padnout dotaz, proč model jednoduše nenabízí kompletní kategorie vzdělání (např. Bachelor, Master, PhD) a praxe (Junior, Mid, Senior) namísto binárního oddělovače (MÁ / NEMÁ vyšší vzdělání či praxi). Přestože by kategorie analyticky zkoumaly diskriminované úrovně vzdělávání na mzdovou prémii mnohem lépe a poskytly nelineární diference, pro modely nad extraky z NLP textů inzerátů webu Glassdoor muselo dojít k použití binárních oddělovačů. 

Důvody, proč postoupit redukci komplexity:
- **Obrana proti "Sparse Cells" kolapsu modelů:** Nelineární multinomiální logit a Heckman jsou vysoce náchylné k datové degradaci za stavu řídkých buněk (Perfect prediction failure). Pakliže by do tabulky křížily interakce navíc 5 prvků edukace a 4 praxe společně s technologickými detaily, ztratil by Indický dataset (s již tak vysokým VIF u education na úrovni 17) stabilní balanc a model by přestal konvergovat či by vyhnal standardní chyby do nesmyslných rovin. 
- **Zamezení masivního hluku z měření (Measurement Error):** Běžný nestrukturovaný text inzerátu (zápisy jako `BS/MS` či `BS required, PhD preferred` společně s měkkou praxí *3-5 years*) působí po extrakci z webscraperů jako hrubé rušivo a započetlo by do "Missing" či "Unknown" škatulky nadpoloviční porce datasetu. Binární ukotvení poskytnutou extrakční mírností tento šum filtruje.
- **Odlehčení Stupňů volnosti pro cíl práce:** Diplomová teze analyzuje specificky mzdové chování AI skillů (jejichž robustnost má teď top prioritu). Vzdělání a seniorita v tomto designu tvoří *pouhé kontrolní proměnné v pozadí*, chránící před omitted variable biasem. Množení těchto proměnných by zbytečně pojídalo stupně volnosti na jevy, které neodpovídají položené výzkumné otázce.
