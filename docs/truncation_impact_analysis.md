# Analýza dopadu zkrácení pracovních inzerátů

## Kontext

Metoda `_prepare_text` v souboru `openai_analyzer.py` zkracuje pracovní inzeráty přesahující `MAX_JOB_DESC_LENGTH` (nastaveno na 6 000 znaků) tím, že provede naivní uříznutí úvodu (`text[:6000]`). To znamená, že u delších inzerátů se pozdější části – které často obsahují explicitní požadavky na dovednosti, předpoklady pro vzdělání a očekávanou praxi – mohou částečně nebo úplně ztratit, než se dostanou do LLM.

Výstupní CSV dokument zachovává původní, nezkrácený text. Nicméně sloupce odvozené pomocí LLM (`desc_tier_llm`, `hardskills`, `softskills`, `edulevel_llm`, `experience_min_llm`) u ovlivněných záznamů byly vytvořeny z neúplného vstupu.

## Rozsah

Z celkových 18 464 pracovních inzerátů přesáhlo **1 031 (5,6 %)** limit 6 000 znaků a bylo zkráceno během inference pomocí LLM. Dataset má průměrně 4 112 znaků na popis pozice (medián 3 939), kde 90. percentil je na 6 941 a 95. percentil na 7 830 – což potvrzuje, že limit ovlivňuje horní odlehlou část u distribuce délek inzerátů.

| Původní délka inzerátu | Zasažených pozic | Přibližný text viditelný pro LLM |
| ---------------------- | ---------------- | -------------------------------- |
| 6 000 – 7 000 znaků    | 290              | ~92 %                            |
| 7 000 – 8 000 znaků    | 391              | ~80 %                            |
| 8 000 – 10 000 znaků   | 258              | ~67 %                            |
| 10 000 – 15 000 znaků  | 84               | ~48 %                            |
| 15 000 – 25 000 znaků  | 8                | ~30 %                            |

Průměrně LLM obdržel v ovlivněných záznamech **77,4 %** původního textu. V nejhorším případě zbyla pouze **26,3 %** textu.

## Riziko dle úlohy v LLM

- **Klasifikace úrovně AI (Nízké riziko):** Signály sloužící k rozhodování (firemní kontext, shrnutí pozice) se zpravidla objevují brzy v inzerátech a tak zůstávají zachovány.
- **Extrakce dovedností (Střední riziko):** Technické požadavky se velmi často objevují v sekcích „Kvalifikace“ v druhé polovině dokumentu. Zhruba 350 inzerátů ztratilo přes 30 % své délky, což riskovalo nezachycení chybějících dovedností.
- **Vzdělání a Zkušenosti (Střední až vysoké riziko):** Tato pole jsou obvykle vyjmenována ke konci pracovního popisu a stávají se tak nejvíce zranitelná vzhledem k úpravě délek. 92 inzerátů, které přesáhly 10 000 znaků, mají tyto položky pravděpodobně nespolehlivé.

## Závěr

Velká většina ovlivněných záznamů (681 z 1 031) ztratila méně než 20 % svého celkového textu a do značné míry omezila praktický dopad přesnosti klasifikátorů. Oproti tomu zhruba 350 záznamů zažilo významnější ztrátu dat (> 30 %), převážně v oblastech požadavků na dovedností a očekávané vzdělání s praxí. Vzhledem na budoucnost může do budoucna implementace v rozmezí 12 000 – 15 000 znaků zabránit zkrácení textu u více jak 99 % dokumentů.
