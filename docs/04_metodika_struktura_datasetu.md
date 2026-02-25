# 4.1.2 Struktura datasetu a popis proměnných

Pro účely empirické analýzy byl na základě dat z platformy Glassdoor sestaven finální dataset o velikosti přes 11 000 unikátních záznamů (pracovních inzerátů z oblasti informačních technologií na území USA). Datová sada obsahuje celkem 75 proměnných, které lze rozdělit do pěti hlavních obsahových kategorií: (1) identifikační a kontextové proměnné, (2) dovednosti a AI klasifikace, (3) vzdělání a praxe, (4) mzdy a (5) atributy společnosti. Níže je uveden detailní popis klíčových proměnných vstupujících do analytických modelů.

### 1. Identifikační a kontextové proměnné

Tato skupina proměnných slouží k základní identifikaci pracovní pozice a jejího geografického či organizačního zasazení.

- **`job_title`**: Původní název pracovní pozice tak, jak byl uveden v inzerátu.
- **`job_family`**: Agregovaná profesní rodina odvozená z názvu pozice (např. Software Engineer, Data Scientist, DevOps & Cloud atd.). Slouží jako důležitá kontrolní proměnná pro očištění regresních modelů od specifik jednotlivých IT profesí.
- **`lokalita` (`state`, `region`)**: Informace o americkém státě a přiřazení do širšího makroregionu v rámci USA (US Census region – Northeast, Midwest, South, West).
- **`is_remote`**: Binární indikátor (0/1) určující, zda inzerát nabízí možnost plně či částečně vzdálené práce (tzv. remote work). Proměnná byla odvozena na základě sloupce `remote_work_types`.

### 2. Dovednosti a AI klasifikace

Nejdůležitější sekci datasetu tvoří extrahované požadavky na dovednosti a proměnné měřící míru orientace pozice na umělou inteligenci.

- **`hardskills` a `softskills`**: Textové řetězce obsahující identifikované tvrdé a měkké dovednosti. Tyto proměnné vznikly robustním deterministickým párováním textu inzerátů a metadat proti předem definovanému slovníku (bez zapojení velkých jazykových modelů pro extrakci).
- **`skill_count`**: Spojitá proměnná udávající celkový kvantitativní počet vyžadovaných tvrdých dovedností na dané pozici.
- **`cluster_*`**: Sada 24 binárních (dummy) proměnných, jež reprezentují agregované rodiny dovedností (např. Data Science & ML, Generative AI, Cloud Computing, Backend Development). Tyto rodiny vznikly na složením mnoha individuálních technologií.
- **`desc_tier_llm` a `ai_tier_num`**: Ordinalizovaná úroveň AI požadavků odvozená pomocí velkého jazykového modelu (LLM) na základě plného textu inzerátu. Nabývá hodnot od absencí AI (`none`), přes integraci nástrojů (`ai_integration`), aplikovanou AI (`applied_ai`) až po vývoj AI řešení (`core_ai`). Numerická podoba slouží pro modelování v analýze rozptylu (ANOVA).
- **`has_ai` / `has_ai_flag`**: Přísný binární indikátor reálné AI pozice. Označuje pozice, které jednak nevykázaly nulový LLM tier (`none`), ale zároveň reálně požadovaly některou ze specifických AI nebo ML technologií, a to po odfiltrování obecných "buzzwords" jako je samotné slovo AI.
- **`ai_level`**: Zjednodušená agregace úrovně AI pro účely multinomiálního logit modelu, nabývající tří hodnot: 0 (None), 1 (AI Integration) a 2 (Applied/Core AI).
- **`desc_conf_llm`**: Kontinuální míra jistoty (confidence score), se kterou LLM přiřadil AI tier. Záznamy s jistotou nižší než 70 % byly z datasetu výlučně odstraněny, aby byla zajištěna robustnost sady.

### 3. Vzdělání a praxe

Lidský kapitál požadovaný zaměstnavatelem je operacionalizován skrze požadavky na formální vzdělání a délku předchozích zkušeností. Jejich extrakce probíhala opět i za pomoci umělé inteligence pro čtení volného textu inzerátů.

- **`education_hybrid` a `edu_cat`**: Kombinovaná a následně ordinalizovaná úroveň požadovaného vzdělání. Predstavuje minimální zmíněnou úroveň vzdělání (od hodnoty 0 pro neuvedené, 3 pro bakalářský stupeň, až po doktorát s hodnotou 5).
- **`experience_min_llm` a `exp_category`**: Minimální počet požadovaných let praxe vytažených z textu a z něj odvozená seniorita pozice (od 0 pro chybějící údaj, přes 1 pro Entry-level, až po 5 označující expertní úroveň s více než 10 lety praxe).

### 4. Mzdové proměnné

Sada proměnných použitá pro analýzu mzdové prémie spojené s AI dovednostmi.

- **`salary_min`, `salary_max`, `salary_mid`**: Finanční ohodnocení pozice definované jako minimum, maximum a střední bod ročního mzdového intervalu v dolarech.
- **`pay_period`**: Proměnná indikující frekvenci uváděné mzdy (hodinová, měsíční, roční), jež byla využita k přepočtu mezd na srovnatelnou roční bázi.
- **`ln_salary`**: Přirozený logaritmus odvozené roční střední mzdy (`salary_mid`). Tato forma je typicky aplikovaná pro interpretaci procentuálních efektů jakožto závislá proměnná pro vícenásobné OLS regresní modely.

### 5. Firemní a sektorové atributy

Proměnné charakterizující vystavující entitu nezbytné jako kontroly v regresních modelech.

- **`company` a `rating`**: Název organizace a její zaměstnanecké hodnocení uvedené uživateli serveru Glassdoor (na škále 0-5).
- **`industry` a `sector`**: Původní oborové a sektorové klasifikace zaměstnavatele.
- **`sector_nace` a `sector_num`**: Pro zachování ekonomické standardizace byla původní kategorizace Glassdoor navázána a formálně zpracována do evropského formátu klasifikace ekonomických činností NACE Rev. 2.
- **`size` a `size_cat`**: Odhadovaný celkový počet zaměstnanců společnosti transformovaný pro modely do ordinálních kategorií.
- **`type` a `type_cat`**: Právní a organizační forma společnosti (např. Private, Public, Subsidiary, Nonprofit / Edu).
- **`year_founded`**: Rok vzniku dané organizace.
