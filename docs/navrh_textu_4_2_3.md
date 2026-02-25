### 4.2.3 Bottom-up klasifikace a čištění dat

Pro finální určení skutečných požadavků na AI dovednosti byl v souladu s metodologií Bone et al. (2025) uplatněn tzv. _bottom-up_ přístup. Tento postup kombinuje kontextové porozumění textu pomocí velkých jazykových modelů (LLM) s reálně a deterministicky identifikovanými AI dovednostmi v textu pracovní nabídky.

Následné čištění dat bylo zaměřeno na eliminaci nerelevantních "buzzwords". Pomocí regulárních výrazů byly odfiltrovány inzeráty obsahující převážně jen obecné pojmy (např. samotné slovo "AI" v názvu pozice či profilu společnosti) bez vazby na skutečné technické kompetence. Výsledkem této procedury je robustní binární indikátor reálné AI pozice (`has_ai_flag`). Pro umožnění standardizovaného srovnání napříč trhy byly nakonec izolované oborové rodiny (`job_family`) a ekonomické sektory namapovány na mezinárodně uznávanou klasifikaci ekonomických činností (NACE Rev. 2).
