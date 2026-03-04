# Hodnocení výstupu a kvality analytických modelů ze Staty

Výstup ze Staty vypadá z metodologického hlediska **velmi profesionálně a kvalitně**. Modely běží, konvergují a jsou nastaveny způsobem, který přesně odpovídá standardům pro socioekonomický / ekonometrický výzkum (např. diplomová práce vysoké úrovně).

Zde je konkrétní rozbor toho, co je dobře uděláno a na co si případně dát pozor (z hlediska struktury a validity modelů):

## Co je uděláno skvěle (Best Practices)

1. **Transformace závislé proměnné:** Použití logaritmu platu (`ln_salary`) pro OLS modely je naprostý standard (Mincerova mzdová rovnice). Pomáhá to normalizovat rozdělení platu a koeficienty se dají interpretovat jako procentuální změny.
2. **Robustní standardní chyby:** U OLS modelů správně používáte `vce(robust)`. To je kriticky důležité pro průřezová data, aby se ošetřila heteroskedasticita (nestejný rozptyl chyb).
3. **Diagnostika modelů (VIF):** Velice chválím, že se po modelech počítá Variance Inflation Factor (`vif`). Všechny průměrné hodnoty jsou kolem 2 a žádná nepřekračuje kritickou mez (často udávanou jako 10). Tím padá obava z multikolinearity.
4. **Práce s kategorickými proměnnými (`i.var`):** Stata je správně instruována, co jsou faktory (např. `i.ai_level`, `i.edu_cat`).
5. **Marginální efekty u nelineárních modelů:** Logit a Multinomiální logit (mlogit) jsou těžké na interpretaci jen z koeficientů (Odds Ratios / RRR). Vy tam správně voláte `margins, dydx(*) atmeans`, což vyhodí přímo procentní body změn pravděpodobnosti. Toto často studenti zanedbávají a dělá to obrovský rozdíl v kvalitě analýzy!
6. **Effect size (Cohenovo d):** U prostého T-testu platu máte navíc dopočítané Cohenovo d (vyšlo 0.587). Opět skvělá analytická praxe – při cca 17 tisících pozorováních vyjde každý t-test statisticky signifikantní ($p < 0.001$), ale `d` dokazuje, že rozdíl má i nezanedbatelnou praktickou velikost (střední efekt).
7. **Base vs. Plný model (A a B):** Inkrementální obohacování modelu (nejprve bez individuálního vzdělání a seniority, a poté s ním) umožňuje pěkně sledovat, o kolik tyto faktory zlepší model. R-squared vám stouplo z 24.6 % na krásných **37.6 %**. Model teda funguje perfektně, $R^2$ kolem 0.35 je na reálná mzdová data vynikající hodnota.

## Co vzít v potaz / Drobné nuance

Tyto body nepředstavují "chyby", spíše věci pro interpretaci nebo mírná vylepšení:

### 1. LR Test po robustních odhadech (sekce 6.3)
V logu je vidět použití `lrtest`:
```stata
quietly regress ...
estimates store model_a_lr
quietly regress ...
estimates store model_b_lr
lrtest model_a_lr model_b_lr
```
Aby Stata provedla `lrtest` (Likelihood-Ratio test), provádíte ve skriptu `quietly regress ...` bez specifikace `vce(robust)`. Stata totiž neumožňuje čistý LR test nad robustními odhady.
Technicky čistší řešení by bylo použít ve velké regresi OLS s `vce(robust)` a pak spustit Waldův test na společnou významnost nových proměnných (často komandem `testparm i.edu_cat i.exp_category i.job_family_num`).
Nemusíte to přepisovat, LR test na standardních (nerobustních) chybách vám taky řekne, že ty proměnné tam patří (p-value je 0.000). Jen abyste věděl, proč byl odstraněn robust flag pře LR testem.

### 2. Logit a Mlogit modely mají velmi malé Pseudo $R^2$
* Logit model (predikce, zda job vyžaduje AI) má Pseudo $R^2 = 0.0142$ (tzn. 1.4 %).
* Mlogit model má $0.0124$ (1.2 %).

Tohle **není chyba modelů**, nestavte se k tomu primárně jako k chybě. Znamená to pouze toto: samotné proměnné jako lokace (remote), vzdělání a zkušenosti nevysvětlují téměř nic z toho, proč zrovna ta která IT pozice vyžaduje AI. Jsou to slabí prediktoři pro `has_ai`. Ty vlivy sice existují (vyjdou signifikantně díky obrovskému počtu pozorování), ale jsou velmi slabé. 

* **Doporučená interpretace do práce:** "Ačkoliv statistické modely nacházejí signifikantní rozdíly (např. Remote práce mírně zvyšuje šanci na AI pozici), celková vysvětlovací síla těchto demografických a lokačních faktorů je mizivá (Pseudo R^2 cca 1.4 %). O nasazení AI nástrojů a technologií tak pravděpodobně rozhodují primárně technické a strategické potřeby (sektor, konkrétní tech stack či business doména) samotné firmy, spíše než plošné ukazatele typu požadovaná formální úroveň vzdělání uchazeče."

### 3. Konvergence a strukturální integrita dat (Závěr empirické části)
Zásadním indikátorem robustnosti představených modelů je jejich ukázková konvergence (např. u složitějšího multinomiálního logit modelu již v rámci tří iterací). Ve statistické praxi toto rychlé dosažení globálního maxima (Maximum Likelihood Estimation) signifikantně vypovídá o vysoké strukturální čistotě databáze. 

Fakt, že algoritmy nenaráží na problémy s nekonkávním tvarem věrohodnostní funkce (non-concave log-likelihood) ani na chyby typu *perfect separation*, potvrzuje, že proměnné netrpí závažnou lineární závislostí a řídké kategorie (sparse data) byly správně agregovány. Absolutní absence výpočetních a iterativních abnormalit napříč všemi specifikacemi dodává prezentovaným výsledkům mimořádnou validitu a umožňuje jejich plnohodnotné využití pro testování výzkumných hypotéz.
