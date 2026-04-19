"""Quick-and-dirty mockups for 7 proposed thesis charts.
Values are approximations from prakticka_cast.md / prakticka_cast_3.md — illustrative only.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

# ---------------------------------------------------------------------------
# 1) Stacked bar: AI tier composition per country
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.4))
countries = ["USA", "Německo", "Indie"]
none = np.array([79.4, 81.7, 93.7])
integ = np.array([13.2, 9.7, 3.5])
appl = np.array([7.4, 8.6, 2.8])
ax.bar(countries, none, label="None", color="#cfd8dc")
ax.bar(countries, integ, bottom=none, label="AI Integration", color="#4a78b8")
ax.bar(countries, appl, bottom=none + integ, label="Applied/Core AI", color="#b84a4a")
for i, c in enumerate(countries):
    ax.text(i, none[i] + integ[i] / 2, f"{integ[i]}%", ha="center", va="center", color="white", fontsize=8)
    ax.text(i, none[i] + integ[i] + appl[i] / 2, f"{appl[i]}%", ha="center", va="center", color="white", fontsize=8)
ax.set_ylabel("Podíl inzerátů (%)")
ax.set_ylim(60, 101)
ax.set_title("1 — Složení AI požadavků podle země")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "01_ai_tier_by_country.png")
plt.close()

# ---------------------------------------------------------------------------
# 2) Horizontal bar: AI share by job family (US)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.6))
jf = ["Data & AI", "Sr. Software Engineer", "Software Engineer",
      "Management", "DevOps & Cloud", "Software Developer", "Other"]
share = [55.7, 33.3, 26.8, 22.0, 17.8, 10.5, 8.0]  # last 3 = odhad
colors = ["#b84a4a" if s > 20.6 else "#4a78b8" for s in share]
y = np.arange(len(jf))
ax.barh(y, share, color=colors)
ax.set_yticks(y)
ax.set_yticklabels(jf)
ax.invert_yaxis()
ax.axvline(20.6, color="gray", linestyle="--", linewidth=1)
ax.text(20.6, -0.6, "průměr 20.6%", color="gray", fontsize=8)
for i, s in enumerate(share):
    ax.text(s + 0.5, i, f"{s}%", va="center", fontsize=8)
ax.set_xlabel("Podíl pozic s AI požadavkem (%)")
ax.set_title("2 — AI požadavek podle profesní skupiny (US)")
plt.tight_layout()
plt.savefig(OUT / "02_ai_share_by_jobfamily.png")
plt.close()

# ---------------------------------------------------------------------------
# 3) Forest plot: binary logit AMEs (US skill clusters)
# ---------------------------------------------------------------------------
clusters = [
    ("Dynamic Web", 9.4), ("Cloud Computing", 6.6), ("Enterprise Platforms", 5.0),
    ("Data Engineering", 4.5), ("Frontend Development", 3.6), ("BI & Analytics", 3.5),
    ("Systems Programming", 3.3), ("DevOps / Containers", 2.6), ("Backend Development", 2.3),
    ("Architecture & Methods", 0.4), ("Security & Identity", -0.6),
    ("Testing / QA", -1.2), ("Mobile / Desktop", -1.4), ("Networking", -2.1),
    ("OS / Embedded", -3.3), ("Databases / Storage", -3.4),
    ("Certifications", -3.8), ("Scripting / Shell", -5.6),
    ("Enterprise / Managed", -6.5),
]
clusters.sort(key=lambda x: x[1])
labels, values = zip(*clusters)
colors = ["#b84a4a" if v < 0 else "#4a78b8" for v in values]
fig, ax = plt.subplots(figsize=(6.5, 5.2))
y = np.arange(len(labels))
ax.barh(y, values, color=colors)
# fake CI whiskers
for i, v in enumerate(values):
    ax.plot([v - 1.0, v + 1.0], [i, i], color="black", linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("AME (procentní body)")
ax.set_title("3 — Binární logit: vliv skill clusterů na P(AI) (US)")
plt.tight_layout()
plt.savefig(OUT / "03_logit_ame_us.png")
plt.close()

# ---------------------------------------------------------------------------
# 4) Dumbbell: mlogit Integration vs Applied/Core AI (US)
# ---------------------------------------------------------------------------
rows = [
    ("Enterprise Platforms", 5.8, 0.0),
    ("Frontend Development", 4.9, 0.0),
    ("Cloud Computing", 4.3, 2.4),
    ("Dynamic Web", 4.4, 5.1),
    ("Backend Development", 2.2, 0.0),
    ("BI & Analytics", 2.0, 1.3),
    ("DevOps / Containers", 1.6, 1.0),
    ("Data Engineering", 0.2, 4.2),
    ("Systems Programming", -1.7, 4.2),
    ("Scripting / Shell", -3.1, -2.5),
    ("Enterprise / Managed", -4.2, -2.7),
]
rows.sort(key=lambda r: r[1] - r[2])
labels, ints, apps = zip(*rows)
y = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(6.8, 4.4))
for i, (lab, iv, av) in enumerate(rows):
    ax.plot([iv, av], [i, i], color="gray", linewidth=1.2, zorder=1)
ax.scatter(ints, y, color="#4a78b8", s=55, zorder=2, label="AI Integration")
ax.scatter(apps, y, color="#b84a4a", s=55, zorder=2, label="Applied/Core AI")
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("AME (procentní body)")
ax.set_title("4 — Mlogit: Integration vs Applied/Core AI (US)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "04_mlogit_dumbbell_us.png")
plt.close()

# ---------------------------------------------------------------------------
# 5) Heatmap: AME skill clusters × countries  (varianta B — ns = šrafování)
# ---------------------------------------------------------------------------
# Hodnoty přesně z Tabulky 3 v prakticka_cast_3.md; sig = True/False (ns).
rows_data = [
    # (label, [US, DE, IN], [sig_US, sig_DE, sig_IN])
    ("Generative AI",        [ 32.8, 36.2, 12.3], [True, True, True]),
    ("Data Science / ML",    [ 28.4, 16.6,  7.1], [True, True, True]),
    ("Dynamic Web",          [  5.6,  7.2,  2.8], [True, True, True]),
    ("Cloud Computing",      [  4.1,  2.9,  1.1], [True, True, True]),
    ("Data Engineering",     [  1.8,  4.6,  0.7], [True, True, True]),
    ("BI & Analytics",       [  2.5,  3.7,  0.9], [True, True, True]),
    ("Frontend Dev",         [  2.8,  0.7,  0.5], [True, False, False]),
    ("Enterprise Platforms", [  2.8,  2.1,  0.3], [True, False, False]),
    ("DevOps & Containers",  [  0.6,  2.5,  0.0], [False, True, False]),
    ("Backend Dev",          [  0.9,  2.3, -0.1], [False, True, False]),
    ("Systems Programming",  [  1.9,  1.4,  0.2], [True, False, False]),
    ("Architecture & Methods", [ 0.6,  0.5,  0.1], [False, False, False]),
    ("Security & Identity",  [ -1.2, -2.4,  0.6], [False, True, False]),
    ("Mobile & Desktop",     [ -0.3, -0.6, -1.8], [False, False, True]),
    ("Testing / QA",         [ -1.0, -1.4, -0.1], [True, False, False]),
    ("Databases & Storage",  [ -1.7, -1.9, -0.6], [True, True, False]),
    ("Networking",           [ -0.5, -4.1, -0.6], [False, True, False]),
    ("OS & Embedded",        [ -2.4, -4.4, -2.0], [True, True, True]),
    ("Certifications",       [ -3.0, -5.4, -0.6], [True, True, False]),
    ("Scripting / Shell",    [ -3.2, -5.2, -1.6], [True, True, True]),
    ("Enterprise / Managed", [ -4.5, -2.6, -0.3], [True, True, False]),
]
rows = [r[0] for r in rows_data]
data = np.array([r[1] for r in rows_data])
sig = np.array([r[2] for r in rows_data])
cols = ["USA", "Německo", "Indie"]
vmax = max(abs(data.min()), abs(data.max()))

fig, ax = plt.subplots(figsize=(5.2, 7.0))
im = ax.imshow(data, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")

# Šrafování přes ns buňky (varianta B)
from matplotlib.patches import Rectangle
for i in range(len(rows)):
    for j in range(len(cols)):
        if not sig[i, j]:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   fill=False, hatch="////",
                                   edgecolor="black", linewidth=0, alpha=0.55))

ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(rows)

for i in range(len(rows)):
    for j in range(len(cols)):
        v = data[i, j]
        txt = f"{v:+.1f}"
        ax.text(j, i, txt, ha="center", va="center",
                color="white" if abs(v) > vmax * 0.55 else "black", fontsize=7,
                fontweight="bold" if sig[i, j] else "normal")

ax.set_title("5 — AME skill clusterů napříč zeměmi\n(šrafované buňky = nesignifikantní, p ≥ 0,05)", fontsize=10)
plt.colorbar(im, ax=ax, shrink=0.7, label="AME (p.b.)")

# Legenda pro šrafování
from matplotlib.patches import Patch
leg = [Patch(facecolor="white", edgecolor="black", hatch="////", label="ns (p ≥ 0,05)")]
ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0, -0.12), fontsize=8, frameon=False)

plt.tight_layout()
plt.savefig(OUT / "05_heatmap_crosscountry.png")
plt.close()

# ---------------------------------------------------------------------------
# 6) Waterfall / grouped bars: AI premium decomposition (US)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.6))
models = ["Model A\n(firma)", "Model B\n(+ lidský kapitál)", "Model C\n(+ dovednosti, JF)"]
integ = [11.8, 10.6, 8.6]
appl = [17.7, 16.4, 11.7]
x = np.arange(len(models))
w = 0.35
b1 = ax.bar(x - w/2, integ, w, label="AI Integration", color="#4a78b8")
b2 = ax.bar(x + w/2, appl, w, label="Applied/Core AI", color="#b84a4a")
ax.plot(x - w/2, integ, "o-", color="#2a4a80", linewidth=1)
ax.plot(x + w/2, appl, "o-", color="#802a2a", linewidth=1)
for bar, v in zip(b1, integ):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v}%", ha="center", fontsize=8)
for bar, v in zip(b2, appl):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v}%", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("AI mzdová prémie (%)")
ax.set_ylim(0, 22)
ax.set_title("6 — Dekompozice AI prémie: Model A → B → C (US)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "06_premium_decomposition.png")
plt.close()

# ---------------------------------------------------------------------------
# 7) Cross-country AI premium (bar with CI, ns = hatched)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 3.6))
countries = ["USA", "Německo", "Indie"]
integ = [11.5, 3.1, 9.0]
integ_err = [1.2, 2.8, 3.5]
appl = [16.2, 7.4, 11.7]
appl_err = [1.5, 4.0, 3.8]
sig_int = [True, False, True]
sig_app = [True, False, True]
x = np.arange(len(countries))
w = 0.35
for i, (v, e, sig) in enumerate(zip(integ, integ_err, sig_int)):
    ax.bar(i - w/2, v, w, yerr=e, color="#4a78b8", alpha=1.0 if sig else 0.35,
           hatch="" if sig else "//", edgecolor="black", capsize=4)
for i, (v, e, sig) in enumerate(zip(appl, appl_err, sig_app)):
    ax.bar(i + w/2, v, w, yerr=e, color="#b84a4a", alpha=1.0 if sig else 0.35,
           hatch="" if sig else "//", edgecolor="black", capsize=4)
for i, (iv, av) in enumerate(zip(integ, appl)):
    ax.text(i - w/2, iv + 0.4, f"{iv}%", ha="center", fontsize=8)
    ax.text(i + w/2, av + 0.4, f"{av}%", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(countries)
ax.set_ylabel("AI prémie (%)")
ax.set_title("7 — AI mzdová prémie napříč zeměmi (šrafované = ns)")
# fake legend
from matplotlib.patches import Patch
leg = [Patch(facecolor="#4a78b8", label="AI Integration"),
       Patch(facecolor="#b84a4a", label="Applied/Core AI")]
ax.legend(handles=leg, fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "07_premium_crosscountry.png")
plt.close()

print("Hotovo — 7 mockups uloženo v:", OUT)
