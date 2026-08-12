# -*- coding: utf-8 -*-
"""1p 리포트 차트 — 동일 결합 규칙에서 Chronos-2f vs BISTRO 부품 교체 비교"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "AppleGothic"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.dirname(os.path.abspath(__file__))
BLUE = "#1c5cab"; BLUE_M = "#9ec5f4"; WARM = "#b0532f"
INK = "#262626"; GREY = "#6b6b6b"; LINE = "#c9c9c9"; GREEN = "#008a3e"

labels = ["XGBoost 단독\n(기준)", "조기 슬롯\nChronos-2f", "조기 슬롯\nBISTRO"]
tot = [0.8000, 0.7889, 0.8259]
rebd = [0.8502, 0.7891, 1.0055]
cols = [BLUE_M, GREEN, WARM]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.3), dpi=200)
for ax, vals, title, ylim in [(a1, tot, "전체 26개 분기", (0.74, 0.88)),
                              (a2, rebd, "경기 반등 분기 (6개 중 가용 5개)", (0.70, 1.10))]:
    x = np.arange(3)
    bars = ax.bar(x, vals, 0.56, color=cols, zorder=3)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + (ylim[1]-ylim[0])*0.015, f"{v:.3f}",
                ha="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.axhline(vals[0], color=GREY, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.8, color=INK)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color="#ececec", lw=0.8); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color(LINE)
    ax.tick_params(colors=GREY, labelsize=8, length=3)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8, fontweight="bold")
fig.suptitle("")
fig.tight_layout()
fig.savefig(f"{OUT}/c2f_vs_bistro.png", bbox_inches="tight", facecolor="white")
print("saved: c2f_vs_bistro.png")
