# -*- coding: utf-8 -*-
"""FM 부품 비교 차트 — 동일 프로토콜 32분기 (조기 구간 진단 + 슬롯 교체 결과)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "AppleGothic"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.dirname(os.path.abspath(__file__))
BLUE = "#1c5cab"; BLUE_M = "#9ec5f4"; WARM = "#b0532f"; GREY_C = "#8a8a8a"
INK = "#262626"; GREY = "#6b6b6b"; LINE = "#c9c9c9"; GREEN = "#008a3e"

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.4), dpi=200)

# (a) 조기 구간 단독 — 슬롯 자격 진단
labels1 = ["GBM\n(파트너)", "Chronos-2f", "XGBoost\n(기준)", "Moirai\n-small", "직접학습\nLSTM"]
vals1 = [0.9243, 0.9238, 0.9460, 1.0206, 1.2861]
cols1 = [BLUE_M, GREEN, BLUE, WARM, GREY_C]
x = np.arange(5)
bars = a1.bar(x, vals1, 0.6, color=cols1, zorder=3)
for b_, v in zip(bars, vals1):
    a1.text(b_.get_x() + b_.get_width()/2, v + 0.012, f"{v:.3f}", ha="center", fontsize=8, color=INK,
            fontweight="bold" if v < 0.93 else "normal")
a1.axhline(0.9243, color=GREY, lw=1, ls=(0, (4, 3)), zorder=2)
a1.text(2.05, 1.28, "점선 = GBM 수준\n(슬롯 자격선)", fontsize=7.5, color=GREY, ha="center")
a1.set_xticks(x); a1.set_xticklabels(labels1, fontsize=7.8, color=INK)
a1.set_ylim(0.85, 1.35)
a1.set_title("① 조기 구간(-19~-14주) 단독 RMSE", fontsize=10, color=INK, loc="left", pad=8, fontweight="bold")

# (b) 슬롯 교체 결과 — 전체 32Q
labels2 = ["XGBoost\n단독(기준)", "슬롯=\nChronos-2f", "슬롯=\nMoirai", "슬롯=\n직접학습 LSTM"]
vals2 = [0.7499, 0.7399, 0.7536, 0.7768]
cols2 = [BLUE, GREEN, WARM, GREY_C]
x2 = np.arange(4)
bars = a2.bar(x2, vals2, 0.58, color=cols2, zorder=3)
for b_, v in zip(bars, vals2):
    a2.text(b_.get_x() + b_.get_width()/2, v + 0.0015, f"{v:.3f}", ha="center", fontsize=8.5, color=INK,
            fontweight="bold" if v in (0.7499, 0.7399) else "normal")
a2.axhline(0.7499, color=GREY, lw=1, ls=(0, (4, 3)), zorder=2)
a2.set_xticks(x2); a2.set_xticklabels(labels2, fontsize=7.8, color=INK)
a2.set_ylim(0.73, 0.785)
a2.set_title("② 조기 슬롯 교체 후 전체 성능 (32개 분기)", fontsize=10, color=INK, loc="left", pad=8, fontweight="bold")

for ax in (a1, a2):
    ax.yaxis.grid(True, color="#ececec", lw=0.8); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color(LINE)
    ax.tick_params(colors=GREY, labelsize=8, length=3)
fig.tight_layout()
fig.savefig(f"{OUT}/fm_parts.png", bbox_inches="tight", facecolor="white")
print("saved: fm_parts.png")
