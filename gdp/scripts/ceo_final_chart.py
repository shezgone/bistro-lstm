# -*- coding: utf-8 -*-
"""대표 보고 차트 — 여정의 기록 경신 계보 (32분기 동일 규약)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "AppleGothic"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.dirname(os.path.abspath(__file__))
BLUE = "#1c5cab"; BLUE_M = "#9ec5f4"; GREEN = "#008a3e"; WARM = "#b0532f"; GREY_C = "#9aa0a6"
INK = "#262626"; GREY = "#6b6b6b"; LINE = "#c9c9c9"

fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=200)
labels = ["DFM 단독\n(현행 중심)", "XGBoost 단독\n(개정 후 기준선)", "+ 조기 6주\nFM 결합 (7~8월)", "+ 적응 임계 규칙\nLoRA 부품 (8월말)", "(참고) 국면전환 v3\n— 채택 보류"]
vals = [0.865, 0.750, 0.740, 0.733, 0.718]
cols = [GREY_C, BLUE, BLUE_M, GREEN, "#d8d8d8"]
x = np.arange(5)
bars = ax.bar(x, vals, 0.58, color=cols, zorder=3)
for b_, v, bold in zip(bars, vals, [0, 1, 0, 1, 0]):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center",
            fontsize=10, color=INK, fontweight="bold" if bold else "normal")
ax.axhline(0.750, color=BLUE, lw=1, ls=(0, (4, 3)), zorder=2, alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.6, color=INK)
ax.set_ylim(0.68, 0.90)
ax.set_ylabel("평균 오차 RMSE (낮을수록 정확)", fontsize=9, color=GREY)
ax.yaxis.grid(True, color="#ececec", lw=0.8); ax.set_axisbelow(True)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: ax.spines[sp].set_color(LINE)
ax.tick_params(colors=GREY, labelsize=8.5, length=3)
ax.annotate("당사 기여분  -2.3%", xy=(3, 0.733), xytext=(2.6, 0.795),
            fontsize=10, color=GREEN, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=GREEN, lw=1))
ax.set_title("성능 계보 — 동일 규약(실시간 빈티지·속보치·32개 분기) 채점", fontsize=10.5,
             color=INK, loc="left", pad=10, fontweight="bold")
fig.savefig(f"{OUT}/ceo_final.png", bbox_inches="tight", facecolor="white")
print("saved: ceo_final.png")
