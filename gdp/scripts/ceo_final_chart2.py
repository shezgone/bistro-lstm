# -*- coding: utf-8 -*-
"""대표 보고 차트 v2 — 성능 계보 (약자 풀이 병기, 쉬운 라벨)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

plt.rcParams["font.family"] = ["NanumSquare", "Apple SD Gothic Neo", "AppleGothic"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.dirname(os.path.abspath(__file__))
BLUE = "#1c5cab"; BLUE_M = "#9ec5f4"; GREEN = "#008a3e"; GREY_C = "#9aa0a6"
INK = "#262626"; GREY = "#6b6b6b"; LINE = "#c9c9c9"

fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=200)
labels = ["한은 현행 중심모형\n(DFM·동적요인모형)", "한은 개선 기준선\n(XGBoost·머신러닝)",
          "+ 분기 초 6주\n파운데이션 모델 결합", "+ 적응학습(LoRA)\n부품 — 최종 구성",
          "(참고) 국면전환 구성\n성능 최고, 채택 보류"]
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
ax.set_ylabel("평균 오차 (낮을수록 정확)", fontsize=9, color=GREY)
ax.yaxis.grid(True, color="#ececec", lw=0.8); ax.set_axisbelow(True)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: ax.spines[sp].set_color(LINE)
ax.tick_params(colors=GREY, labelsize=8.5, length=3)
ax.annotate("당사 기여  -2.3%", xy=(3, 0.733), xytext=(2.6, 0.795),
            fontsize=10, color=GREEN, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=GREEN, lw=1))
ax.set_title("성능 계보 — 같은 채점표(실시간 데이터 · 2018~2025년 32개 분기)로 잰 평균 오차(RMSE)",
             fontsize=10.5, color=INK, loc="left", pad=10, fontweight="bold")
fig.savefig(f"{OUT}/ceo_final2.png", bbox_inches="tight", facecolor="white")
print("saved: ceo_final2.png")
