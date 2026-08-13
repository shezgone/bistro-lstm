# -*- coding: utf-8 -*-
"""정보 공백 개념도 — 공식 통계 도착 시간표 vs 일별 신호 vs 사용 부품 (Q1 예시)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np, os

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "AppleGothic"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.dirname(os.path.abspath(__file__))
BLUE = "#1c5cab"; BLUE_M = "#9ec5f4"; BLUE_L = "#cde2fb"; WARM = "#b0532f"
INK = "#262626"; GREY = "#6b6b6b"; LINE = "#c9c9c9"; GREEN = "#008a3e"
GAP = "#f5e6e0"

fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=200)
ax.set_xlim(-19.8, 0.8); ax.set_ylim(-0.4, 5.4)
ax.axis("off")

# ── 시간축 구간 배경 (분기 전 / 분기 / 발표 대기)
for x0, x1, lab in [(-19, -17, "분기 시작 전\n2주"), (-17, -4, "대상 분기 13주  (예: 1~3월)"), (-4, 0, "발표 대기\n4주")]:
    ax.axvspan(x0, x1, ymin=0.02, ymax=0.10, color=BLUE_L if x0 != -17 else BLUE_M, alpha=0.9)
    ax.text((x0 + x1) / 2, 0.03, lab, ha="center", va="bottom", fontsize=8.5, color=INK)
ax.annotate("", xy=(0.6, 0.33), xytext=(-19.6, 0.33), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))
for w, lab in [(-19, "19주 전"), (-14, "14주 전"), (-9, "9주 전"), (-4, "4주 전"), (0, "속보 발표")]:
    ax.plot([w, w], [0.33, 0.48], color=GREY, lw=1)
    ax.text(w, 0.52, lab, ha="center", fontsize=8.5, color=WARM if w == 0 else GREY,
            fontweight="bold" if w in (0, -14) else "normal")

# ── Row 1: 공식 월간통계 도착 (약 1개월 시차)
y1 = 4.4
ax.text(-19.7, y1 + 0.42, "이 분기의 공식 월간통계  (산업생산·수출입 등 — 약 1개월 시차로 발표)", fontsize=10.5, color=INK, fontweight="bold")
# 정보 공백 음영
ax.axvspan(-19, -14, ymin=(y1 - 0.32 + 0.4) / 5.8, ymax=(y1 + 0.32 + 0.4) / 5.8, color=GAP)
arrivals = [(-13, "직전 분기\n12월치 도착"), (-8.5, "1월치 도착"), (-4, "2월치 도착"), (-0.5, "3월치 도착")]
for w, lab in arrivals:
    ax.plot(w, y1, marker="s", ms=13, color=BLUE, zorder=3)
    ax.text(w, y1 - 0.52, lab, ha="center", fontsize=8, color=BLUE)
ax.text(-16.5, y1, "0장", ha="center", va="center", fontsize=13, color=WARM, fontweight="bold")
ax.text(-16.5, y1 - 0.52, "정보 공백 — 새 공식 통계 없음", ha="center", fontsize=8.5, color=WARM, fontweight="bold")

# ── Row 2: 일별 신호
y2 = 3.0
ax.text(-19.7, y2 + 0.42, "일별 신호  (주가·환율·심리 — 매일 확정, 개정 없음)", fontsize=10.5, color=INK, fontweight="bold")
xs = np.arange(-19, 0.5, 0.5)
ax.plot(xs, y2 + 0.10 * np.sin(np.linspace(0, 9 * np.pi, len(xs))), color=GREEN, lw=1.8, zorder=3)
ax.text(0.4, y2, "매일 도착", fontsize=8.5, color=GREEN, va="center")

# ── Row 3: 사용 부품
y3 = 1.55
ax.text(-19.7, y3 + 0.47, "주차별 결합 — 그때그때 있는 정보를 읽을 수 있는 부품 사용", fontsize=10.5, color=INK, fontweight="bold")
ax.add_patch(mpatches.FancyBboxPatch((-19, y3 - 0.26), 5, 0.52, boxstyle="round,pad=0.02", fc=WARM, ec="none"))
ax.add_patch(mpatches.FancyBboxPatch((-13.9, y3 - 0.26), 13.9, 0.52, boxstyle="round,pad=0.02", fc=BLUE, ec="none"))
ax.text(-16.5, y3, "(GBM+Chronos-2f)÷2", ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
ax.text(-16.5, y3 - 0.56, "일별 신호를 읽는 부품", ha="center", fontsize=8, color=WARM)
ax.text(-7, y3, "XGBoost 단독 (현행 그대로)", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
ax.text(-7, y3 - 0.56, "공식 통계를 읽는 부품", ha="center", fontsize=8, color=BLUE)

# ── 경계 정렬 강조
ax.plot([-14, -14], [1.1, 5.0], color=GREY, lw=1.2, ls=(0, (5, 4)))
ax.annotate("교대 경계(14주 전) ≈ 첫 공식 통계가\n도착하기 시작하는 시점", xy=(-14, 4.85), xytext=(-11.3, 5.15),
            fontsize=9, color=INK, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=GREY, lw=1))

fig.tight_layout()
fig.savefig(f"{OUT}/info_gap.png", bbox_inches="tight", facecolor="white")
print("saved: info_gap.png")
