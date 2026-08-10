# -*- coding: utf-8 -*-
"""한은 종합 리포트 차트 — schema v2 리더보드 + 전망시계 진단"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "AppleGothic"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.dirname(os.path.abspath(__file__))
BLUE = "#1c5cab"; BLUE_M = "#9ec5f4"; BLUE_L = "#cde2fb"; WARM = "#b0532f"
INK = "#262626"; GREY = "#6b6b6b"; LINE = "#c9c9c9"; GREEN = "#008a3e"; GREY_L = "#d8d8d8"

# ---------- 차트 1: schema v2 리더보드 ----------
rows = [
    ("조기주차 (GBM+Chronos-2)/2\n→ 이후 XGBoost (당사 제안)",  0.740, GREEN),
    ("XGBoost 단독 (개정 후 기준선)",           0.750, BLUE),
    ("XGBoost+GBM 평균",                       0.752, BLUE_M),
    ("GBM 단독",                               0.757, BLUE_M),
    ("DFM+XGBoost 평균 (종전 최고 구성)",       0.787, BLUE_L),
    ("Chronos-2 + 일별신호 (zero-shot)",        0.808, BLUE_L),
    ("Chronos-2 (zero-shot)",                  0.836, BLUE_L),
    ("TTM few-shot / 합성 미세조정",            0.854, BLUE_L),
    ("DFM 단독",                               0.865, BLUE_L),
    ("LSTM (기준모형)",                         0.922, GREY_L),
]
fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=200)
ys = np.arange(len(rows))[::-1]
for y, (name, v, c) in zip(ys, rows):
    ax.barh(y, v, color=c, height=0.6, zorder=3)
    ax.text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=9.5, color=INK,
            fontweight="bold" if v in (0.740, 0.750) else "normal")
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9.2, color=INK)
ax.set_xlim(0.70, 0.96)
ax.xaxis.grid(True, color="#ececec", lw=0.8); ax.yaxis.grid(False)
for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(LINE)
ax.tick_params(colors=GREY, labelsize=8.5, length=3)
ax.set_axisbelow(True)
ax.text(0.746, ys[0] + 0.55, "-1.3% (통계적 유의성 미달 — 비열등+개선 방향)", fontsize=8.8, color=GREEN, fontweight="bold")
ax.set_title("주요 구성 리더보드 — 평균 오차 RMSE (낮을수록 정확) · 예측단위 개정(schema v2) 반영 · 동일 규약 32개 분기",
             fontsize=10, color=INK, loc="left", pad=12, fontweight="bold")
fig.savefig(f"{OUT}/bokf_leader.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- 차트 2: 전망시계(주차) 구간별 진단 ----------
buckets = ["조기\n(발표 19~14주 전)", "중반\n(13~8주 전)", "후반\n(7~1주 전)"]
data = {"XGBoost":  ([0.888, 0.711, 0.615], BLUE),
        "GBM":      ([0.869, 0.754, 0.615], BLUE_M),
        "Chronos-2f": ([0.869, 0.838, 0.682], WARM)}
fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=200)
x = np.arange(3); w = 0.26
for i, (name, (vals, c)) in enumerate(data.items()):
    bars = ax.bar(x + (i - 1) * w, vals, w - 0.02, color=c, zorder=3, label=name)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center",
                fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels(buckets, fontsize=9.5, color=INK)
ax.set_ylim(0.55, 0.95)
ax.yaxis.grid(True, color="#ececec", lw=0.8); ax.set_axisbelow(True)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: ax.spines[sp].set_color(LINE)
ax.tick_params(colors=GREY, labelsize=8.5, length=3)
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax.set_title("전망주차 구간별 RMSE — 조기 구간에서는 XGBoost가 최강이 아님", fontsize=10.5,
             color=INK, loc="left", pad=10, fontweight="bold")
ax.annotate("해당 분기 지표 공백 구간:\n사전학습 모델·GBM 우위", xy=(0 - w, 0.872), xytext=(-0.42, 0.93),
            fontsize=8.8, color=GREEN, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=GREEN, lw=1))
fig.savefig(f"{OUT}/bokf_horizon.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved: bokf_leader.png, bokf_horizon.png")
