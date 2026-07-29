# -*- coding: utf-8 -*-
"""한은 공식 보고 덱 차트 — 중립 톤, 라이트 배경"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, glob, os

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "AppleGothic"]
plt.rcParams["axes.unicode_minus"] = False

GDP = "/Users/user/vibe/gdp-nowcasting-renewal"
OUT = os.path.dirname(os.path.abspath(__file__))
BLUE = "#1c5cab"; BLUE_M = "#9ec5f4"; BLUE_L = "#cde2fb"; WARM = "#b0532f"
INK = "#262626"; GREY = "#6b6b6b"; LINE = "#c9c9c9"; GREEN = "#008a3e"; GREY_L = "#d8d8d8"

# ---------- 차트 A: 검증 리더보드 ----------
rows = [
    ("국면 조건부 결합 v3 (연구 보류)", 0.718, GREY_L),
    ("상시 3자 결합 (DFM+XGB+Chronos-2f)/3", 0.746, BLUE),
    ("현행 시스템 (DFM+XGBoost)", 0.765, BLUE_M),
    ("Chronos-2 + 일별신호 (zero-shot)", 0.808, BLUE_L),
    ("DFM+TabPFN 결합", 0.815, BLUE_L),
    ("Chronos-2 (zero-shot)", 0.836, BLUE_L),
    ("TTM (few-shot 미세조정)", 0.854, BLUE_L),
    ("DFM 단독", 0.865, BLUE_L),
]
fig, ax = plt.subplots(figsize=(7.0, 3.9), dpi=200)
ys = np.arange(len(rows))[::-1]
for y, (name, v, c) in zip(ys, rows):
    ax.barh(y, v, color=c, height=0.58, zorder=3)
    ax.text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=9.5, color=INK,
            fontweight="bold" if v in (0.746, 0.765) else "normal")
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9.3, color=INK)
ax.set_xlim(0.70, 0.90)
ax.xaxis.grid(True, color="#ececec", lw=0.8); ax.yaxis.grid(False)
for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(LINE)
ax.tick_params(colors=GREY, labelsize=8.5, length=3)
ax.set_axisbelow(True)
ax.text(0.752, ys[1] + 0.52, "현행 대비 -2.5% (비열등 확인)", fontsize=9, color=GREEN, fontweight="bold")
ax.set_title("주요 구성 검증 결과 — 평균 오차 RMSE (낮을수록 정확), 동일 규약", fontsize=10.5,
             color=INK, loc="left", pad=10, fontweight="bold")
fig.savefig(f"{OUT}/bok_leader.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- 차트 B: 2020년 사례 (중립 톤) ----------
px = pd.read_parquet(f"{GDP}/data/fast_signals_daily.parquet")
k = px["kospi"][px.index <= "2020-08-15"]
mret = (k.resample("ME").last().pct_change() * 100).loc["2020-01":"2020-07"]
files = sorted(glob.glob(f"{GDP}/data/vintages/*.xlsx"))
vf = [f for f in files if os.path.basename(f)[:-5] <= "2020-08-15"][-1]
raw = pd.read_excel(vf)
if "date" in raw.columns: raw = raw.rename(columns={"date": "Date"})
raw["Date"] = pd.to_datetime(raw["Date"]); raw = raw.set_index("Date")
esi = raw["new_esi"].loc["2020-01":"2020-07"]

def style(ax):
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color(LINE)
    ax.tick_params(colors=GREY, labelsize=9, length=3)
    ax.yaxis.grid(True, color="#ececec", lw=0.8)
    ax.set_axisbelow(True)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 3.9), dpi=200, sharex=True,
                             gridspec_kw={"hspace": 0.42})
xs = np.arange(len(mret))
cols = [BLUE if v > 0 else WARM for v in mret.values]
a1.bar(xs, mret.values, color=cols, width=0.62, zorder=3)
a1.axhline(0, color=LINE, lw=1)
for i, v in enumerate(mret.values):
    a1.text(i, v + (0.9 if v > 0 else -0.9), f"{v:+.0f}%", ha="center",
            va="bottom" if v > 0 else "top", fontsize=8.5, color=INK)
a1.set_title("일별 금융지표 — KOSPI 월간 수익률 (당일 확정 · 무개정)", fontsize=10,
             color=INK, loc="left", pad=8, fontweight="bold")
a1.set_ylim(-16, 17); style(a1)
a1.annotate("4월부터 4개월 연속 반등", xy=(3, 11.6), xytext=(4.0, 14.5),
            fontsize=9, color=BLUE, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=1))
a2.plot(xs, esi.values, color=GREY, lw=2, marker="o", ms=5,
        markerfacecolor="white", markeredgecolor=GREY, zorder=3)
for i, v in enumerate(esi.values):
    a2.text(i, v - 4.5, f"{v:.0f}", ha="center", va="top", fontsize=8.5, color=GREY)
a2.set_title("경제심리지수(ESI) 원지수 — 월 1회 발표 (100=평년)", fontsize=10,
             color=INK, loc="left", pad=8, fontweight="bold")
a2.set_ylim(38, 108); style(a2)
a2.annotate("4월 저점 55.7", xy=(3, 57), xytext=(3.8, 88),
            fontsize=9, color=WARM, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=WARM, lw=1))
a2.set_xticks(xs); a2.set_xticklabels([f"{m}월" for m in range(1, 8)])
fig.text(0.01, 0.005, "2020년 1~7월 — 2020-08-15 시점에 확보 가능했던 데이터만 사용 (실시간 재현)", fontsize=7.5, color=GREY)
fig.savefig(f"{OUT}/bok_case2020.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved: bok_leader.png, bok_case2020.png")
