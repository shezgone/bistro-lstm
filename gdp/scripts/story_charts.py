# -*- coding: utf-8 -*-
"""상사 보고용 스토리 덱 차트 3장 — 라이트 배경 PNG"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, glob, os

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "AppleGothic"]
plt.rcParams["axes.unicode_minus"] = False

GDP = "/Users/user/vibe/gdp-nowcasting-renewal"
OUT = os.path.dirname(os.path.abspath(__file__))

BLUE = "#1c5cab"; BLUE_M = "#9ec5f4"; BLUE_L = "#cde2fb"; WARM = "#b0532f"
INK = "#262626"; GREY = "#6b6b6b"; LINE = "#c9c9c9"; GREEN = "#008a3e"

def style(ax):
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color(LINE)
    ax.tick_params(colors=GREY, labelsize=9, length=3)
    ax.yaxis.grid(True, color="#ececec", lw=0.8)
    ax.set_axisbelow(True)

# ---------- 차트 1: 2020년 시세판 vs 공식 심리지표 ----------
px = pd.read_parquet(f"{GDP}/data/fast_signals_daily.parquet")
k = px["kospi"][px.index <= "2020-08-15"]
mret = (k.resample("ME").last().pct_change() * 100).loc["2020-01":"2020-07"]
files = sorted(glob.glob(f"{GDP}/data/vintages/*.xlsx"))
vf = [f for f in files if os.path.basename(f)[:-5] <= "2020-08-15"][-1]
raw = pd.read_excel(vf)
if "date" in raw.columns: raw = raw.rename(columns={"date": "Date"})
raw["Date"] = pd.to_datetime(raw["Date"]); raw = raw.set_index("Date")
esi = raw["new_esi"].loc["2020-01":"2020-07"]
print("KOSPI 월수익률:", mret.round(1).to_dict())
print("ESI:", esi.round(1).to_dict())

fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 3.9), dpi=200, sharex=True,
                             gridspec_kw={"hspace": 0.42})
xs = np.arange(len(mret))
cols = [BLUE if v > 0 else WARM for v in mret.values]
a1.bar(xs, mret.values, color=cols, width=0.62, zorder=3)
a1.axhline(0, color=LINE, lw=1)
for i, v in enumerate(mret.values):
    a1.text(i, v + (0.9 if v > 0 else -0.9), f"{v:+.0f}%", ha="center",
            va="bottom" if v > 0 else "top", fontsize=8.5, color=INK,
            fontweight="bold" if abs(v) > 9 else "normal")
a1.set_title("시세판 — KOSPI 월간 수익률 (예측 시점에 당일 확인 가능)", fontsize=10,
             color=INK, loc="left", pad=8, fontweight="bold")
a1.set_ylim(-16, 17); style(a1)
a1.annotate("4월부터 넉 달 연속 반등", xy=(3, 11.6), xytext=(4.0, 14.5),
            fontsize=9, color=BLUE, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=1))

a2.plot(xs, esi.values, color=GREY, lw=2, marker="o", ms=5,
        markerfacecolor="white", markeredgecolor=GREY, zorder=3)
for i, v in enumerate(esi.values):
    a2.text(i, v - 4.5, f"{v:.0f}", ha="center", va="top", fontsize=8.5, color=GREY)
a2.set_title("공식 심리지표 — ESI (월 1회 발표, 100=평년)", fontsize=10,
             color=INK, loc="left", pad=8, fontweight="bold")
a2.set_ylim(38, 108); style(a2)
a2.annotate("4월에도 여전히 바닥", xy=(3, 57), xytext=(3.6, 88),
            fontsize=9, color=WARM, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=WARM, lw=1))
a2.set_xticks(xs); a2.set_xticklabels([f"{m}월" for m in range(1, 8)])
fig.text(0.01, 0.005, "2020년 1~7월, 2020-08-15 시점에 확보 가능했던 데이터만 사용", fontsize=7.5, color=GREY)
fig.savefig(f"{OUT}/story_c1.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- 차트 2: 원로+컨설턴트 평균의 함정 (2020Q3, 발표 16주 전) ----------
fig, ax = plt.subplots(figsize=(5.4, 3.5), dpi=200)
names = ["원로(DFM)\n단독", "원로+컨설턴트\n평균", "컨설턴트(C2f)\n단독"]
vals = [-0.50, -0.385, -0.27]
cols = [BLUE_M, WARM, BLUE]
bars = ax.bar(np.arange(3), vals, color=cols, width=0.52, zorder=3)
ax.axhline(0, color=LINE, lw=1)
ax.axhline(1.9, color=INK, lw=1.6, ls=(0, (5, 3)))
ax.text(2.32, 1.98, "실제 속보치 +1.9%", fontsize=9.5, color=INK, ha="right", fontweight="bold")
for i, v in enumerate(vals):
    ax.text(i, v - 0.09, f"{v:+.2f}%", ha="center", va="top", fontsize=9.5,
            color=INK, fontweight="bold")
for i, d in enumerate([2.40, 2.29, 2.17]):
    ax.text(i, 0.14, f"정답과 거리 {d:.2f}", ha="center", fontsize=8.2, color=GREY)
ax.annotate("섞었더니 컨설턴트\n단독보다 정답에서 멀어짐", xy=(1, -0.52), xytext=(0.62, -1.45),
            fontsize=9, color=WARM, fontweight="bold", ha="center")
ax.set_xticks(np.arange(3)); ax.set_xticklabels(names, fontsize=9.5, color=INK)
ax.set_ylim(-1.7, 2.4); style(ax)
ax.set_title("2020년 3분기 예측 — 발표 16주 전 시점", fontsize=10.5, color=INK,
             loc="left", pad=8, fontweight="bold")
ax.set_ylabel("전기비 성장률 예측 (%)", fontsize=8.5, color=GREY)
fig.savefig(f"{OUT}/story_c2.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- 차트 3: 성적표 계보 (RMSE, 낮을수록 정확) ----------
fig, ax = plt.subplots(figsize=(6.4, 3.1), dpi=200)
names = ["세 명의 위원회\n(DFM+XGB+C2f)/3", "원로+조수\nDFM+XGB (현행 최고)", "컨설턴트 단독\nChronos-2f", "원로 단독\nDFM"]
vals = [0.7457, 0.7650, 0.8075, 0.8650]
cols = [BLUE, BLUE_M, BLUE_L, BLUE_L]
ys = np.arange(4)[::-1]
ax.barh(ys, vals, color=cols, height=0.56, zorder=3)
for y, v, bold in zip(ys, vals, [True, False, False, False]):
    ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=10,
            color=INK, fontweight="bold" if bold else "normal")
ax.text(0.787, ys[0], "← 신기록, 현행 최고 대비 -2.5%", fontsize=10.5, color=GREEN,
        fontweight="bold", ha="left", va="center")
ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=9.5, color=INK)
ax.set_xlim(0.70, 0.90)
ax.xaxis.grid(True, color="#ececec", lw=0.8); ax.yaxis.grid(False)
for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(LINE)
ax.tick_params(colors=GREY, labelsize=9, length=3)
ax.set_axisbelow(True)
ax.set_title("8년(32개 분기) 실시간 재현 — 평균 오차 RMSE, 낮을수록 정확", fontsize=10.5,
             color=INK, loc="left", pad=8, fontweight="bold")
fig.savefig(f"{OUT}/story_c3.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved 3 charts →", OUT)
