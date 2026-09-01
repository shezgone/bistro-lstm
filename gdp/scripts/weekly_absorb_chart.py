# -*- coding: utf-8 -*-
"""주차별 RMSE — 빈티지 정보 흡수 비교 (이창훈 그림 1 형식 + Chronos-2f 추가)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, os

plt.rcParams["font.family"] = ["NanumSquare", "Apple SD Gothic Neo", "AppleGothic"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.dirname(os.path.abspath(__file__))
BLUE = "#1c5cab"; GREEN = "#008a3e"; WARM = "#b0532f"; GREY_C = "#9aa0a6"
INK = "#262626"; GREY = "#6b6b6b"; LINE = "#c9c9c9"

P = pd.read_csv("/Users/user/vibe/gdp-nowcasting-renewal/output/csv/_phase_b_weekly_profile.csv", index_col=0)
P = P.loc[-19:-8]   # TSFM 자체 예측 구간만

fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=200)
series = [("bfm", "BISTRO", WARM, "-", 2.2),
          ("moi", "Moirai-small", WARM, (0, (3, 2)), 1.3),
          ("c2f", "Chronos-2f", GREEN, "-", 2.2),
          ("dfm", "DFM", GREY_C, "-", 1.5),
          ("xgb", "XGBoost(신)", BLUE, "-", 2.0)]
for c, name, col, ls, lw in series:
    ax.plot(P.index, P[c], color=col, ls=ls, lw=lw, marker="o", ms=3.5,
            label=name, alpha=0.55 if c == "moi" else 1.0)
labels = {"bfm": ("BISTRO  -4.7% (평탄 — 새 정보 미흡수)", 0.0),
          "c2f": ("Chronos-2f  -9.3% (우하향 — 흡수)", 0.0),
          "xgb": ("XGBoost  -24.8%", 0.0)}
for c, (txt, dy) in labels.items():
    ax.annotate(txt, xy=(-8, P[c].iloc[-1]), xytext=(-7.8, P[c].iloc[-1] + dy),
                fontsize=9, color={"bfm": WARM, "c2f": GREEN, "xgb": BLUE}[c], fontweight="bold", va="center")
ax.set_xlim(-19.3, -3.4); ax.set_ylim(0.55, 1.12)
ax.set_xticks(range(-19, -7))
ax.set_xlabel("속보 발표까지 남은 주 (좌→우로 정보 축적)", fontsize=9.5, color=GREY)
ax.set_ylabel("주차별 RMSE", fontsize=9.5, color=GREY)
ax.yaxis.grid(True, color="#ececec", lw=0.8); ax.set_axisbelow(True)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: ax.spines[sp].set_color(LINE)
ax.tick_params(colors=GREY, labelsize=8.5, length=3)
ax.legend(loc="lower left", fontsize=8.5, frameon=False, ncol=2)
ax.set_title("주차별 RMSE — 새 빈티지 정보의 흡수 (TSFM 자체 예측 구간, w=-19~-8)",
             fontsize=11, color=INK, loc="left", pad=10, fontweight="bold")
fig.text(0.01, -0.02, "주: 32개 분기 평균, 감소율 = 초반 6주 대비 다음 6주. w>=-7은 분기말 월이 관측창에 들어와 "
                       "TSFM이 DFM 값을 그대로 쓰는 구간이라 제외. 괄호 % = 정보 흡수(우하향) 정도.", fontsize=7.5, color=GREY)
fig.savefig(f"{OUT}/weekly_absorb.png", bbox_inches="tight", facecolor="white")
print("saved: weekly_absorb.png")
