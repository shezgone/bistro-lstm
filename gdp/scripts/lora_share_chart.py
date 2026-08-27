# -*- coding: utf-8 -*-
"""이과장 공유용 차트 — LoRA 적응 전후 주차별 RMSE (BISTRO / Chronos-2, 3-seed 앙상블)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, os

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "AppleGothic"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.dirname(os.path.abspath(__file__))
BLUE = "#1c5cab"; GREEN = "#008a3e"; WARM = "#b0532f"; GREY_C = "#9aa0a6"
INK = "#262626"; GREY = "#6b6b6b"; LINE = "#c9c9c9"

P = pd.read_csv("/Users/user/vibe/gdp-nowcasting-renewal/output/csv/_phase_b_lora_weekly_profile.csv", index_col=0)
P = P.loc[-19:-8]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.5), dpi=200, sharey=True)
for ax, zs, lo, title in [(a1, "bzs", "blo", "BISTRO (BIS WP 1337)"),
                          (a2, "czs", "clo", "Chronos-2f")]:
    ax.plot(P.index, P["xgb"], color=GREY_C, lw=1.4, marker="o", ms=3, label="XGBoost (참고)")
    ax.plot(P.index, P[zs], color=WARM, lw=2.0, marker="o", ms=3.5, label="zero-shot")
    ax.plot(P.index, P[lo], color=GREEN, lw=2.2, marker="o", ms=3.5, label="LoRA 적응 (3-seed 평균)")
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8, fontweight="bold")
    ax.set_xlim(-19.4, -7.6); ax.set_ylim(0.42, 0.78)
    ax.set_xticks(range(-19, -7, 2))
    ax.yaxis.grid(True, color="#ececec", lw=0.8); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color(LINE)
    ax.tick_params(colors=GREY, labelsize=8.5, length=3)
    ax.set_xlabel("속보 발표까지 남은 주", fontsize=8.5, color=GREY)
a1.set_ylabel("주차별 RMSE", fontsize=8.5, color=GREY)
a1.legend(loc="upper left", fontsize=8, frameon=False)
a1.annotate("적응 후: 레벨 하락 +\n평탄성 부분 해소", xy=(-11, P.loc[-11, "blo"]), xytext=(-13.5, 0.47),
            fontsize=8.5, color=GREEN, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=GREEN, lw=1))
a2.annotate("적응 후: 레벨 하락,\n흡수 기울기는 유지", xy=(-11, P.loc[-11, "clo"]), xytext=(-13.5, 0.47),
            fontsize=8.5, color=GREEN, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=GREEN, lw=1))
fig.tight_layout()
fig.savefig(f"{OUT}/lora_share.png", bbox_inches="tight", facecolor="white")
print("saved: lora_share.png")
