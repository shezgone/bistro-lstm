# -*- coding: utf-8 -*-
"""LoRA 학습 데이터 모양 다이어그램 — 학습 시리즈 1개의 생김새 + 연도별 재적응"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np, os

plt.rcParams["font.family"] = ["NanumSquare", "Apple SD Gothic Neo", "AppleGothic"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.dirname(os.path.abspath(__file__))
BLUE = "#1c5cab"; BLUE_L = "#cde2fb"; GREEN = "#008a3e"; WARM = "#b0532f"
INK = "#262626"; GREY = "#6b6b6b"; LINE = "#c9c9c9"; BG = "#f2f2f0"

fig = plt.figure(figsize=(7.6, 6.4), dpi=200)

# ── (상) 학습 시리즈 1개의 생김새 ──
ax = fig.add_axes([0.06, 0.44, 0.9, 0.5]); ax.axis("off")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.text(0, 97, "학습 시리즈 한 개의 생김새  (예: 2023년 1분기)", fontsize=12, color=INK, fontweight="bold")

rows = [("GDP 월별 경로 (N_gdp)", BLUE),
        ("공식지표 10종  (생산·수출·심리 ...)", "#6b8fb5"),
        ("빠른신호 4종  (주가·환율·심리 원지수)", GREEN),
        ("관측 플래그  (실제=0 / 추정=1)", GREY)]
x0, x1 = 8, 78          # 과거 구간
xl = 90                  # 라벨 위치(분기말)
y = 78
for name, c in rows:
    ax.add_patch(mp.FancyBboxPatch((x0, y-4), x1-x0, 8, boxstyle="round,pad=0.3", fc=c, ec="none", alpha=0.85))
    ax.text(x0-1.5, y, name, fontsize=8.5, color=INK, ha="right", va="center")
    y -= 13
# GDP 행에만 라벨 셀
ax.add_patch(mp.FancyBboxPatch((x1+3, 74), 12, 8, boxstyle="round,pad=0.3", fc=WARM, ec="none"))
ax.text(x1+9, 78, "라벨", fontsize=9, color="white", ha="center", va="center", fontweight="bold")
ax.annotate("분기말 달의 값만 실제 속보치로 교체 (=정답)", xy=(x1+9, 82.5), xytext=(30, 93),
            fontsize=8.5, color=WARM, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=WARM, lw=1))
# 시간축
ax.annotate("", xy=(97, 22), xytext=(5, 22), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))
for x, t in [(x0, "약 11년 전"), (x1, "예측 시작"), (x1+9, "분기말")]:
    ax.plot([x, x], [22, 25], color=GREY, lw=1)
    ax.text(x, 17, t, fontsize=8.5, color=GREY, ha="center")
ax.text((x0+x1)/2, 28, "과거 구간 (최대 128개월) — 그 시점에 알 수 있던 값만", fontsize=9, color=INK, ha="center")
ax.text(x1+9, 30, "6개월 예측", fontsize=8, color=WARM, ha="center")
ax.text(0, 4, "이런 시리즈가 분기당 1개씩 — 2024년 적응이라면 23개가 전부입니다.", fontsize=9.5, color=INK)

# ── (하) 연도별 재적응 ──
ax2 = fig.add_axes([0.06, 0.04, 0.9, 0.32]); ax2.axis("off")
ax2.set_xlim(0, 100); ax2.set_ylim(0, 100)
ax2.text(0, 92, "연도별로 새로 적응하고, 배운 적 없는 미래만 채점", fontsize=12, color=INK, fontweight="bold")
years = [("2021", 12), ("2022", 15), ("2023", 19), ("2024", 23), ("2025", 27)]
y0 = 62
for i, (yr, n) in enumerate(years):
    yy = y0 - i*13
    ax2.add_patch(mp.FancyBboxPatch((8, yy-4), 30, 8, boxstyle="round,pad=0.3", fc=BLUE_L, ec="none"))
    ax2.text(23, yy, f"~{int(yr)-1}년 분기 {n}개로 적응", fontsize=8, color=INK, ha="center", va="center")
    ax2.annotate("", xy=(46, yy), xytext=(39, yy), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1))
    ax2.add_patch(mp.FancyBboxPatch((47, yy-4), 22, 8, boxstyle="round,pad=0.3", fc=GREEN, ec="none", alpha=0.9))
    ax2.text(58, yy, f"{yr}년 4개 분기 예측", fontsize=8, color="white", ha="center", va="center", fontweight="bold")
ax2.text(74, y0, "매년 원본 모델에서\n다시 시작 (누적 없음)", fontsize=8.5, color=GREY, va="center")
ax2.text(74, y0-26, "학습에 쓴 분기는\n채점에 절대 안 들어감", fontsize=8.5, color=GREY, va="center")
fig.savefig(f"{OUT}/lora_dataset.png", bbox_inches="tight", facecolor="white")
print("saved: lora_dataset.png")
