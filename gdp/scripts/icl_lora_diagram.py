# -*- coding: utf-8 -*-
"""대표 덱 — 파운데이션 모델 사용법 2가지 (즉석 사용 ICL vs 가벼운 적응 LoRA) 개념도"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import os

plt.rcParams["font.family"] = ["NanumSquare", "Apple SD Gothic Neo", "AppleGothic"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.dirname(os.path.abspath(__file__))
BLUE = "#1c5cab"; BLUE_BG = "#e8f1fb"; GREEN = "#008a3e"; WARM = "#b0532f"
INK = "#262626"; GREY = "#6b6b6b"; BODY = "#e6e6e3"; LINE = "#c9c9c9"

fig = plt.figure(figsize=(7.5, 5.2), dpi=200)
ax = fig.add_axes([0.005, 0.005, 0.99, 0.99]); ax.axis("off")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)

def box(x, y, w, h, fc, ec="none", lw=1.0, ls="-"):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                   fc=fc, ec=ec, lw=lw, ls=ls))

def arrow(x0, y0, x1, y1, c=GREY, lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw))

# ── 공통 입력 ──
box(5, 86, 90, 12, BLUE_BG, BLUE, 1.2)
ax.text(50, 94.6, "들어가는 데이터 — 두 방법 모두 동일", fontsize=10, ha="center", color=BLUE, fontweight="bold")
ax.text(50, 91.0, "월별 GDP 경로 · 공식지표 10종(생산·수출·심리 …) · 주가·환율 등 일별 빠른신호 4종", fontsize=8.4, ha="center", color=INK)
ax.text(50, 87.7, "— 각 예측 시점에 \"그때 실제로 알 수 있었던\" 값만 (미래 정보 차단)", fontsize=8.2, ha="center", color=GREY)

arrow(27, 85.4, 27, 81)
arrow(67, 85.4, 67, 81)

# ── 열 제목 ──
ax.text(27, 77.3, "방법 1 · 즉석 사용", fontsize=10.5, ha="center", color=INK, fontweight="bold")
ax.text(27, 73.8, "(ICL · In-Context Learning · 문맥 내 학습)", fontsize=8, ha="center", color=GREY)
ax.text(70, 77.3, "방법 2 · 가벼운 적응", fontsize=10.5, ha="center", color=INK, fontweight="bold")
ax.text(70, 73.8, "(LoRA · Low-Rank Adaptation · 저계수 적응)", fontsize=8, ha="center", color=GREY)

# ── 모델 상자: ICL ──
box(6, 56, 42, 14, BODY, GREY, 1.0, (0, (4, 3)))
ax.text(27, 64.3, "사전학습 모델 본체 — 그대로", fontsize=9.5, ha="center", color=INK, fontweight="bold")
ax.text(27, 59.6, "약 1억 개 파라미터 · 학습하지 않음 (동결)", fontsize=8, ha="center", color=GREY)

# ── 모델 상자: LoRA = 동결 본체 + 우회 회로 ──
box(52, 56, 29, 14, BODY, GREY, 1.0, (0, (4, 3)))
ax.text(66.5, 64.3, "본체 — 동결", fontsize=9.5, ha="center", color=INK, fontweight="bold")
ax.text(66.5, 59.6, "99.4%는 그대로", fontsize=8, ha="center", color=GREY)
ax.text(83, 62.5, "+", fontsize=13, ha="center", color=INK, fontweight="bold")
box(85, 56, 10.5, 14, "#f9ece5", WARM, 1.2)
ax.text(90.2, 65.0, "우회 회로", fontsize=8.3, ha="center", color=WARM, fontweight="bold")
ax.text(90.2, 61.5, "0.6%만", fontsize=8.3, ha="center", color=WARM, fontweight="bold")
ax.text(90.2, 58.2, "학습", fontsize=8.3, ha="center", color=WARM, fontweight="bold")

# LoRA 학습 재료 (우회 회로로 올라가는 화살표)
arrow(90.2, 47.5, 90.2, 55.2, WARM)
ax.text(90.2, 44.7, "과거 분기 사례", fontsize=7.8, ha="center", color=WARM)
ax.text(90.2, 41.6, "12~27개로 연습", fontsize=7.8, ha="center", color=WARM)

# ── 예측 상자 ──
arrow(27, 55.4, 27, 45.6)
arrow(66.5, 55.4, 66.5, 45.6)
box(16, 37.5, 22, 7.5, "white", GREEN, 1.3)
ax.text(27, 41.3, "즉석 예측", fontsize=9.5, ha="center", color=GREEN, fontweight="bold")
box(55.5, 37.5, 22, 7.5, "white", GREEN, 1.3)
ax.text(66.5, 41.3, "적응된 예측", fontsize=9.5, ha="center", color=GREEN, fontweight="bold")

# ── 캡션 ──
ax.text(6, 31.0, "과거 흐름을 보여주면 그 자리에서 이어 그림 — 준비 0분", fontsize=8.4, color=INK)
ax.text(6, 26.8, "장점: 바로 쓸 수 있고 관리가 단순함", fontsize=8.4, color=INK)
ax.text(6, 22.6, "한계: 새 지표가 발표돼도 예측을 잘 바꾸지 않음 (한은 지적)", fontsize=8.4, color=WARM)

ax.text(52, 31.0, "\"분기말 실제 값을 맞혀 보라\"를 500번 연습 — 한 번에 몇 분", fontsize=8.4, color=INK)
ax.text(52, 26.8, "효과: 새 지표 반영이 살아나고, 오차 6~9% 개선", fontsize=8.4, color=INK)
ax.text(52, 22.6, "규칙: 사례가 12분기 넘게 쌓인 뒤에만 사용 (과적합 방지)", fontsize=8.4, color=GREEN)

# ── 하단 요약 ──
box(5, 5, 90, 11, "#f5f5f4")
ax.text(50, 12.2, "운영 구성에서는 두 방법이 교대합니다", fontsize=8.8, ha="center", color=INK, fontweight="bold")
ax.text(50, 8.0, "적응 사례가 부족한 시기 = 방법 1(즉석 사용)  →  사례가 12분기 넘게 쌓이면 = 방법 2(가벼운 적응)", fontsize=8.6, ha="center", color=INK)

fig.savefig(f"{OUT}/icl_lora.png", bbox_inches="tight", facecolor="white")
print("saved: icl_lora.png")
