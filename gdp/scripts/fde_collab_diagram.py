# -*- coding: utf-8 -*-
"""대표 덱 — FDE 협업 구조도: 저장소를 가운데 둔 양방향 커밋 + 상호 재현 검증"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import os

plt.rcParams["font.family"] = ["NanumSquare", "Apple SD Gothic Neo", "AppleGothic"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.dirname(os.path.abspath(__file__))
BLUE = "#1c5cab"; GREEN = "#008a3e"; WARM = "#b0532f"
INK = "#262626"; GREY = "#6b6b6b"; LINE = "#c9c9c9"

fig = plt.figure(figsize=(7.5, 4.4), dpi=200)
ax = fig.add_axes([0.005, 0.005, 0.99, 0.99]); ax.axis("off")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)

def box(x, y, w, h, fc, ec="none", lw=1.0):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                   fc=fc, ec=ec, lw=lw))

# ── 좌: 한국은행 ──
box(2, 46, 28, 48, "#f3f8fd", BLUE, 1.2)
ax.text(16, 88.3, "한국은행", fontsize=10.5, ha="center", color=BLUE, fontweight="bold")
ax.text(16, 84.0, "디지털혁신실", fontsize=8.3, ha="center", color=GREY)
for i, t in enumerate(["· 실시간 빈티지 데이터셋", "· 전처리·평가 코드",
                       "· 채점 기준 스스로 개정", "· 실험 코드 3.7만 줄",
                       "· 당사 리포트를 저장소에 수록"]):
    ax.text(4, 77.5 - i * 6.3, t, fontsize=8.0, color=INK)

# ── 우: 네이버클라우드 ──
box(70, 46, 28, 48, "#f0f8f2", GREEN, 1.2)
ax.text(84, 88.3, "네이버클라우드", fontsize=10.5, ha="center", color=GREEN, fontweight="bold")
ax.text(84, 84.0, "AX Forward Lab", fontsize=8.3, ha="center", color=GREY)
for i, t in enumerate(["· 실험 스크립트 50여 종", "· 40여 개 모형 전면 재검증",
                       "· 분석 리포트 6건 (PDF)", "· 적응학습(LoRA) 첫 실증",
                       "· 커밋 90여 건 (6~9월)"]):
    ax.text(72, 77.5 - i * 6.3, t, fontsize=8.0, color=INK)

# ── 중앙: 저장소 ──
box(36, 55, 28, 30, "#f0f0ee", "#9aa0a6", 1.2)
ax.text(50, 77.8, "GitHub 저장소", fontsize=11, ha="center", color=INK, fontweight="bold")
ax.text(50, 71.5, "커밋이 곧 회의록이자", fontsize=8.3, ha="center", color=GREY)
ax.text(50, 67.8, "요구사항 문서", fontsize=8.3, ha="center", color=GREY)
ax.text(50, 61.5, "데이터·코드·결과가 모두", fontsize=8.3, ha="center", color=INK)
ax.text(50, 57.8, "기록으로 남음", fontsize=8.3, ha="center", color=INK)

# ── 양방향 화살표 ──
for x0, x1 in [(30.8, 35.2), (64.8, 69.2)]:
    ax.annotate("", xy=(x1, 66), xytext=(x0, 66),
                arrowprops=dict(arrowstyle="<|-|>", color=GREY, lw=1.4))
for xc in (33, 67):
    ax.text(xc, 70.5, "커밋", fontsize=7.6, ha="center", color=GREY)
    ax.text(xc, 60.3, "리뷰", fontsize=7.6, ha="center", color=GREY)

# ── 하단: 상호 재현 검증 ──
box(2, 4, 96, 34, "#eef5f0", GREEN, 1.0)
ax.text(50, 31.5, "상호 재현 검증 — 신뢰의 근거가 사람이 아니라 기록", fontsize=10, ha="center", color=GREEN, fontweight="bold")
ax.text(50, 24.0, "한은이 개정한 채점 결과를 당사가 재계산해 일치 확인 — 같은 데이터, 같은 채점표", fontsize=8.6, ha="center", color=INK)
ax.text(50, 17.0, "당사가 제안한 \"분기 초 6주 교대\" 구성을 한은이 자체 코드로 재현", fontsize=8.6, ha="center", color=INK)
ax.text(50, 10.0, "실패한 실험도 기각 근거와 함께 커밋 — 재현 가능한 기록으로 남김", fontsize=8.4, ha="center", color=GREY)

for x in [16, 84]:
    ax.annotate("", xy=(x, 44.5), xytext=(x, 39),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.2))

fig.savefig(f"{OUT}/fde_collab.png", bbox_inches="tight", facecolor="white")
print("saved: fde_collab.png")
