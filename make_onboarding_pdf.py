"""상사용 온보딩 PDF 생성 — BISTRO 프로젝트 개념 / 한국은행 협업목표 / 작업내역 / 로드맵.
슬라이드(16:9) 형식, matplotlib PdfPages, 한글 폰트(AppleSDGothicNeo).
실행: .venv/bin/python make_onboarding_pdf.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager as fm

# ---- 한글 폰트 등록 ----
KFONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
fm.fontManager.addfont(KFONT)
KO = fm.FontProperties(fname=KFONT).get_name()
plt.rcParams["font.family"] = KO
plt.rcParams["axes.unicode_minus"] = False

# ---- 팔레트 ----
NAVY = "#1f3a5f"
BLUE = "#2e6da4"
ACCENT = "#e8743b"     # 강조(ours / 핵심)
GREEN = "#3c9a5f"
GRAY = "#8a929b"
LIGHT = "#eef2f6"
RED = "#c0392b"

W, H = 13.33, 7.5  # 16:9


def new_slide():
    fig = plt.figure(figsize=(W, H), dpi=150)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    return fig, ax


def header(ax, kicker, title):
    ax.add_patch(plt.Rectangle((0, 92), 100, 8, color=NAVY, zorder=1))
    ax.text(4, 96, kicker, color="#a9c4e0", fontsize=11, va="center", fontweight="bold")
    ax.text(4, 86, title, color=NAVY, fontsize=24, va="center", fontweight="bold")
    ax.plot([4, 96], [82, 82], color=GRAY, lw=0.8, alpha=0.5)


def footer(ax, page):
    ax.text(96, 3, f"BISTRO 온보딩 · {page}", color=GRAY, fontsize=8.5, ha="right")
    ax.text(4, 3, "한국은행 × NAVER · 2026-06", color=GRAY, fontsize=8.5)


def box(ax, x, y, w, h, fc, ec=None, alpha=1.0, rad=0.02):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={rad*100}",
                       fc=fc, ec=ec or fc, lw=1.5, alpha=alpha, zorder=2,
                       mutation_aspect=H / W)
    ax.add_patch(p)


def arrow(ax, x1, y1, x2, y2, color=NAVY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=22, lw=2.2, color=color, zorder=3))


pages = []

# ============================================================ 1. 표지
fig, ax = new_slide()
ax.add_patch(plt.Rectangle((0, 0), 100, 100, color=NAVY))
ax.add_patch(plt.Rectangle((0, 0), 100, 38, color="#16293f"))
ax.text(8, 70, "BISTRO", color="white", fontsize=66, fontweight="bold")
ax.text(8, 60, "한국은행 거시지표 예측 모델 — 온보딩 자료", color="#d8e6f2", fontsize=22)
ax.text(8, 52, "Task-Specific 소형 모델 vs Foundation Model vs LLM In-Context",
        color="#a9c4e0", fontsize=14)
ax.add_patch(plt.Rectangle((8, 30), 40, 0.6, color=ACCENT))
ax.text(8, 22, "작은 과제특화 모델이 거대 모델과 어떻게 경쟁하는가,", color="white", fontsize=13)
ax.text(8, 17, "그리고 '언제 LLM을 호출할 가치가 있는가'를 정량화한다.", color="white", fontsize=13)
ax.text(8, 7, "한국은행 × NAVER 협업  |  작성: kim.yongmin@navercorp.com  |  2026-06-17",
        color="#8fa9c4", fontsize=11)
footer(ax, "표지"); pages.append(fig)

# ============================================================ 2. 개념
fig, ax = new_slide()
header(ax, "CONCEPT — 무엇을 하는 프로젝트인가", "세 가지 예측 접근의 비교")
ax.text(4, 76, "거시지표(물가·성장률)를 예측하는 데는 성격이 다른 세 갈래가 있다. 이 프로젝트는 셋을 같은 잣대로 비교한다.",
        fontsize=12.5, color="#333")

cards = [
    ("Task-Specific\n소형 모델", "우리가 직접 학습\n(LSTM·Transformer)", "170K~327K 파라미터\n과제에 특화", BLUE),
    ("Foundation\nModel", "사전학습 거대 모델\n(Chronos·Moirai)", "91M~200M 파라미터\n학습 없이 zero-shot", GRAY),
    ("LLM\nIn-Context", "거대언어모델이 예측\n(HCX-32B·Claude)", "패널을 표로 입력\n추론으로 예측", ACCENT),
]
for i, (t, s, d, c) in enumerate(cards):
    x = 6 + i * 31
    box(ax, x, 26, 26, 40, "white", ec=c)
    ax.add_patch(plt.Rectangle((x + 1.2, 60), 23.6, 5.5, color=c, alpha=0.95, zorder=4))
    ax.text(x + 13, 62.5, t, color="white", fontsize=14.5, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(x + 13, 50, s, fontsize=12, ha="center", va="center", color="#222")
    ax.text(x + 13, 36, d, fontsize=10.5, ha="center", va="center", color=GRAY)

box(ax, 6, 9, 88, 12, LIGHT, ec=LIGHT)
ax.text(9, 15, "핵심 질문", fontsize=12.5, fontweight="bold", color=NAVY)
ax.text(9, 11.5, "① 작은 과제특화 모델이 1000배 큰 거대 모델을 이길 수 있는가?   "
                 "② 거대 모델·LLM은 '언제' 호출할 가치가 있는가?", fontsize=11.5, color="#333")
footer(ax, "2 / 개념"); pages.append(fig)

# ============================================================ 3. 한국은행 협업 목표
fig, ax = new_slide()
header(ax, "GOAL — 한국은행과의 협업 목표", "왜 이 비교가 중요한가")
goals = [
    ("예측 품질", "OOS(표본외) 구간에서 신뢰할 수 있는 거시지표 예측 정확도 확보", GREEN),
    ("운영 비용 구조", "매 예측마다 거대 모델을 돌리는 대신, 필요한 순간에만 호출 → 비용 절감", BLUE),
    ("신뢰성 · 설명가능성", "예측 근거를 경제적 서사로 제시 (어떤 변수가, 얼마나, 어떤 경로로)", ACCENT),
    ("증분 유지보수", "지표 개정 시 전체 재학습 없이 변경분만 갱신", NAVY),
]
for i, (t, d, c) in enumerate(goals):
    y = 64 - i * 13
    ax.add_patch(plt.Rectangle((6, y), 1.4, 9, color=c))
    ax.text(10, y + 6.2, t, fontsize=15, fontweight="bold", color=c)
    ax.text(10, y + 1.8, d, fontsize=12, color="#333")

box(ax, 70, 24, 26, 50, LIGHT, ec=BLUE)
ax.text(83, 70, "방향 전환", fontsize=13, fontweight="bold", color=RED, ha="center")
ax.text(83, 64, "2026-06\n한국은행 요청", fontsize=11.5, ha="center", color="#333")
arrow(ax, 83, 60, 83, 52, color=RED)
ax.text(83, 48, "CPI 예측", fontsize=15, fontweight="bold", ha="center", color=GRAY)
ax.text(83, 43, "↓", fontsize=18, ha="center", color=RED)
ax.text(83, 37, "GDP 예측", fontsize=17, fontweight="bold", ha="center", color=NAVY)
ax.text(83, 29, "물가 → 성장률\n(상세 로드맵 후반)", fontsize=10.5, ha="center", color=GRAY)
footer(ax, "3 / 목표"); pages.append(fig)

# ============================================================ 4. 작업내역 + Key Results
fig, ax = new_slide()
header(ax, "WORK — 지금까지 한 일", "결과 ①  2023·2024 표본외(OOS) 예측 정확도")
ax.text(4, 76, "작은 과제특화 Transformer(170K)가 91~200M 거대 모델을 평균 정확도에서 앞섰다. (낮을수록 좋음, RMSE)",
        fontsize=12.5, color="#333")

axc = fig.add_axes([0.30, 0.12, 0.62, 0.52])
models = ["Transformer\n170K (Ours)", "LSTM\n327K (Ours)", "Chronos-2\n120M", "Moirai\n91M"]
vals = [0.556, 0.703, 0.932, 0.986]
colors = [ACCENT, BLUE, GRAY, GRAY]
bars = axc.bar(models, vals, color=colors, width=0.6)
axc.set_ylabel("평균 RMSE (2023+2024)", fontsize=11)
axc.set_ylim(0, 1.1)
for b, v in zip(bars, vals):
    axc.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
axc.spines[["top", "right"]].set_visible(False)
axc.tick_params(labelsize=10)

ax.text(4, 64, "한 일 요약", fontsize=12.5, fontweight="bold", color=NAVY)
for i, t in enumerate([
    "• 과제특화 모델 학습\n  (Attention-LSTM /\n   Transformer)",
    "• Foundation Model\n  zero-shot 평가\n  (Chronos·Moirai)",
    "• LLM in-context 예측\n  (HCX·Claude) +\n  다수 ablation",
    "• 경제 서사 분석\n  (인과·시차·경로)",
]):
    ax.text(4, 58 - i * 11, t, fontsize=10.8, color="#333", va="top")
footer(ax, "4 / 작업내역"); pages.append(fig)

# ============================================================ 5. 핵심 발견: Regime-gated
fig, ax = new_slide()
header(ax, "KEY INSIGHT — 핵심 발견", "결과 ②  'LLM은 충격 구간에서만 가치가 있다'")
ax.text(4, 76, "평온한 구간에선 단순 baseline(Trend12)이 더 정확하고, LLM(HCX)은 오히려 손해. "
               "그러나 충격(shock) 구간에선 HCX가 baseline을 확실히 앞선다.", fontsize=12.5, color="#333")

axc = fig.add_axes([0.08, 0.13, 0.5, 0.52])
groups = ["평온 구간\n(calm)", "충격 구간\n(shock)"]
hcx = [0.293, 0.441]; tr = [0.248, 0.568]
import numpy as np
x = np.arange(2); bw = 0.34
axc.bar(x - bw / 2, tr, bw, label="Trend12 (baseline)", color=GRAY)
axc.bar(x + bw / 2, hcx, bw, label="HCX (LLM)", color=ACCENT)
for xi, (a, b) in enumerate(zip(tr, hcx)):
    axc.text(xi - bw / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=10.5, fontweight="bold")
    axc.text(xi + bw / 2, b + 0.012, f"{b:.3f}", ha="center", fontsize=10.5, fontweight="bold")
axc.set_xticks(x); axc.set_xticklabels(groups, fontsize=11)
axc.set_ylabel("RMSE (낮을수록 좋음)", fontsize=11); axc.set_ylim(0, 0.65)
axc.legend(fontsize=10, loc="upper left"); axc.spines[["top", "right"]].set_visible(False)

box(ax, 62, 30, 34, 40, LIGHT, ec=NAVY)
ax.text(79, 65, "그래서 — Regime-Gated\nEscalation", fontsize=14, fontweight="bold", color=NAVY, ha="center", va="center")
ax.text(64, 54, "① 평소엔 저렴한\n   Trend12 사용", fontsize=12, color="#333", va="top")
ax.text(64, 45, "② '충격 탐지기'가\n   경보를 울릴 때만\n   LLM으로 escalate", fontsize=12, color=ACCENT, va="top", fontweight="bold")
ax.text(64, 34, "→ 비용↓  정확도↑\n   (탐지기 AUC 0.70)", fontsize=11.5, color=GREEN, va="top")
footer(ax, "5 / 핵심 발견"); pages.append(fig)

# ============================================================ 6. Rolling-2025 비교
fig, ax = new_slide()
header(ax, "WORK — 지금까지 한 일", "결과 ③  Rolling 2025 실시간 1개월 예측 (학습 컷오프 이후)")
ax.text(4, 76, "데이터 오염이 없는 2025년 구간에서 1개월 앞 예측 비교. LLM·거대모델·단순 baseline이 박빙, "
               "과제특화 모델은 이 짧은 구간에선 열위.", fontsize=12.5, color="#333")
axc = fig.add_axes([0.10, 0.14, 0.8, 0.5])
labels = ["HCX\n(CoT+CSI)", "Moirai-small\n+CSI", "Trend12", "Chronos\n-tiny", "LSTM\n(in-domain)", "TFM\n(in-domain)"]
vals = [0.250, 0.254, 0.258, 0.281, 0.484, 0.773]
cols = [ACCENT, BLUE, GREEN, BLUE, GRAY, GRAY]
bars = axc.bar(labels, vals, color=cols, width=0.62)
for b, v in zip(bars, vals):
    axc.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=10.5, fontweight="bold")
axc.set_ylabel("RMSE (rolling 2025, 1-step)", fontsize=11); axc.set_ylim(0, 0.85)
axc.spines[["top", "right"]].set_visible(False); axc.tick_params(labelsize=9.5)
footer(ax, "6 / 작업내역"); pages.append(fig)

# ============================================================ 7. 멀티에이전트 루프
fig, ax = new_slide()
header(ax, "AUTOMATION — 자동 개선 루프", "멀티에이전트 실험 파이프라인 (agents/)")
ax.text(4, 76, "사람이 일일이 실험을 설계하는 대신, AI 에이전트들이 가설→실행→분석을 반복하며 "
               "예측 전략을 스스로 개선한다.", fontsize=12.5, color="#333")
nodes = [
    ("Researcher", "Claude Opus", "다음 실험 가설 제안", BLUE, 16),
    ("Engineer", "HCX 러너", "예측 실행 → 성능 측정", ACCENT, 44),
    ("Analyzer", "Claude Opus", "결과 해석 → 교훈 추출", GREEN, 72),
]
for t, who, d, c, x in nodes:
    box(ax, x, 42, 22, 22, "white", ec=c)
    ax.add_patch(plt.Rectangle((x + 1, 58), 20, 4.5, color=c, zorder=4))
    ax.text(x + 11, 60.2, t, color="white", fontsize=13, fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(x + 11, 52, who, fontsize=11, ha="center", color="#222", fontweight="bold")
    ax.text(x + 11, 46.5, d, fontsize=9.8, ha="center", color=GRAY)
arrow(ax, 38, 53, 44, 53)
arrow(ax, 66, 53, 72, 53)
arrow(ax, 83, 42, 83, 33, color=GRAY)
ax.annotate("", xy=(27, 41), xytext=(83, 31),
            arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2, connectionstyle="arc3,rad=0.25"))
ax.text(55, 27, "N라운드 반복 — 교훈을 experiments.db / cognition.json에 누적",
        fontsize=11, ha="center", color=GRAY, style="italic")
box(ax, 18, 10, 64, 11, LIGHT, ec=LIGHT)
ax.text(50, 15.5, "결과물: 어떤 입력 포맷·few-shot·CSI 조합이 HCX 예측을 개선하는지 자동 탐색",
        fontsize=11.5, ha="center", color=NAVY)
footer(ax, "7 / 자동화"); pages.append(fig)

# ============================================================ 8. 로드맵 CPI→GDP
fig, ax = new_slide()
header(ax, "ROADMAP — 다음 단계", "CPI → GDP 전환 계획")
ax.text(4, 76, "현 코드베이스는 CPI(물가) 기준. 한국은행 요청에 따라 GDP(성장률)로 전환한다. "
               "하네스는 대부분 재사용, 데이터·타깃은 재설계.", fontsize=12.5, color="#333")
steps = [
    ("1. 데이터 레이어", "분기 GDP + 혼합빈도 공변량 수집", "재설계", RED),
    ("2. 공변량 재선정", "GDP 드라이버(투자·소비·순수출·심리)", "재실행", RED),
    ("3. DFM 베이스라인", "중앙은행 표준 nowcasting 기준선 구축", "신규", BLUE),
    ("4. 하네스 이식", "rolling 평가 · FM 비교 · regime-gating", "대부분 재사용", GREEN),
    ("5. LLM in-context", "HCX·Claude 프롬프트를 GDP·분기로 교체", "구조 재사용", GREEN),
]
for i, (t, d, tag, c) in enumerate(steps):
    y = 64 - i * 10.5
    ax.add_patch(plt.Circle((7, y + 2.3), 0.9, color=c, transform=ax.transData))
    ax.text(10, y + 4, t, fontsize=13, fontweight="bold", color=NAVY)
    ax.text(10, y + 0.5, d, fontsize=11, color="#333")
    ax.add_patch(plt.Rectangle((78, y + 0.5), 16, 5, color=c, alpha=0.15))
    ax.text(86, y + 3, tag, fontsize=10.5, ha="center", va="center", color=c, fontweight="bold")
ax.plot([7, 7], [12, 66], color=GRAY, lw=1, alpha=0.4, zorder=0)
box(ax, 6, 7, 88, 6.5, "#fff4ee", ec=ACCENT)
ax.text(8, 10.2, "핵심 제약: GDP는 분기 데이터(~90개)로 CPI 월별(276개)보다 소표본 → "
                "DFM/MIDAS를 베이스라인, 과제특화 모델은 챌린저로 두는 구도가 현실적.",
        fontsize=10.8, color="#333")
footer(ax, "8 / 로드맵"); pages.append(fig)

# ---- 저장 ----
OUT = Path("BISTRO_온보딩_상사용.pdf")
with PdfPages(OUT) as pdf:
    for f in pages:
        pdf.savefig(f, facecolor="white")
        plt.close(f)
print(f"OK: {OUT.resolve()}  ({len(pages)} pages)")
