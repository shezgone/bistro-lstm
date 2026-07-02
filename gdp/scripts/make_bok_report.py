"""한은 GDP Nowcasting 중간결과 보고서 PDF (2p, A4, 정직 톤)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager as fm

KF="/System/Library/Fonts/AppleSDGothicNeo.ttc"
fm.fontManager.addfont(KF); KO=fm.FontProperties(fname=KF).get_name()
plt.rcParams["font.family"]=KO; plt.rcParams["axes.unicode_minus"]=False
NAVY="#1f3a5f"; BLUE="#2e6da4"; ACC="#c0392b"; GRAY="#666"; LIGHT="#eef2f6"; GREEN="#2e7d32"
W,H=8.27,11.69

def page():
    fig=plt.figure(figsize=(W,H),dpi=150); fig.patch.set_facecolor("white")
    ax=fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,100)
    return fig,ax
def head(ax,title,sub=None):
    ax.add_patch(plt.Rectangle((0,94),100,6,color=NAVY))
    ax.text(6,96.7,title,color="white",fontsize=13,fontweight="bold",va="center")
    if sub: ax.text(94,96.7,sub,color="#c9d8e8",fontsize=8.5,va="center",ha="right")
def h2(ax,y,t): ax.text(7,y,t,fontsize=12.5,fontweight="bold",color=NAVY); return y-1.2
def body(ax,y,t,x=8,fs=9.6,color="#222"): ax.text(x,y,t,fontsize=fs,color=color,va="top");

pages=[]
# ===== page 1 =====
fig,ax=page()
ax.add_patch(plt.Rectangle((0,94),100,6,color=NAVY))
ax.text(6,97,"GDP Nowcasting 시스템 고도화 — AI/ML 협업 중간 결과",color="white",fontsize=12.5,fontweight="bold",va="center")
ax.text(7,89,"네이버클라우드 × 한국은행 데이터사이언스팀",fontsize=10.5,color=GRAY)
ax.text(7,86.5,"작성일 2026-07-02  |  평가: 속보치(flash) 기준, 전망주차 w[-19,-1] 평균 RMSE, 2018Q1–2025Q4(32분기)",fontsize=8.3,color=GRAY)
ax.plot([7,93],[84.5,84.5],color=GRAY,lw=0.6,alpha=0.5)

y=81
y=h2(ax,y,"1. 요약")
for t in [
 "• 기존 DFM/LSTM 전망체계 및 AI/ML 후보군을 동일 실시간 빈티지 검증체계에서 재현·확장 평가하였습니다.",
 "• 재현 결과 DFM(0.865)·DFM+XGBoost(0.765) 등 기존 성과를 소수점 수준까지 일치시켰습니다.",
 "• 국면전환(regime-gated) 앙상블로 기존 최고 DFM+XGBoost(0.765)를 개선한 0.7548을 확인했습니다.",
 "• 최신 딥러닝(Transformer·시계열 파운데이션 모델)은 소표본 특성상 정확도 개선 효과가 확인되지 않았으며,",
 "   설명가능성(어텐션 기반 해석) 역할로의 활용을 제안합니다.",
]:
    ax.text(8,y-0.3,t,fontsize=9.4,color="#222",va="top"); y-=2.5

y-=1.2
y=h2(ax,y,"2. 핵심 결과 (평균 RMSE, 낮을수록 우수)")
rows=[("모형","전체 32분기","COVID(20–22)","최근(23–25)"),
 ("★ regime-gated (제안)","0.7548","0.9917","0.5198"),
 ("DFM+XGBoost (기존 최고)","0.765","0.999","0.542"),
 ("DFM (기준선)","0.865","1.255","0.541"),
 ("XGBoost / RF (단독)","0.937 / —","—","—"),
 ("Transformer·Foundation (단독)","1.02–1.45","열위","열위")]
yt=y-0.6; rh=2.3
for i,r in enumerate(rows):
    yy=yt-i*rh
    if i==0: ax.add_patch(plt.Rectangle((7,yy-1.5),86,rh,color=NAVY)); c="white"; fw="bold"
    elif i==1: ax.add_patch(plt.Rectangle((7,yy-1.5),86,rh,color="#e8f0e8")); c=GREEN; fw="bold"
    else: c="#222"; fw="normal"
    xs=[9,42,62,80]
    for xi,cell in zip(xs,r):
        ax.text(xi,yy,cell,fontsize=8.8,color=c,fontweight=fw,va="center")
y=yt-len(rows)*rh-1.5
ax.add_patch(plt.Rectangle((7,y-6.5),86,6.3,fc="#e8f0e8",ec=GREEN,lw=1))
ax.text(9,y-1.3,"핵심: 국면전환 앙상블이 기존 최고 대비 개선 + 두 국면 모두 우위",fontsize=9.6,fontweight="bold",color=GREEN)
ax.text(9,y-3.3,"실시간 충격탐지기가 국면을 판정 → shock: DFM+XGBoost / 평온: DFM+RF 로 전환.",fontsize=9,color="#222")
ax.text(9,y-5.2,"변동성 윈도우 K∈{3–8} 전부 0.765 하회(0.7548–0.7594), 견고성 확인.",fontsize=9,color="#222")
ax.text(50,3,"1 / 2",fontsize=8,color=GRAY,ha="center")
pages.append(fig)

# ===== page 2 =====
fig,ax=page(); head(ax,"GDP Nowcasting 중간 결과","2026-07-02")
y=88
y=h2(ax,y,"3. 방법")
for t in ["• 데이터: 제공된 실시간 빈티지 + DFM/LSTM 운영 산출물. 모든 모형 동일 목표분기·빈티지·전망주차에서 비교.",
          "• 검증: walk-forward(정보누수 방지), 평가 target = 속보치(flash), 후보모형은 DFM 보정 월별 데이터를 공통입력.",
          "• AI/ML 후보를 우리 환경에서 재학습해 기존 결과를 재현(신뢰성 확인) 후 확장 실험 진행."]:
    ax.text(8,y-0.3,t,fontsize=9.3,color="#222",va="top"); y-=2.4
y-=1.2

y=h2(ax,y,"4. 주요 발견 (정직 보고)")
findings=[("① 재현 성공","기존 DFM·DFM+XGBoost 성과를 소수점 수준으로 재현 — 검증체계 신뢰성 확인."),
 ("② 국면전환 앙상블","기존 최고(0.765)를 0.7548로 개선. COVID·최근 양 국면 모두 우위, 하이퍼파라미터에 견고."),
 ("③ 최신 딥러닝의 한계","Transformer·파운데이션 모델은 소표본(분기 GDP)에서 정확도 개선 없음 — 예측·탐지 양쪽 확인."),
 ("④ 시사점","정확도 기반은 DFM+트리로 견고. 최신기술은 '설명가능성(어텐션 해석)·국면 진단' 역할이 적합.")]
for tt,dd in findings:
    ax.text(8,y-0.3,tt,fontsize=9.8,fontweight="bold",color=BLUE,va="top")
    ax.text(8,y-2.0,dd,fontsize=9.1,color="#222",va="top"); y-=5.0

y-=0.5
y=h2(ax,y,"5. 다음 단계 (제안)")
for t in ["• Diebold–Mariano 검정으로 개선의 통계적 유의성 점검 (소표본 한계 함께 명시).",
          "• 충격탐지기 고도화: DFM 뉴스분해·월별 지표 기반 조기 감지.",
          "• 설명가능성 레이어(어텐션 기반 변수 기여도)로 전망 근거 제시 — 최신기술의 실질 활용처.",
          "• 산업모형 기반 생산접근 nowcasting 결합 검토."]:
    ax.text(8,y-0.3,t,fontsize=9.3,color="#222",va="top"); y-=2.4

y-=1.5
ax.add_patch(plt.Rectangle((7,y-5.5),86,5.2,fc=LIGHT,ec=LIGHT))
ax.text(9,y-1.3,"종합",fontsize=10,fontweight="bold",color=NAVY)
ax.text(9,y-3.4,"'국면전환 앙상블'로 정확도를 개선하고, Transformer는 설명가능성·국면진단에 투입하는 구성을 제안합니다.\n소표본 환경에서는 단순·견고한 방법이 우수하다는 점이 예측·탐지 실험 전반에서 일관되게 확인되었습니다.",
        fontsize=9,color="#222",va="top")
ax.text(50,3,"2 / 2",fontsize=8,color=GRAY,ha="center")
pages.append(fig)

OUT=Path("/Users/user_1/vibe/bistro-lstm/docs/GDP_Nowcasting_중간보고_2026-07-02.pdf")
with PdfPages(OUT) as pdf:
    for f in pages: pdf.savefig(f,facecolor="white"); plt.close(f)
print(f"OK: {OUT} ({len(pages)}p)")
