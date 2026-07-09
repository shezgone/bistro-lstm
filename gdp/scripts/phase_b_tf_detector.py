"""Transformer(어텐션) 기반 충격 탐지기 실험.
- 입력: DFM 보정 월별 패널 (최근 L개월 × 변수)  [q의 첫 빈티지 CSV]
- 라벨: 라우팅 정답 = 그 분기에 DFM+XGB가 DFM+RF보다 나으면 1(shock), 아니면 0(calm)
- 학습: walk-forward (과거 분기 라벨만) → 현재 분기 국면 예측 → 게이트
- 비교: 휴리스틱 게이트(0.7548) / DFM+XGBoost(0.765)
"""
import os, sys, warnings, glob
warnings.filterwarnings("ignore")
sys.path.insert(0,"."); sys.path.insert(0,"/Users/user/vibe/bistro-lstm")
import numpy as np, pandas as pd, torch, torch.nn as nn
from core.lstm_model import AttentionLSTMForecaster
import phase_b_harness as H

COVID=[str(p) for p in pd.period_range("2020Q1","2022Q4",freq="Q")]
RECENT=[str(p) for p in pd.period_range("2023Q1","2025Q4",freq="Q")]
L=24; DFM_ROOT=H.DFM_ROOT

grid,refdf=H.load_grid(); KEY=["tq","vintage","week_idx"]
def norm(d): d=d.copy(); d["vintage"]=pd.to_datetime(d["vintage"]).dt.strftime("%Y-%m-%d"); return d
b=norm(H.load_baseline(grid,"dfm"))[KEY+["y_pred","flash"]].rename(columns={"y_pred":"dfm"})
for m in ["xgboost","rf"]:
    b=b.merge(norm(refdf[refdf.model_name==m])[KEY+["y_pred"]].rename(columns={"y_pred":m}),on=KEY,how="left")
b=b.dropna().reset_index(drop=True)
b["e_xgb"]=((b.dfm+b.xgboost)/2-b.flash)**2
b["e_rf"] =((b.dfm+b.rf)/2 -b.flash)**2

# 분기별 라우팅 정답 (오라클 라벨): DFM+XGB가 더 나으면 1
q_err=b.groupby("tq").agg(exgb=("e_xgb","mean"), erf=("e_rf","mean")).reset_index()
q_err["label"]=(q_err.exgb<q_err.erf).astype(int)
labels=dict(zip(q_err.tq,q_err.label))
targets=sorted(b.tq.unique(), key=lambda x: pd.Period(x,"Q"))
print(f"라우팅 정답 분포: shock(DFM+XGB우세)={sum(labels.values())} / calm={len(labels)-sum(labels.values())}")

# 분기별 패널 피처 (q의 첫 빈티지 CSV, 최근 L개월, N_gdp 제외)
def q_first_vintage(q):
    v=b[b.tq==q].vintage
    return sorted(v.unique())[0]
def panel_feat(q):
    csv=DFM_ROOT/q/f"{q_first_vintage(q)}.csv"
    if not csv.exists():
        c=sorted(glob.glob(str(DFM_ROOT/q/'*.csv')))
        if not c: return None
        csv=c[0]
    df=pd.read_csv(csv); df["Time"]=pd.to_datetime(df["Time"]); df=df.set_index("Time").sort_index()
    cols=[c for c in df.columns if c!="N_gdp"]
    sub=df[cols].tail(L)
    if len(sub)<L: return None
    return sub.to_numpy(np.float32)
feats={q:panel_feat(q) for q in targets}

def train_detect(q):
    past=[qq for qq in targets if pd.Period(qq,"Q")<pd.Period(q,"Q") and feats[qq] is not None]
    if len(past)<6 or feats[q] is None:
        return 1  # 정보부족 → 보수적 shock(DFM+XGB), 휴리스틱과 동일
    X=np.stack([feats[qq] for qq in past]); y=np.array([labels[qq] for qq in past],np.float32)
    if y.sum()==0 or y.sum()==len(y):  # 한 클래스만 → 그 클래스로
        return int(y[0])
    mu=X.reshape(-1,X.shape[-1]).mean(0); sd=X.reshape(-1,X.shape[-1]).std(0)+1e-6
    Xn=(X-mu)/sd; Xt=((feats[q]-mu)/sd)[None]
    torch.manual_seed(0)
    m=AttentionLSTMForecaster(n_vars=X.shape[-1], d_model=16, hidden_dim=32, n_layers=1,
                              n_heads=2, pred_len=1, dropout=0.3)
    opt=torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-3)
    xt=torch.tensor(Xn); yt=torch.tensor(y).view(-1,1)
    pos=float(y.mean()); w=torch.tensor([(1-pos)/max(pos,1e-3)])  # 불균형 보정
    m.train()
    for _ in range(120):
        opt.zero_grad(); logit=m(xt)["mu"]
        loss=nn.functional.binary_cross_entropy_with_logits(logit,yt,pos_weight=w)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p=torch.sigmoid(m(torch.tensor(Xt))["mu"]).item()
    return int(p>0.5)

pred_shock={q:train_detect(q) for q in targets}
nshock=sum(pred_shock.values())
print(f"TF 탐지기 shock 판정: {nshock}/{len(targets)}")
# 정답 대비 정확도
acc=np.mean([pred_shock[q]==labels[q] for q in targets])
print(f"라우팅 라벨 정확도(참고, 사후): {acc:.2%}")

b["gated_tf"]=np.where(b.tq.map(pred_shock),(b.dfm+b.xgboost)/2,(b.dfm+b.rf)/2)
def rmse(col,sub=None):
    d=b if sub is None else b[b.tq.isin(sub)]
    t=pd.DataFrame({"model_name":"c","tq":d.tq,"vintage":d.vintage,"week_idx":d.week_idx,"flash":d.flash,"y_pred":d[col]})
    s=H.score(t); return float(s.iloc[0]) if len(s) else np.nan
b["dfm_xgb"]=(b.dfm+b.xgboost)/2
print("\n=== 결과 (flash w[-19,-1] avg RMSE) ===")
print(f"  TF-gated       전체 {rmse('gated_tf'):.4f} | COVID {rmse('gated_tf',COVID):.4f} | 최근 {rmse('gated_tf',RECENT):.4f}")
print(f"  [비교] 휴리스틱-gated 0.7548 | DFM+XGBoost {rmse('dfm_xgb'):.4f} | 사후상한 0.7607")
print("TF_DETECTOR_DONE")
