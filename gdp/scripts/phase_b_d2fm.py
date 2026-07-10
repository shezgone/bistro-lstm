"""T3: D²FM-lite (Andreini–Izzo–Ricco, arXiv:2007.11887 축소판) 공정 주입.

원 논문: 오토인코더로 비선형 요인 추출 + 요인 동역학 + 상태공간 (US 실시간에서 DFM 상회).
축소판: 월별 패널 → 잡음주입 오토인코더 요인 k개 → 요인 lag 시퀀스 + AR 동역학 →
        선형 readout으로 월별화 N_gdp 예측. XGBoost와 동일한 quarter_batch 하네스 주입.
(상태공간 칼만 대신 요인 lag 회귀 — 완전판 대비 근사임을 명시)
"""
import os, sys, re, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"src"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin

import gdp_nowcasting.ml_models as MM
import gdp_nowcasting.hyperparameter_tuning as HT
import gdp_nowcasting.pipeline_ai_ml as PL
from gdp_nowcasting.ml_models import SklearnNowcastModel, ModelSpec
from gdp_nowcasting.pipeline_ai_ml import AIRunContext, BacktestConfig, run_tree_ensemble_backtest
from gdp_nowcasting.ml_preprocessing import TabularFeatureConfig
from gdp_nowcasting.hyperparameter_tuning import TuningConfig
import phase_b_harness as H

LAGRE = re.compile(r"^(.+)__lag(\d+)$")

class D2FMLite(BaseEstimator, RegressorMixin):
    """잡음주입 AE 요인 + 요인 lag 회귀 readout."""
    def __init__(self, k=4, hidden=32, ae_epochs=300, head_epochs=200, seeds=3, noise=0.1):
        self.k=k; self.hidden=hidden; self.ae_epochs=ae_epochs
        self.head_epochs=head_epochs; self.seeds=seeds; self.noise=noise
    def _reshape(self, Xdf):
        n=len(Xdf); arr=np.zeros((n,len(self.lags_),len(self.vars_)),np.float32)
        for vi,v in enumerate(self.vars_):
            for li,lg in enumerate(self.lags_):
                c=f"{v}__lag{lg:02d}"
                if c in Xdf.columns: arr[:,li,vi]=Xdf[c].to_numpy(np.float32)
        return np.nan_to_num(arr)
    def fit(self, X, y):
        Xdf=X if isinstance(X,pd.DataFrame) else pd.DataFrame(X)
        parsed=[(m.group(1),int(m.group(2))) for c in Xdf.columns for m in [LAGRE.match(c)] if m]
        self.vars_=sorted(set(v for v,_ in parsed)); self.lags_=sorted(set(l for _,l in parsed))
        A=self._reshape(Xdf); n,L,V=A.shape
        flat=A.reshape(-1,V); self.mu_=flat.mean(0); self.sd_=flat.std(0)+1e-6
        Z=((A-self.mu_)/self.sd_).astype(np.float32)
        y=np.asarray(y,np.float32); self.ym_=float(y.mean()); self.ys_=float(y.std()+1e-6)
        yn=((y-self.ym_)/self.ys_).reshape(-1,1)
        self.models_=[]
        for s in range(self.seeds):
            torch.manual_seed(s)
            enc=nn.Sequential(nn.Linear(V,self.hidden),nn.Tanh(),nn.Linear(self.hidden,self.k))
            dec=nn.Sequential(nn.Linear(self.k,self.hidden),nn.Tanh(),nn.Linear(self.hidden,V))
            opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=1e-3)
            Xm=torch.tensor(Z.reshape(-1,V))
            for _ in range(self.ae_epochs):        # denoising AE (요인 추출)
                opt.zero_grad()
                xin=Xm+self.noise*torch.randn_like(Xm)
                loss=nn.functional.mse_loss(dec(enc(xin)),Xm); loss.backward(); opt.step()
            with torch.no_grad():
                F=enc(torch.tensor(Z.reshape(-1,V))).reshape(n,L,self.k)   # 요인 lag 경로
            feats=torch.cat([F.reshape(n,-1),F[:,-1,:]-F[:,0,:]],dim=1)   # lag 요인 + 추세
            head=nn.Linear(feats.shape[1],1)
            opt2=torch.optim.Adam(head.parameters(),lr=1e-2,weight_decay=1e-2)
            yt=torch.tensor(yn)
            for _ in range(self.head_epochs):
                opt2.zero_grad(); loss=nn.functional.mse_loss(head(feats),yt); loss.backward(); opt2.step()
            self.models_.append((enc,head))
        return self
    def predict(self, X):
        Xdf=X if isinstance(X,pd.DataFrame) else pd.DataFrame(X)
        A=self._reshape(Xdf); n,L,V=A.shape
        Z=torch.tensor(((A-self.mu_)/self.sd_).astype(np.float32))
        outs=[]
        with torch.no_grad():
            for enc,head in self.models_:
                F=enc(Z.reshape(-1,V)).reshape(n,L,-1)
                feats=torch.cat([F.reshape(n,-1),F[:,-1,:]-F[:,0,:]],dim=1)
                outs.append(head(feats).numpy().ravel())
        return np.mean(outs,axis=0)*self.ys_+self.ym_

_orig=MM.create_model
def cm(model_name, model_params=None):
    spec=model_name if isinstance(model_name,ModelSpec) else None
    name=(spec.model_name if spec else model_name).lower()
    if name=="our_d2fm":
        params=dict(spec.model_params) if spec else dict(model_params or {})
        return SklearnNowcastModel(model_name="our_d2fm", estimator=D2FMLite(**params))
    return _orig(model_name, model_params)
MM.create_model=cm; PL.create_model=cm
_og=HT.get_default_parameter_grid
HT.get_default_parameter_grid=lambda n: {} if str(n).lower()=="our_d2fm" else _og(n)

repo=os.getcwd(); os.environ["GDP_NOWCAST_USE_S3"]="0"; os.environ["GDP_NOWCAST_LOCAL_PATH"]=repo
ctx=AIRunContext(specfile=str(Path(repo)/"data/meta/Spec_kim_34var_transformchange_named.xlsx"),
    input_data=str(Path(repo)/"data/vintages"), repo_root=Path(repo),
    forecasting_week="2026-01-23", sample_yrs=12, dfm_suffix="11")
fc=TabularFeatureConfig(lookback_months=13, add_missing_mask=False, add_release_lag=False,
                        rolling_windows=(), delta_lags=())
tuning=TuningConfig(min_train_size=8, max_candidates=10)
QSUB=os.environ.get("QSUB"); SEEDS=int(os.environ.get("SEEDS","3"))
quarters=QSUB.split(",") if QSUB else [str(p) for p in pd.period_range("2018Q1","2025Q4",freq="Q")]
print(f"[run] our_d2fm (denoising-AE k=4 + factor-lag head) quarters={len(quarters)} seeds={SEEDS}",flush=True)
res=run_tree_ensemble_backtest(ctx, feature_config=fc, linear_specs=[],
    tree_specs=[ModelSpec("our_d2fm",{"seeds":SEEDS},model_id="our_d2fm")],
    backtest_config=BacktestConfig(tuning_mode="quarter_batch", quarter_batch_selection_mode="fvintage_window",
    min_train_rows=8, verbose=True), tuning_config=tuning,
    dfm_snapshot_root=Path(repo)/"output/model/DFM/11", gdp_release_file=str(Path(repo)/"data/GDP_releases.pkl"),
    quarters=quarters, news_config=None)
res.predictions.to_csv("output/csv/_phase_b_d2fm_predictions.csv",index=False)
sc=H.score(res.predictions)
print("\n=== T3 D2FM-lite (공정 주입) flash w[-19,-1] avg RMSE ===")
print(sc[sc.index.str.contains("d2fm")].to_string())
print("[기준] DFM 0.865 / DFM+XGB 0.765 / DFM+TabPFN 0.815")
print("PHASE_B_D2FM_DONE")
